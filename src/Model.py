"""
完整模型：Qwen2.5-VL + 冻结 SAE + 可训练 FeatureRouter。

每层结构：
    h'     = Qwen原层(h)                          # 冻结
    z      = SAE.encode(h'[image_tokens])         # 冻结
    q_vec  = h'[text_tokens].mean(dim=1)          # 同层 text hidden state 作问题向量
    alpha  = Router(q_vec, z, SAE.decoder.weight) # 可训练
    delta  = SAE.decode(z*alpha) - SAE.decode(z)  # 残差
    h_out  = h' + delta                           # 叠加回去

冻结：Qwen 所有参数 + SAE 所有参数
训练：每层的 FeatureRouter（参数量极小）
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

from src.SAE  import SAE
from src.router import FeatureRouter


class QwenLayerWithSAERouter(nn.Module):
    """
    包装单个 Qwen transformer layer，在其后插入 SAE + Router。
    """

    def __init__(
        self,
        qwen_layer:   nn.Module,
        sae:          SAE,
        router:       FeatureRouter,
        layer_idx:    int,
    ):
        super().__init__()
        self.qwen_layer = qwen_layer
        self.sae        = sae
        self.router     = router
        self.layer_idx  = layer_idx

        # side-channel：由 QwenWithSAERouter.forward() 在调用前设置
        self.vision_pos:     int = 0
        self.num_img_tokens: int = 0
        self.text_positions: Optional[torch.Tensor] = None   # text token indices

    def set_context(
        self,
        vision_pos:     int,
        num_img_tokens: int,
        text_positions: torch.Tensor,   # (num_text_tokens,) text token 的位置
    ):
        self.vision_pos     = vision_pos
        self.num_img_tokens = num_img_tokens
        self.text_positions = text_positions

    def forward(self, hidden_states, *args, **kwargs):
        # ── Qwen 原层前向（冻结）──────────────────────────────
        output = self.qwen_layer(hidden_states, *args, **kwargs)

        if isinstance(output, tuple):
            h_prime = output[0]
        else:
            h_prime = output

        # ── 跳过：没有设置 context 或图像 token 为空 ──────────
        if self.num_img_tokens == 0 or self.text_positions is None:
            return output

        vp  = self.vision_pos
        nit = self.num_img_tokens

        # ── 取 image token hidden state ────────────────────────
        img_h = h_prime[:, vp : vp + nit, :]           # (1, nit, hidden_dim)
        flat  = img_h.reshape(-1, img_h.shape[-1]).float()   # (nit, hidden_dim)

        # ── 取同层 text token hidden state 作为问题向量 ────────
        # 直接用当前层的 text hidden state，天然与 image token 语义对齐
        text_h = h_prime[:, self.text_positions, :]    # (1, n_text, hidden_dim)
        q_vec  = text_h.float().mean(dim=1)            # (1, hidden_dim) 均值池化

        # ── SAE 编码（冻结，no_grad）───────────────────────────
        with torch.no_grad():
            z_original     = self.sae.encode(flat)           # (nit, latent_dim)
            recon_original = self.sae.decoder(z_original)    # (nit, hidden_dim)
            # decoder weight: (hidden_dim, latent_dim)
            dec_weight = self.sae.decoder.weight             # (hidden_dim, latent_dim)

        # ── Router（可训练）────────────────────────────────────
        alpha = self.router(q_vec, z_original, dec_weight)   # (nit, latent_dim)

        # ── 增强后的重建 ───────────────────────────────────────
        z_steered     = z_original * alpha
        recon_steered = self.sae.decoder(z_steered)          # (nit, hidden_dim)

        # ── 残差叠加（只有 delta 参与梯度）────────────────────
        delta = (recon_steered - recon_original).to(h_prime.dtype)
        h_out = h_prime.clone()
        h_out[:, vp : vp + nit, :] += delta

        if isinstance(output, tuple):
            return (h_out,) + output[1:]
        return h_out


class QwenWithSAERouter(nn.Module):
    """
    完整模型：Qwen2.5-VL + 多层冻结 SAE + 多层可训练 Router。

    Args:
        base_model:       原始 Qwen2_5_VLForConditionalGeneration
        layers:           插入层索引，如 [4, 8, 16, 24]
        sae_ckpt_dir:     预训练 SAE 权重目录
        latent_mult:      SAE latent 倍数
        topk:             SAE top-k
        topk_route:       Router 每次增强的最大 feature 数
        max_alpha:        Router 最大增强倍数
    """

    def __init__(
        self,
        base_model,
        layers:        List[int],
        sae_ckpt_dir:  str,
        latent_mult:   int   = 16,
        topk:          int   = 32,
        topk_route:    int   = 64,
        max_alpha:     float = 3.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.layers     = layers
        self.processor  = None   # 由外部设置

        hidden_dim = base_model.config.text_config.hidden_size
        latent_dim = hidden_dim * latent_mult
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.wrapped_layers: Dict[int, QwenLayerWithSAERouter] = {}

        for l in layers:
            # 加载并冻结 SAE
            sae       = SAE(hidden_dim, latent_dim, topk).float()
            ckpt_path = os.path.join(sae_ckpt_dir, f"sae_layer{l}.pt")
            assert os.path.exists(ckpt_path), \
                f"SAE checkpoint not found: {ckpt_path}\n" \
                f"Please run `python -m src.train` first."
            sae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            sae.eval()
            for p in sae.parameters():
                p.requires_grad = False
            print(f"  [layer {l:2d}] SAE loaded & frozen: {ckpt_path}")

            # 构建 Router（唯一需要训练的部分）
            router = FeatureRouter(
                hidden_dim = hidden_dim,
                latent_dim = latent_dim,
                topk_route = topk_route,
                max_alpha  = max_alpha,
            )

            # 替换 Qwen 对应层
            orig_layer = base_model.model.layers[l]
            wrapped    = QwenLayerWithSAERouter(orig_layer, sae, router, l)
            base_model.model.layers[l] = wrapped
            self.wrapped_layers[l]     = wrapped

        # 冻结 Qwen 所有参数（Router 在 wrapped_layers 里，不受影响）
        for name, param in base_model.named_parameters():
            if "router" not in name:
                param.requires_grad = False

        # 统计参数量
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"\nModel ready.")
        print(f"  Trainable : {trainable:>12,}  (Router only)")
        print(f"  Total     : {total:>12,}")
        print(f"  Ratio     : {100*trainable/total:.4f}%")

    # ── context 设置 ──────────────────────────────────────────────────────────

    def set_context(
        self,
        vision_pos:     int,
        num_img_tokens: int,
        text_positions: torch.Tensor,
    ):
        """每次 forward 前调用，把 image/text token 位置传给各层。"""
        for layer in self.wrapped_layers.values():
            layer.set_context(vision_pos, num_img_tokens, text_positions)

    def clear_context(self):
        for layer in self.wrapped_layers.values():
            layer.num_img_tokens = 0
            layer.text_positions = None

    # ── 前向 ──────────────────────────────────────────────────────────────────

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)

    # ── 参数访问 ──────────────────────────────────────────────────────────────

    def router_parameters(self):
        """只返回 Router 参数，用于优化器。"""
        params = []
        for layer in self.wrapped_layers.values():
            params.extend(layer.router.parameters())
        return params

    # ── 权重保存与加载 ────────────────────────────────────────────────────────

    def save_routers(self, save_dir: str, tag: str = "latest"):
        os.makedirs(save_dir, exist_ok=True)
        for l, layer in self.wrapped_layers.items():
            path = os.path.join(save_dir, f"router_layer{l}_{tag}.pt")
            torch.save(layer.router.state_dict(), path)
        print(f"  Routers saved -> {save_dir}  (tag={tag})")

    def load_routers(self, save_dir: str, tag: str = "best"):
        for l, layer in self.wrapped_layers.items():
            path = os.path.join(save_dir, f"router_layer{l}_{tag}.pt")
            if os.path.exists(path):
                layer.router.load_state_dict(
                    torch.load(path, map_location="cpu")
                )
                print(f"  [layer {l}] Router loaded: {path}")
            else:
                print(f"  [layer {l}] Router ckpt not found: {path}")