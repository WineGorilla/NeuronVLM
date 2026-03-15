import os
import torch
import torch.nn as nn
from typing import List, Optional

from src.SAE import SAE
from src.Router import FeatureRouter


class QwenLayerWithSAERouter(nn.Module):
    """
    包装单个 Qwen transformer layer，在其后插入 SAE + Router。

    每层逻辑：
        h'     = Qwen原层(h)
        z      = SAE.encode(h'[image_tokens])
        q_vec  = h'[text_tokens].mean(dim=1)
        alpha  = Router(q_vec, z, SAE.decoder.weight)
        delta  = SAE.decode(z*alpha) - SAE.decode(z)
        h_out  = h' + delta(image tokens only)
    """

    def __init__(
        self,
        qwen_layer: nn.Module,
        sae: SAE,
        router: FeatureRouter,
        layer_idx: int,
    ):
        super().__init__()
        self.qwen_layer = qwen_layer
        self.sae = sae
        self.router = router
        self.layer_idx = layer_idx

        self.vision_pos: int = 0
        self.num_img_tokens: int = 0
        self.text_positions: Optional[torch.Tensor] = None

    def __getattr__(self, name: str):
        # 先走 nn.Module 默认的 __getattr__
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 找不到时去原始 layer 里找
            return getattr(self.qwen_layer, name)

    def set_context(
        self,
        vision_pos: int,
        num_img_tokens: int,
        text_positions: torch.Tensor,
    ):
        self.vision_pos = vision_pos
        self.num_img_tokens = num_img_tokens
        self.text_positions = text_positions

    def clear_context(self):
        self.vision_pos = 0
        self.num_img_tokens = 0
        self.text_positions = None

    def forward(self, hidden_states, *args, **kwargs):
        # 1) 原始 Qwen layer 前向
        output = self.qwen_layer(hidden_states, *args, **kwargs)

        if isinstance(output, tuple):
            h_prime = output[0]
            rest = output[1:]
        else:
            h_prime = output
            rest = None

        # 2) 如果没有有效 context，直接返回
        if self.num_img_tokens == 0 or self.text_positions is None:
            return output

        vp = self.vision_pos
        nit = self.num_img_tokens

        # 边界保护
        if vp < 0 or nit <= 0 or (vp + nit) > h_prime.shape[1]:
            return output

        # 统一 device
        device = h_prime.device
        text_pos = self.text_positions.to(device)

        # 3) 取 image tokens hidden states
        img_h = h_prime[:, vp: vp + nit, :]
        B, N_img, H = img_h.shape
        flat = img_h.reshape(B * N_img, H).float()

        # 4) 取 text tokens hidden states 作为 query vector
        text_h = h_prime[:, text_pos, :]
        q_vec = text_h.float().mean(dim=1)             # (B, H)

        if q_vec.shape[0] != 1:
            raise ValueError(
                f"Current QwenLayerWithSAERouter assumes batch_size=1, "
                f"but got batch_size={q_vec.shape[0]} at layer {self.layer_idx}."
            )

        # 5) SAE frozen
        with torch.no_grad():
            z_original = self.sae.encode(flat)
            recon_original = self.sae.decoder(z_original)

        dec_weight = self.sae.decoder.weight           # (H, latent_dim)

        # 6) Router trainable
        alpha = self.router(q_vec, z_original, dec_weight)
        self.last_alpha = alpha.detach()
        # 7) Steering
        z_steered = z_original * alpha
        recon_steered = self.sae.decoder(z_steered)

        delta = (recon_steered - recon_original).to(h_prime.dtype)
        delta = delta.view(B, N_img, H)

        # 8) 只加回 image token 区域
        h_out = h_prime.clone()
        h_out[:, vp: vp + nit, :] += delta

        if rest is not None:
            return (h_out,) + rest
        return h_out


class QwenWithSAERouter(nn.Module):
    """
    完整模型：Qwen2.5-VL + 多层冻结 SAE + 多层可训练 Router
    """

    def __init__(
        self,
        base_model,
        layers: List[int],
        sae_ckpt_dir: str,
        latent_mult: int = 16,
        topk: int = 32,
        topk_route: int = 64,
        max_alpha: float = 3.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.layers = layers

        device = next(base_model.parameters()).device

        hidden_dim = base_model.config.text_config.hidden_size
        latent_dim = hidden_dim * latent_mult
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.wrapped_layers = nn.ModuleDict()

        # 正确路径
        layers_ref = base_model.model.language_model.layers

        print("Injecting SAE + Router into layers:", layers)

        for l in layers:
            if l < 0 or l >= len(layers_ref):
                raise IndexError(
                    f"Layer index {l} out of range. "
                    f"Model has {len(layers_ref)} language layers."
                )

            # 1) SAE
            sae = SAE(hidden_dim, latent_dim, topk).float()
            ckpt_path = os.path.join(sae_ckpt_dir, f"sae_layer{l}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"SAE checkpoint not found: {ckpt_path}\n"
                    f"Please run `python -m src.train` first."
                )
            sae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            sae.eval()
            sae = sae.to(device)
            for p in sae.parameters():
                p.requires_grad = False
            print(f"  [layer {l:2d}] SAE loaded & frozen: {ckpt_path}")

            # 2) Router
            router = FeatureRouter(
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                topk_route=topk_route,
                max_alpha=max_alpha,
            ).to(device)

            # 3) Wrap 原始 layer
            orig_layer = layers_ref[l]
            wrapped = QwenLayerWithSAERouter(orig_layer, sae, router, l).to(device)
            layers_ref[l] = wrapped
            self.wrapped_layers[str(l)] = wrapped

        # 冻结 base_model 中除 router 外的参数
        for name, param in self.base_model.named_parameters():
            if "router" not in name:
                param.requires_grad = False

        # 再保险：确保 SAE / qwen_layer 冻结，router 可训练
        for layer in self.wrapped_layers.values():
            for p in layer.qwen_layer.parameters():
                p.requires_grad = False
            for p in layer.sae.parameters():
                p.requires_grad = False
            for p in layer.router.parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print("\nModel ready.")
        print(f"  Trainable : {trainable:>12,}  (Router only)")
        print(f"  Total     : {total:>12,}")
        print(f"  Ratio     : {100 * trainable / total:.6f}%")

    def set_context(
        self,
        vision_pos: int,
        num_img_tokens: int,
        text_positions: torch.Tensor,
    ):
        for layer in self.wrapped_layers.values():
            layer.set_context(vision_pos, num_img_tokens, text_positions)

    def clear_context(self):
        for layer in self.wrapped_layers.values():
            layer.clear_context()

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)

    def router_parameters(self):
        params = []
        for layer in self.wrapped_layers.values():
            params.extend(list(layer.router.parameters()))
        return params

    def save_routers(self, save_dir: str, tag: str = "latest"):
        os.makedirs(save_dir, exist_ok=True)
        for l, layer in self.wrapped_layers.items():
            path = os.path.join(save_dir, f"router_layer{l}_{tag}.pt")
            torch.save(layer.router.state_dict(), path)
        print(f"  Routers saved -> {save_dir}  (tag={tag})")

    def load_routers(self, save_dir: str, tag: str = "best"):
        device = next(self.base_model.parameters()).device
        for l, layer in self.wrapped_layers.items():
            path = os.path.join(save_dir, f"router_layer{l}_{tag}.pt")
            if os.path.exists(path):
                layer.router.load_state_dict(
                    torch.load(path, map_location="cpu")
                )
                layer.router.to(device)
                print(f"  [layer {int(l)}] Router loaded: {path}")
            else:
                print(f"  [layer {int(l)}] Router ckpt not found: {path}")