"""
LlavaNextWithClusterPredictorAndSAE — LLaVA-NeXT-LLaMA3 适配版

适配 llava-hf/llama3-llava-next-8b-hf

与 LLaVA-OneVision 版的关键区别：
  1. 模型类：LlavaNextForConditionalGeneration (不是 LlavaOnevision)
  2. Processor：LlavaNextProcessor
  3. LLM backbone：LLaMA3-8B (hidden_size=4096, 32 layers)
  4. Vision encoder：CLIP ViT-L/14-336 (不是 SigLIP)
  5. image_token_index：128256 (不是 151646)
  6. Chat template：LLaMA3 格式 (<|start_header_id|>...<|end_header_id|>)
  7. SAE 必须重新训练（hidden_size 从 3584 → 4096）
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from src.SAE import SAE

# ── 复用的子模块（与 Qwen/OV 版完全相同）─────────────────────────────────────

class ClusterPredictor(nn.Module):
    def __init__(self, dim: int, n_clusters: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, n_clusters),
        )

    def forward(self, h, text_pos):
        text_h_all = h[:, text_pos, :]
        mean_text_h = text_h_all.mean(dim=1).float()
        return self.head(mean_text_h)


class ImageClusterScorer(nn.Module):
    def __init__(self, dim: int, n_clusters: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, n_clusters),
        )

    def forward(self, h_vision):
        return self.head(h_vision.float().mean(dim=0, keepdim=True))


class ExtraProjector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x):
        return self.proj(self.norm(x))


class SemanticCrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm     = nn.LayerNorm(dim)
        nn.init.normal_(self.out_proj.weight, std=0.001)
        self.lambda_param = nn.Parameter(torch.tensor(0.75))

    def forward(self, vision, extra):
        lam = F.softplus(self.lambda_param)
        Q = self.q_proj(vision)
        K = self.k_proj(extra)
        V = self.v_proj(extra)
        attn = F.softmax(Q @ K.T / self.dim**0.5, dim=-1)
        return vision + lam * self.norm(self.out_proj(attn @ V))


class PrincipalComponentSuppressor(nn.Module):
    def __init__(self, n_suppress: int = 3):
        super().__init__()
        self.n_suppress = n_suppress
        self.alpha_param = nn.Parameter(torch.tensor(-3.0))

    def forward(self, vision_h):
        alpha = F.softplus(self.alpha_param)
        N, D = vision_h.shape
        if N <= self.n_suppress:
            return vision_h
        mean = vision_h.mean(dim=0, keepdim=True)
        centered = vision_h - mean
        U, S, V = torch.svd_lowrank(centered, q=self.n_suppress)
        proj = centered @ V @ V.T
        return vision_h - alpha * proj


class SemanticCompleter(nn.Module):
    def __init__(self, dim: int, latent_dim: int, n_clusters: int = None,
                 bottleneck: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_to_delta = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, bottleneck),
        )
        self.z_to_gate = nn.Sequential(
            nn.Linear(latent_dim, bottleneck),
            nn.Sigmoid(),
        )
        self.delta_proj = nn.Linear(bottleneck, latent_dim, bias=False)
        nn.init.normal_(self.delta_proj.weight, std=0.001)
        self.beta_param = nn.Parameter(torch.tensor(-3.0))

    def forward(self, text_h, z_cluster):
        beta = F.softplus(self.beta_param)
        text_bn = self.text_to_delta(text_h)
        gate = self.z_to_gate(z_cluster)
        fused = gate * text_bn.unsqueeze(0)
        delta_z = beta * self.delta_proj(fused)
        return delta_z


# ── 自动探测 transformer layers（LLaVA-NeXT-LLaMA3 适配）─────────────────────

def _find_layers(model):
    """
    LLaVA-NeXT 的 layers 路径探测。
    LlavaNextForConditionalGeneration 的结构：
      model.language_model.model.layers   (最常见)
      model.model.language_model.layers
      model.model.layers
    """
    for path in [
        lambda m: m.language_model.model.layers,
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
    ]:
        try:
            layers = path(model)
            if hasattr(layers, '__len__') and len(layers) > 0:
                return layers
        except AttributeError:
            pass
    raise AttributeError(
        "Cannot find transformer layers in LLaVA-NeXT model. Run:\n"
        "  print([n for n,_ in model.named_modules() if 'layers.0' in n][:5])"
    )


# ── 主模型 ────────────────────────────────────────────────────────────────────

class LlavaNextWithClusterPredictorAndSAE(nn.Module):

    HOOK_OFF       = 0
    HOOK_READ_ONLY = 1
    HOOK_READ_WRITE = 2

    def __init__(self, base_model, sae, processor, cluster_info,
                 inject_layer=8, suppress_layer=-8, n_suppress_pcs=3,
                 top_n_patches=60, top_k_clusters=10,
                 cluster_threshold=0.5,
                 bce_lambda=0.5, align_lambda=0.3, max_tokens=128):
        super().__init__()
        self.base_model  = base_model
        self.sae         = sae
        self.processor   = processor
        self.inject_layer      = inject_layer
        self.top_n_patches     = top_n_patches
        self.top_k_clusters    = top_k_clusters
        self.cluster_threshold = cluster_threshold
        self.bce_lambda        = bce_lambda
        self.align_lambda      = align_lambda
        self.max_tokens        = max_tokens

        self.clusters = {int(k): v for k, v in cluster_info["clusters"].items()}
        self.feature_to_cluster = {
            int(k): int(v) for k, v in cluster_info["feature_to_cluster"].items()
        }
        self.cluster_to_features = {}
        for fid, cid in self.feature_to_cluster.items():
            self.cluster_to_features.setdefault(cid, []).append(fid)
        self.n_clusters = len(self.clusters)

        # ── LLaVA-NeXT: hidden_size 从 text_config 获取 ──────────────────────
        dim = base_model.config.text_config.hidden_size
        latent_dim = sae.latent_dim
        device = next(base_model.parameters()).device

        print(f"  LLM hidden_size: {dim}")

        self.cluster_predictor    = ClusterPredictor(dim, self.n_clusters).to(device)
        self.image_cluster_scorer = ImageClusterScorer(dim, self.n_clusters).to(device)
        self.extra_projector      = ExtraProjector(dim).to(device)
        self.semantic_cross_attn  = SemanticCrossAttention(dim).to(device)
        self.pc_suppressor        = PrincipalComponentSuppressor(n_suppress=n_suppress_pcs).to(device)
        self.semantic_completer   = SemanticCompleter(dim, latent_dim).to(device)

        self._layers = _find_layers(base_model)
        n_layers = len(self._layers)
        print(f"  Found {n_layers} transformer layers")

        self.suppress_layer = suppress_layer if suppress_layer >= 0 else n_layers + suppress_layer
        print(f"  Inject @ layer {self.inject_layer}, Suppress @ layer {self.suppress_layer}")

        self.sae.eval()
        for p in self.sae.parameters():
            p.requires_grad = False
        for p in self.base_model.parameters():
            p.requires_grad = False

        # hook 运行时状态
        self._hook_mode: int = self.HOOK_OFF
        self._hook_fired: bool = False
        self._suppress_hook_fired: bool = False
        self._cached_positions: Optional[Tuple] = None
        self._stashed_logits: Optional[torch.Tensor] = None
        self._stashed_h_vision: Optional[torch.Tensor] = None
        self._last_cluster_ids: List[int] = []
        self._last_cluster_probs: Optional[torch.Tensor] = None

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    # ── 输入构建（LLaVA-NeXT-LLaMA3 适配）────────────────────────────────────

    def _build_inputs(self, image_path, question, answer=None, for_generation=True):
        """
        LLaVA-NeXT 使用 LlavaNextProcessor。
        和 LLaVA-OV 的区别：processor 类不同，但 apply_chat_template 接口相同。
        """
        from PIL import Image

        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        if answer is not None:
            conversation.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            })
            for_generation = False

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=for_generation,
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        return inputs.to(self.device)

    def _get_token_positions(self, inputs):
        """
        LLaVA-NeXT-LLaMA3 vision token 定位。

        image_token_index = 128256（LLaMA3 的扩展 vocab）
        """
        ids = inputs["input_ids"][0]

        # ── 获取 image token ID ───────────────────────────────────────────────
        image_token_id = getattr(
            self.base_model.config, "image_token_index",
            None,
        )
        if image_token_id is None:
            # fallback：尝试从 tokenizer 获取
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")

        image_mask = (ids == image_token_id)
        image_positions = image_mask.nonzero(as_tuple=False).squeeze(-1)

        if image_positions.numel() == 0:
            raise ValueError(
                f"No image tokens found in input_ids. "
                f"image_token_id={image_token_id}, "
                f"unique ids: {ids.unique().tolist()[:20]}"
            )

        v_pos = image_positions[0].item()
        n_img = image_positions.numel()

        # ── text token 位置 ───────────────────────────────────────────────────
        # LLaMA3 的 special tokens
        tokenizer = self.processor.tokenizer
        special_ids = set()

        # LLaMA3 特有的 special tokens
        for token_name in [
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "<|eot_id|>",
            "<image>",
            "<pad>",
        ]:
            tid = tokenizer.convert_tokens_to_ids(token_name)
            if tid is not None and tid != tokenizer.unk_token_id:
                special_ids.add(tid)

        for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_ids.add(tid)

        mask = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
        mask[v_pos:v_pos + n_img] = False
        for s in special_ids:
            mask &= (ids != s)

        text_pos = mask.nonzero(as_tuple=False).squeeze(-1)

        return v_pos, n_img, text_pos

    # ── 中间层 Hook ───────────────────────────────────────────────────────────

    def _mid_layer_hook(self, module, input, output):
        if self._hook_fired:
            return output

        is_tuple = isinstance(output, tuple)
        raw_hs = output[0] if is_tuple else output
        was_2d = (raw_hs.dim() == 2)
        hs = raw_hs.unsqueeze(0) if was_2d else raw_hs

        if hs.shape[1] <= 1:
            return output

        self._hook_fired = True
        v_pos, n_img, text_pos = self._cached_positions

        # ── Cluster 预测 ──────────────────────────────────────────────────────
        logits = self.cluster_predictor(hs, text_pos)
        self._stashed_logits = logits

        h_vision_float = hs[0, v_pos:v_pos + n_img, :].float()
        self._stashed_h_vision = h_vision_float

        probs_with_grad = torch.sigmoid(logits)[0]
        probs_detached = probs_with_grad.detach()

        top_k_clusters = min(self.top_k_clusters, len(probs_with_grad))
        _, top_cids = torch.topk(probs_detached, top_k_clusters)
        top_cids = top_cids.tolist()

        self._last_cluster_ids = [cid for cid in top_cids if probs_detached[cid] > 0.1]
        self._last_cluster_probs = probs_with_grad

        if self._hook_mode == self.HOOK_READ_ONLY:
            return output

        # ── 稀疏注入 ─────────────────────────────────────────────────────────
        injection = self._build_sparse_injection(
            hs, top_cids, probs_with_grad, v_pos, n_img, text_pos,
        )
        if injection is None:
            return output

        active_indices, recon_projected = injection

        hs_mod = hs.clone()
        vision_active = hs_mod[0, v_pos + active_indices, :].float()
        updated = self.semantic_cross_attn(vision_active, recon_projected)
        hs_mod[0, v_pos + active_indices, :] = updated.to(hs.dtype)

        if was_2d:
            hs_mod = hs_mod.squeeze(0)
        if is_tuple:
            return (hs_mod,) + output[1:]
        return hs_mod

    def _suppress_hook(self, module, input, output):
        if self._suppress_hook_fired:
            return output

        is_tuple = isinstance(output, tuple)
        raw_hs = output[0] if is_tuple else output
        was_2d = (raw_hs.dim() == 2)
        hs = raw_hs.unsqueeze(0) if was_2d else raw_hs

        if hs.shape[1] <= 1:
            return output

        self._suppress_hook_fired = True
        v_pos, n_img, _ = self._cached_positions

        hs_mod = hs.clone()
        vision_h = hs_mod[0, v_pos:v_pos + n_img, :].float()
        suppressed = self.pc_suppressor(vision_h)
        hs_mod[0, v_pos:v_pos + n_img, :] = suppressed.to(hs.dtype)

        if was_2d:
            hs_mod = hs_mod.squeeze(0)
        if is_tuple:
            return (hs_mod,) + output[1:]
        return hs_mod

    # ── Hook 生命周期管理 ─────────────────────────────────────────────────────

    def _activate_hook(self, mode: int):
        self._hook_mode = mode
        self._hook_fired = False
        self._suppress_hook_fired = False
        self._stashed_logits = None
        self._stashed_h_vision = None
        self._last_cluster_ids = []
        self._last_cluster_probs = None
        handles = []
        handles.append(self._layers[self.inject_layer].register_forward_hook(self._mid_layer_hook))
        if mode == self.HOOK_READ_WRITE:
            handles.append(self._layers[self.suppress_layer].register_forward_hook(self._suppress_hook))
        return handles

    def _deactivate_hook(self, handles):
        for h in handles:
            h.remove()
        self._hook_mode = self.HOOK_OFF

    # ── 稀疏注入构造 ─────────────────────────────────────────────────────────

    def _build_sparse_injection(self, h, cluster_ids, cluster_probs, v_pos, n_img, text_pos):
        flat = h[:, v_pos:v_pos + n_img, :].reshape(-1, h.shape[-1]).float()
        n_patches = flat.shape[0]

        text_h = h[0, text_pos[-1], :].float()

        with torch.no_grad():
            z = self.sae.encode(flat)

        patch_recon = torch.zeros(n_patches, flat.shape[-1], device=flat.device)
        patch_activated = torch.zeros(n_patches, dtype=torch.bool, device=flat.device)

        for cid in cluster_ids:
            fids = [f for f in self.cluster_to_features.get(cid, []) if f < z.shape[-1]]
            if not fids:
                continue

            acts = z[:, fids].sum(dim=-1)
            if acts.max() <= 0:
                continue

            k = min(self.top_n_patches, (acts > 0).sum().item())
            if k == 0:
                continue
            topk_vals, topk_idx = torch.topk(acts, k)

            z_cluster = torch.zeros_like(z[topk_idx])
            z_sub = z[topk_idx]
            z_cluster[:, fids] = z_sub[:, fids]

            delta_z = self.semantic_completer(text_h, z_cluster)
            z_completed = z_cluster + delta_z

            with torch.no_grad():
                recon = self.sae.decoder(z_completed)

            if cid < len(cluster_probs):
                recon = recon * cluster_probs[cid]

            patch_recon[topk_idx] += recon
            patch_activated[topk_idx] = True

        active_indices = patch_activated.nonzero(as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            return None

        recon_active = patch_recon[active_indices]
        recon_projected = self.extra_projector(recon_active)

        return active_indices, recon_projected

    def _compute_alignment_loss(self, h_vision, text_logits):
        img_probs  = torch.sigmoid(self.image_cluster_scorer(h_vision)).clamp(1e-7, 1 - 1e-7)
        text_probs = torch.sigmoid(text_logits.detach()).clamp(1e-7, 1 - 1e-7)
        return (
            F.kl_div(img_probs.log(), text_probs, reduction='batchmean')
            + F.kl_div(text_probs.log(), img_probs, reduction='batchmean')
        ) / 2

    # ── 推理 ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, image_path, question, max_new_tokens=None, verbose=False):
        if max_new_tokens is None:
            max_new_tokens = self.max_tokens

        inputs = self._build_inputs(image_path, question)
        v_pos, n_img, text_pos = self._get_token_positions(inputs)
        self._cached_positions = (v_pos, n_img, text_pos)

        handles = self._activate_hook(self.HOOK_READ_WRITE)
        try:
            input_len = inputs["input_ids"].shape[1]
            out_ids = self.base_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        finally:
            self._deactivate_hook(handles)

        answer = self.processor.decode(
            out_ids[0, input_len:], skip_special_tokens=True,
        ).strip()

        cids   = self._last_cluster_ids
        cnames = [self.clusters[c]["name"] for c in cids if c in self.clusters]

        if verbose:
            print(f"  Vision tokens : {n_img}")
            print(f"  Clusters      : {cids} → {cnames}")
            lam = F.softplus(self.semantic_cross_attn.lambda_param).item()
            print(f"  λ (softplus)  : {lam:.4f}")

        return {
            "cluster_ids":   cids,
            "cluster_names": cnames,
            "final_answer":  answer,
        }

    # ── 训练 ──────────────────────────────────────────────────────────────────

    def compute_loss(self, image_path, question, answer, focus_clusters,
                     stage=2, include_bce=True):
        if stage == 1:
            return self._compute_loss_stage1(image_path, question, focus_clusters)
        else:
            return self._compute_loss_stage2(
                image_path, question, answer, focus_clusters, include_bce,
            )

    def _compute_loss_stage1(self, image_path, question, focus_clusters):
        inputs = self._build_inputs(image_path, question, for_generation=False)
        v_pos, n_img, text_pos = self._get_token_positions(inputs)
        self._cached_positions = (v_pos, n_img, text_pos)

        handles = self._activate_hook(self.HOOK_READ_ONLY)
        try:
            self.base_model(**inputs, return_dict=True)
        finally:
            self._deactivate_hook(handles)

        logits   = self._stashed_logits
        h_vision = self._stashed_h_vision

        if logits is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, 0.0

        target = torch.zeros(1, self.n_clusters, device=self.device)
        for c in focus_clusters:
            if 0 <= c < self.n_clusters:
                target[0, c] = 1.0
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)
        align_loss = self._compute_alignment_loss(h_vision, logits)
        total = bce_loss + self.align_lambda * align_loss
        return total, bce_loss.item(), align_loss.item()

    def _compute_loss_stage2(self, image_path, question, answer,
                             focus_clusters, include_bce=True):
        inputs = self._build_inputs(image_path, question, answer=answer)
        v_pos, n_img, text_pos = self._get_token_positions(inputs)
        labels = self._build_labels(inputs["input_ids"])
        self._cached_positions = (v_pos, n_img, text_pos)

        handles = self._activate_hook(self.HOOK_READ_WRITE)
        try:
            lm_out = self.base_model(**inputs, labels=labels, return_dict=True)
        finally:
            self._deactivate_hook(handles)

        lm_loss = lm_out.loss
        bce_val = 0.0
        bce_tensor = torch.tensor(0.0, device=self.device)

        if include_bce and self._stashed_logits is not None:
            logits = self._stashed_logits
            target = torch.zeros(1, self.n_clusters, device=self.device)
            for c in focus_clusters:
                if 0 <= c < self.n_clusters:
                    target[0, c] = 1.0
            bce_tensor = F.binary_cross_entropy_with_logits(logits, target)
            bce_val = bce_tensor.item()

            if self._stashed_h_vision is not None:
                align = self._compute_alignment_loss(self._stashed_h_vision, logits)
                bce_tensor = bce_tensor + self.align_lambda * align

        total = lm_loss + self.bce_lambda * bce_tensor if include_bce else lm_loss
        return total, lm_loss.item(), bce_val

    def _build_labels(self, input_ids):
        """
        LLaMA3 chat template label 构建。

        LLaMA3 格式：
          <|begin_of_text|><|start_header_id|>user<|end_header_id|>
          ...
          <|eot_id|><|start_header_id|>assistant<|end_header_id|>
          ...answer...
          <|eot_id|>

        只对最后一个 assistant 回复内容计算 loss。
        """
        labels = input_ids.clone()
        tokenizer = self.processor.tokenizer

        # ── 尝试 LLaMA3 格式 ─────────────────────────────────────────────────
        # 找 <|start_header_id|>assistant<|end_header_id|> 的位置
        start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")

        if (start_header_id is not None and start_header_id != tokenizer.unk_token_id
                and end_header_id is not None and end_header_id != tokenizer.unk_token_id):
            ast_ids = tokenizer.encode("assistant", add_special_tokens=False)

            for b in range(input_ids.shape[0]):
                ids = input_ids[b].tolist()
                answer_start = None

                # 从后往前找最后一个 <|start_header_id|> assistant <|end_header_id|>
                for i in range(len(ids) - 3, -1, -1):
                    if ids[i] == start_header_id:
                        # 检查后面是否跟着 "assistant" 和 <|end_header_id|>
                        ast_end = i + 1 + len(ast_ids)
                        if (ids[i+1:ast_end] == ast_ids
                                and ast_end < len(ids)
                                and ids[ast_end] == end_header_id):
                            # answer 内容从 <|end_header_id|> 后面的 \n 开始
                            answer_start = ast_end + 1
                            # 跳过可能的换行符 token
                            while (answer_start < len(ids)
                                   and tokenizer.decode([ids[answer_start]]).strip() == ""):
                                answer_start += 1
                            break

                if answer_start is not None:
                    labels[b, :answer_start] = -100
                else:
                    # fallback
                    labels[b, :int(len(ids) * 0.8)] = -100
        else:
            # ── fallback: chatml 格式（以防万一）──────────────────────────────
            im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
            if im_start is not None and im_start != tokenizer.unk_token_id:
                ast_ids = tokenizer.encode("assistant", add_special_tokens=False)
                for b in range(input_ids.shape[0]):
                    ids = input_ids[b].tolist()
                    start = None
                    for i in range(len(ids) - 2, -1, -1):
                        if ids[i] == im_start and ids[i+1:i+1+len(ast_ids)] == ast_ids:
                            start = i + 1 + len(ast_ids) + 1
                            break
                    labels[b, :start if start else len(ids)] = -100
            else:
                for b in range(input_ids.shape[0]):
                    labels[b, :int(len(input_ids[b]) * 0.8)] = -100

        return labels.to(self.device)

    # ── 保存 / 加载 ──────────────────────────────────────────────────────────

    def save_predictor(self, path):
        torch.save({
            "cluster_predictor":    self.cluster_predictor.state_dict(),
            "image_cluster_scorer": self.image_cluster_scorer.state_dict(),
            "extra_projector":      self.extra_projector.state_dict(),
            "semantic_cross_attn":  self.semantic_cross_attn.state_dict(),
            "pc_suppressor":        self.pc_suppressor.state_dict(),
            "semantic_completer":   self.semantic_completer.state_dict(),
        }, path)
        print(f"  Saved: {path}")

    def load_predictor(self, path):
        state = torch.load(path, map_location="cpu")
        if "cluster_predictor" in state:
            for k in [
                "cluster_predictor", "image_cluster_scorer",
                "extra_projector", "semantic_cross_attn",
                "pc_suppressor", "semantic_completer",
            ]:
                if k in state:
                    getattr(self, k).load_state_dict(state[k])
        else:
            self.cluster_predictor.load_state_dict(state)
        for m in [
            self.cluster_predictor, self.image_cluster_scorer,
            self.extra_projector, self.semantic_cross_attn,
            self.pc_suppressor, self.semantic_completer,
        ]:
            m.to(self.device)
        print(f"  Loaded: {path}")

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    @classmethod
    def from_pretrained(cls, model_id, sae_ckpt_dir, cluster_path,
                        inject_layer=8, suppress_layer=-8, n_suppress_pcs=2,
                        latent_mult=8, topk=32, top_n_patches=60,
                        top_k_clusters=10, cluster_threshold=0.5,
                        bce_lambda=0.5, align_lambda=0.3,
                        predictor_ckpt=None, device="cuda"):
        """
        加载 LLaVA-NeXT-LLaMA3 模型。

        model_id: "llava-hf/llama3-llava-next-8b-hf"
        """
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        print(f"Loading LLaVA-NeXT: {model_id}...")
        processor = LlavaNextProcessor.from_pretrained(model_id)
        base_model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(device)

        dim = base_model.config.text_config.hidden_size
        sae = SAE(dim, dim * latent_mult, topk).float().to(device)
        sae.load_state_dict(torch.load(
            os.path.join(sae_ckpt_dir, f"sae_layer{inject_layer}.pt"),
            map_location=device,
        ), strict=False)

        with open(cluster_path) as f:
            cluster_info = json.load(f)

        model = cls(
            base_model, sae, processor, cluster_info,
            inject_layer=inject_layer, suppress_layer=suppress_layer,
            n_suppress_pcs=n_suppress_pcs,
            top_n_patches=top_n_patches,
            top_k_clusters=top_k_clusters,
            cluster_threshold=cluster_threshold,
            bce_lambda=bce_lambda, align_lambda=align_lambda,
        )

        if predictor_ckpt and os.path.exists(predictor_ckpt):
            model.load_predictor(predictor_ckpt)

        model.eval()
        total = sum(
            sum(p.numel() for p in m.parameters())
            for m in [
                model.cluster_predictor, model.image_cluster_scorer,
                model.extra_projector, model.semantic_cross_attn,
                model.pc_suppressor, model.semantic_completer,
            ]
        )
        print(f"  Trainable modules: {total:,} params")
        return model