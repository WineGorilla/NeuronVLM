"""
QwenWithClusterPredictorAndSAE — 消融版：无 PCA 主成分抑制

与完整版唯一区别：去掉 PrincipalComponentSuppressor，不注册 suppress_hook。
用于验证 PCA suppression 的独立贡献。
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from src.SAE import SAE


# ── 子模块（与完整版一致）─────────────────────────────────────────────────────

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


class SemanticCompleter(nn.Module):
    def __init__(self, dim: int, latent_dim: int, n_clusters: int = None, bottleneck: int = 128):
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


# ── 自动探测 transformer layers ───────────────────────────────────────────────

def _find_layers(model):
    for path in [
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
    raise AttributeError("Cannot find transformer layers.")


# ── 主模型（无 PCA suppression）───────────────────────────────────────────────

class QwenWithClusterPredictorAndSAE(nn.Module):

    HOOK_OFF       = 0
    HOOK_READ_ONLY = 1
    HOOK_READ_WRITE = 2

    def __init__(self, base_model, sae, processor, cluster_info,
                 inject_layer=8, top_n_patches=60, top_k_clusters=10,
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

        dim    = base_model.config.text_config.hidden_size
        latent_dim = sae.latent_dim
        device = next(base_model.parameters()).device

        self.cluster_predictor    = ClusterPredictor(dim, self.n_clusters).to(device)
        self.image_cluster_scorer = ImageClusterScorer(dim, self.n_clusters).to(device)
        self.extra_projector      = ExtraProjector(dim).to(device)
        self.semantic_cross_attn  = SemanticCrossAttention(dim).to(device)
        self.semantic_completer   = SemanticCompleter(dim, latent_dim).to(device)
        # ▲ 无 PrincipalComponentSuppressor

        self._layers = _find_layers(base_model)
        print(f"  Found {len(self._layers)} transformer layers")
        print(f"  Inject @ layer {self.inject_layer} (NO suppress hook)")

        self.sae.eval()
        for p in self.sae.parameters():
            p.requires_grad = False
        for p in self.base_model.parameters():
            p.requires_grad = False

        self._hook_mode: int = self.HOOK_OFF
        self._hook_fired: bool = False
        self._cached_positions: Optional[Tuple] = None
        self._stashed_logits: Optional[torch.Tensor] = None
        self._stashed_h_vision: Optional[torch.Tensor] = None
        self._last_cluster_ids: List[int] = []
        self._last_cluster_probs: Optional[torch.Tensor] = None

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    # ── 输入构建 ──────────────────────────────────────────────────────────────

    def _build_inputs(self, image_path, question, answer=None, for_generation=True):
        from qwen_vl_utils import process_vision_info
        msg = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text":  question},
            ],
        }]
        if answer is not None:
            msg.append({"role": "assistant", "content": answer})
            for_generation = False
        messages = [msg]
        texts = [self.processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=for_generation
        ) for m in messages]
        img_in, vid_in = process_vision_info(messages)
        inputs = self.processor(
            text=texts, images=img_in, videos=vid_in,
            padding=True, return_tensors="pt",
        )
        return inputs.to(self.device)

    def _get_token_positions(self, inputs):
        grid = inputs["image_grid_thw"]
        merge = self.base_model.config.vision_config.spatial_merge_size
        n_img = int(grid[0, 1].item() * grid[0, 2].item() / merge**2)

        vs_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ids   = inputs["input_ids"][0]
        v_pos = (ids == vs_id).nonzero()[0].item() + 1

        special = {
            self.processor.tokenizer.convert_tokens_to_ids(t)
            for t in [
                "<|im_start|>", "<|im_end|>",
                "<|vision_start|>", "<|vision_end|>",
            ]
            if self.processor.tokenizer.convert_tokens_to_ids(t) is not None
        }
        mask = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
        mask[v_pos:v_pos + n_img] = False
        for s in special:
            mask &= (ids != s)
        return v_pos, n_img, mask.nonzero(as_tuple=False).squeeze(-1)

    # ── Hook（只有 inject，无 suppress）────────────────────────────────────────

    def _mid_layer_hook(self, module, input, output):
        if self._hook_fired:
            return output

        hs = output[0]
        if hs.shape[1] <= 1:
            return output

        self._hook_fired = True
        v_pos, n_img, text_pos = self._cached_positions

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

        return (hs_mod,) + output[1:]

    # ── Hook 生命周期（只注册 inject hook）─────────────────────────────────────

    def _activate_hook(self, mode: int):
        self._hook_mode = mode
        self._hook_fired = False
        self._stashed_logits = None
        self._stashed_h_vision = None
        self._last_cluster_ids = []
        self._last_cluster_probs = None
        handles = []
        handles.append(self._layers[self.inject_layer].register_forward_hook(self._mid_layer_hook))
        # ▲ 无 suppress hook
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
        labels = input_ids.clone()
        im_start = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        ast_ids  = self.processor.tokenizer.encode("assistant", add_special_tokens=False)
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            start = None
            for i in range(len(ids) - 2, -1, -1):
                if ids[i] == im_start and ids[i+1:i+1+len(ast_ids)] == ast_ids:
                    start = i + 1 + len(ast_ids) + 1
                    break
            labels[b, :start if start else len(ids)] = -100
        return labels.to(self.device)

    # ── 保存 / 加载 ──────────────────────────────────────────────────────────

    def save_predictor(self, path):
        torch.save({
            "cluster_predictor":    self.cluster_predictor.state_dict(),
            "image_cluster_scorer": self.image_cluster_scorer.state_dict(),
            "extra_projector":      self.extra_projector.state_dict(),
            "semantic_cross_attn":  self.semantic_cross_attn.state_dict(),
            "semantic_completer":   self.semantic_completer.state_dict(),
        }, path)
        print(f"  Saved: {path}")

    def load_predictor(self, path):
        state = torch.load(path, map_location="cpu")
        if "cluster_predictor" in state:
            for k in [
                "cluster_predictor", "image_cluster_scorer",
                "extra_projector", "semantic_cross_attn",
                "semantic_completer",
            ]:
                if k in state:
                    getattr(self, k).load_state_dict(state[k])
        else:
            self.cluster_predictor.load_state_dict(state)
        for m in [
            self.cluster_predictor, self.image_cluster_scorer,
            self.extra_projector, self.semantic_cross_attn,
            self.semantic_completer,
        ]:
            m.to(self.device)
        print(f"  Loaded: {path}")

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    @classmethod
    def from_pretrained(cls, model_id, sae_ckpt_dir, cluster_path,
                        inject_layer=8, latent_mult=8, topk=32, top_n_patches=60,
                        top_k_clusters=10, cluster_threshold=0.5,
                        bce_lambda=0.5, align_lambda=0.3,
                        predictor_ckpt=None, device="cuda"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"Loading Qwen: {model_id}...")
        processor  = AutoProcessor.from_pretrained(model_id)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(device)

        dim = base_model.config.text_config.hidden_size
        sae = SAE(dim, dim * latent_mult, topk).float().to(device)
        sae.load_state_dict(torch.load(
            os.path.join(sae_ckpt_dir, f"sae_layer{inject_layer}.pt"),
            map_location=device,
        ))

        with open(cluster_path) as f:
            cluster_info = json.load(f)

        model = cls(
            base_model, sae, processor, cluster_info,
            inject_layer=inject_layer,
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
                model.semantic_completer,
            ]
        )
        print(f"  Trainable modules: {total:,} params (no PCS)")
        return model