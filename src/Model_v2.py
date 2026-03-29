"""
QwenWithClusterPredictorAndSAE — Single-Forward, Sparse Local Injection
+ Spatial Patch Interaction (v2)

新增模块 — SpatialPatchInteraction：
  在 SemanticCrossAttention 之后、写回 hidden state 之前，
  让被激活的 patches 之间做一次带 2D 位置编码的 self-attention。

  为什么需要：
    SemanticCrossAttention 是 patch-wise 独立的（每个 patch 只和自己的 SAE 重构交互），
    无法捕捉 patch 间的空间关系。但计数、空间推理（"左边"、"之间"、"更大"）
    本质上需要 patch 之间的信息交换。

  设计：
    1. 2D Sinusoidal PE：根据 vision grid 的 (row, col) 生成位置编码，
       让 attention 知道 patch 的绝对位置和相对距离。
    2. Multi-head self-attention (2-4 heads)：轻量但足够捕捉空间模式。
    3. Zero-init output：初始等价于恒等，训练初期不干扰。
    4. 只在 activated patches 上运行：稀疏，计算量小。

  流程变化：
    原：vision[active] = CrossAttn(vision[active], recon[active])
    新：vision[active] = CrossAttn(vision[active], recon[active])
        vision[active] = SpatialInteract(vision[active], pos_2d[active])

训练注意：
  SpatialPatchInteraction 的参数在 Stage 2 一起训练，
  Stage 1 不涉及（因为 Stage 1 不修改 hidden state）。
"""
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from src.SAE import SAE


# ── 子模块 ────────────────────────────────────────────────────────────────────

class ClusterPredictor(nn.Module):
    """Text hidden → cluster logits"""
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
    """Vision hidden → cluster logits（用于 alignment loss）"""
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
    """SAE reconstructed features → projected for cross-attention K/V"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x):
        return self.proj(self.norm(x))


class SemanticCrossAttention(nn.Module):
    """Sparse local cross-attention: 只更新被选中的 vision patches"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm     = nn.LayerNorm(dim)
        nn.init.normal_(self.out_proj.weight, std=0.001)
        self.lambda_param = nn.Parameter(torch.tensor(3.0))

    def forward(self, vision, extra):
        lam = F.softplus(self.lambda_param)
        Q = self.q_proj(vision)
        K = self.k_proj(extra)
        V = self.v_proj(extra)
        attn = F.softmax(Q @ K.T / self.dim**0.5, dim=-1)
        return vision + lam * self.norm(self.out_proj(attn @ V))


class SpatialPatchInteraction(nn.Module):
    """带 2D 位置编码的稀疏 patch 间 self-attention，用于空间推理。

    核心思想：
      activated patches 之间需要知道彼此的空间位置才能推理
      "A 在 B 左边"、"有 3 个红色物体"等空间/计数问题。

    位置编码方案 — 2D Sinusoidal + Learnable Relative Bias：
      1. 绝对位置：2D sinusoidal PE (row, col) → 拼接到 hidden 前做 QKV 投影
         不增加 hidden dim，而是通过一个小的 pos_proj 混入 hidden。
      2. 相对位置 bias：learnable log-spaced distance bias，
         加到 attention logits 上，让模型直接学习"距离 d 的 patch 对
         应该有多大的 attention"。

    Multi-head 设计：
      n_heads=4，head_dim = dim // 4（或更小的 bottleneck）。
      为了控制参数量，QKV 投影用 bottleneck dim 而非 full dim。

    Zero-init：
      out_proj 初始化为近零 + gamma 初始化极小，训练初期 ≈ identity。
    """
    def __init__(self, dim: int, n_heads: int = 4, max_grid: int = 40,
                 n_dist_bins: int = 32):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_grid = max_grid
        self.n_dist_bins = n_dist_bins

        # 位置编码混入：2D pos → dim
        # sinusoidal PE 维度 = dim // 2 each for row/col, 拼接 = dim
        self.pos_proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.pos_proj.weight, std=0.01)

        # QKV 投影（full dim → full dim, multi-head）
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.001)

        # Learnable relative distance bias (per head)
        # 把连续的 2D 距离离散化到 n_dist_bins 个 bin
        self.dist_bias = nn.Parameter(torch.zeros(n_heads, n_dist_bins))

        # 输出 gate，初始极小
        self.gamma_param = nn.Parameter(torch.tensor(-3.0))

        self.norm = nn.LayerNorm(dim)

    def _sinusoidal_pe_2d(self, rows, cols, dim, device):
        """生成 2D sinusoidal positional encoding。

        Args:
            rows: (N,) 每个 patch 的行号
            cols: (N,) 每个 patch 的列号
            dim:  总维度（row 用 dim//2, col 用 dim//2）

        Returns:
            (N, dim) positional encoding
        """
        half_dim = dim // 2
        freq = torch.exp(
            torch.arange(0, half_dim, 2, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / half_dim)
        )  # (half_dim // 2,)

        rows_f = rows.float().unsqueeze(-1)  # (N, 1)
        cols_f = cols.float().unsqueeze(-1)

        row_pe = torch.zeros(rows.shape[0], half_dim, device=device)
        col_pe = torch.zeros(cols.shape[0], half_dim, device=device)

        row_pe[:, 0::2] = torch.sin(rows_f * freq)
        row_pe[:, 1::2] = torch.cos(rows_f * freq)
        col_pe[:, 0::2] = torch.sin(cols_f * freq)
        col_pe[:, 1::2] = torch.cos(cols_f * freq)

        return torch.cat([row_pe, col_pe], dim=-1)  # (N, dim)

    def _distance_to_bin(self, dist_matrix):
        """将连续 L2 距离映射到离散 bin index。

        使用 log-spaced binning：近距离分辨率高，远距离分辨率低。
        bin_0 = dist < 1, bin_1 = dist < 2, ..., log-spaced after that.

        Args:
            dist_matrix: (N, N) 非负距离矩阵

        Returns:
            (N, N) LongTensor, 每个元素是 bin index ∈ [0, n_dist_bins-1]
        """
        # log(1 + dist) 然后线性映射到 [0, n_bins-1]
        log_dist = torch.log1p(dist_matrix)
        max_log_dist = math.log1p(self.max_grid * math.sqrt(2))  # 对角线最大距离
        bins = (log_dist / max_log_dist * (self.n_dist_bins - 1)).long()
        return bins.clamp(0, self.n_dist_bins - 1)

    def forward(self, vision_h, patch_rows, patch_cols):
        """
        Args:
            vision_h:   (N, dim)  — activated patches 的 hidden states (float)
            patch_rows: (N,) LongTensor — 每个 patch 在 vision grid 中的行号
            patch_cols: (N,) LongTensor — 列号

        Returns:
            (N, dim) — 空间交互后的 hidden states
        """
        N, D = vision_h.shape
        gamma = F.softplus(self.gamma_param)

        if N <= 1:
            return vision_h  # 只有 1 个 patch，无法交互

        # ── 2D Positional Encoding ────────────────────────────────────────────
        pe = self._sinusoidal_pe_2d(patch_rows, patch_cols, D, vision_h.device)
        h_with_pos = vision_h + self.pos_proj(pe)  # (N, D)

        # ── Multi-head Self-Attention ─────────────────────────────────────────
        Q = self.q_proj(h_with_pos).view(N, self.n_heads, self.head_dim).transpose(0, 1)  # (H, N, d)
        K = self.k_proj(h_with_pos).view(N, self.n_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(vision_h).view(N, self.n_heads, self.head_dim).transpose(0, 1)
        # 注意：V 不加位置编码，保持内容纯净；位置信息只影响 "谁关注谁"

        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (H, N, N)

        # ── Relative Distance Bias ────────────────────────────────────────────
        # 计算 2D 欧氏距离
        row_diff = patch_rows.float().unsqueeze(1) - patch_rows.float().unsqueeze(0)  # (N, N)
        col_diff = patch_cols.float().unsqueeze(1) - patch_cols.float().unsqueeze(0)
        dist_matrix = torch.sqrt(row_diff ** 2 + col_diff ** 2)  # (N, N)

        bin_indices = self._distance_to_bin(dist_matrix)  # (N, N) LongTensor
        # 每个 head 有自己的 distance bias
        rel_bias = self.dist_bias[:, bin_indices]  # (H, N, N)

        attn_logits = attn_logits + rel_bias

        attn_weights = F.softmax(attn_logits, dim=-1)  # (H, N, N)
        attn_out = torch.matmul(attn_weights, V)  # (H, N, d)
        attn_out = attn_out.transpose(0, 1).contiguous().view(N, D)  # (N, D)

        return vision_h + gamma * self.norm(self.out_proj(attn_out))


class PrincipalComponentSuppressor(nn.Module):
    """在靠后层削弱 vision tokens 的主成分。"""
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
    """根据问题语义，预测 SAE latent 空间中应该补充的缺失信息 Δz。"""
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
    raise AttributeError(
        "Cannot find transformer layers. Run: "
        "print([n for n,_ in model.named_modules() if 'layers.0' in n][:5])"
    )


# ── 主模型 ────────────────────────────────────────────────────────────────────

class QwenWithClusterPredictorAndSAE(nn.Module):

    HOOK_OFF       = 0
    HOOK_READ_ONLY = 1
    HOOK_READ_WRITE = 2

    def __init__(self, base_model, sae, processor, cluster_info,
                 inject_layer=8, suppress_layer=-8, n_suppress_pcs=3,
                 top_n_patches=60, top_k_clusters=10,
                 cluster_threshold=0.5,
                 bce_lambda=0.5, align_lambda=0.3, max_tokens=128,
                 spatial_n_heads=4, spatial_n_dist_bins=32):
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
        self.spatial_interaction  = SpatialPatchInteraction(
            dim, n_heads=spatial_n_heads, n_dist_bins=spatial_n_dist_bins,
        ).to(device)
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
        self._cached_grid_hw: Optional[Tuple[int, int]] = None  # (grid_h, grid_w) for spatial
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
        grid_h = int(grid[0, 1].item() // merge)
        grid_w = int(grid[0, 2].item() // merge)
        n_img = grid_h * grid_w

        # 缓存 grid 尺寸，供 spatial interaction 使用
        self._cached_grid_hw = (grid_h, grid_w)

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

    # ── Patch index → 2D grid coordinates ────────────────────────────────────

    def _patch_indices_to_2d(self, indices):
        """将 flat patch indices 转换为 (row, col) grid 坐标。

        Qwen-VL 的 vision tokens 按 row-major 排列：
        patch_0 = (0,0), patch_1 = (0,1), ..., patch_{W-1} = (0, W-1),
        patch_W = (1,0), ...

        Args:
            indices: (K,) LongTensor — patch 在 flat vision token 序列中的索引

        Returns:
            rows: (K,) LongTensor
            cols: (K,) LongTensor
        """
        grid_h, grid_w = self._cached_grid_hw
        rows = indices // grid_w
        cols = indices % grid_w
        return rows, cols

    # ── 中间层 Hook（核心）────────────────────────────────────────────────────

    def _mid_layer_hook(self, module, input, output):
        if self._hook_fired:
            return output

        hs = output[0]
        if hs.shape[1] <= 1:
            return output

        self._hook_fired = True
        v_pos, n_img, text_pos = self._cached_positions

        # ── Step 1: Cluster 预测 ──────────────────────────────────────────────
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

        # ── Stage 1: 只读 ────────────────────────────────────────────────────
        if self._hook_mode == self.HOOK_READ_ONLY:
            return output

        # ── Stage 2 / 推理: 连续加权稀疏注入 + 空间交互 ─────────────────────
        injection = self._build_sparse_injection(
            hs, top_cids, probs_with_grad, v_pos, n_img, text_pos,
        )
        if injection is None:
            return output

        active_indices, recon_projected = injection

        hs_mod = hs.clone()
        vision_active = hs_mod[0, v_pos + active_indices, :].float()

        # Step A: Semantic Cross-Attention（patch-wise 语义注入）
        updated = self.semantic_cross_attn(vision_active, recon_projected)

        # Step B: Spatial Patch Interaction（patch 间空间推理）
        rows, cols = self._patch_indices_to_2d(active_indices)
        updated = self.spatial_interaction(updated, rows, cols)

        # 写回
        hs_mod[0, v_pos + active_indices, :] = updated.to(hs.dtype)

        return (hs_mod,) + output[1:]

    def _suppress_hook(self, module, input, output):
        if self._suppress_hook_fired:
            return output

        hs = output[0]
        if hs.shape[1] <= 1:
            return output

        self._suppress_hook_fired = True
        v_pos, n_img, _ = self._cached_positions

        hs_mod = hs.clone()
        vision_h = hs_mod[0, v_pos:v_pos + n_img, :].float()
        suppressed = self.pc_suppressor(vision_h)
        hs_mod[0, v_pos:v_pos + n_img, :] = suppressed.to(hs.dtype)

        return (hs_mod,) + output[1:]

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
            print(f"  Grid (H×W)    : {self._cached_grid_hw}")
            print(f"  Clusters      : {cids} → {cnames}")
            lam = F.softplus(self.semantic_cross_attn.lambda_param).item()
            gam = F.softplus(self.spatial_interaction.gamma_param).item()
            print(f"  λ_semantic    : {lam:.4f}")
            print(f"  γ_spatial     : {gam:.4f}")

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
            "spatial_interaction":  self.spatial_interaction.state_dict(),
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
                "spatial_interaction",
                "pc_suppressor", "semantic_completer",
            ]:
                if k in state:
                    getattr(self, k).load_state_dict(state[k])
        else:
            self.cluster_predictor.load_state_dict(state)
        for m in [
            self.cluster_predictor, self.image_cluster_scorer,
            self.extra_projector, self.semantic_cross_attn,
            self.spatial_interaction,
            self.pc_suppressor, self.semantic_completer,
        ]:
            m.to(self.device)
        print(f"  Loaded: {path}")

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    @classmethod
    def from_pretrained(cls, model_id, sae_ckpt_dir, cluster_path,
                        inject_layer=8, suppress_layer=-8, n_suppress_pcs=3,
                        latent_mult=8, topk=32, top_n_patches=60,
                        top_k_clusters=10, cluster_threshold=0.5,
                        bce_lambda=0.5, align_lambda=0.3,
                        spatial_n_heads=4, spatial_n_dist_bins=32,
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
            inject_layer=inject_layer, suppress_layer=suppress_layer,
            n_suppress_pcs=n_suppress_pcs,
            top_n_patches=top_n_patches,
            top_k_clusters=top_k_clusters,
            cluster_threshold=cluster_threshold,
            bce_lambda=bce_lambda, align_lambda=align_lambda,
            spatial_n_heads=spatial_n_heads,
            spatial_n_dist_bins=spatial_n_dist_bins,
        )

        if predictor_ckpt and os.path.exists(predictor_ckpt):
            model.load_predictor(predictor_ckpt)

        model.eval()
        total = sum(
            sum(p.numel() for p in m.parameters())
            for m in [
                model.cluster_predictor, model.image_cluster_scorer,
                model.extra_projector, model.semantic_cross_attn,
                model.spatial_interaction,
                model.pc_suppressor, model.semantic_completer,
            ]
        )
        print(f"  Trainable modules: {total:,} params")
        return model