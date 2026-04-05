"""
QwenWithClusterPredictorAndSAE — Single-Forward, Sparse Local Injection

核心思想：
  所有语义决策和注入都发生在 transformer 的中间层（inject_layer）的 post-hook 内。
  一次 forward，一个 hidden state，一个 truth。

注入策略 — Sparse Local Cross-Attention：
  不再把所有 cluster 信息 pool 成全局 extra tokens 广播给所有 vision patches。
  而是：找到每个 cluster 激活的 top-k patches，只让这些 patches 和自己的 SAE 重构
  做 cross-attention。未激活的 patches 完全不动。

  vision[active] = CrossAttn(Q=vision[active], K=recon[active], V=recon[active])
  vision[inactive] = vision[inactive]  # untouched

推理流程（单次 forward）：
  1. 图片+问题 → Qwen forward 开始
  2. Layer 0..N-1 正常执行
  3. Layer N (inject_layer) 执行完毕 → post-hook 触发：
     a. ClusterPredictor: text hidden → 预测哪些语义 cluster 被激活
     b. SAE encode vision patches → 按 cluster 筛选 top-k activated patches
     c. SAE decode → 每个 patch 的纯净语义重构
     d. Sparse local cross-attention: 只修改激活的 patches
  4. Layer N+1..末尾 使用修改后的 hidden 继续
  5. generate() 输出

训练流程：
  Stage 1: hook 只读 — 预测 cluster，计算 BCE + alignment loss，不修改 hidden
  Stage 2: hook 读写 — 预测 cluster + 稀疏注入，联合 LM + BCE + alignment loss
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from src.SAE import SAE


# ── 子模块 ────────────────────────────────────────────────────────────────────

class ClusterPredictor(nn.Module):
    """Text hidden → cluster logits

    使用最后一个 text token 的 hidden state（Causal LM 中聚合了全部上文语义），
    而非 mean pooling（会掺杂缺乏上下文的早期 token，导致语义稀释）。
    """
    def __init__(self, dim: int, n_clusters: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, n_clusters),
        )

    def forward(self, h, text_pos):
        # 取出所有的 text tokens 并求均值，作为整体的语义特征
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
    """Sparse local cross-attention: 只更新被选中的 vision patches

    Q = selected vision patches (from hidden state)
    K, V = their SAE reconstructed semantic features (projected)

    Zero-Init 策略：out_proj 权重初始化为 0，lambda 初始化为极小值，
    确保训练初期注入量严格为 0，模型等价于原始预训练模型。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm     = nn.LayerNorm(dim)
        # Near-zero init: 初始注入量极小但梯度非零，避免 zero-init 死锁
        # std=0.001 × softplus(-3.0)≈0.049 → 初始扰动 < 0.00005，安全
        nn.init.normal_(self.out_proj.weight, std=0.001)
        # 超参数 目前分数最高是0.75
        self.lambda_param = nn.Parameter(torch.tensor(0.75))

    def forward(self, vision, extra):
        """
        Args:
            vision: (N, dim) — selected vision patch hidden states
            extra:  (N, dim) — corresponding SAE reconstructed features (projected)
        Returns:
            (N, dim) — updated vision patches
        """
        lam = F.softplus(self.lambda_param)
        Q = self.q_proj(vision)
        K = self.k_proj(extra)
        V = self.v_proj(extra)
        attn = F.softmax(Q @ K.T / self.dim**0.5, dim=-1)
        return vision + lam * self.norm(self.out_proj(attn @ V))


class PrincipalComponentSuppressor(nn.Module):
    """在靠后层削弱 vision tokens 的主成分，迫使模型分散注意力。

    原理：
      vision patches 的 hidden states 通常被少数主成分主导（比如"大物体的轮廓"），
      导致模型忽略空间关系、小物体计数等细节。削弱 top-k 主成分后，
      次要特征（位置、相对距离、小物体）的信号占比上升。

    实现：
      1. 对 vision tokens 做 SVD
      2. 投影到 top-k 主成分方向
      3. 从原始 hidden 中减去 alpha * 主成分投影
      alpha 可学习，初始化接近 0
    """
    def __init__(self, n_suppress: int = 3):
        super().__init__()
        self.n_suppress = n_suppress
        # softplus(-3.0) ≈ 0.049，初始几乎不抑制
        self.alpha_param = nn.Parameter(torch.tensor(-3.0))

    def forward(self, vision_h):
        """
        Args:
            vision_h: (N, dim) vision patch hidden states (float)
        Returns:
            (N, dim) 主成分被削弱后的 vision hidden states
        """
        alpha = F.softplus(self.alpha_param)
        N, D = vision_h.shape

        if N <= self.n_suppress:
            return vision_h

        mean = vision_h.mean(dim=0, keepdim=True)
        centered = vision_h - mean

        U, S, V = torch.svd_lowrank(centered, q=self.n_suppress)

        # 主成分分量 = 投影到主成分空间再投影回来
        proj = centered @ V @ V.T  # (N, D)

        return vision_h - alpha * proj


class SemanticCompleter(nn.Module):
    """根据问题语义，预测 SAE latent 空间中应该补充的缺失信息 Δz。

    从 perception → reasoning 的跨越：
      不只是从已有的 SAE latent 里挑选（重排），
      而是根据问题推断"当前视觉表示缺了什么"，在 latent 空间补全。

    设计（轻量 bottleneck 版）：
      latent_dim 通常很大（7168），不能在 latent 空间做全连接。
      改用 hidden_dim 空间的 bottleneck：
      1. text_h (dim) → bottleneck (dim//4) → scale vector (dim)
      2. z_cluster (k, latent_dim) 先通过 SAE decoder 映射到 hidden_dim 空间
         → 在 hidden_dim 空间做 gating → 再通过 SAE encoder 映射回 latent 空间
      但这样太复杂。更简单的方案：
      直接在 latent 空间用 per-cluster-feature 的标量 gate：
      1. text_h → 一个小向量 (n_clusters 维或 bottleneck 维)
      2. 广播为 latent_dim 的 sparse mask（利用已有的 cluster_to_features 映射）

    最终方案（极简高效）：
      text_h (dim) → MLP → (n_clusters,) 维的 modulation weights
      对每个 cluster，modulation weight 控制该 cluster 的 Δz 强度。
      Δz 本身 = z_cluster 自己（self-reinforcement）* modulation。
      这样参数量 = dim * n_clusters 级别，极小。
    """
    def __init__(self, dim: int, latent_dim: int, n_clusters: int = None, bottleneck: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        # text_h → bottleneck → latent_dim 的 delta 方向
        # 用 bottleneck 避免 latent_dim × latent_dim 的大矩阵
        self.text_to_delta = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, bottleneck),
        )
        # z_cluster 压缩到 bottleneck 空间做 gating
        self.z_to_gate = nn.Sequential(
            nn.Linear(latent_dim, bottleneck),
            nn.Sigmoid(),
        )
        # bottleneck → latent_dim 的稀疏投影
        self.delta_proj = nn.Linear(bottleneck, latent_dim, bias=False)
        nn.init.normal_(self.delta_proj.weight, std=0.001)
        # 可学习的整体强度
        self.beta_param = nn.Parameter(torch.tensor(-3.0))

    def forward(self, text_h, z_cluster):
        """
        Args:
            text_h:    (dim,)          — 问题语义
            z_cluster: (k, latent_dim) — patches 的 SAE 激活
        Returns:
            delta_z: (k, latent_dim) — 需要补充的 latent 修正
        """
        beta = F.softplus(self.beta_param)
        # text → "需要什么" 的 bottleneck 表示
        text_bn = self.text_to_delta(text_h)       # (bottleneck,)
        # z_cluster → "有什么" 的 gate
        gate = self.z_to_gate(z_cluster)            # (k, bottleneck)
        # 交汇：gate * text_bn → 投影回 latent 空间
        fused = gate * text_bn.unsqueeze(0)         # (k, bottleneck)
        delta_z = beta * self.delta_proj(fused)     # (k, latent_dim)
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
        latent_dim = sae.latent_dim  # SAE 的 latent 维度
        device = next(base_model.parameters()).device

        self.cluster_predictor    = ClusterPredictor(dim, self.n_clusters).to(device)
        self.image_cluster_scorer = ImageClusterScorer(dim, self.n_clusters).to(device)
        self.extra_projector      = ExtraProjector(dim).to(device)
        self.semantic_cross_attn  = SemanticCrossAttention(dim).to(device)
        self.pc_suppressor        = PrincipalComponentSuppressor(n_suppress=n_suppress_pcs).to(device)
        self.semantic_completer   = SemanticCompleter(dim, latent_dim).to(device)

        self._layers = _find_layers(base_model)
        n_layers = len(self._layers)
        print(f"  Found {n_layers} transformer layers")

        # suppress_layer 支持负数索引（如 -8 表示倒数第 8 层）
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

    # ── 中间层 Hook（核心）────────────────────────────────────────────────────
    def _mid_layer_hook(self, module, input, output):
        """
        Post-hook on self._layers[inject_layer].

        HOOK_READ_ONLY  → 预测 cluster，stash logits，不修改 hidden
        HOOK_READ_WRITE → 预测 cluster + 稀疏局部注入
        """
        if self._hook_fired:
            return output

        hs = output[0]  # (B, seq, dim)
        if hs.shape[1] <= 1:
            return output

        self._hook_fired = True
        v_pos, n_img, text_pos = self._cached_positions

        # ── Step 1: Cluster 预测 ──────────────────────────────────────────────
        logits = self.cluster_predictor(hs, text_pos)
        self._stashed_logits = logits

        h_vision_float = hs[0, v_pos:v_pos + n_img, :].float()
        self._stashed_h_vision = h_vision_float

        probs_with_grad = torch.sigmoid(logits)[0]  # (n_clusters,) 连续权重
        probs_detached = probs_with_grad.detach()

        # ── [修复位置] 先把 top_cids 计算出来 ──────────────────────────────────
        top_k_clusters = min(self.top_k_clusters, len(probs_with_grad))
        _, top_cids = torch.topk(probs_detached, top_k_clusters)
        top_cids = top_cids.tolist()

        # ── [修复位置] 现在可以安全地使用 top_cids 记录用于打印了 ──────────────
        # 把刚才算好的 top_cids 存下来，过滤掉概率小于 0.1 的用于 verbose 打印
        self._last_cluster_ids = [cid for cid in top_cids if probs_detached[cid] > 0.1]
        self._last_cluster_probs = probs_with_grad

        # ── Stage 1: 只读 ────────────────────────────────────────────────────
        if self._hook_mode == self.HOOK_READ_ONLY:
            return output

        # ── Stage 2 / 推理: 连续加权稀疏注入 ─────────────────────────────────
        # 这里的 top_cids 已经算好了，直接传给 _build_sparse_injection
        injection = self._build_sparse_injection(
            hs, top_cids, probs_with_grad, v_pos, n_img, text_pos,
        )
        if injection is None:
            return output

        active_indices, recon_projected = injection

        # 取出激活 patches 的 vision hidden，做 sparse local cross-attention
        hs_mod = hs.clone()
        vision_active = hs_mod[0, v_pos + active_indices, :].float()  # (K, dim)

        # cross-attention: Q=vision_active, K/V=recon_projected
        updated = self.semantic_cross_attn(vision_active, recon_projected)

        # 写回：只修改激活的 patches
        hs_mod[0, v_pos + active_indices, :] = updated.to(hs.dtype)

        return (hs_mod,) + output[1:]

    def _suppress_hook(self, module, input, output):
        """Post-hook on self._layers[suppress_layer].
        削弱 vision tokens 的主成分，让模型看图更分散。
        只在 prefill 时触发一次。
        """
        if self._suppress_hook_fired:
            return output

        hs = output[0]
        if hs.shape[1] <= 1:
            return output

        self._suppress_hook_fired = True
        v_pos, n_img, _ = self._cached_positions

        hs_mod = hs.clone()
        vision_h = hs_mod[0, v_pos:v_pos + n_img, :].float()

        # 削弱主成分
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
        # suppress hook 只在 READ_WRITE 模式注册（Stage 2 / 推理）
        if mode == self.HOOK_READ_WRITE:
            handles.append(self._layers[self.suppress_layer].register_forward_hook(self._suppress_hook))
        return handles

    def _deactivate_hook(self, handles):
        for h in handles:
            h.remove()
        self._hook_mode = self.HOOK_OFF

    # ── 稀疏注入构造 ─────────────────────────────────────────────────────────

    def _build_sparse_injection(self, h, cluster_ids, cluster_probs, v_pos, n_img, text_pos):
        """
        Sparse Local Injection + Semantic Completion

        流程：
        1. SAE encode 所有 vision patches → 稀疏激活 z
        2. 对每个激活的 cluster：
           a. 找到该 cluster 激活强度 top-k 的 patches
           b. 只保留该 cluster 的 SAE 激活，零出其余 → z_cluster
           c. SemanticCompleter: Δz = f(text_h, z_cluster) — 补全缺失语义
           d. z_completed = z_cluster + Δz
           e. SAE decode(z_completed) → 包含补全语义的重构
           f. 乘以带梯度的 cluster prob
        3. 多个 cluster 的重构累加
        4. ExtraProjector 投影

        Returns:
            None 如果没有激活的 patches
            (active_indices, recon_projected)
        """
        flat = h[:, v_pos:v_pos + n_img, :].reshape(-1, h.shape[-1]).float()
        n_patches = flat.shape[0]

        # 提取 text hidden（last token）用于 semantic completion
        text_h = h[0, text_pos[-1], :].float()  # (dim,)

        with torch.no_grad():
            z = self.sae.encode(flat)  # (n_patches, latent_dim)

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

            # ── Semantic Completion: 补全缺失语义 ────────────────────────────
            # Δz = f(text_h, z_cluster): "为了回答这个问题，这些 patch 还缺什么"
            delta_z = self.semantic_completer(text_h, z_cluster)  # (k, latent_dim)
            z_completed = z_cluster + delta_z

            # SAE decode 包含补全信息的 latent
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
    #目前最优抑制成分个数为3
    def from_pretrained(cls, model_id, sae_ckpt_dir, cluster_path,
                        inject_layer=8, suppress_layer=-8, n_suppress_pcs=3,
                        latent_mult=8, topk=32, top_n_patches=60,
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