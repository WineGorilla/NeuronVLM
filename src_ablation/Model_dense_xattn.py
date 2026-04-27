"""
DenseCrossAttention Baseline — 在 dense embedding space 做 cross-attention 增强。

审稿人要求的 baseline：
  和 NeuronEye 相同的设置（Layer 8 post-hook, 5K 训练数据, frozen backbone），
  但不使用 SAE 分解和 neuron cluster routing，
  直接在 dense hidden states 上做 text→vision cross-attention。

与 NeuronEye 的对比点：
  - NeuronEye: SAE encode → neuron cluster routing → semantic completion → SAE decode → cross-attention
  - DenseXAttn: 直接用 text hidden 做 Q, vision hidden 做 K/V → cross-attention

  两者参数量相当，训练数据相同，唯一区别是有无 sparse neuron space。
  如果 NeuronEye >> DenseXAttn，证明 sparse neuron space 的结构化推理是关键，
  而非 cross-attention 本身的增益。

推理流程：
  1. 图片+问题 → Qwen forward
  2. Layer 0..7 正常执行
  3. Layer 8 post-hook 触发：
     a. 提取 text hidden states → mean pool 得到 query
     b. 选出与 query 最相关的 top-K vision patches (cosine similarity)
     c. Cross-attention: Q=selected_vision, K/V=text_hidden → 更新 vision patches
  4. Layer 9..末尾 使用修改后的 hidden 继续
  5. generate() 输出

训练流程：
  Stage 1: hook 只读 — 不修改 hidden（可跳过）
  Stage 2: hook 读写 — cross-attention 注入, LM loss
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ── 子模块 ────────────────────────────────────────────────────────────────────

class DenseCrossAttention(nn.Module):
    """Dense space cross-attention: text attend to vision, 更新 vision patches.

    Q = selected vision patches
    K, V = text hidden states (projected)

    与 NeuronEye 的 SemanticCrossAttention 对应，但 K/V 来源不同：
      NeuronEye: K/V = SAE decoded + projected semantic features
      DenseXAttn: K/V = text hidden states (dense, unprojected through SAE)
    """
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

    def forward(self, vision, text_kv):
        """
        Args:
            vision:  (N, dim) — selected vision patch hidden states
            text_kv: (M, dim) — text hidden states as K/V source
        Returns:
            (N, dim) — updated vision patches
        """
        lam = F.softplus(self.lambda_param)
        Q = self.q_proj(vision)     # (N, dim)
        K = self.k_proj(text_kv)    # (M, dim)
        V = self.v_proj(text_kv)    # (M, dim)
        attn = F.softmax(Q @ K.T / self.dim**0.5, dim=-1)  # (N, M)
        return vision + lam * self.norm(self.out_proj(attn @ V))


class TextQueryProjector(nn.Module):
    """Text hidden → query for vision patch selection."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )

    def forward(self, text_h):
        return self.proj(text_h.float())


class PrincipalComponentSuppressor(nn.Module):
    """与 NeuronEye 的 PCS 完全相同，用于公平对比。"""
    def __init__(self, n_suppress: int = 2):
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


# ── 主模型 ────────────────────────────────────────────────────────────────────

class QwenWithDenseCrossAttention(nn.Module):
    """
    Dense Cross-Attention Baseline.

    与 NeuronEye 的 QwenWithClusterPredictorAndSAE 对应，但：
      - 无 SAE
      - 无 neuron clusters
      - 无 ClusterPredictor / ImageClusterScorer
      - 无 SemanticCompleter
      - 有 DenseCrossAttention（直接 text→vision）
      - 有 PCS（与 NeuronEye 相同）
      - 有 TextQueryProjector（用于 vision patch selection）
    """

    HOOK_OFF = 0
    HOOK_READ_WRITE = 2

    def __init__(self, base_model, processor,
                 inject_layer=8, suppress_layer=-8, n_suppress_pcs=2,
                 top_n_patches=60, max_tokens=128):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.inject_layer = inject_layer
        self.top_n_patches = top_n_patches
        self.max_tokens = max_tokens

        dim = base_model.config.text_config.hidden_size
        device = next(base_model.parameters()).device

        self.text_query_proj = TextQueryProjector(dim).to(device)
        self.dense_cross_attn = DenseCrossAttention(dim).to(device)
        self.pc_suppressor = PrincipalComponentSuppressor(n_suppress=n_suppress_pcs).to(device)

        self._layers = _find_layers(base_model)
        n_layers = len(self._layers)
        self.suppress_layer = suppress_layer if suppress_layer >= 0 else n_layers + suppress_layer
        print(f"  Found {n_layers} transformer layers")
        print(f"  Inject @ layer {self.inject_layer}, Suppress @ layer {self.suppress_layer}")

        for p in self.base_model.parameters():
            p.requires_grad = False

        self._hook_mode = self.HOOK_OFF
        self._hook_fired = False
        self._suppress_hook_fired = False
        self._cached_positions = None

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
        ids = inputs["input_ids"][0]
        v_pos = (ids == vs_id).nonzero()[0].item() + 1

        special = {
            self.processor.tokenizer.convert_tokens_to_ids(t)
            for t in ["<|im_start|>", "<|im_end|>",
                       "<|vision_start|>", "<|vision_end|>"]
            if self.processor.tokenizer.convert_tokens_to_ids(t) is not None
        }
        mask = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
        mask[v_pos:v_pos + n_img] = False
        for s in special:
            mask &= (ids != s)
        return v_pos, n_img, mask.nonzero(as_tuple=False).squeeze(-1)

    # ── 中间层 Hook ───────────────────────────────────────────────────────────

    def _mid_layer_hook(self, module, input, output):
        """
        Dense Cross-Attention: 直接用 text hidden 做 K/V，
        选出与 query 最相关的 vision patches 做 cross-attention 更新。
        """
        if self._hook_fired:
            return output

        hs = output[0]
        if hs.shape[1] <= 1:
            return output

        self._hook_fired = True
        v_pos, n_img, text_pos = self._cached_positions

        if self._hook_mode != self.HOOK_READ_WRITE:
            return output

        # === Step 1: 提取 text query ===
        text_h_all = hs[0, text_pos, :].float()  # (n_text, dim)
        query = self.text_query_proj(text_h_all.mean(dim=0))  # (dim,)

        # === Step 2: 选择与 query 最相关的 vision patches ===
        vision_h = hs[0, v_pos:v_pos + n_img, :].float()  # (n_img, dim)
        sim = F.cosine_similarity(
            query.unsqueeze(0).expand(n_img, -1),
            vision_h, dim=-1
        )  # (n_img,)
        top_k = min(self.top_n_patches, n_img)
        _, top_indices = sim.topk(top_k)

        # === Step 3: Dense Cross-Attention ===
        # Q = selected vision patches, K/V = all text hidden states
        selected_vision = vision_h[top_indices]  # (top_k, dim)
        text_kv = text_h_all  # (n_text, dim)

        updated = self.dense_cross_attn(selected_vision, text_kv)  # (top_k, dim)

        # === Step 4: 写回 ===
        hs_mod = hs.clone()
        global_indices = v_pos + top_indices
        hs_mod[0, global_indices, :] = updated.to(hs.dtype)

        return (hs_mod,) + output[1:]

    def _suppress_hook(self, module, input, output):
        """PCS: 与 NeuronEye 完全相同。"""
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

    # ── Hook 管理 ─────────────────────────────────────────────────────────────

    def _activate_hook(self, mode):
        self._hook_mode = mode
        self._hook_fired = False
        self._suppress_hook_fired = False
        handles = []
        handles.append(self._layers[self.inject_layer].register_forward_hook(self._mid_layer_hook))
        if mode == self.HOOK_READ_WRITE:
            handles.append(self._layers[self.suppress_layer].register_forward_hook(self._suppress_hook))
        return handles

    def _deactivate_hook(self, handles):
        for h in handles:
            h.remove()
        self._hook_mode = self.HOOK_OFF

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

        if verbose:
            print(f"  Vision tokens: {n_img}")
            lam = F.softplus(self.dense_cross_attn.lambda_param).item()
            print(f"  λ (softplus): {lam:.4f}")

        return {"final_answer": answer}

    # ── 训练 ──────────────────────────────────────────────────────────────────

    def compute_loss(self, image_path, question, answer, stage=2):
        inputs = self._build_inputs(image_path, question, answer=answer)
        v_pos, n_img, text_pos = self._get_token_positions(inputs)
        labels = self._build_labels(inputs["input_ids"])
        self._cached_positions = (v_pos, n_img, text_pos)

        handles = self._activate_hook(self.HOOK_READ_WRITE)
        try:
            lm_out = self.base_model(**inputs, labels=labels, return_dict=True)
        finally:
            self._deactivate_hook(handles)

        return lm_out.loss, lm_out.loss.item(), 0.0

    def _build_labels(self, input_ids):
        labels = input_ids.clone()
        im_start = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        ast_ids = self.processor.tokenizer.encode("assistant", add_special_tokens=False)
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
            "text_query_proj":   self.text_query_proj.state_dict(),
            "dense_cross_attn":  self.dense_cross_attn.state_dict(),
            "pc_suppressor":     self.pc_suppressor.state_dict(),
        }, path)
        print(f"  Saved: {path}")

    def load_predictor(self, path):
        state = torch.load(path, map_location="cpu")
        for k in ["text_query_proj", "dense_cross_attn", "pc_suppressor"]:
            if k in state:
                getattr(self, k).load_state_dict(state[k])
        for m in [self.text_query_proj, self.dense_cross_attn, self.pc_suppressor]:
            m.to(self.device)
        print(f"  Loaded: {path}")

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    @classmethod
    def from_pretrained(cls, model_id, inject_layer=8, suppress_layer=-8,
                        n_suppress_pcs=2, top_n_patches=60,
                        predictor_ckpt=None, device="cuda"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"Loading Qwen: {model_id}...")
        processor = AutoProcessor.from_pretrained(model_id)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(device)

        model = cls(
            base_model, processor,
            inject_layer=inject_layer,
            suppress_layer=suppress_layer,
            n_suppress_pcs=n_suppress_pcs,
            top_n_patches=top_n_patches,
        )

        if predictor_ckpt and os.path.exists(predictor_ckpt):
            model.load_predictor(predictor_ckpt)

        model.eval()
        total = sum(
            sum(p.numel() for p in m.parameters())
            for m in [model.text_query_proj, model.dense_cross_attn, model.pc_suppressor]
        )
        print(f"  Trainable modules: {total:,} params")
        return model