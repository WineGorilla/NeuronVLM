"""
QwenWithClusterPredictorAndSAE：完整模型。

架构：
    1. Qwen 跑到 layer 7（layer 之前）
    2. ClusterPredictor 预测需要关注的 cluster
    3. SAE encode layer 7 的 image token hidden state → z
       找每个预测 cluster 里激活最强的 patch
    4. 注入 feature 语义增强
    5. 继续 layer 8 → 后面的层 → 生成答案

训练：
    BCE loss（cluster 预测）+ LM loss（答案生成）联合训练
    预测头参数量极小，收敛快
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from src.SAE import SAE


# ── Cluster 预测头 ────────────────────────────────────────────────────────────

class ClusterPredictor(nn.Module):
    """
    轻量 MLP，在 layer inject_layer 之前的 hidden state 上预测 cluster。
    输入：text token hidden state 的均值
    输出：每个 cluster 的 logit（多标签分类）
    """

    def __init__(self, hidden_dim: int, n_clusters: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_clusters),
        )

    def forward(self, h: torch.Tensor, text_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:              (1, seq_len, hidden_dim)
            text_positions: text token 的索引
        Returns:
            logits: (1, n_clusters)
        """
        text_h = h[:, text_positions, :]   # (1, n_text, hidden_dim)
        q      = text_h.float().mean(dim=1)  # (1, hidden_dim)
        return self.head(q)                  # (1, n_clusters)


# ── Patch 注入 Hook ───────────────────────────────────────────────────────────

class PatchInjectionHook:
    """
    通过 forward hook 在指定 layer 注入 feature 语义增强。
    在 layer 执行之前注入（pre-hook），让 layer 处理增强后的 hidden state。
    """

    def __init__(
        self,
        inject_ops:     list,   # [(patch_idx, feature_dir, strength), ...]
        vision_pos:     int,
        num_img_tokens: int,
        inject_scale:   float = 0.3,
    ):
        self.inject_ops     = inject_ops
        self.vision_pos     = vision_pos
        self.num_img_tokens = num_img_tokens
        self.inject_scale   = inject_scale
        self.handle         = None

    def hook_fn(self, module, input):
        """pre_forward_hook：在 layer forward 之前修改输入。"""
        if isinstance(input, tuple):
            h    = input[0]
            rest = input[1:]
        else:
            return input

        vp, nit = self.vision_pos, self.num_img_tokens
        if vp < 0 or nit <= 0 or (vp + nit) > h.shape[1]:
            return input

        h_new = h.clone()
        for patch_idx, feature_dir, strength in self.inject_ops:
            if patch_idx >= nit:
                continue
            feature_dir = feature_dir.to(h.device).to(h.dtype)
            h_new[:, vp + patch_idx, :] += \
                self.inject_scale * strength * feature_dir.unsqueeze(0)

        return (h_new,) + rest

    def register(self, layer_module):
        # 用 pre-hook，在 layer 执行之前注入
        self.handle = layer_module.register_forward_pre_hook(self.hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ── 完整模型 ──────────────────────────────────────────────────────────────────

class QwenWithClusterPredictorAndSAE(nn.Module):
    """
    完整模型：Qwen + ClusterPredictor + 冻结 SAE + patch 注入。

    单次 forward 流程：
        1. Qwen 跑到 inject_layer 之前
        2. ClusterPredictor 预测 cluster
        3. SAE encode → 找注入操作
        4. 注册 pre-hook 到 inject_layer
        5. 继续 forward → 生成答案
    """

    def __init__(
        self,
        base_model,
        sae:            SAE,
        processor,
        cluster_info:   dict,
        inject_layer:   int   = 8,
        inject_scale:   float = 0.3,
        top_n_patches:  int   = 60,
        cluster_threshold: float = 0.5,   # sigmoid 输出大于此值则选择该 cluster
        bce_lambda:     float = 0.5,      # BCE loss 权重
        max_tokens:     int   = 128,
    ):
        super().__init__()
        self.base_model         = base_model
        self.sae                = sae
        self.processor          = processor
        self.inject_layer       = inject_layer
        self.inject_scale       = inject_scale
        self.top_n_patches      = top_n_patches
        self.cluster_threshold  = cluster_threshold
        self.bce_lambda         = bce_lambda
        self.max_tokens         = max_tokens

        # cluster 信息
        self.clusters            = {int(k): v for k, v in cluster_info["clusters"].items()}
        self.feature_to_cluster  = {int(k): int(v) for k, v in cluster_info["feature_to_cluster"].items()}
        self.cluster_to_features = {}
        for fid, cid in self.feature_to_cluster.items():
            self.cluster_to_features.setdefault(cid, []).append(fid)
        self.n_clusters = len(self.clusters)

        # ClusterPredictor（可训练）
        hidden_dim = base_model.config.text_config.hidden_size
        self.cluster_predictor = ClusterPredictor(hidden_dim, self.n_clusters).to(
            next(base_model.parameters()).device  # 和 base_model 同设备
        )

        # 冻结 SAE
        self.sae.eval()
        for p in self.sae.parameters():
            p.requires_grad = False

        # 冻结 base_model（除了 cluster_predictor）
        for p in self.base_model.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    # ── 输入构建 ──────────────────────────────────────────────────────────────

    def _build_inputs(self, image_path: str, question: str, for_generation: bool = True):
        from qwen_vl_utils import process_vision_info
        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text":  question},
            ],
        }]]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=for_generation
            )
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        return inputs.to(self.device)

    def _get_token_positions(self, inputs):
        """获取 image token 位置和 text token 位置。"""
        image_grid      = inputs["image_grid_thw"]
        H_grid          = image_grid[0, 1].item()
        W_grid          = image_grid[0, 2].item()
        spatial_merge   = self.base_model.config.vision_config.spatial_merge_size
        num_img_tokens  = int(H_grid * W_grid / (spatial_merge ** 2))
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        input_ids       = inputs["input_ids"][0]
        vision_pos      = (input_ids == vision_start_id).nonzero()[0].item() + 1

        # text token 位置（排除 image token 和特殊 token）
        special_ids = {
            self.processor.tokenizer.convert_tokens_to_ids(t)
            for t in ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
            if self.processor.tokenizer.convert_tokens_to_ids(t) is not None
        }
        text_mask = torch.ones(len(input_ids), dtype=torch.bool, device=input_ids.device)
        text_mask[vision_pos : vision_pos + num_img_tokens] = False
        for sid in special_ids:
            text_mask = text_mask & (input_ids != sid)
        text_positions = text_mask.nonzero(as_tuple=False).squeeze(-1)

        return vision_pos, num_img_tokens, text_positions

    # ── 获取 layer N 之前的 hidden state ─────────────────────────────────────

    def _get_hidden_before_layer(self, inputs, layer_idx: int):
        """
        获取 inject_layer 之前（即 layer_idx - 1 输出）的 hidden state。
        用 output_hidden_states=True，取 hidden_states[layer_idx]。
        注意：hidden_states[0] 是 embedding，hidden_states[i] 是第 i 层的输出。
        """
        with torch.no_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        # hidden_states[inject_layer] = inject_layer 之前那层的输出
        return outputs.hidden_states[layer_idx]

    # ── 找注入操作 ────────────────────────────────────────────────────────────

    def _get_inject_ops(self, h: torch.Tensor, cluster_ids: list,
                        vision_pos: int, num_img_tokens: int) -> list:
        """
        用 inject_layer 之前的 hidden state 做 SAE encode，
        找每个选中 cluster 里激活最强的 patch，构建注入操作。
        """
        flat = h[:, vision_pos : vision_pos + num_img_tokens, :].reshape(
            -1, h.shape[-1]
        ).float()

        with torch.no_grad():
            z = self.sae.encode(flat)   # (num_tokens, latent_dim)

        dec_weight = self.sae.decoder.weight.detach()   # (hidden_dim, latent_dim)
        all_ops    = []

        for cid in cluster_ids:
            fids = self.cluster_to_features.get(cid, [])
            for fid in fids:
                if fid >= z.shape[-1]:
                    continue
                patch_acts = z[:, fid]
                if patch_acts.max().item() <= 0:
                    continue
                top_patch   = patch_acts.argmax().item()
                strength    = float(patch_acts[top_patch])
                feature_dir = dec_weight[:, fid]
                all_ops.append((strength, top_patch, feature_dir))

        # 按激活强度降序，取 top_n_patches
        all_ops.sort(key=lambda x: x[0], reverse=True)
        all_ops = all_ops[:self.top_n_patches]

        return [(patch_idx, feature_dir, strength)
                for strength, patch_idx, feature_dir in all_ops]

    # ── 训练 loss ─────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        image_path:     str,
        question:       str,
        answer:         str,
        focus_clusters: list,
    ):
        """
        联合训练：
            BCE loss：cluster 预测
            LM loss： 注入增强后生成答案
            total = LM loss + bce_lambda * BCE loss
        """
        from qwen_vl_utils import process_vision_info

        inputs = self._build_inputs(image_path, question, for_generation=False)

        # 加上答案的 inputs（用于 LM loss）
        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text":  question},
            ],
        }, {
            "role":    "assistant",
            "content": answer,
        }]]
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs_with_ans = self.processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        labels = self._build_labels(inputs_with_ans["input_ids"])

        # 获取 inject_layer 之前的 hidden state
        vision_pos, num_img_tokens, text_positions = self._get_token_positions(inputs)
        h_before = self._get_hidden_before_layer(inputs, self.inject_layer)

        # ClusterPredictor 前向（可训练，需要梯度）
        logits = self.cluster_predictor(h_before, text_positions)   # (1, n_clusters)

        # BCE loss
        target = torch.zeros(1, self.n_clusters, device=self.device)
        for cid in focus_clusters:
            if 0 <= cid < self.n_clusters:
                target[0, cid] = 1.0
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)

        # 用预测的 cluster 找注入操作（推理时用预测，训练时用真实 label）
        pred_clusters = (torch.sigmoid(logits[0]) > self.cluster_threshold).nonzero(
            as_tuple=False
        ).squeeze(-1).tolist()
        # 训练时用 ground truth cluster（更稳定）
        inject_ops = self._get_inject_ops(
            h_before, focus_clusters, vision_pos, num_img_tokens
        )

        # 注入后的 LM loss
        target_layer = self.base_model.model.language_model.layers[self.inject_layer]
        hook = PatchInjectionHook(
            inject_ops, vision_pos, num_img_tokens, self.inject_scale
        )
        hook.register(target_layer)

        try:
            outputs  = self.base_model(**inputs_with_ans, labels=labels, return_dict=True)
            lm_loss  = outputs.loss
        finally:
            hook.remove()

        total_loss = lm_loss + self.bce_lambda * bce_loss

        return total_loss, lm_loss.item(), bce_loss.item()

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        labels        = input_ids.clone()
        im_start_id   = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_ids = self.processor.tokenizer.encode("assistant", add_special_tokens=False)

        for b in range(input_ids.shape[0]):
            ids          = input_ids[b].tolist()
            answer_start = None
            for i in range(len(ids) - 2, -1, -1):
                if ids[i] == im_start_id:
                    if ids[i+1:i+1+len(assistant_ids)] == assistant_ids:
                        answer_start = i + 1 + len(assistant_ids) + 1
                        break
            if answer_start is None:
                labels[b, :] = -100
            else:
                labels[b, :answer_start] = -100

        return labels.to(self.device)

    # ── 推理 ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, image_path: str, question: str, verbose: bool = False) -> dict:
        """单次 forward 推理。"""
        inputs = self._build_inputs(image_path, question, for_generation=True)
        vision_pos, num_img_tokens, text_positions = self._get_token_positions(inputs)

        # 获取 inject_layer 之前的 hidden state
        h_before = self._get_hidden_before_layer(inputs, self.inject_layer)

        # ClusterPredictor 预测
        logits       = self.cluster_predictor(h_before, text_positions)
        probs        = torch.sigmoid(logits[0])
        cluster_ids  = (probs > self.cluster_threshold).nonzero(
            as_tuple=False
        ).squeeze(-1).tolist()

        if verbose:
            cluster_names = [self.clusters[c]["name"] for c in cluster_ids if c in self.clusters]
            print(f"  Predicted clusters: {cluster_ids} → {cluster_names}")

        # 找注入操作
        inject_ops = self._get_inject_ops(
            h_before, cluster_ids, vision_pos, num_img_tokens
        )

        if verbose:
            print(f"  Inject ops: {len(inject_ops)} (max {self.top_n_patches})")

        # base answer（不注入）
        base_output_ids = self.base_model.generate(
            **inputs, max_new_tokens=self.max_tokens, do_sample=False,
        )
        input_len   = inputs["input_ids"].shape[1]
        base_answer = self.processor.decode(
            base_output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        # 注入后的 answer
        if inject_ops:
            target_layer = self.base_model.model.language_model.layers[self.inject_layer]
            hook = PatchInjectionHook(
                inject_ops, vision_pos, num_img_tokens, self.inject_scale
            )
            hook.register(target_layer)
            try:
                output_ids = self.base_model.generate(
                    **inputs, max_new_tokens=self.max_tokens, do_sample=False,
                )
                final_answer = self.processor.decode(
                    output_ids[0, input_len:], skip_special_tokens=True
                ).strip()
            finally:
                hook.remove()
        else:
            final_answer = base_answer

        cluster_names = [self.clusters[c]["name"] for c in cluster_ids if c in self.clusters]

        return {
            "cluster_ids":   cluster_ids,
            "cluster_names": cluster_names,
            "base_answer":   base_answer,
            "final_answer":  final_answer,
        }

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    # ── 保存 / 加载 ───────────────────────────────────────────────────────────

    def save_predictor(self, path: str):
        torch.save(self.cluster_predictor.state_dict(), path)
        print(f"  ClusterPredictor saved: {path}")

    def load_predictor(self, path: str):
        self.cluster_predictor.load_state_dict(
            torch.load(path, map_location="cpu")
        )
        self.cluster_predictor.to(self.device)
        print(f"  ClusterPredictor loaded: {path}")

    @classmethod
    def from_pretrained(
        cls,
        model_id:          str,
        sae_ckpt_dir:      str,
        cluster_path:      str,
        inject_layer:      int   = 8,
        latent_mult:       int   = 8,
        topk:              int   = 32,
        inject_scale:      float = 0.3,
        top_n_patches:     int   = 60,
        cluster_threshold: float = 0.5,
        bce_lambda:        float = 0.5,
        predictor_ckpt:    str   = None,
        device:            str   = "cuda",
    ):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"Loading Qwen: {model_id}...")
        processor  = AutoProcessor.from_pretrained(model_id)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(device)

        print(f"Loading SAE layer {inject_layer}...")
        hidden_dim = base_model.config.text_config.hidden_size
        latent_dim = hidden_dim * latent_mult
        sae        = SAE(hidden_dim, latent_dim, topk).float().to(device)
        sae.load_state_dict(torch.load(
            os.path.join(sae_ckpt_dir, f"sae_layer{inject_layer}.pt"),
            map_location=device,
        ))

        print(f"Loading cluster info: {cluster_path}...")
        with open(cluster_path) as f:
            cluster_info = json.load(f)

        model = cls(
            base_model         = base_model,
            sae                = sae,
            processor          = processor,
            cluster_info       = cluster_info,
            inject_layer       = inject_layer,
            inject_scale       = inject_scale,
            top_n_patches      = top_n_patches,
            cluster_threshold  = cluster_threshold,
            bce_lambda         = bce_lambda,
        )

        if predictor_ckpt and os.path.exists(predictor_ckpt):
            model.load_predictor(predictor_ckpt)

        model.eval()
        trainable = sum(p.numel() for p in model.cluster_predictor.parameters())
        print(f"Model ready. ClusterPredictor trainable params: {trainable:,}")
        return model