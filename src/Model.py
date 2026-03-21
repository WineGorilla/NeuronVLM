"""
QwenWithClusterPredictorAndSAE：完整模型。

架构：
    1. 获取 layer 7 的 hidden state
    2. ClusterPredictor 预测需要关注的 cluster
    3. SAE encode → 找每个 cluster 激活最强的 patch
    4. 把这些 patch 的 layer 7 hidden state 作为 extra token
       直接在 inputs_embeds 层 concat 到 image token 后面
    5. 用 inputs_embeds 调用模型 → 所有 attention/mask 自动对齐
    6. 生成答案

核心思想：
    在 embedding 层操作，不需要 hook
    extra token 的 hidden state 来自 layer 7（和 image token 同一空间）
    所有下游计算自动对齐，不会有维度不匹配问题
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
    def __init__(self, hidden_dim: int, n_clusters: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_clusters),
        )

    def forward(self, h: torch.Tensor, text_positions: torch.Tensor) -> torch.Tensor:
        text_h = h[:, text_positions, :]
        q      = text_h.float().mean(dim=1)
        return self.head(q)


# ── 完整模型 ──────────────────────────────────────────────────────────────────

class QwenWithClusterPredictorAndSAE(nn.Module):

    def __init__(
        self,
        base_model,
        sae:               SAE,
        processor,
        cluster_info:      dict,
        inject_layer:      int   = 8,
        top_n_patches:     int   = 60,
        cluster_threshold: float = 0.5,
        bce_lambda:        float = 0.5,
        max_tokens:        int   = 128,
    ):
        super().__init__()
        self.base_model        = base_model
        self.sae               = sae
        self.processor         = processor
        self.inject_layer      = inject_layer
        self.top_n_patches     = top_n_patches
        self.cluster_threshold = cluster_threshold
        self.bce_lambda        = bce_lambda
        self.max_tokens        = max_tokens

        self.clusters            = {int(k): v for k, v in cluster_info["clusters"].items()}
        self.feature_to_cluster  = {int(k): int(v) for k, v in cluster_info["feature_to_cluster"].items()}
        self.cluster_to_features = {}
        for fid, cid in self.feature_to_cluster.items():
            self.cluster_to_features.setdefault(cid, []).append(fid)
        self.n_clusters = len(self.clusters)

        hidden_dim = base_model.config.text_config.hidden_size
        self.cluster_predictor = ClusterPredictor(hidden_dim, self.n_clusters).to(
            next(base_model.parameters()).device
        )

        self.sae.eval()
        for p in self.sae.parameters():
            p.requires_grad = False

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
        image_grid      = inputs["image_grid_thw"]
        H_grid          = image_grid[0, 1].item()
        W_grid          = image_grid[0, 2].item()
        spatial_merge   = self.base_model.config.vision_config.spatial_merge_size
        num_img_tokens  = int(H_grid * W_grid / (spatial_merge ** 2))
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        input_ids       = inputs["input_ids"][0]
        vision_pos      = (input_ids == vision_start_id).nonzero()[0].item() + 1

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

    def _get_hidden_before_layer(self, inputs, layer_idx: int):
        with torch.no_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[layer_idx]

    # ── 找 extra patch hidden state ───────────────────────────────────────────

    def _get_extra_tokens(
        self,
        h:              torch.Tensor,
        cluster_ids:    list,
        vision_pos:     int,
        num_img_tokens: int,
    ) -> Optional[torch.Tensor]:
        """
        找每个 cluster 激活最强的 patch
        返回这些 patch 的 layer inject_layer 的 hidden state
        shape: (K, hidden_dim)
        """
        flat = h[:, vision_pos : vision_pos + num_img_tokens, :].reshape(
            -1, h.shape[-1]
        ).float()

        with torch.no_grad():
            z = self.sae.encode(flat)

        all_ops = []
        for cid in cluster_ids:
            fids = self.cluster_to_features.get(cid, [])
            for fid in fids:
                if fid >= z.shape[-1]:
                    continue
                patch_acts = z[:, fid]
                if patch_acts.max().item() <= 0:
                    continue
                top_patch = patch_acts.argmax().item()
                strength  = float(patch_acts[top_patch])
                all_ops.append((strength, top_patch))

        if not all_ops:
            return None

        all_ops.sort(key=lambda x: x[0], reverse=True)
        all_ops = all_ops[:self.top_n_patches]

        patch_indices = [idx for _, idx in all_ops]
        extra_hidden  = flat[patch_indices].detach()   # (K, hidden_dim)
        return extra_hidden

    # ── 构建 inputs_embeds + 扩展 attention_mask 和 labels ───────────────────

    def _build_with_extra(
        self,
        inputs:       dict,
        extra_hidden: torch.Tensor,
        vision_pos:   int,
        num_img_tokens: int,
        labels:       Optional[torch.Tensor] = None,
    ):
        """
        在 inputs_embeds 层把 extra token concat 到 image token 后面。
        同步扩展 attention_mask 和 labels。

        Returns:
            dict：可直接传给 base_model 的 kwargs
        """
        input_ids      = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # 获取 inputs_embeds（通过 embedding 层）
        with torch.no_grad():
            inputs_embeds = self.base_model.model.language_model.embed_tokens(input_ids)

        extra     = extra_hidden.to(inputs_embeds.device).to(inputs_embeds.dtype)
        extra     = extra.unsqueeze(0)             # (1, K, hidden_dim)
        insert_pos = vision_pos + num_img_tokens
        extra_len  = extra.shape[1]

        # concat extra token 到 image token 后面
        inputs_embeds_new = torch.cat([
            inputs_embeds[:, :insert_pos, :],
            extra,
            inputs_embeds[:, insert_pos:, :],
        ], dim=1)

        # 扩展 attention_mask
        attention_mask_new = None
        if attention_mask is not None:
            extra_mask = torch.ones(
                attention_mask.shape[0], extra_len,
                dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask_new = torch.cat([
                attention_mask[:, :insert_pos],
                extra_mask,
                attention_mask[:, insert_pos:],
            ], dim=1)

        # 扩展 labels
        labels_new = None
        if labels is not None:
            extra_labels = torch.full(
                (labels.shape[0], extra_len), -100,
                dtype=labels.dtype, device=labels.device
            )
            labels_new = torch.cat([
                labels[:, :insert_pos],
                extra_labels,
                labels[:, insert_pos:],
            ], dim=1)

        kwargs = {
            "inputs_embeds": inputs_embeds_new,
            "attention_mask": attention_mask_new,
        }
        # 不传 input_ids，用 inputs_embeds 替代
        # 传递其他必要的字段
        for k, v in inputs.items():
            if k not in ["input_ids", "attention_mask", "inputs_embeds"]:
                kwargs[k] = v

        if labels_new is not None:
            kwargs["labels"] = labels_new

        return kwargs, extra_len   # ← 同时返回 extra_len，方便后续计算偏移

    # ── 训练 loss ─────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        image_path:     str,
        question:       str,
        answer:         str,
        focus_clusters: list,
    ):
        from qwen_vl_utils import process_vision_info

        inputs = self._build_inputs(image_path, question, for_generation=False)
        vision_pos, num_img_tokens, text_positions = self._get_token_positions(inputs)
        h_before = self._get_hidden_before_layer(inputs, self.inject_layer)

        # ClusterPredictor 前向（需要梯度）
        logits = self.cluster_predictor(h_before, text_positions)

        # BCE loss
        target = torch.zeros(1, self.n_clusters, device=self.device)
        for cid in focus_clusters:
            if 0 <= cid < self.n_clusters:
                target[0, cid] = 1.0
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)

        # 找 extra token
        extra_hidden = self._get_extra_tokens(
            h_before, focus_clusters, vision_pos, num_img_tokens
        )

        # 构建带答案的 inputs
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

        if extra_hidden is not None and len(extra_hidden) > 0:
            kwargs, _ = self._build_with_extra(
                inputs_with_ans, extra_hidden, vision_pos, num_img_tokens, labels
            )
        else:
            kwargs = dict(inputs_with_ans)
            kwargs["labels"] = labels

        outputs = self.base_model(**kwargs, return_dict=True)
        lm_loss = outputs.loss

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
        inputs = self._build_inputs(image_path, question, for_generation=True)
        vision_pos, num_img_tokens, text_positions = self._get_token_positions(inputs)
        h_before = self._get_hidden_before_layer(inputs, self.inject_layer)

        # ClusterPredictor 预测
        logits      = self.cluster_predictor(h_before, text_positions)
        probs       = torch.sigmoid(logits[0])
        cluster_ids = (probs > self.cluster_threshold).nonzero(
            as_tuple=False
        ).squeeze(-1).tolist()

        cluster_names = [self.clusters[c]["name"] for c in cluster_ids if c in self.clusters]

        if verbose:
            print(f"  Predicted clusters: {cluster_ids} → {cluster_names}")

        # base answer（不注入）
        base_output_ids = self.base_model.generate(
            **inputs, max_new_tokens=self.max_tokens, do_sample=False,
        )
        input_len   = inputs["input_ids"].shape[1]
        base_answer = self.processor.decode(
            base_output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        if not cluster_ids:
            return {
                "cluster_ids":   [],
                "cluster_names": [],
                "base_answer":   base_answer,
                "final_answer":  base_answer,
            }

        # 找 extra token
        extra_hidden = self._get_extra_tokens(
            h_before, cluster_ids, vision_pos, num_img_tokens
        )

        if verbose:
            k = len(extra_hidden) if extra_hidden is not None else 0
            print(f"  Extra tokens: {k} (max {self.top_n_patches})")

        # ── 注入后推理（修复解码偏移） ────────────────────────────────────────
        if extra_hidden is not None and len(extra_hidden) > 0:
            kwargs, extra_len = self._build_with_extra(
                inputs, extra_hidden, vision_pos, num_img_tokens
            )

            output_ids = self.base_model.generate(
                **kwargs, max_new_tokens=self.max_tokens, do_sample=False,
            )

            # ── 关键修复 ──────────────────────────────────────────────────────
            # 用 inputs_embeds 调 generate() 时，Qwen 返回的 output_ids
            # 前面会填充 embed_len 个 dummy token id（通常是 0）。
            # 但 embed_len = 原始 input_ids 长度 + extra_len，
            # 所以正确的切分方式是：原始长度 + extra_len。
            original_input_len = inputs["input_ids"].shape[1]
            prefix_len = original_input_len + extra_len

            if verbose:
                print(f"  output_ids.shape={output_ids.shape}, "
                      f"original_input_len={original_input_len}, "
                      f"extra_len={extra_len}, prefix_len={prefix_len}")

            if output_ids.shape[1] > prefix_len:
                generated_ids = output_ids[0, prefix_len:]
            elif output_ids.shape[1] > original_input_len:
                # 回退：也许 generate 没有把 extra token 算进去
                generated_ids = output_ids[0, original_input_len:]
            else:
                # 最后兜底：整个 output 都是生成的
                generated_ids = output_ids[0]

            final_answer = self.processor.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
        else:
            final_answer = base_answer

        return {
            "cluster_ids":   cluster_ids,
            "cluster_names": cluster_names,
            "base_answer":   base_answer,
            "final_answer":  final_answer,
        }

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

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
            base_model        = base_model,
            sae               = sae,
            processor         = processor,
            cluster_info      = cluster_info,
            inject_layer      = inject_layer,
            top_n_patches     = top_n_patches,
            cluster_threshold = cluster_threshold,
            bce_lambda        = bce_lambda,
        )

        if predictor_ckpt and os.path.exists(predictor_ckpt):
            model.load_predictor(predictor_ckpt)

        model.eval()
        trainable = sum(p.numel() for p in model.cluster_predictor.parameters())
        print(f"Model ready. ClusterPredictor trainable params: {trainable:,}")
        return model