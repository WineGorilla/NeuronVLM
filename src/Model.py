"""
QwenWithClusterPredictorAndSAE - Latent Semantic Injection v2

核心改进：
    1. ExtraProjector: hidden state → embedding space (修复 216x 空间不匹配)
    2. Soft Weighted Sum: 不再选 top-K patch，而是对所有 patch 加权求和
       extra = Σ (patch_hidden * cluster_weight)  → 连续控制，不是 hard routing
    3. Image-Cluster Alignment: question→cluster 和 image→cluster 必须一致

Architecture:
    h_vision → SAE encode → z (sparse features)
    per cluster: weight_c = Σ z[:, fids] → softmax over patches → weighted sum
    extra_tokens = [weighted_sum_cluster_0, weighted_sum_cluster_1, ...]
    → ExtraProjector → embedding space → inject → LM decode
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


# ── Image-based Cluster Scorer ────────────────────────────────────────────────

class ImageClusterScorer(nn.Module):
    """
    从 image hidden state 预测 cluster 分布。
    用于 alignment loss：question→cluster 和 image→cluster 要一致。
    """
    def __init__(self, hidden_dim: int, n_clusters: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_clusters),
        )

    def forward(self, h_vision: torch.Tensor) -> torch.Tensor:
        """
        h_vision: (num_patches, hidden_dim)
        returns: (1, n_clusters) logits
        """
        pooled = h_vision.float().mean(dim=0, keepdim=True)
        return self.head(pooled)


# ── Extra Token 投影层 ────────────────────────────────────────────────────────

class ExtraProjector(nn.Module):
    """将 hidden state 空间投影到 embedding 空间"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


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
        align_lambda:      float = 0.3,
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
        self.align_lambda      = align_lambda
        self.max_tokens        = max_tokens

        self.clusters            = {int(k): v for k, v in cluster_info["clusters"].items()}
        self.feature_to_cluster  = {int(k): int(v) for k, v in cluster_info["feature_to_cluster"].items()}
        self.cluster_to_features = {}
        for fid, cid in self.feature_to_cluster.items():
            self.cluster_to_features.setdefault(cid, []).append(fid)
        self.n_clusters = len(self.clusters)

        hidden_dim = base_model.config.text_config.hidden_size
        device     = next(base_model.parameters()).device

        self.cluster_predictor   = ClusterPredictor(hidden_dim, self.n_clusters).to(device)
        self.image_cluster_scorer = ImageClusterScorer(hidden_dim, self.n_clusters).to(device)
        self.extra_projector     = ExtraProjector(hidden_dim).to(device)

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

    def _build_inputs_with_answer(self, image_path: str, question: str, answer: str):
        from qwen_vl_utils import process_vision_info
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

    def _get_hidden_at_layer(self, inputs, layer_idx: int):
        with torch.no_grad():
            outputs = self.base_model(
                **inputs, output_hidden_states=True, return_dict=True,
            )
        return outputs.hidden_states[layer_idx + 1]

    # ── Cluster 预测 ──────────────────────────────────────────────────────────

    def _predict_clusters(self, h, text_positions):
        logits = self.cluster_predictor(h, text_positions)
        probs  = torch.sigmoid(logits.detach())
        cluster_ids = (probs[0] > self.cluster_threshold).nonzero(
            as_tuple=False
        ).squeeze(-1).tolist()
        return logits, cluster_ids, probs[0]

    # ── Soft Weighted Sum（核心改动）─────────────────────────────────────────

    def _get_extra_tokens_soft(
        self,
        h:              torch.Tensor,
        cluster_ids:    list,
        cluster_probs:  torch.Tensor,
        vision_pos:     int,
        num_img_tokens: int,
    ) -> Optional[torch.Tensor]:
        """
        Soft weighted sum：不选 top-K，而是对每个 cluster 做 weighted sum。

        对每个激活的 cluster c:
            1. 收集该 cluster 所有 feature 在每个 patch 上的激活值
            2. 对 patch 维度做 softmax → attention weights
            3. weighted sum: extra_c = Σ_p (w_p * h_p)
            4. 乘以 cluster_prob 作为 gate

        返回: (num_active_clusters, hidden_dim)
        """
        if not cluster_ids:
            return None

        flat = h[:, vision_pos : vision_pos + num_img_tokens, :].reshape(
            -1, h.shape[-1]
        )  # (num_patches, hidden_dim)
        flat_f = flat.float()

        with torch.no_grad():
            z = self.sae.encode(flat_f)  # (num_patches, latent_dim)

        extra_list = []
        for cid in cluster_ids:
            fids = self.cluster_to_features.get(cid, [])
            if not fids:
                continue

            # 收集该 cluster 所有 feature 的激活，汇总到 patch 级别
            valid_fids = [f for f in fids if f < z.shape[-1]]
            if not valid_fids:
                continue

            # (num_patches, num_features_in_cluster) → sum → (num_patches,)
            cluster_acts = z[:, valid_fids].sum(dim=-1)

            if cluster_acts.max().item() <= 0:
                continue

            # softmax over patches → attention weights
            attn_weights = F.softmax(cluster_acts, dim=0)  # (num_patches,)

            # weighted sum of patch hidden states
            # attn_weights: (num_patches,) × flat: (num_patches, hidden_dim)
            weighted = (attn_weights.unsqueeze(-1) * flat_f).sum(dim=0)  # (hidden_dim,)

            # gate by cluster probability
            if cid < len(cluster_probs):
                weighted = weighted * float(cluster_probs[cid])

            extra_list.append(weighted)

        if not extra_list:
            return None

        extra = torch.stack(extra_list, dim=0)  # (num_clusters, hidden_dim)

        # 投影到 embedding 空间
        extra_projected = self.extra_projector(extra)

        return extra_projected

    # ── 构建 inputs_embeds ────────────────────────────────────────────────────

    def _build_with_extra(
        self,
        inputs:       dict,
        extra_hidden: torch.Tensor,
        vision_pos:   int,
        num_img_tokens: int,
        labels:       Optional[torch.Tensor] = None,
    ):
        input_ids      = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        with torch.no_grad():
            inputs_embeds = self.base_model.model.language_model.embed_tokens(input_ids)

        extra     = extra_hidden.to(inputs_embeds.device).to(inputs_embeds.dtype)
        extra     = extra.unsqueeze(0)
        insert_pos = vision_pos + num_img_tokens
        extra_len  = extra.shape[1]

        inputs_embeds_new = torch.cat([
            inputs_embeds[:, :insert_pos, :],
            extra,
            inputs_embeds[:, insert_pos:, :],
        ], dim=1)

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
        for k, v in inputs.items():
            if k not in ["input_ids", "attention_mask", "inputs_embeds"]:
                kwargs[k] = v
        if labels_new is not None:
            kwargs["labels"] = labels_new

        return kwargs, extra_len

    # ── Image-Cluster Alignment Loss ──────────────────────────────────────────

    def _compute_alignment_loss(self, h_vision_flat, text_logits):
        """
        question→cluster 和 image→cluster 必须一致。
        用 KL divergence 对齐两个分布。
        """
        image_logits = self.image_cluster_scorer(h_vision_flat)
        text_probs   = torch.sigmoid(text_logits.detach())
        image_probs  = torch.sigmoid(image_logits)

        # 双向 KL
        eps = 1e-7
        text_probs_  = text_probs.clamp(eps, 1 - eps)
        image_probs_ = image_probs.clamp(eps, 1 - eps)

        kl_forward  = F.kl_div(image_probs_.log(), text_probs_, reduction='batchmean')
        kl_backward = F.kl_div(text_probs_.log(), image_probs_, reduction='batchmean')

        return (kl_forward + kl_backward) / 2

    # ── 训练 loss ─────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        image_path:     str,
        question:       str,
        answer:         str,
        focus_clusters: list,
        include_bce:    bool = True,
    ):
        inputs = self._build_inputs(image_path, question, for_generation=False)
        vision_pos, num_img_tokens, text_positions = self._get_token_positions(inputs)
        h = self._get_hidden_at_layer(inputs, self.inject_layer)

        # Cluster 预测
        text_logits, pred_cluster_ids, cluster_probs = self._predict_clusters(h, text_positions)

        # BCE loss（Stage 1）
        bce_loss_val = 0.0
        bce_loss = torch.tensor(0.0, device=self.device)
        if include_bce:
            logits_with_grad = self.cluster_predictor(h, text_positions)
            target = torch.zeros(1, self.n_clusters, device=self.device)
            for cid in focus_clusters:
                if 0 <= cid < self.n_clusters:
                    target[0, cid] = 1.0
            bce_loss = F.binary_cross_entropy_with_logits(logits_with_grad, target)
            bce_loss_val = bce_loss.item()

            # Image-Cluster alignment loss（Stage 1 一起训练）
            h_vision = h[0, vision_pos:vision_pos + num_img_tokens, :].float()
            align_loss = self._compute_alignment_loss(h_vision, logits_with_grad)
            bce_loss = bce_loss + self.align_lambda * align_loss

        # Soft weighted sum injection
        extra_hidden = self._get_extra_tokens_soft(
            h, pred_cluster_ids, cluster_probs, vision_pos, num_img_tokens
        )

        # 构建带答案的 inputs
        inputs_with_ans = self._build_inputs_with_answer(image_path, question, answer)
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

        if include_bce:
            total_loss = lm_loss + self.bce_lambda * bce_loss
        else:
            total_loss = lm_loss

        return total_loss, lm_loss.item(), bce_loss_val

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
        h = self._get_hidden_at_layer(inputs, self.inject_layer)

        _, cluster_ids, cluster_probs = self._predict_clusters(h, text_positions)
        cluster_names = [self.clusters[c]["name"] for c in cluster_ids if c in self.clusters]

        if verbose:
            print(f"  Predicted clusters: {cluster_ids} → {cluster_names}")
            print(f"  Cluster probs: {[f'{cluster_probs[c]:.3f}' for c in cluster_ids]}")

        # base answer
        base_output_ids = self.base_model.generate(
            **inputs, max_new_tokens=self.max_tokens, do_sample=False,
        )
        input_len   = inputs["input_ids"].shape[1]
        base_answer = self.processor.decode(
            base_output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        if not cluster_ids:
            return {
                "cluster_ids": [], "cluster_names": [],
                "base_answer": base_answer, "final_answer": base_answer,
            }

        # Soft weighted sum
        extra_hidden = self._get_extra_tokens_soft(
            h, cluster_ids, cluster_probs, vision_pos, num_img_tokens
        )

        if verbose:
            k = len(extra_hidden) if extra_hidden is not None else 0
            print(f"  Extra tokens: {k} (one per active cluster)")

        if extra_hidden is not None and len(extra_hidden) > 0:
            kwargs, extra_len = self._build_with_extra(
                inputs, extra_hidden, vision_pos, num_img_tokens
            )
            output_ids = self.base_model.generate(
                **kwargs, max_new_tokens=self.max_tokens, do_sample=False,
            )
            original_input_len = inputs["input_ids"].shape[1]
            prefix_len = original_input_len + extra_len

            if verbose:
                print(f"  output_ids.shape={output_ids.shape}, prefix_len={prefix_len}")

            if output_ids.shape[1] > prefix_len:
                generated_ids = output_ids[0, prefix_len:]
            elif output_ids.shape[1] > original_input_len:
                generated_ids = output_ids[0, original_input_len:]
            else:
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

    # ── 保存 / 加载 ──────────────────────────────────────────────────────────

    def save_predictor(self, path: str):
        state = {
            "cluster_predictor":    self.cluster_predictor.state_dict(),
            "image_cluster_scorer": self.image_cluster_scorer.state_dict(),
            "extra_projector":      self.extra_projector.state_dict(),
        }
        torch.save(state, path)
        print(f"  Saved predictor+scorer+projector: {path}")

    def load_predictor(self, path: str):
        state = torch.load(path, map_location="cpu")

        if "cluster_predictor" in state:
            self.cluster_predictor.load_state_dict(state["cluster_predictor"])
            if "image_cluster_scorer" in state:
                self.image_cluster_scorer.load_state_dict(state["image_cluster_scorer"])
            if "extra_projector" in state:
                self.extra_projector.load_state_dict(state["extra_projector"])
            print(f"  Loaded predictor+scorer+projector: {path}")
        else:
            self.cluster_predictor.load_state_dict(state)
            print(f"  Loaded predictor (legacy format): {path}")

        self.cluster_predictor.to(self.device)
        self.image_cluster_scorer.to(self.device)
        self.extra_projector.to(self.device)

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
        align_lambda:      float = 0.3,
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
            align_lambda      = align_lambda,
        )

        if predictor_ckpt and os.path.exists(predictor_ckpt):
            model.load_predictor(predictor_ckpt)

        model.eval()
        print(f"Model ready. Trainable modules:")
        for name, mod in [("ClusterPredictor", model.cluster_predictor),
                          ("ImageClusterScorer", model.image_cluster_scorer),
                          ("ExtraProjector", model.extra_projector)]:
            n = sum(p.numel() for p in mod.parameters())
            print(f"  {name}: {n:,} params")
        return model