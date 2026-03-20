"""
QwenWithFocusAndSAE：完整模型。

注入逻辑：
    对每个选中 cluster 里的每个 feature（神经元）：
        找这张图里该 feature 激活最强的 patch
        注入：h[patch] += scale * z[patch, fid] * decoder.weight[:, fid]
    
    所有 cluster 合计最多注入 top_n_patches 个 patch（按激活强度排序取 top）
    每个 feature 只取激活最强的 1 个 patch
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from src.SAE import SAE
from src.dataset import parse_think, strip_think


# ── Patch 注入 Hook ───────────────────────────────────────────────────────────

class PatchInjectionHook:
    """
    通过 forward hook 在指定 layer 注入 feature 语义增强。

    对每个 (patch_idx, feature_dir, activation_strength)：
        h[patch_idx] += scale * activation_strength * feature_dir
    只增强对应 patch，不污染其他区域。
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

    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None

        vp, nit = self.vision_pos, self.num_img_tokens
        if vp < 0 or nit <= 0 or (vp + nit) > h.shape[1]:
            return output

        h_new = h.clone()

        for patch_idx, feature_dir, strength in self.inject_ops:
            if patch_idx >= nit:
                continue
            feature_dir = feature_dir.to(h.device).to(h.dtype)
            h_new[:, vp + patch_idx, :] += \
                self.inject_scale * strength * feature_dir.unsqueeze(0)

        return (h_new,) + rest if rest is not None else h_new

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ── 完整模型 ──────────────────────────────────────────────────────────────────

class QwenWithFocusAndSAE(nn.Module):

    def __init__(
        self,
        base_model,
        sae:               SAE,
        processor,
        cluster_info:      dict,
        layer:             int   = 8,
        inject_scale:      float = 0.3,
        top_n_patches:     int   = 60,   # 最多注入的 patch 数，和 CFG.top_n_patches 一致
        aux_lambda:        float = 0.1,
        aux_margin:        float = 0.1,
        max_think_tokens:  int   = 32,
        max_answer_tokens: int   = 128,
    ):
        super().__init__()
        self.base_model        = base_model
        self.sae               = sae
        self.processor         = processor
        self.layer             = layer
        self.inject_scale      = inject_scale
        self.top_n_patches     = top_n_patches
        self.aux_lambda        = aux_lambda
        self.aux_margin        = aux_margin
        self.max_think_tokens  = max_think_tokens
        self.max_answer_tokens = max_answer_tokens

        # cluster 信息
        self.clusters           = {int(k): v for k, v in cluster_info["clusters"].items()}
        self.feature_to_cluster = {int(k): int(v) for k, v in cluster_info["feature_to_cluster"].items()}
        self.cluster_to_features = {}
        for fid, cid in self.feature_to_cluster.items():
            self.cluster_to_features.setdefault(cid, []).append(fid)

        # 冻结 SAE
        self.sae.eval()
        for p in self.sae.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    # ── 输入构建 ──────────────────────────────────────────────────────────────

    def _build_text_inputs(self, question: str):
        messages = [[{"role": "user", "content": question}]]
        texts    = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages
        ]
        inputs = self.processor(text=texts, padding=True, return_tensors="pt")
        return inputs.to(self.device)

    def _build_image_inputs(self, image_path: str, question: str, for_generation: bool = True):
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

    def _get_image_token_info(self, inputs):
        image_grid      = inputs["image_grid_thw"]
        H_grid          = image_grid[0, 1].item()
        W_grid          = image_grid[0, 2].item()
        spatial_merge   = self.base_model.config.vision_config.spatial_merge_size
        num_img_tokens  = int(H_grid * W_grid / (spatial_merge ** 2))
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        input_ids       = inputs["input_ids"][0]
        vision_pos      = (input_ids == vision_start_id).nonzero()[0].item() + 1
        return vision_pos, num_img_tokens

    # ── 第一次 forward：只用问题文字 → <think> ───────────────────────────────

    def forward_first(self, question: str) -> str:
        inputs = self._build_text_inputs(question)
        with torch.no_grad():
            output_ids = self.base_model.generate(
                **inputs,
                max_new_tokens = self.max_think_tokens,
                do_sample      = False,
            )
        input_len = inputs["input_ids"].shape[1]
        new_ids   = output_ids[:, input_len:]
        return self.processor.decode(new_ids[0], skip_special_tokens=True).strip()

    # ── 找注入操作 ────────────────────────────────────────────────────────────

    def get_inject_ops(self, image_path: str, cluster_ids: list) -> tuple:
        """
        对每个选中 cluster 里的每个 feature：
            找该 feature 在当前图片激活最强的 patch（1 个）
            构建注入操作：(patch_idx, feature_direction, activation_strength)

        所有 feature 合计，按激活强度排序，取 top_n_patches 个。

        Returns:
            inject_ops:     [(patch_idx, feature_dir, strength), ...]
            vision_pos:     image token 起始位置
            num_img_tokens: image token 数量
        """
        inputs = self._build_image_inputs(
            image_path, "Describe the image.", for_generation=False
        )

        with torch.no_grad():
            outputs = self.base_model(
                **inputs, output_hidden_states=True, return_dict=True,
            )

        h               = outputs.hidden_states[self.layer + 1]
        vision_pos, nit = self._get_image_token_info(inputs)
        flat            = h[:, vision_pos : vision_pos + nit, :].reshape(-1, h.shape[-1]).float()

        with torch.no_grad():
            z = self.sae.encode(flat)   # (num_tokens, latent_dim)

        # decoder.weight[:, fid] = feature fid 在 hidden space 的语义方向
        dec_weight = self.sae.decoder.weight.detach()   # (hidden_dim, latent_dim)

        # 收集所有选中 cluster 里的所有 feature 的注入操作
        all_ops = []   # [(strength, patch_idx, feature_dir), ...]

        for cid in cluster_ids:
            fids = self.cluster_to_features.get(cid, [])
            for fid in fids:
                if fid >= z.shape[-1]:
                    continue

                patch_activations = z[:, fid]   # (num_tokens,)

                # 只处理有正激活的 feature
                max_act = patch_activations.max().item()
                if max_act <= 0:
                    continue

                # 每个 feature 只取激活最强的 1 个 patch
                top_patch = patch_activations.argmax().item()
                strength  = float(patch_activations[top_patch])

                feature_dir = dec_weight[:, fid]   # (hidden_dim,)

                all_ops.append((strength, top_patch, feature_dir))

        # 按激活强度降序排序，取 top_n_patches 个
        all_ops.sort(key=lambda x: x[0], reverse=True)
        all_ops = all_ops[:self.top_n_patches]

        # 转成 hook 需要的格式
        inject_ops = [(patch_idx, feature_dir, strength)
                      for strength, patch_idx, feature_dir in all_ops]

        return inject_ops, vision_pos, nit

    # ── 第二次 forward：注入增强 ──────────────────────────────────────────────

    def forward_second(
        self,
        image_path:     str,
        question:       str,
        inject_ops:     list,
        vision_pos:     int,
        num_img_tokens: int,
        labels:         torch.Tensor = None,
    ):
        target_layer = self.base_model.model.language_model.layers[self.layer]
        hook = PatchInjectionHook(
            inject_ops, vision_pos, num_img_tokens, self.inject_scale
        )
        hook.register(target_layer)

        try:
            if labels is not None:
                inputs  = self._build_image_inputs(image_path, question, for_generation=False)
                outputs = self.base_model(**inputs, labels=labels, return_dict=True)
                return outputs.loss
            else:
                inputs = self._build_image_inputs(image_path, question, for_generation=True)
                with torch.no_grad():
                    output_ids = self.base_model.generate(
                        **inputs,
                        max_new_tokens = self.max_answer_tokens,
                        do_sample      = False,
                    )
                input_len = inputs["input_ids"].shape[1]
                new_ids   = output_ids[:, input_len:]
                return self.processor.decode(new_ids[0], skip_special_tokens=True).strip()
        finally:
            hook.remove()

    # ── 训练 loss ─────────────────────────────────────────────────────────────

    def compute_loss(self, image_path: str, question: str, answer: str, focus_clusters: list):
        """
        total = loss_inject + aux_lambda * relu(loss_inject - loss_base + margin)
        """
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
        ).to(self.device)

        labels = self._build_labels(inputs["input_ids"])

        inject_ops, vision_pos, num_img_tokens = self.get_inject_ops(
            image_path, focus_clusters
        )

        loss_inject = self.forward_second(
            image_path, question, inject_ops, vision_pos, num_img_tokens,
            labels=labels,
        )

        with torch.no_grad():
            outputs_base = self.base_model(**inputs, labels=labels, return_dict=True)
            loss_base    = outputs_base.loss

        aux_loss   = F.relu(loss_inject - loss_base + self.aux_margin)
        total_loss = loss_inject + self.aux_lambda * aux_loss

        return total_loss, loss_inject.item(), loss_base.item(), aux_loss.item()

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

    # ── 完整推理 ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, image_path: str, question: str, verbose: bool = False) -> dict:
        if verbose:
            print("[Step 1] First forward (text only)...")
        think_output = self.forward_first(question)
        cluster_ids  = parse_think(think_output)

        if verbose:
            print(f"  Think      : {think_output}")
            print(f"  Cluster ids: {cluster_ids}")

        # base answer
        inputs     = self._build_image_inputs(image_path, question, for_generation=True)
        output_ids = self.base_model.generate(
            **inputs, max_new_tokens=self.max_answer_tokens, do_sample=False,
        )
        input_len   = inputs["input_ids"].shape[1]
        base_answer = self.processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        if not cluster_ids:
            return {
                "think_output":  think_output,
                "cluster_ids":   [],
                "cluster_names": [],
                "base_answer":   base_answer,
                "final_answer":  base_answer,
            }

        cluster_names = [self.clusters[c]["name"] for c in cluster_ids if c in self.clusters]

        if verbose:
            print(f"[Step 2] Building inject ops for: {cluster_names}...")
        inject_ops, vision_pos, num_img_tokens = self.get_inject_ops(
            image_path, cluster_ids
        )
        if verbose:
            print(f"  Total inject ops: {len(inject_ops)} (max {self.top_n_patches})")

        if verbose:
            print("[Step 3] Second forward with feature injection...")
        final_answer = self.forward_second(
            image_path, question, inject_ops, vision_pos, num_img_tokens,
        )
        final_answer = strip_think(final_answer)

        return {
            "think_output":  think_output,
            "cluster_ids":   cluster_ids,
            "cluster_names": cluster_names,
            "base_answer":   base_answer,
            "final_answer":  final_answer,
        }

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    # ── 加载 ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_id:      str,
        sae_ckpt_dir:  str,
        cluster_path:  str,
        layer:         int   = 8,
        latent_mult:   int   = 8,
        topk:          int   = 32,
        inject_scale:  float = 0.3,
        top_n_patches: int   = 60,
        aux_lambda:    float = 0.1,
        aux_margin:    float = 0.1,
        focus_ckpt:    str   = None,
        device:        str   = "cuda",
    ):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"Loading Qwen: {model_id}...")
        processor  = AutoProcessor.from_pretrained(model_id)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device)

        if focus_ckpt and os.path.exists(focus_ckpt):
            print(f"Loading focus weights: {focus_ckpt}...")
            state = torch.load(focus_ckpt, map_location="cpu")
            base_model.load_state_dict(state, strict=False)

        print(f"Loading SAE layer {layer}...")
        hidden_dim = base_model.config.text_config.hidden_size
        latent_dim = hidden_dim * latent_mult
        sae        = SAE(hidden_dim, latent_dim, topk).float().to(device)
        sae.load_state_dict(torch.load(
            os.path.join(sae_ckpt_dir, f"sae_layer{layer}.pt"), map_location=device,
        ))

        print(f"Loading cluster info: {cluster_path}...")
        with open(cluster_path) as f:
            cluster_info = json.load(f)

        model = cls(
            base_model     = base_model,
            sae            = sae,
            processor      = processor,
            cluster_info   = cluster_info,
            layer          = layer,
            inject_scale   = inject_scale,
            top_n_patches  = top_n_patches,
            aux_lambda     = aux_lambda,
            aux_margin     = aux_margin,
        )
        model.eval()
        print("Model ready.")
        return model