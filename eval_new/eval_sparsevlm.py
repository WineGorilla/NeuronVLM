"""
SparseVLM 评估脚本 — 在 Qwen2.5-VL 上复现 SparseVLM 算法。

SparseVLM 核心思想 (Zhang et al., ICML 2025):
  1. Text-Guided Sparsification: 用 text token 对 visual token 的 cross-attention
     来评估视觉 token 的重要性（而非 FastV 那样 text-agnostic）
  2. 逐层渐进剪枝: 在多个层中逐步减少视觉 token 数量
  3. Token Recycling: 被剪掉的 token 不是直接丢弃，而是聚类压缩后合并回去

本脚本通过 monkey-patch Qwen2.5-VL 的 decoder layers 实现 SparseVLM，
无需 clone SparseVLM 仓库，可在同一基座模型上公平对比。

支持 4 个 benchmark：CV-Bench, BLINK, RealWorldQA, MMStar。

用法：
    python eval/eval_sparsevlm.py --benchmarks cvbench mmstar
    python eval/eval_sparsevlm.py --retain_tokens 192
    python eval/eval_sparsevlm.py --retain_tokens 128 --benchmarks blink --depth_spatial_only
    python eval/eval_sparsevlm.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --max_samples 50

依赖：
    pip install transformers>=4.45.0 torch accelerate qwen-vl-utils datasets
"""
import os
import sys
import re
import json
import argparse
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# 默认配置
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Qwen2.5-VL 的特殊 token id
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656


# ══════════════════════════════════════════════════════════════════════════════
# SparseVLM: Text-Guided Visual Token Sparsification
# ══════════════════════════════════════════════════════════════════════════════

class SparseVLMWrapper:
    """
    SparseVLM 实现：text-guided 逐层渐进式视觉 token 剪枝 + token recycling。

    算法步骤：
      1. Rater Selection: 从 text token 中选出与视觉信号强相关的 token（raters），
         通过计算 text→visual 的 cross-attention 分数来选取
      2. Visual Token Scoring: 用 raters 对每个 visual token 评分
         （即 raters 给每个 visual token 的 attention 之和）
      3. Rank-Based Pruning: 按分数排序，保留 top-K 个 visual token
      4. Token Recycling: 被剪掉的 token 通过均值聚类压缩成少量代表 token，
         追加回序列中以减少信息损失

    与 FastV 的关键区别：
      - FastV 是 text-agnostic（不看问题内容），SparseVLM 是 text-guided
      - FastV 一次性剪枝，SparseVLM 逐层渐进
      - SparseVLM 有 token recycling，不直接丢弃信息
    """

    def __init__(self, model, retain_tokens=192, n_recycle_clusters=8,
                 prune_start_layer=2, prune_end_layer=None, n_prune_steps=4):
        """
        Args:
            model: Qwen2_5_VLForConditionalGeneration 实例
            retain_tokens: 最终保留的视觉 token 数量（论文默认 192/128/96/64）
            n_recycle_clusters: token recycling 时的聚类数
            prune_start_layer: 从第几层开始剪枝
            prune_end_layer: 到第几层结束剪枝（None = 总层数的一半）
            n_prune_steps: 分几步逐渐剪枝到目标数量
        """
        self.model = model
        self.retain_tokens = retain_tokens
        self.n_recycle_clusters = n_recycle_clusters
        self.prune_start_layer = prune_start_layer
        self.n_prune_steps = n_prune_steps

        n_layers = len(model.model.language_model.layers)
        self.prune_end_layer = prune_end_layer or (n_layers // 2)

        # 计算每一步剪枝到多少 token
        self._prune_schedule = None  # 在运行时根据实际 img token 数量计算

        self._hooks = []
        self._image_token_mask = None
        self._current_n_visual = None  # 当前剩余的视觉 token 数量
        self._pruned_at_step = 0
        self._prune_layers = []  # 哪些层需要执行剪枝

    def _compute_prune_schedule(self, n_visual_tokens):
        """计算逐层剪枝的 schedule: 从 n_visual_tokens 逐步减少到 retain_tokens。"""
        if n_visual_tokens <= self.retain_tokens:
            return []  # 不需要剪枝

        # 在 [prune_start, prune_end] 之间均匀分配 n_prune_steps 个剪枝点
        layer_span = self.prune_end_layer - self.prune_start_layer
        step_interval = max(1, layer_span // self.n_prune_steps)

        prune_layers = []
        for step in range(self.n_prune_steps):
            layer_idx = self.prune_start_layer + step * step_interval
            if layer_idx < self.prune_end_layer:
                prune_layers.append(layer_idx)

        if not prune_layers:
            prune_layers = [self.prune_start_layer]

        # 每一步保留的 token 数量（线性递减）
        schedule = []
        for i, layer_idx in enumerate(prune_layers):
            # 线性插值：从 n_visual_tokens 到 retain_tokens
            ratio = (i + 1) / len(prune_layers)
            n_keep = int(n_visual_tokens - ratio * (n_visual_tokens - self.retain_tokens))
            n_keep = max(n_keep, self.retain_tokens)
            schedule.append((layer_idx, n_keep))

        self._prune_layers = [s[0] for s in schedule]
        return schedule

    def set_image_token_mask(self, input_ids):
        """根据 input_ids 定位 image/video token 的位置。"""
        self._image_token_mask = (
            (input_ids == IMAGE_TOKEN_ID) | (input_ids == VIDEO_TOKEN_ID)
        )
        n_visual = self._image_token_mask.sum().item()
        self._current_n_visual = n_visual
        self._prune_schedule = self._compute_prune_schedule(n_visual)
        self._pruned_at_step = 0

    def install_hooks(self):
        self._hooks = []
        layers = self.model.model.language_model.layers
        for layer_idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, args, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            seq_len = hidden_states.shape[1]
            if seq_len <= 1:
                return output

            # 检查是否需要在此层剪枝
            if self._prune_schedule is None or self._pruned_at_step >= len(self._prune_schedule):
                return output

            target_layer, n_keep = self._prune_schedule[self._pruned_at_step]
            if layer_idx != target_layer:
                return output

            self._do_text_guided_pruning(hidden_states, n_keep)
            self._pruned_at_step += 1

            return output
        return hook_fn

    def _do_text_guided_pruning(self, hidden_states, n_keep):
        """
        SparseVLM 的 text-guided 剪枝 + token recycling。

        步骤：
          1. 找到 visual token 和 text token 的位置
          2. 计算 text→visual 的 attention 相似度（用 dot product 代理）
          3. 选出与视觉信号最相关的 text token 作为 raters
          4. 用 raters 给 visual token 评分
          5. 保留得分最高的 n_keep 个 visual token
          6. 被剪掉的 token 聚类压缩（token recycling）后均值替换
        """
        if self._image_token_mask is None:
            return

        batch_size = hidden_states.shape[0]

        for b in range(batch_size):
            img_mask = self._image_token_mask[b]
            if img_mask.shape[0] != hidden_states.shape[1]:
                continue

            img_positions = img_mask.nonzero(as_tuple=True)[0]
            text_positions = (~img_mask).nonzero(as_tuple=True)[0]

            n_img = len(img_positions)
            n_text = len(text_positions)

            if n_img <= n_keep or n_text == 0:
                continue

            # === Step 1-2: Text-Visual Correlation (Rater Selection) ===
            img_hidden = hidden_states[b, img_positions, :].float()   # [n_img, D]
            text_hidden = hidden_states[b, text_positions, :].float() # [n_text, D]

            # 归一化后计算余弦相似度
            img_norm = F.normalize(img_hidden, dim=-1)
            text_norm = F.normalize(text_hidden, dim=-1)

            # text→visual attention: [n_text, n_img]
            cross_attn = torch.matmul(text_norm, img_norm.T)

            # 选择与 visual token 最相关的 text token 作为 raters
            # 方式：取每个 text token 对所有 visual token 的平均 attention，选 top-K
            text_visual_relevance = cross_attn.mean(dim=1)  # [n_text]
            n_raters = min(max(4, n_text // 4), n_text)
            _, rater_indices = text_visual_relevance.topk(n_raters)

            # === Step 3: Visual Token Scoring by Raters ===
            rater_attn = cross_attn[rater_indices, :]  # [n_raters, n_img]
            visual_scores = rater_attn.sum(dim=0)       # [n_img]

            # === Step 4: Rank-Based Pruning ===
            _, keep_indices = visual_scores.topk(n_keep)
            _, prune_indices = visual_scores.topk(n_img - n_keep, largest=False)

            keep_positions = img_positions[keep_indices]
            prune_positions = img_positions[prune_indices]

            # === Step 5: Token Recycling ===
            # 将被剪掉的 token 聚类压缩成 n_recycle_clusters 个代表 token
            pruned_hidden = hidden_states[b, prune_positions, :].float()
            n_pruned = len(prune_positions)

            if n_pruned > 0 and self.n_recycle_clusters > 0:
                n_clusters = min(self.n_recycle_clusters, n_pruned)

                # 简单均匀分组聚类（比 K-Means 快，效果接近）
                chunk_size = n_pruned // n_clusters
                recycled_tokens = []
                for c in range(n_clusters):
                    start = c * chunk_size
                    end = start + chunk_size if c < n_clusters - 1 else n_pruned
                    cluster_mean = pruned_hidden[start:end].mean(dim=0)
                    recycled_tokens.append(cluster_mean)

                # 将 recycled tokens 写入被剪掉位置的前几个位置
                # （保持序列长度不变，剩余位置置零）
                recycled = torch.stack(recycled_tokens).to(hidden_states.dtype)
                for i, pos in enumerate(prune_positions[:n_clusters]):
                    hidden_states[b, pos, :] = recycled[i]

                # 剩余被剪掉的位置 → scatter-copy 到最近的保留 token
                if n_clusters < n_pruned:
                    remaining_prune = prune_positions[n_clusters:]
                    rem_hidden = hidden_states[b, remaining_prune, :].float()
                    all_keep = hidden_states[b, keep_positions, :].float()
                    sim = torch.matmul(
                        F.normalize(rem_hidden, dim=-1),
                        F.normalize(all_keep, dim=-1).T
                    )
                    nearest = sim.argmax(dim=1)
                    for ii, pos in enumerate(remaining_prune):
                        hidden_states[b, pos, :] = hidden_states[b, keep_positions[nearest[ii]], :]
            else:
                # 没有 recycling → scatter-copy 到最近的保留 token
                keep_hidden_f = hidden_states[b, keep_positions, :].float()
                prune_hidden_f = hidden_states[b, prune_positions, :].float()
                sim = torch.matmul(
                    F.normalize(prune_hidden_f, dim=-1),
                    F.normalize(keep_hidden_f, dim=-1).T
                )
                nearest = sim.argmax(dim=1)
                for ii, pos in enumerate(prune_positions):
                    hidden_states[b, pos, :] = hidden_states[b, keep_positions[nearest[ii]], :]

    def patch_generate(self):
        original_generate = self.model.generate

        @wraps(original_generate)
        def patched_generate(*args, **kwargs):
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            if input_ids is not None:
                self.set_image_token_mask(input_ids)

            self.install_hooks()
            try:
                result = original_generate(*args, **kwargs)
            finally:
                self.remove_hooks()
                self._pruned_at_step = 0
            return result

        self.model.generate = patched_generate
        self._original_generate = original_generate

    def unpatch_generate(self):
        if hasattr(self, "_original_generate"):
            self.model.generate = self._original_generate


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen25vl(model_id, retain_tokens=192, n_recycle_clusters=8):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading Qwen2.5-VL: {model_id}")
    # 限制 max_pixels 避免超大图 OOM（默认最大 16384 token，这里限制到 1280*28*28）
    processor = AutoProcessor.from_pretrained(
        model_id,
        max_pixels=1280 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    n_layers = len(model.model.language_model.layers)
    print(f"  Model loaded. Layers: {n_layers}")

    sparse = SparseVLMWrapper(
        model, retain_tokens=retain_tokens,
        n_recycle_clusters=n_recycle_clusters,
    )
    sparse.patch_generate()
    print(f"  SparseVLM installed: retain={retain_tokens}, "
          f"recycle_clusters={n_recycle_clusters}")
    if sparse._prune_schedule:
        print(f"  Prune schedule (example): {sparse._prune_schedule}")

    return model, processor, sparse


def qwen25vl_generate(model, processor, image, question, max_new_tokens=32):
    """Qwen2.5-VL 推理。"""
    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            return ""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(
            output_ids[0, input_len:],
            skip_special_tokens=True,
        ).strip()
        return response

    except Exception as e:
        print(f"  [error] generate failed: {e}")
        torch.cuda.empty_cache()
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 答案匹配工具
# ══════════════════════════════════════════════════════════════════════════════

def extract_choice_letter(response, max_letter="Z"):
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None
    pat = f'[A-{max_letter}]'
    m = re.search(rf'\(({pat})\)', text)
    if m:
        return m.group(1)
    m = re.search(rf'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?({pat})\)?', text)
    if m:
        return m.group(1)
    m = re.match(rf'^({pat})(?:[\s.,):]|$)', text)
    if m:
        return m.group(1)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CV-Bench
# ══════════════════════════════════════════════════════════════════════════════

def eval_cvbench(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  CV-Bench Evaluation\n{'='*60}")

    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for item in tqdm(ds, desc="CV-Bench"):
        choices = item["choices"]
        choice_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
        prompt = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter.\n"
            "Do NOT output explanation.\n\n"
            f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
        )

        response = qwen25vl_generate(model, processor, item["image"], prompt)
        extracted = extract_choice_letter(response)
        answer = item["answer"]
        extracted_fmt = f"({extracted})" if extracted else None
        correct = extracted_fmt == answer

        results.append({
            "idx": item["idx"], "task": item["task"],
            "source": item["source"], "type": item["type"],
            "answer": answer, "response": response,
            "extracted": extracted_fmt, "correct": correct,
        })

    by_source = defaultdict(list)
    by_task = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)
        by_task[r["task"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    acc_ade = acc(by_source.get("ADE20K", []))
    acc_coco = acc(by_source.get("COCO", []))
    acc_omni = acc(by_source.get("Omni3D", []))
    acc_2d = (acc_ade + acc_coco) / 2
    acc_3d = acc_omni
    cv_bench = (acc_2d + acc_3d) / 2
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  CV-Bench Overall : {cv_bench:.2f}")
    print(f"  2D Accuracy      : {acc_2d:.2f}  (ADE={acc_ade:.2f}, COCO={acc_coco:.2f})")
    print(f"  3D Accuracy      : {acc_3d:.2f}  (Omni3D={acc_omni:.2f})")
    print(f"  Per-Task:")
    for t in sorted(by_task.keys()):
        print(f"    {t:20s}: {acc(by_task[t]):6.2f}  (n={len(by_task[t])})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {
        "benchmark": "CV-Bench", "cv_bench": cv_bench,
        "acc_2d": acc_2d, "acc_3d": acc_3d,
        "per_task": {t: acc(items) for t, items in by_task.items()},
        "n_unparsed": n_unparsed, "n_total": len(results),
    }, results


# ══════════════════════════════════════════════════════════════════════════════
# MMStar
# ══════════════════════════════════════════════════════════════════════════════

def eval_mmstar(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  MMStar Evaluation\n{'='*60}")

    ds = load_dataset("Lin-Chen/MMStar", split="val")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for item in tqdm(ds, desc="MMStar"):
        prompt = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter (A, B, C, or D).\n"
            "Do NOT output explanation.\n\n"
            f"{item['question']}\n\nAnswer:"
        )

        response = qwen25vl_generate(model, processor, item["image"], prompt)
        extracted = extract_choice_letter(response, max_letter="D")
        answer = item["answer"].strip().upper()
        correct = extracted == answer

        results.append({
            "index": item["index"], "category": item["category"],
            "l2_category": item["l2_category"],
            "answer": answer, "response": response,
            "extracted": extracted, "correct": correct,
        })

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    overall = acc(results)
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  Overall Accuracy : {overall:.2f}  (n={len(results)})")
    print(f"  Per Category:")
    cat_accs = {}
    for cat in sorted(by_category.keys()):
        a = acc(by_category[cat])
        cat_accs[cat] = a
        print(f"    {cat:30s}: {a:6.2f}  (n={len(by_category[cat])})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {
        "benchmark": "MMStar", "overall": overall,
        "per_category": cat_accs,
        "n_unparsed": n_unparsed, "n_total": len(results),
    }, results


# ══════════════════════════════════════════════════════════════════════════════
# RealWorldQA
# ══════════════════════════════════════════════════════════════════════════════

def detect_question_type(question, answer):
    ans = answer.strip()
    if ans in ("A", "B", "C", "D"):
        return "mcq"
    if ans.lower() in ("yes", "no"):
        return "yes_no"
    return "short"


def match_realworldqa(response, answer, q_type):
    resp = response.strip()
    ans = answer.strip()

    if q_type == "mcq":
        extracted = extract_choice_letter(resp, max_letter="D")
        if extracted:
            return extracted == ans, extracted
        return False, None

    if q_type == "yes_no":
        resp_lower = resp.lower()
        if resp_lower.startswith("yes"):
            return "Yes" == ans, "Yes"
        if resp_lower.startswith("no"):
            return "No" == ans, "No"
        return False, None

    resp_norm = resp.split("\n")[0].strip().lower().rstrip(".")
    ans_norm = ans.lower().rstrip(".")
    if resp_norm == ans_norm or resp_norm.startswith(ans_norm):
        return True, resp
    try:
        if float(resp_norm) == float(ans_norm):
            return True, resp
    except ValueError:
        pass
    return False, resp


def eval_realworldqa(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  RealWorldQA Evaluation\n{'='*60}")

    ds = load_dataset("xai-org/RealworldQA", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for i, item in enumerate(tqdm(ds, desc="RealWorldQA")):
        question = item["question"]
        answer = item["answer"]
        q_type = detect_question_type(question, answer)

        response = qwen25vl_generate(model, processor, item["image"], question)
        correct, extracted = match_realworldqa(response, answer, q_type)

        results.append({
            "idx": i, "q_type": q_type, "answer": answer,
            "response": response, "extracted": extracted, "correct": correct,
        })

    by_type = defaultdict(list)
    for r in results:
        by_type[r["q_type"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    overall = acc(results)
    n_unparsed = sum(1 for r in results if r.get("extracted") is None and r["q_type"] == "mcq")

    print(f"\n  Overall Accuracy : {overall:.2f}  (n={len(results)})")
    print(f"  Per Type:")
    per_type = {}
    for t in ["mcq", "yes_no", "short"]:
        if t in by_type:
            a = acc(by_type[t])
            per_type[t] = a
            print(f"    {t:10s}: {a:6.2f}  (n={len(by_type[t])})")
    print(f"  Unparsed MCQ: {n_unparsed}/{len(by_type.get('mcq', []))}")

    return {
        "benchmark": "RealWorldQA", "overall": overall,
        "per_type": per_type,
        "n_unparsed": n_unparsed, "n_total": len(results),
    }, results


# ══════════════════════════════════════════════════════════════════════════════
# BLINK
# ══════════════════════════════════════════════════════════════════════════════

ALL_BLINK_SUBTASKS = [
    "Art_Style", "Counting", "Forensic_Detection",
    "Functional_Correspondence", "IQ_Test", "Jigsaw",
    "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance",
    "Semantic_Correspondence", "Spatial_Relation",
    "Visual_Correspondence", "Visual_Similarity",
]

DEPTH_SPATIAL_SUBTASKS = [
    "Relative_Depth", "Spatial_Relation",
    "Multi-view_Reasoning", "Object_Localization",
]


def concat_images_horizontal(image_paths, max_height=768):
    if len(image_paths) == 1:
        return image_paths[0]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    target_h = min(max_height, max(img.height for img in images))
    resized = []
    for img in images:
        ratio = target_h / img.height
        resized.append(img.resize((int(img.width * ratio), target_h), Image.LANCZOS))
    total_w = sum(img.width for img in resized)
    concat = Image.new("RGB", (total_w, target_h))
    x = 0
    for img in resized:
        concat.paste(img, (x, 0))
        x += img.width
    out = "/tmp/blink_eval_concat.png"
    concat.save(out)
    return out


def eval_blink(model, processor, subtasks=None, max_samples=None):
    print(f"\n{'='*60}\n  BLINK Evaluation\n{'='*60}")

    if subtasks is None:
        subtasks = ALL_BLINK_SUBTASKS

    all_items = []
    for subtask in subtasks:
        try:
            ds = load_dataset("BLINK-Benchmark/BLINK", subtask, split="val")
            for i, item in enumerate(ds):
                choices = item.get("choices", item.get("options", []))
                if isinstance(choices, str):
                    try:
                        choices = json.loads(choices)
                    except json.JSONDecodeError:
                        choices = [c.strip() for c in choices.split(",")]
                all_items.append({
                    "idx": f"{subtask}_{i}", "subtask": subtask,
                    "prompt": item.get("prompt", item.get("question", "")),
                    "answer": item.get("answer", ""), "choices": choices,
                    "image_1": item.get("image_1", item.get("image", None)),
                    "image_2": item.get("image_2", None),
                    "image_3": item.get("image_3", None),
                    "image_4": item.get("image_4", None),
                })
            print(f"  {subtask}: {len(ds)} samples")
        except Exception as e:
            print(f"  [error] {subtask}: {e}")

    if max_samples and len(all_items) > max_samples:
        per_task = max(1, max_samples // len(subtasks))
        by_task = defaultdict(list)
        for item in all_items:
            by_task[item["subtask"]].append(item)
        sampled = []
        for items in by_task.values():
            sampled.extend(items[:per_task])
        all_items = sampled[:max_samples]

    print(f"  Total: {len(all_items)} samples")

    results = []
    for item in tqdm(all_items, desc="BLINK"):
        choices = item["choices"]
        has_image_choices = (
            isinstance(choices, list) and len(choices) > 0
            and isinstance(choices[0], Image.Image)
        )
        if has_image_choices:
            choice_text = "\n".join(
                [f"{chr(ord('A')+i)}. (Image {i+1})" for i in range(len(choices))]
            )
        else:
            choice_text = "\n".join(
                [f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)]
            )
        prompt_text = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter.\n"
            "Do NOT output explanation.\n\n"
            f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
        )

        tmp_dir = "/tmp/blink_eval"
        os.makedirs(tmp_dir, exist_ok=True)
        image_paths = []
        for key in ["image_1", "image_2", "image_3", "image_4"]:
            img = item.get(key)
            if img is not None and isinstance(img, Image.Image):
                p = os.path.join(tmp_dir, f"{key}.png")
                img.save(p)
                image_paths.append(p)

        if not image_paths:
            results.append({
                "idx": item["idx"], "subtask": item["subtask"],
                "answer": item["answer"], "response": "",
                "extracted": None, "correct": False,
            })
            continue

        concat_path = concat_images_horizontal(image_paths)
        response = qwen25vl_generate(model, processor, concat_path, prompt_text)

        extracted = extract_choice_letter(response)
        answer = item["answer"]
        extracted_fmt = f"({extracted})" if extracted else None
        correct = extracted_fmt == answer

        results.append({
            "idx": item["idx"], "subtask": item["subtask"],
            "answer": answer, "response": response,
            "extracted": extracted_fmt, "correct": correct,
        })

    by_subtask = defaultdict(list)
    for r in results:
        by_subtask[r["subtask"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    per_subtask = {t: acc(by_subtask[t]) for t in sorted(by_subtask.keys())}
    overall = sum(per_subtask.values()) / len(per_subtask) if per_subtask else 0
    ds_items = [r for t in DEPTH_SPATIAL_SUBTASKS for r in by_subtask.get(t, [])]
    acc_ds = acc(ds_items)
    other_items = [r for t in by_subtask if t not in DEPTH_SPATIAL_SUBTASKS for r in by_subtask[t]]
    acc_other = acc(other_items)
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  BLINK Overall (macro) : {overall:.2f}")
    print(f"  Depth/Spatial         : {acc_ds:.2f}  (n={len(ds_items)})")
    print(f"  Other Tasks           : {acc_other:.2f}  (n={len(other_items)})")
    print(f"  Per-Subtask:")
    for t in sorted(by_subtask.keys()):
        marker = " *" if t in DEPTH_SPATIAL_SUBTASKS else ""
        print(f"    {t:30s}: {acc(by_subtask[t]):6.2f}  (n={len(by_subtask[t])}){marker}")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {
        "benchmark": "BLINK", "overall": overall,
        "acc_depth_spatial": acc_ds, "acc_other": acc_other,
        "per_subtask": per_subtask,
        "n_unparsed": n_unparsed, "n_total": len(results),
    }, results


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SparseVLM on Qwen2.5-VL Evaluation")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["cvbench", "mmstar", "realworldqa", "blink"],
                        choices=["cvbench", "mmstar", "realworldqa", "blink"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/sparsevlm_qwen25vl_results")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--retain_tokens", type=int, default=192,
                        help="SparseVLM: 最终保留的视觉 token 数量 (192/128/96/64)")
    parser.add_argument("--n_recycle_clusters", type=int, default=8,
                        help="SparseVLM: token recycling 的聚类数")
    parser.add_argument("--blink_subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    model, processor, sparse = load_qwen25vl(
        args.model_id,
        retain_tokens=args.retain_tokens,
        n_recycle_clusters=args.n_recycle_clusters,
    )

    config = {
        "method": "SparseVLM",
        "model_id": args.model_id,
        "retain_tokens": args.retain_tokens,
        "n_recycle_clusters": args.n_recycle_clusters,
        "n_layers": len(model.model.language_model.layers),
        "prune_start_layer": sparse.prune_start_layer,
        "prune_end_layer": sparse.prune_end_layer,
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_summary = {}

    if "cvbench" in args.benchmarks:
        metrics, results = eval_cvbench(model, processor, args.max_samples)
        all_summary["cvbench"] = metrics
        with open(os.path.join(args.save_dir, "cvbench_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if "mmstar" in args.benchmarks:
        metrics, results = eval_mmstar(model, processor, args.max_samples)
        all_summary["mmstar"] = metrics
        with open(os.path.join(args.save_dir, "mmstar_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if "realworldqa" in args.benchmarks:
        metrics, results = eval_realworldqa(model, processor, args.max_samples)
        all_summary["realworldqa"] = metrics
        with open(os.path.join(args.save_dir, "realworldqa_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if "blink" in args.benchmarks:
        subtasks = args.blink_subtasks
        if args.depth_spatial_only:
            subtasks = DEPTH_SPATIAL_SUBTASKS
        metrics, results = eval_blink(model, processor, subtasks, args.max_samples)
        all_summary["blink"] = metrics
        with open(os.path.join(args.save_dir, "blink_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  SparseVLM (retain={args.retain_tokens}) on Qwen2.5-VL — Summary")
    print(f"{'='*60}")
    for name, m in all_summary.items():
        key = "cv_bench" if name == "cvbench" else "overall"
        score = m.get(key, 0)
        print(f"  {name:15s}: {score:.2f}")
    print(f"{'='*60}")

    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nDone. Results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()