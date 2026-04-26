"""
LLaVA-PruMerge 评估脚本 — 在 Qwen2.5-VL 上复现 PruMerge 算法。

PruMerge 核心思想 (Shang et al., ICCV 2025):
  在视觉 token 进入 LLM decoder 之前，做 Prune + Merge：
  1. Pruning: 根据 token 重要性（原版用 CLS-spatial attention sparsity），
     选出最重要的 top-K 个 visual token 保留
  2. Merging: 将被剪掉的 token 按 key similarity 聚类，
     每个聚类的均值合并到最近的保留 token 上，补充信息

与 FastV/SparseVLM 的关键区别：
  - PruMerge 作用在 ViT 输出 → LLM 输入之间（encoder 端）
  - FastV/SparseVLM 作用在 LLM decoder 内部（decoder 端）
  - PruMerge 有 merge 步骤，被剪掉的 token 信息不丢失

适配说明：
  Qwen2.5-VL 的 ViT 没有 CLS token（用 windowed attention），
  因此用 hidden states L2 norm 作为 token 重要性的代理指标。
  这与原论文精神一致：选择信息密度高的 token 保留。

支持 4 个 benchmark：CV-Bench, BLINK, RealWorldQA, MMStar。

用法：
    python eval/eval_prumerge.py --benchmarks cvbench mmstar
    python eval_new/eval_prumerge.py --retain_ratio 0.25
    python eval/eval_prumerge.py --retain_ratio 0.055 --benchmarks blink --depth_spatial_only
    python eval/eval_prumerge.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --max_samples 50

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
# PruMerge: Prune + Merge Visual Tokens
# ══════════════════════════════════════════════════════════════════════════════

class PruMergeWrapper:
    """
    PruMerge 实现：在 LLM 第一层之前，对视觉 token 做 Prune + Merge。

    算法步骤：
      1. 定位 input embeddings 中的 visual token 位置
      2. 计算每个 visual token 的重要性分数
         - 原版用 CLS-spatial attention（CLIP ViT 有 CLS token）
         - 适配版用 hidden states L2 norm（Qwen2.5-VL 无 CLS token）
      3. 保留重要性最高的 top-K 个 visual token（Pruning）
      4. 将被剪掉的 token 按 cosine similarity 聚类到最近的保留 token
      5. 每个保留 token = 原始值 + 其聚类中被剪 token 的加权均值（Merging）
      6. 将被剪掉位置的 hidden states 置零（保持序列长度不变）

    这样 LLM decoder 后续层只会从保留 + 合并后的 token 获取有效信息。
    """

    def __init__(self, model, retain_ratio=0.25, merge_weight=0.5):
        """
        Args:
            model: Qwen2_5_VLForConditionalGeneration 实例
            retain_ratio: 保留的视觉 token 比例（0.25 = 保留 25%，约 4x 压缩）
                          原论文：PruMerge 平均保留 5.5%，PruMerge+ 保留 25%
            merge_weight: merge 时被剪 token 信息的权重 (0~1)
        """
        self.model = model
        self.retain_ratio = retain_ratio
        self.merge_weight = merge_weight
        self._hooks = []
        self._image_token_mask = None
        self._applied = False

    def set_image_token_mask(self, input_ids):
        """根据 input_ids 定位 image/video token 的位置。"""
        self._image_token_mask = (
            (input_ids == IMAGE_TOKEN_ID) | (input_ids == VIDEO_TOKEN_ID)
        )
        self._applied = False

    def install_hooks(self):
        """在 LLM 的第一层 decoder layer 上安装 hook。"""
        self._hooks = []
        layers = self.model.model.language_model.layers

        # 只在第 0 层安装 hook — PruMerge 作用在 LLM 入口处
        hook = layers[0].register_forward_pre_hook(self._prumerge_hook)
        self._hooks.append(hook)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _prumerge_hook(self, module, args):
        """
        在 LLM 第一层 forward 之前执行 PruMerge。
        此时 hidden_states 已经是 ViT 输出 + MLP projector 后的 embeddings。
        """
        if self._applied:
            return args

        # args 可能是 tuple 或包含 hidden_states 的结构
        if isinstance(args, tuple) and len(args) > 0:
            hidden_states = args[0]
        else:
            return args

        seq_len = hidden_states.shape[1]
        if seq_len <= 1:
            return args

        if self._image_token_mask is None:
            return args

        self._do_prumerge(hidden_states)
        self._applied = True

        return args

    def _do_prumerge(self, hidden_states):
        """
        PruMerge 核心算法：Prune + Merge。

        Step 1: 计算每个 visual token 的重要性（L2 norm）
        Step 2: 保留 top-K 个重要 token (Pruning)
        Step 3: 将被剪 token 按 cosine similarity 分配到最近的保留 token (Clustering)
        Step 4: 将聚类的均值加权合并到保留 token 上 (Merging)
        Step 5: 被剪位置置零
        """
        batch_size = hidden_states.shape[0]

        for b in range(batch_size):
            img_mask = self._image_token_mask[b]
            if img_mask.shape[0] != hidden_states.shape[1]:
                continue

            img_positions = img_mask.nonzero(as_tuple=True)[0]
            n_img = len(img_positions)

            if n_img == 0:
                continue

            n_keep = max(1, int(n_img * self.retain_ratio))
            if n_keep >= n_img:
                continue

            img_hidden = hidden_states[b, img_positions, :].float()  # [n_img, D]

            # === Step 1: Token Importance Scoring ===
            # 原版 PruMerge 用 CLS-spatial attention sparsity
            # Qwen2.5-VL 无 CLS token，用 L2 norm 作为代理
            importance = img_hidden.norm(dim=-1)  # [n_img]

            # === Step 2: Pruning — 保留最重要的 n_keep 个 ===
            _, keep_indices = importance.topk(n_keep)
            all_indices = torch.arange(n_img, device=hidden_states.device)
            prune_mask = torch.ones(n_img, dtype=torch.bool, device=hidden_states.device)
            prune_mask[keep_indices] = False
            prune_indices = all_indices[prune_mask]

            keep_hidden = img_hidden[keep_indices]    # [n_keep, D]
            prune_hidden = img_hidden[prune_indices]  # [n_prune, D]

            n_prune = len(prune_indices)
            if n_prune == 0:
                continue

            # === Step 3: Clustering — 将被剪 token 分配到最近的保留 token ===
            keep_norm = F.normalize(keep_hidden, dim=-1)    # [n_keep, D]
            prune_norm = F.normalize(prune_hidden, dim=-1)  # [n_prune, D]

            # 余弦相似度: [n_prune, n_keep]
            similarity = torch.matmul(prune_norm, keep_norm.T)
            # 每个被剪 token 分配到最相似的保留 token
            assignments = similarity.argmax(dim=1)  # [n_prune]

            # === Step 4: Merging — 将聚类信息合并到保留 token ===
            for k_idx in range(n_keep):
                cluster_mask = (assignments == k_idx)
                if cluster_mask.sum() == 0:
                    continue

                cluster_tokens = prune_hidden[cluster_mask]  # [n_cluster, D]
                cluster_mean = cluster_tokens.mean(dim=0)     # [D]

                # 加权合并：保留 token + merge_weight * 聚类均值
                merged = keep_hidden[k_idx] + self.merge_weight * cluster_mean
                keep_pos = img_positions[keep_indices[k_idx]]
                hidden_states[b, keep_pos, :] = merged.to(hidden_states.dtype)

            # === Step 5: 被剪位置 → scatter-copy 到最近的保留 token ===
            prune_positions = img_positions[prune_indices]
            keep_positions = img_positions[keep_indices]
            prune_hidden_f = hidden_states[b, prune_positions, :].float()
            keep_hidden_f = hidden_states[b, keep_positions, :].float()
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
                self._applied = False
            return result

        self.model.generate = patched_generate
        self._original_generate = original_generate

    def unpatch_generate(self):
        if hasattr(self, "_original_generate"):
            self.model.generate = self._original_generate


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen25vl(model_id, retain_ratio=0.25, merge_weight=0.5):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading Qwen2.5-VL: {model_id}")
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

    prumerge = PruMergeWrapper(
        model, retain_ratio=retain_ratio, merge_weight=merge_weight,
    )
    prumerge.patch_generate()
    print(f"  PruMerge installed: retain_ratio={retain_ratio}, "
          f"merge_weight={merge_weight}")

    return model, processor, prumerge


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
    parser = argparse.ArgumentParser(description="PruMerge on Qwen2.5-VL Evaluation")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["cvbench", "mmstar", "realworldqa", "blink"],
                        choices=["cvbench", "mmstar", "realworldqa", "blink"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/prumerge_qwen25vl_results")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--retain_ratio", type=float, default=0.25,
                        help="PruMerge: 保留的视觉 token 比例 "
                             "(0.25=PruMerge+, 0.055=PruMerge)")
    parser.add_argument("--merge_weight", type=float, default=0.5,
                        help="PruMerge: merge 时被剪 token 的权重")
    parser.add_argument("--blink_subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    model, processor, prumerge = load_qwen25vl(
        args.model_id,
        retain_ratio=args.retain_ratio,
        merge_weight=args.merge_weight,
    )

    config = {
        "method": "PruMerge",
        "model_id": args.model_id,
        "retain_ratio": args.retain_ratio,
        "merge_weight": args.merge_weight,
        "n_layers": len(model.model.language_model.layers),
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
    print(f"  PruMerge (retain={args.retain_ratio}) on Qwen2.5-VL — Summary")
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