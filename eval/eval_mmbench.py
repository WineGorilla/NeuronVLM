"""
MMBench 评估脚本。

MMBench 包含 ~3000 道多选题，覆盖 20 个能力维度。
数据集: https://huggingface.co/datasets/HuggingFaceM4/MMBench_dev
官方 TSV: https://github.com/open-compass/MMBench
论文: https://arxiv.org/abs/2307.06281 (ECCV 2024)

能力维度 (L-2):
  Perception: Coarse Perception (CP), Fine-grained Single (FP-S),
              Fine-grained Cross (FP-C)
  Reasoning:  Attribute Reasoning (AR), Relation Reasoning (RR),
              Logic Reasoning (LR)

L-3 维度 (20个): 包含 Spatial Relationship, Object Localization,
  Physical Relation 等与空间/距离相关的维度。

支持两种数据源:
  1. HuggingFace datasets (自动下载)
  2. 官方 TSV 文件 (手动下载)

用法：
    # 使用 HuggingFace 数据集
    CUDA_VISIBLE_DEVICES=0 python eval/eval_mmbench.py --mode base
    python eval/eval_mmbench.py --mode enhanced
    CUDA_VISIBLE_DEVICES=0 python eval/eval_mmbench.py --mode both
    python eval/eval_mmbench.py --mode spatial --spatial_qwen_ckpt outputs/focus_v2_ckpt/qwen_best.pt
    python eval/eval_mmbench.py --mode both --max_samples 100
    python eval/eval_mmbench.py --mode both --tsv_path data/MMBench_TEST_EN.tsv

    # 使用官方 TSV 文件
    CUDA_VISIBLE_DEVICES=0 python eval/eval_mmbench.py --mode both --tsv_path data/MMBench_DEV_EN.tsv

    # 只看空间/距离相关维度的表现
    python eval/eval_mmbench.py --mode both --spatial_focus
"""
import os
import sys
import re
import json
import base64
import argparse
from io import BytesIO
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import CFG


# ── 与空间/距离相关的 L-3 能力维度 ───────────────────────────────────────────

SPATIAL_DIMENSIONS = {
    "Spatial Relationship",
    "Object Localization",
    "Physical Relation",
    "Relative Position",       # MMBench v1.1 可能新增
}

# L-2 到 L-1 映射
L2_TO_L1 = {
    "Coarse Perception": "Perception",
    "Fine-grained Single-instance Perception": "Perception",
    "Fine-grained Cross-instance Perception": "Perception",
    "Attribute Reasoning": "Reasoning",
    "Relation Reasoning": "Reasoning",
    "Logic Reasoning": "Reasoning",
}


# ── Prompt 构造 ───────────────────────────────────────────────────────────────

def build_mmbench_prompt(question: str, hint: str,
                         options: dict[str, str]) -> str:
    """
    构造 MMBench 评估 prompt。
    question: 问题文本
    hint:     可选提示（可能为空）
    options:  {'A': '...', 'B': '...', 'C': '...', 'D': '...'} 部分题目可能只有 A/B
    """
    instruction = (
        "Answer the following question.\n"
        "Select the correct option and output ONLY the letter.\n"
        "Do NOT output explanation.\n"
    )
    parts = [instruction]

    if hint and str(hint).strip() and str(hint).strip().lower() != "nan":
        parts.append(f"Hint: {hint}")

    parts.append(f"\n{question}\n")

    choice_lines = []
    for key in ["A", "B", "C", "D"]:
        if key in options and options[key] and str(options[key]).strip() \
                and str(options[key]).strip().lower() != "nan":
            choice_lines.append(f"{key}. {options[key]}")
    parts.append("Choices:\n" + "\n".join(choice_lines))

    parts.append("\nAnswer:")
    return "\n".join(parts)


# ── 答案匹配 ──────────────────────────────────────────────────────────────────

def extract_choice_letter(response: str) -> str | None:
    """从模型输出中提取选项字母。"""
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None

    # (A) 格式
    m = re.search(r'\(([A-D])\)', text)
    if m:
        return m.group(1)

    # "ANSWER IS A" 格式
    m = re.search(r'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?([A-D])\)?', text)
    if m:
        return m.group(1)

    # 开头就是字母
    m = re.match(r'^([A-D])(?:[\s.,):]|$)', text)
    if m:
        return m.group(1)

    return None


def match_by_content(response: str, options: dict[str, str]) -> str | None:
    """尝试通过内容匹配答案。"""
    resp = response.strip().split("\n")[0].strip().lower()
    for key in ["A", "B", "C", "D"]:
        if key in options and str(options[key]).strip().lower() == resp:
            return key
    return None


def match_answer(response: str, options: dict[str, str],
                 answer: str) -> tuple[bool, str | None]:
    """匹配模型响应与标准答案。MMBench answer 格式为 'A','B','C','D'。"""
    extracted = extract_choice_letter(response)
    if extracted is not None:
        return extracted == answer, extracted

    extracted = match_by_content(response, options)
    if extracted is not None:
        return extracted == answer, extracted

    return False, None


# ── 图片处理工具 ──────────────────────────────────────────────────────────────

def decode_base64_image(b64_str: str, save_path: str) -> str:
    """将 base64 编码的图片保存为文件。"""
    img_data = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    img.save(save_path)
    return save_path


def get_image_path(item, tmp_dir="/tmp/mmbench_eval") -> str | None:
    """从 MMBench 数据项获取图片路径。"""
    os.makedirs(tmp_dir, exist_ok=True)
    idx = item.get("index", item.get("idx", 0))
    save_path = os.path.join(tmp_dir, f"img_{idx}.png")

    # HuggingFace datasets 格式: image 字段是 PIL Image
    if "image" in item and isinstance(item["image"], Image.Image):
        item["image"].save(save_path)
        return save_path

    # TSV 格式: image 字段是 base64 字符串
    if "image" in item and isinstance(item["image"], str):
        try:
            return decode_base64_image(item["image"], save_path)
        except Exception:
            pass

    # image_path 字段
    if "image_path" in item and item["image_path"]:
        return item["image_path"]

    return None


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_mmbench_from_hf(max_samples: int | None = None):
    """从 HuggingFace 加载 MMBench dev 集。"""
    from datasets import load_dataset

    print("Loading MMBench from HuggingFace (HuggingFaceM4/MMBench_dev)...")
    ds = load_dataset("HuggingFaceM4/MMBench_dev", split="train")

    columns = ds.column_names
    print(f"  columns: {columns}")

    items = []
    for i, row in enumerate(ds):
        item = {
            "index": row.get("index", i),
            "image": row.get("image"),
            "question": row.get("question", ""),
            "hint": row.get("hint", ""),
            "A": row.get("A", ""),
            "B": row.get("B", ""),
            "C": row.get("C", ""),
            "D": row.get("D", ""),
            "answer": row.get("answer", ""),
            "category": row.get("category", ""),        # L-2
            "l2_category": row.get("l2-category",
                           row.get("l2_category", "")),  # L-3
        }
        items.append(item)

    if max_samples:
        items = items[:max_samples]

    print(f"  Loaded {len(items)} samples")
    return items


def load_mmbench_from_tsv(tsv_path: str, max_samples: int | None = None):
    """从官方 TSV 文件加载 MMBench。"""
    print(f"Loading MMBench from TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"  columns: {list(df.columns)}")

    items = []
    for _, row in df.iterrows():
        item = {
            "index": row.get("index", len(items)),
            "image": row.get("image", ""),      # base64
            "question": row.get("question", ""),
            "hint": str(row.get("hint", "")),
            "A": str(row.get("A", "")),
            "B": str(row.get("B", "")),
            "C": str(row.get("C", "")),
            "D": str(row.get("D", "")),
            "answer": str(row.get("answer", "")),
            "category": str(row.get("category", "")),
            "l2_category": str(row.get("l2-category",
                            row.get("l2_category", ""))),
        }
        items.append(item)

    if max_samples:
        items = items[:max_samples]

    print(f"  Loaded {len(items)} samples")
    return items


# ── 加载模型 ──────────────────────────────────────────────────────────────────

def load_base_model():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading vanilla Qwen2.5-VL for baseline...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16, device_map=CFG.device,
    )
    base_model.eval()
    return base_model, processor


def load_finetune_baseline(qwen_ckpt):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading finetune baseline...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16, device_map=CFG.device,
    )
    if qwen_ckpt and os.path.exists(qwen_ckpt):
        print(f"  Loading finetuned weights: {qwen_ckpt}")
        state = torch.load(qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"  Loaded {len(state)} weight tensors.")
    else:
        print("  WARNING: --qwen_ckpt not found, using vanilla model!")
    model.eval()
    return model, processor


def load_enhanced_model(args):
    from src.Model import QwenWithClusterPredictorAndSAE
    print("Loading enhanced model (ClusterPredictor + SAE)...")
    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id, sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path, inject_layer=args.layer,
        latent_mult=CFG.latent_mult, topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.predictor_ckpt, device=CFG.device,
    )
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("  Enhanced model weights (Stage 2) loaded.")
    model.to(CFG.device)
    model.eval()
    return model


def load_spatial_model(args):
    from src.Model_v2 import QwenWithClusterPredictorAndSAE
    print("Loading spatial model (Model_v2)...")
    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id, sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path, inject_layer=args.layer,
        latent_mult=CFG.latent_mult, topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.spatial_predictor_ckpt, device=CFG.device,
    )
    if args.spatial_qwen_ckpt and os.path.exists(args.spatial_qwen_ckpt):
        state = torch.load(args.spatial_qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("  Spatial model weights loaded.")
    model.to(CFG.device)
    model.eval()
    return model


def load_enhanced_model_no_pcs(args):
    from ablation.Model_no_pcs import QwenWithClusterPredictorAndSAE
    print("Loading enhanced model (NO PCS ablation)...")
    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id, sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path, inject_layer=args.layer,
        latent_mult=CFG.latent_mult, topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.no_pcs_predictor_ckpt, device=CFG.device,
    )
    if args.no_pcs_qwen_ckpt and os.path.exists(args.no_pcs_qwen_ckpt):
        state = torch.load(args.no_pcs_qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("  No-PCS model weights loaded.")
    model.to(CFG.device)
    model.eval()
    return model


# ── 评估循环 ──────────────────────────────────────────────────────────────────

def evaluate_base(base_model, processor, dataset, save_path=None, label="base"):
    from qwen_vl_utils import process_vision_info

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        options = {k: item[k] for k in ["A", "B", "C", "D"]
                   if item.get(k) and str(item[k]).strip()
                   and str(item[k]).strip().lower() != "nan"}
        prompt_text = build_mmbench_prompt(
            item["question"], item.get("hint", ""), options
        )

        img_path = get_image_path(item)
        if not img_path:
            print(f"  [warning] No image for idx={item['index']}")
            results.append({
                "index": item["index"],
                "category": item.get("category", ""),
                "l2_category": item.get("l2_category", ""),
                "answer": item["answer"],
                "response": "", "extracted": None, "correct": False,
            })
            continue

        response = ""
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(base_model.device)

            with torch.no_grad():
                output_ids = base_model.generate(
                    **inputs, max_new_tokens=32, do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [error] idx={item['index']}: {e}")

        correct, extracted = match_answer(response, options, item["answer"])
        results.append({
            "index": item["index"],
            "category": item.get("category", ""),
            "l2_category": item.get("l2_category", ""),
            "answer": item["answer"],
            "response": response,
            "extracted": extracted,
            "correct": correct,
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")
    return results


def evaluate_enhanced(model, dataset, save_path=None, label="enhanced"):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        options = {k: item[k] for k in ["A", "B", "C", "D"]
                   if item.get(k) and str(item[k]).strip()
                   and str(item[k]).strip().lower() != "nan"}
        prompt_text = build_mmbench_prompt(
            item["question"], item.get("hint", ""), options
        )

        img_path = get_image_path(item)
        if not img_path:
            results.append({
                "index": item["index"],
                "category": item.get("category", ""),
                "l2_category": item.get("l2_category", ""),
                "answer": item["answer"],
                "response": "", "extracted": None, "correct": False,
                "cluster_ids": [],
            })
            continue

        response = ""
        cluster_ids = []
        try:
            result = model.generate(
                image_path=img_path, question=prompt_text, verbose=False,
            )
            response = result["final_answer"]
            cluster_ids = result.get("cluster_ids", [])
        except Exception as e:
            print(f"  [error] idx={item['index']}: {e}")

        correct, extracted = match_answer(response, options, item["answer"])
        results.append({
            "index": item["index"],
            "category": item.get("category", ""),
            "l2_category": item.get("l2_category", ""),
            "answer": item["answer"],
            "response": response,
            "extracted": extracted,
            "correct": correct,
            "cluster_ids": cluster_ids,
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")
    return results


# ── 统计 ──────────────────────────────────────────────────────────────────────

def compute_metrics(results, label=""):
    by_l2 = defaultdict(list)       # L-2 (category)
    by_l3 = defaultdict(list)       # L-3 (l2_category)
    for r in results:
        cat = r.get("category", "")
        l3  = r.get("l2_category", "")
        if cat:
            by_l2[cat].append(r)
        if l3:
            by_l3[l3].append(r)

    def acc(items):
        if not items:
            return 0.0
        return sum(1 for i in items if i["correct"]) / len(items) * 100

    n_unparsed = sum(1 for r in results if r.get("extracted") is None)
    n_total = len(results)

    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) "
              f"responses could not be parsed")
        for s in [r for r in results if r.get("extracted") is None][:3]:
            print(f"    example: cat={s.get('category','')} "
                  f"ans={s['answer']}")
            print(f"      resp=\"{s['response'][:100]}\"")

    overall_acc = acc(results)

    # 空间/距离相关维度
    spatial_items = []
    for r in results:
        l3 = r.get("l2_category", "")
        if l3 in SPATIAL_DIMENSIONS:
            spatial_items.append(r)
    acc_spatial = acc(spatial_items)

    # L-2 准确率
    per_l2 = {cat: acc(items) for cat, items in sorted(by_l2.items())}

    # L-3 准确率
    per_l3 = {cat: acc(items) for cat, items in sorted(by_l3.items())}

    # L-1 准确率
    per_l1 = defaultdict(list)
    for cat, items in by_l2.items():
        l1 = L2_TO_L1.get(cat, "Other")
        per_l1[l1].extend(items)
    per_l1_acc = {l1: acc(items) for l1, items in per_l1.items()}

    print(f"\n{'='*60}")
    print(f"  MMBench Results: {label}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy      : {overall_acc:.2f}  (n={n_total})")
    print(f"  Spatial Dimensions    : {acc_spatial:.2f}  "
          f"(n={len(spatial_items)})")

    if per_l1_acc:
        print(f"{'─'*60}")
        print(f"  L-1 Abilities:")
        for l1 in ["Perception", "Reasoning"]:
            if l1 in per_l1_acc:
                n = len(per_l1[l1])
                print(f"    {l1:30s}: {per_l1_acc[l1]:6.2f}  (n={n})")

    if per_l2:
        print(f"{'─'*60}")
        print(f"  L-2 Abilities:")
        for cat, a in per_l2.items():
            n = len(by_l2[cat])
            print(f"    {cat:42s}: {a:6.2f}  (n={n})")

    if per_l3:
        print(f"{'─'*60}")
        print(f"  L-3 Abilities:")
        for cat, a in per_l3.items():
            n = len(by_l3[cat])
            marker = " *" if cat in SPATIAL_DIMENSIONS else ""
            print(f"    {cat:42s}: {a:6.2f}  (n={n}){marker}")
        print(f"  (* = spatial/distance related)")

    print(f"  Unparsed: {n_unparsed}/{n_total}")
    print(f"{'='*60}")

    return {
        "overall": overall_acc,
        "acc_spatial": acc_spatial,
        "per_l1": per_l1_acc,
        "per_l2": per_l2,
        "per_l3": per_l3,
        "n_unparsed": n_unparsed,
        "n_total": n_total,
    }


def print_comparison(metrics_dict):
    labels = list(metrics_dict.keys())
    if len(labels) < 2:
        return

    print(f"\n{'='*75}")
    print(f"  MMBench Comparison")
    print(f"{'='*75}")

    header = f"  {'Metric':<40s}"
    for label in labels:
        header += f" {label:>12s}"
    if len(labels) >= 2:
        header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*70}")

    # Overall + Spatial
    for name, key in [("Overall", "overall"),
                      ("Spatial Dimensions", "acc_spatial")]:
        line = f"  {name:<40s}"
        vals = [metrics_dict[l][key] for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    # L-1
    print(f"  {'─'*70}")
    for l1 in ["Perception", "Reasoning"]:
        line = f"  {l1:<40s}"
        vals = [metrics_dict[l]["per_l1"].get(l1, 0) for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    # L-2
    all_l2 = sorted(set(
        c for m in metrics_dict.values() for c in m["per_l2"]
    ))
    if all_l2:
        print(f"  {'─'*70}")
        for cat in all_l2:
            line = f"    {cat:<38s}"
            vals = [metrics_dict[l]["per_l2"].get(cat, 0) for l in labels]
            for v in vals:
                line += f" {v:12.2f}"
            if len(vals) >= 2:
                d = vals[-1] - vals[0]
                line += f" {'+'if d>0 else ''}{d:7.2f}"
            print(line)

    # L-3 (spatial only for brevity)
    spatial_l3 = sorted(set(
        c for m in metrics_dict.values()
        for c in m["per_l3"] if c in SPATIAL_DIMENSIONS
    ))
    if spatial_l3:
        print(f"  {'─'*70}")
        print(f"  Spatial L-3 dimensions:")
        for cat in spatial_l3:
            line = f"  * {cat:<37s}"
            vals = [metrics_dict[l]["per_l3"].get(cat, 0) for l in labels]
            for v in vals:
                line += f" {v:12.2f}"
            if len(vals) >= 2:
                d = vals[-1] - vals[0]
                line += f" {'+'if d>0 else ''}{d:7.2f}"
            print(line)

    print(f"  {'─'*70}")
    line = f"  {'Unparsed':<40s}"
    for l in labels:
        line += f" {metrics_dict[l]['n_unparsed']:>12d}"
    print(line)
    print(f"{'='*75}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MMBench Evaluation")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "spatial", "no_pcs",
                                 "finetune_baseline", "both", "all"])
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/focus_ckpt/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--spatial_predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_spatial/predictor_best.pt")
    parser.add_argument("--spatial_qwen_ckpt", type=str, default=None)
    parser.add_argument("--no_pcs_predictor_ckpt", type=str,
                        default="outputs/ablation_no_pcs/predictor_best.pt")
    parser.add_argument("--no_pcs_qwen_ckpt", type=str, default=None)
    # ── MMBench 特有参数 ──
    parser.add_argument("--tsv_path", type=str, default=None,
                        help="官方 TSV 文件路径。不指定则从 HuggingFace 加载。")
    parser.add_argument("--spatial_focus", action="store_true",
                        help="在输出中重点展示空间/距离相关维度")
    parser.add_argument("--save_dir", type=str,
                        default="outputs/mmbench_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载数据集 ──
    if args.tsv_path and os.path.exists(args.tsv_path):
        dataset = load_mmbench_from_tsv(args.tsv_path, args.max_samples)
    else:
        dataset = load_mmbench_from_hf(args.max_samples)

    # 统计
    cat_counts = defaultdict(int)
    l3_counts  = defaultdict(int)
    for item in dataset:
        if item.get("category"):
            cat_counts[item["category"]] += 1
        if item.get("l2_category"):
            l3_counts[item["l2_category"]] += 1

    print(f"\n  L-2 distribution:")
    for c, n in sorted(cat_counts.items()):
        print(f"    {c}: {n}")
    print(f"\n  L-3 distribution:")
    for c, n in sorted(l3_counts.items()):
        marker = " *" if c in SPATIAL_DIMENSIONS else ""
        print(f"    {c}: {n}{marker}")

    # ── 评估 ──
    all_metrics = {}
    run_base     = args.mode in ["base", "both", "all"]
    run_ft_base  = args.mode in ["finetune_baseline", "all"]
    run_no_pcs   = args.mode in ["no_pcs", "all"]
    run_enhanced = args.mode in ["enhanced", "both", "all"]
    run_spatial  = args.mode in ["spatial", "all"]

    if run_base:
        base_model, processor = load_base_model()
        res = evaluate_base(
            base_model, processor, dataset,
            save_path=os.path.join(args.save_dir, "base_results.json"),
            label="base (vanilla)",
        )
        all_metrics["base"] = compute_metrics(res, label="Base Qwen2.5-VL")
        del base_model, processor; torch.cuda.empty_cache()

    if run_ft_base:
        ft_ckpt = args.baseline_ckpt or args.qwen_ckpt
        ft_model, ft_proc = load_finetune_baseline(ft_ckpt)
        res = evaluate_base(
            ft_model, ft_proc, dataset,
            save_path=os.path.join(args.save_dir, "ft_baseline_results.json"),
            label="finetune_baseline",
        )
        all_metrics["ft_baseline"] = compute_metrics(
            res, label="Finetune Baseline")
        del ft_model, ft_proc; torch.cuda.empty_cache()

    if run_no_pcs:
        no_pcs_model = load_enhanced_model_no_pcs(args)
        res = evaluate_enhanced(
            no_pcs_model, dataset,
            save_path=os.path.join(args.save_dir, "no_pcs_results.json"),
            label="enhanced (no PCS)",
        )
        all_metrics["no_pcs"] = compute_metrics(res, label="Enhanced (no PCS)")
        del no_pcs_model; torch.cuda.empty_cache()

    if run_enhanced:
        enhanced_model = load_enhanced_model(args)
        res = evaluate_enhanced(
            enhanced_model, dataset,
            save_path=os.path.join(args.save_dir, "enhanced_results.json"),
            label="enhanced (v1)",
        )
        all_metrics["enhanced"] = compute_metrics(res, label="Enhanced (v1)")
        del enhanced_model; torch.cuda.empty_cache()

    if run_spatial:
        spatial_model = load_spatial_model(args)
        res = evaluate_enhanced(
            spatial_model, dataset,
            save_path=os.path.join(args.save_dir, "spatial_results.json"),
            label="spatial (v2)",
        )
        all_metrics["spatial"] = compute_metrics(res, label="Spatial (v2)")
        del spatial_model; torch.cuda.empty_cache()

    if len(all_metrics) >= 2:
        print_comparison(all_metrics)

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nDone. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()