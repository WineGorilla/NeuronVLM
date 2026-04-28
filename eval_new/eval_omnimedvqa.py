"""
OmniMedVQA-Mini 评估脚本：对比 base 和 enhanced 两种模式。

数据集：simwit/omni-med-vqa-mini
  - 2000 条测试样本，单一 test split
  - 覆盖 8 种影像模态（CT、MRI、X-Ray、病理等）
  - 5 种题型：Disease Diagnosis、Anatomy Identification、
              Lesion Grading、Modality Recognition、Other
  - 选项：option_A / option_B / option_C / option_D（C/D 可为空，即两选题）
  - 答案：label 字段，值为 "A" / "B" / "C" / "D"

支持模式:
  base      —— 原始 Qwen2.5-VL
  enhanced  —— ClusterPredictor + SAE 增强模型
  both      —— 同时跑两种，输出对比

用法示例:
  # 跑 both 模式，完整 2000 条
  python eval_new/eval_omnimedvqa.py --mode both --layer 8

  # 快速验证（只跑 100 条）
  python eval_omnimedvqa.py --mode both --max_samples 100

  # 多次随机子采样，报告 mean±std
  python eval_omnimedvqa.py --mode both --num_runs 3 --subsample_ratio 0.8

  # 只跑 base
  python eval_new/eval_omnimedvqa.py --mode base
  python eval_new/eval_omnimedvqa.py --mode enhanced

  # 指定输出目录和增强模型权重
  python eval_omnimedvqa.py --mode both --layer 8 \\
      --predictor_ckpt outputs/focus_ckpt/predictor_best.pt \\
      --qwen_ckpt outputs/stage2/qwen_best.pt \\
      --save_dir outputs/eval_omnimedvqa
"""

import os
import sys
import re
import json
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import CFG


# ══════════════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════════════

QUESTION_TYPES = [
    "Disease Diagnosis",
    "Anatomy Identification",
    "Lesion Grading",
    "Modality Recognition",
    "Other",
]

MODALITIES = [
    "CT(Computed Tomography)",
    "MRI(Magnetic Resonance Imaging)",
    "X-Ray",
    "Pathology",
    "Fundus Photography",
    "Dermoscopy",
    "Ultrasound",
    "Other",
]


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def extract_choice_letter(response: str, valid_choices: list[str]) -> str | None:
    """
    从模型输出中提取选项字母，只接受 valid_choices 中存在的字母
    （有些题只有 A/B 两个选项，C/D 无效）。
    """
    max_letter = valid_choices[-1]  # e.g. "B" or "D"
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None

    pat = f"[A-{max_letter}]"

    # 优先匹配括号格式 (A)
    m = re.search(rf'\(({pat})\)', text)
    if m and m.group(1) in valid_choices:
        return m.group(1)

    # 匹配 "Answer: A" 或 "Answer is A"
    m = re.search(rf'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?({pat})\)?', text)
    if m and m.group(1) in valid_choices:
        return m.group(1)

    # 匹配行首单个字母
    m = re.match(rf'^({pat})(?:[\s.,):]|$)', text)
    if m and m.group(1) in valid_choices:
        return m.group(1)

    return None


def match_by_content(response: str, options: dict) -> str | None:
    """尝试将模型输出与选项文本内容直接匹配。"""
    resp = response.strip().split("\n")[0].strip().lower()
    for letter, text in options.items():
        if text and resp == text.strip().lower():
            return letter
    return None


def build_options(item: dict) -> dict:
    """
    从样本中提取有效选项（过滤掉空的 C/D）。
    返回 {"A": "...", "B": "...", "C": "..."(可选), "D": "..."(可选)}
    """
    options = {}
    for letter in ["A", "B", "C", "D"]:
        val = item.get(f"option_{letter}", "")
        if val and val.strip():
            options[letter] = val.strip()
    return options


def build_prompt(question: str, options: dict) -> str:
    """构造送入模型的 prompt。"""
    choice_lines = "\n".join(
        [f"{letter}. {text}" for letter, text in options.items()]
    )
    return (
        "You are a medical image analysis expert.\n"
        "Answer the following question about the medical image.\n"
        "Select the correct option and output ONLY the letter (e.g. A, B, C, or D).\n"
        "Do NOT output any explanation.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{choice_lines}\n\n"
        "Answer:"
    )


def subsample(ds, ratio: float, seed: int):
    """对 HuggingFace Dataset 随机采样 ratio 比例的数据。"""
    random.seed(seed)
    n = len(ds)
    k = int(n * ratio)
    indices = random.sample(range(n), k)
    return ds.select(indices)


def save_results(results: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_base_model():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading vanilla Qwen2.5-VL (base)...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16, device_map=CFG.device,
    )
    model.eval()
    return model, processor


def load_enhanced_model(args):
    from src.Model import QwenWithClusterPredictorAndSAE
    print("Loading enhanced model (ClusterPredictor + SAE)...")
    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id,
        sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path,
        inject_layer=args.layer,
        latent_mult=CFG.latent_mult,
        topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.predictor_ckpt,
        device=CFG.device,
    )
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"  Enhanced weights loaded from: {args.qwen_ckpt}")
    model.to(CFG.device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 推理封装
# ══════════════════════════════════════════════════════════════════════════════

def base_generate(model, processor, image_path: str, prompt: str,
                  max_new_tokens: int = 32) -> str:
    from qwen_vl_utils import process_vision_info
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        return processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()
    except Exception as e:
        print(f"  [error] base_generate: {e}")
        return ""


def enhanced_generate(model, image_path: str, prompt: str) -> dict:
    try:
        return model.generate(image_path=image_path, question=prompt, verbose=False)
    except Exception as e:
        print(f"  [error] enhanced_generate: {e}")
        return {"final_answer": "", "cluster_ids": []}


# ══════════════════════════════════════════════════════════════════════════════
# 核心评估逻辑
# ══════════════════════════════════════════════════════════════════════════════

def eval_omnimedvqa(model, processor, ds, is_enhanced: bool, label: str) -> list:
    """
    跑一遍 OmniMedVQA-Mini 评估。

    Args:
        model:        base 模型 或 enhanced 模型
        processor:    base 模式下的 processor（enhanced 模式传 None）
        ds:           HuggingFace Dataset
        is_enhanced:  True → 使用 enhanced_generate；False → 使用 base_generate
        label:        用于日志打印的模型标签

    Returns:
        results: list of dict，每条包含预测结果和元信息
    """
    print(f"\n  Evaluating OmniMedVQA-Mini [{label}] — {len(ds)} samples")
    tmp_path = "/tmp/omnimedvqa_eval_img.png"
    results = []

    for item in tqdm(ds, desc=f"OmniMedVQA [{label}]"):
        # ── 构造选项和 prompt ──
        options = build_options(item)
        if not options:
            # 极少数情况：选项全为空，跳过
            continue
        valid_choices = list(options.keys())   # e.g. ["A", "B"] 或 ["A","B","C","D"]
        prompt = build_prompt(item["question"], options)

        # ── 保存图像到临时文件 ──
        img = item["image"]
        if isinstance(img, Image.Image):
            img.save(tmp_path)
        else:
            tmp_path = str(img)

        # ── 推理 ──
        cluster_ids = []
        if is_enhanced:
            res = enhanced_generate(model, tmp_path, prompt)
            response = res["final_answer"]
            cluster_ids = res.get("cluster_ids", [])
        else:
            response = base_generate(model, processor, tmp_path, prompt)

        # ── 解析答案 ──
        extracted = extract_choice_letter(response, valid_choices)
        if extracted is None:
            extracted = match_by_content(response, options)

        gt = item["label"].strip().upper()   # "A" / "B" / "C" / "D"
        correct = (extracted == gt)

        record = {
            "question_id":    item["question_id"],
            "dataset":        item["dataset"],
            "modality":       item["modality"],
            "question_type":  item["question_type"],
            "question":       item["question"],
            "options":        options,
            "gt_answer":      item["gt_answer"],
            "gt_label":       gt,
            "response":       response,
            "extracted":      extracted,
            "correct":        correct,
        }
        if is_enhanced:
            record["cluster_ids"] = cluster_ids
        results.append(record)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 指标计算与打印
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list, label: str) -> dict:
    """计算总体、按题型、按模态的准确率。"""
    def acc(items):
        if not items:
            return 0.0
        return sum(1 for i in items if i["correct"]) / len(items) * 100

    by_qtype    = defaultdict(list)
    by_modality = defaultdict(list)

    for r in results:
        by_qtype[r["question_type"]].append(r)
        by_modality[r["modality"]].append(r)

    overall      = acc(results)
    per_qtype    = {t: acc(items) for t, items in sorted(by_qtype.items())}
    per_modality = {m: acc(items) for m, items in sorted(by_modality.items())}
    n_unparsed   = sum(1 for r in results if r["extracted"] is None)

    # ── 打印摘要 ──
    print(f"\n  ── OmniMedVQA-Mini Results [{label}] ──")
    print(f"  Overall Accuracy : {overall:.2f}%  "
          f"({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"  Unparsed         : {n_unparsed}/{len(results)}")

    print(f"\n  By Question Type:")
    for t, a in per_qtype.items():
        n = len(by_qtype[t])
        print(f"    {t:<28s}: {a:6.2f}%  (n={n})")

    print(f"\n  By Modality:")
    for m, a in per_modality.items():
        n = len(by_modality[m])
        print(f"    {m:<35s}: {a:6.2f}%  (n={n})")

    return {
        "overall":       overall,
        "per_qtype":     per_qtype,
        "per_modality":  per_modality,
        "n_unparsed":    n_unparsed,
        "n_total":       len(results),
    }


def print_comparison(metrics_base: dict, metrics_enh: dict):
    """并排打印 base vs enhanced 对比。"""
    print(f"\n{'═'*70}")
    print(f"  Comparison: base  vs  enhanced")
    print(f"{'═'*70}")

    def row(name, b, e):
        delta = e - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<35s}  {b:6.2f}%   {e:6.2f}%   {sign}{delta:.2f}%")

    print(f"  {'Metric':<35s}  {'base':>7s}   {'enhanced':>8s}   {'Delta':>7s}")
    print(f"  {'─'*65}")

    row("Overall", metrics_base["overall"], metrics_enh["overall"])

    print(f"\n  By Question Type:")
    all_qtypes = sorted(
        set(metrics_base["per_qtype"]) | set(metrics_enh["per_qtype"])
    )
    for t in all_qtypes:
        b = metrics_base["per_qtype"].get(t, 0.0)
        e = metrics_enh["per_qtype"].get(t, 0.0)
        row(f"  {t}", b, e)

    print(f"\n  By Modality:")
    all_modalities = sorted(
        set(metrics_base["per_modality"]) | set(metrics_enh["per_modality"])
    )
    for m in all_modalities:
        b = metrics_base["per_modality"].get(m, 0.0)
        e = metrics_enh["per_modality"].get(m, 0.0)
        row(f"  {m}", b, e)

    print(f"{'═'*70}")


def print_summary_with_std(aggregated: dict, model_labels: list):
    """多次运行的 mean±std 汇总打印。"""
    print(f"\n{'═'*70}")
    print(f"  Grand Summary (mean ± std) — OmniMedVQA-Mini")
    print(f"{'═'*70}")

    header = f"  {'Metric':<35s}"
    for ml in model_labels:
        header += f"  {ml:>18s}"
    if len(model_labels) >= 2:
        header += f"  {'Delta(mean)':>12s}"
    print(header)
    print(f"  {'─'*65}")

    def stat_row(name, key_path):
        line = f"  {name:<35s}"
        means = []
        for ml in model_labels:
            stats = aggregated.get(ml, {}).get(key_path, None)
            if stats:
                line += f"  {stats['mean']:6.2f}±{stats['std']:4.2f}   "
                means.append(stats["mean"])
            else:
                line += f"  {'N/A':>18s}"
                means.append(None)
        if len(means) >= 2 and all(m is not None for m in means):
            d = means[-1] - means[0]
            line += f"  {'+'if d>=0 else ''}{d:.2f}%"
        print(line)

    stat_row("Overall", "overall")
    print(f"{'═'*70}")


# ══════════════════════════════════════════════════════════════════════════════
# 多次运行聚合
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_runs(all_runs: list, model_labels: list) -> dict:
    """
    all_runs: list of {model_label: metrics_dict}
    返回: {model_label: {metric_key: {"mean": x, "std": x, "runs": [...]}}}
    """
    aggregated = {}
    for ml in model_labels:
        run_metrics = [run[ml] for run in all_runs if ml in run]
        if not run_metrics:
            continue
        agg = {}
        for key in run_metrics[0]:
            vals = [m[key] for m in run_metrics if key in m]
            if vals and isinstance(vals[0], (int, float)):
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "runs": [float(v) for v in vals],
                }
        aggregated[ml] = agg
    return aggregated


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base / enhanced models on OmniMedVQA-Mini"
    )

    # 模式
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "both"],
                        help="Which model(s) to evaluate")

    # 模型路径
    parser.add_argument("--layer", type=int, default=CFG.vis_layer,
                        help="SAE injection layer for enhanced model")
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/qwen_layer8_old/focus_ckpt_0.75_64_5000/predictor_best.pt",
                        help="ClusterPredictor checkpoint path")
    parser.add_argument("--qwen_ckpt", type=str, default=None,
                        help="Stage-2 finetuned Qwen weights for enhanced model")

    # 数据
    parser.add_argument("--max_samples", type=int, default=None,
                        help="截断样本数（调试用）")

    # 多次运行
    parser.add_argument("--num_runs", type=int, default=1,
                        help="重复运行次数，配合 --subsample_ratio 报告 mean±std")
    parser.add_argument("--subsample_ratio", type=float, default=0.8,
                        help="每次运行随机采样的比例（默认 0.8）")

    # 输出
    parser.add_argument("--save_dir", type=str,
                        default="outputs/eval_omnimedvqa",
                        help="结果保存目录")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    run_base     = args.mode in ["base", "both"]
    run_enhanced = args.mode in ["enhanced", "both"]

    # ── 加载数据集（只加载一次）──
    print("Loading simwit/omni-med-vqa-mini (test split)...")
    ds_full = load_dataset("simwit/omni-med-vqa-mini", split="test")
    if args.max_samples:
        ds_full = ds_full.select(range(min(args.max_samples, len(ds_full))))
    print(f"  Total samples: {len(ds_full)}")

    # ── 预加载模型（只加载一次）──
    base_model, processor = None, None
    enhanced_model        = None

    if run_base:
        base_model, processor = load_base_model()
    if run_enhanced:
        enhanced_model = load_enhanced_model(args)

    # ── 多次运行循环 ──
    all_runs     = []   # list of {label: metrics}
    model_labels = []

    for run_idx in range(args.num_runs):
        seed = 42 + run_idx

        if args.num_runs > 1:
            print(f"\n{'▶'*25} Run {run_idx+1}/{args.num_runs} "
                  f"(seed={seed}, ratio={args.subsample_ratio}) {'◀'*25}")
            ds = subsample(ds_full, ratio=args.subsample_ratio, seed=seed)
            print(f"  Sampled {len(ds)}/{len(ds_full)} samples")
        else:
            ds = ds_full

        run_save_dir = (
            os.path.join(args.save_dir, f"run_{run_idx}")
            if args.num_runs > 1 else args.save_dir
        )
        os.makedirs(run_save_dir, exist_ok=True)

        run_result   = {}
        model_labels = []

        # ── Base ──
        if run_base:
            label   = "base"
            model_labels.append(label)
            results = eval_omnimedvqa(base_model, processor, ds,
                                      is_enhanced=False, label=label)
            metrics = compute_metrics(results, label)
            run_result[label] = metrics
            save_results(results, os.path.join(run_save_dir, f"results_{label}.json"))

        # ── Enhanced ──
        if run_enhanced:
            label   = "enhanced"
            model_labels.append(label)
            results = eval_omnimedvqa(enhanced_model, None, ds,
                                      is_enhanced=True, label=label)
            metrics = compute_metrics(results, label)
            run_result[label] = metrics
            save_results(results, os.path.join(run_save_dir, f"results_{label}.json"))

        # ── 单次对比打印 ──
        if run_base and run_enhanced:
            print_comparison(run_result["base"], run_result["enhanced"])

        all_runs.append(run_result)

    # ── 释放显存 ──
    del base_model, processor, enhanced_model
    torch.cuda.empty_cache()

    # ── 多次运行汇总 ──
    if args.num_runs > 1:
        aggregated = aggregate_runs(all_runs, model_labels)
        print_summary_with_std(aggregated, model_labels)

        summary = {
            "config": {
                "num_runs":        args.num_runs,
                "subsample_ratio": args.subsample_ratio,
                "seeds":           [42 + i for i in range(args.num_runs)],
            },
            "aggregated": aggregated,
            "per_run":    all_runs,
        }
        out_path = os.path.join(args.save_dir, "summary_mean_std.json")
    else:
        summary  = all_runs[0]
        out_path = os.path.join(args.save_dir, "summary.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Summary saved to {out_path}")


if __name__ == "__main__":
    main()