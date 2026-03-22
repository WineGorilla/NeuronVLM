"""
CV-Bench 评估脚本。

用法：
    # 只评估 base Qwen（不注入 extra token）
    python eval/eval_cvbench.py --mode base

    # 只评估 enhanced（注入 extra token）
    python eval/eval_cvbench.py --mode enhanced

    # 两者都评估并对比
    python eval/eval_cvbench.py --mode both

    # 指定权重
    python eval/eval_cvbench.py --mode both \
        --predictor_ckpt outputs/focus_ckpt/predictor_best.pt \
        --qwen_ckpt outputs/focus_ckpt/qwen_best.pt

CV-Bench 数据集：
    来自 Cambrian-1，包含 2638 道多选 VQA 题目。
    2D 任务：Count（计数）、Spatial Relation（空间关系）  来源: ADE20K, COCO
    3D 任务：Depth（深度顺序）、Distance（相对距离）    来源: Omni3D

评估指标：
    CV-Bench Accuracy = 0.5 * ((acc_2d_ade + acc_2d_coco) / 2 + acc_3d_omni)
    同时输出 Count / Depth / Distance / Spatial Relation 分项准确率
"""
import os
import sys
import re
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import CFG
from src.Model import QwenWithClusterPredictorAndSAE


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def extract_choice(response: str, num_choices: int) -> str:
    """
    从模型输出中提取选项字母，如 (A), A, a 等。
    """
    response = response.strip()

    # 尝试匹配 (A), (B) 等格式
    match = re.search(r'\(([A-Z])\)', response)
    if match:
        return f"({match.group(1)})"

    # 尝试匹配开头的单个字母
    match = re.match(r'^([A-Z])\b', response.upper())
    if match:
        return f"({match.group(1)})"

    # 如果模型直接输出了选项内容（数字等），尝试匹配
    # 这种情况下返回原始文本，后面用内容匹配
    return response


def match_answer(response: str, choices: list, answer: str) -> bool:
    """
    判断模型输出是否正确。
    answer 格式如 "(C)"，choices 是选项列表。
    """
    extracted = extract_choice(response, len(choices))

    # 直接匹配选项字母
    if extracted == answer:
        return True

    # 尝试通过内容匹配
    # answer 是 "(C)" 形式，提取索引
    ans_match = re.match(r'\(([A-Z])\)', answer)
    if ans_match:
        ans_idx = ord(ans_match.group(1)) - ord('A')
        if 0 <= ans_idx < len(choices):
            correct_content = str(choices[ans_idx]).strip()
            # 检查模型输出是否包含正确答案内容
            if correct_content in response or response.strip() == correct_content:
                return True

    return False


def build_prompt(question: str, choices: list) -> str:
    """
    构建带选项的 prompt。
    """
    letters = [chr(ord('A') + i) for i in range(len(choices))]
    options = " ".join([f"({l}) {c}" for l, c in zip(letters, choices)])
    return f"{question} Select from the following choices. {options}"


# ── 评估函数 ──────────────────────────────────────────────────────────────────

def evaluate_base(model, dataset, save_path=None):
    """
    评估 base Qwen（不注入 extra token）。
    直接用 base_model.generate()。
    """
    print("\n" + "=" * 60)
    print("Evaluating: Base Qwen (no extra token injection)")
    print("=" * 60)

    results = []
    for item in tqdm(dataset, desc="Base eval"):
        image = item["image"]
        prompt_text = item["prompt"]
        answer = item["answer"]
        choices = item["choices"]
        task = item["task"]
        source = item["source"]

        # 保存临时图片
        tmp_path = "/tmp/cvbench_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        try:
            inputs = model._build_inputs(tmp_path, prompt_text, for_generation=True)
            output_ids = model.base_model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
            )
            input_len = inputs["input_ids"].shape[1]
            response = model.processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")
            response = ""

        correct = match_answer(response, choices, answer)
        results.append({
            "idx": item["idx"],
            "task": task,
            "source": source,
            "type": item["type"],
            "answer": answer,
            "response": response,
            "correct": correct,
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")

    return results


def evaluate_enhanced(model, dataset, save_path=None):
    """
    评估 enhanced 模型（注入 extra token）。
    使用 model.generate() 完整流程。
    """
    print("\n" + "=" * 60)
    print("Evaluating: Enhanced (with extra token injection)")
    print("=" * 60)

    results = []
    for item in tqdm(dataset, desc="Enhanced eval"):
        image = item["image"]
        prompt_text = item["prompt"]
        answer = item["answer"]
        choices = item["choices"]
        task = item["task"]
        source = item["source"]

        tmp_path = "/tmp/cvbench_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        try:
            result = model.generate(
                image_path=tmp_path,
                question=prompt_text,
                verbose=False,
            )
            response = result["final_answer"]
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")
            response = ""

        correct = match_answer(response, choices, answer)
        results.append({
            "idx": item["idx"],
            "task": task,
            "source": source,
            "type": item["type"],
            "answer": answer,
            "response": response,
            "correct": correct,
            "cluster_ids": result.get("cluster_ids", []) if 'result' in dir() else [],
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")

    return results


# ── 统计函数 ──────────────────────────────────────────────────────────────────

def compute_metrics(results, label=""):
    """
    按 CV-Bench 官方公式计算准确率：
        CV-Bench = 0.5 * ((acc_2d_ade + acc_2d_coco) / 2 + acc_3d_omni)

    同时输出 Count / Depth / Distance / Spatial Relation 分项。
    """
    # 按 source 分组
    by_source = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)

    # 按 task 分组
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)

    # 按 type 分组
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    def acc(items):
        if not items:
            return 0.0
        return sum(1 for i in items if i["correct"]) / len(items) * 100

    acc_ade = acc(by_source.get("ADE20K", []))
    acc_coco = acc(by_source.get("COCO", []))
    acc_omni = acc(by_source.get("Omni3D", []))

    acc_2d = (acc_ade + acc_coco) / 2
    acc_3d = acc_omni
    cv_bench = (acc_2d + acc_3d) / 2

    print(f"\n{'='*60}")
    print(f"  CV-Bench Results: {label}")
    print(f"{'='*60}")
    print(f"  CV-Bench Overall : {cv_bench:.2f}")
    print(f"  2D Accuracy      : {acc_2d:.2f}")
    print(f"    ADE20K         : {acc_ade:.2f}  (n={len(by_source.get('ADE20K', []))})")
    print(f"    COCO           : {acc_coco:.2f}  (n={len(by_source.get('COCO', []))})")
    print(f"  3D Accuracy      : {acc_3d:.2f}")
    print(f"    Omni3D         : {acc_omni:.2f}  (n={len(by_source.get('Omni3D', []))})")
    print(f"{'─'*60}")
    print(f"  Per-Task Breakdown:")
    for task_name in sorted(by_task.keys()):
        items = by_task[task_name]
        print(f"    {task_name:20s}: {acc(items):6.2f}  (n={len(items)})")
    print(f"{'='*60}")

    return {
        "cv_bench": cv_bench,
        "acc_2d": acc_2d,
        "acc_3d": acc_3d,
        "acc_ade": acc_ade,
        "acc_coco": acc_coco,
        "acc_omni": acc_omni,
        "per_task": {t: acc(items) for t, items in by_task.items()},
    }


def print_comparison(base_metrics, enhanced_metrics):
    """打印两组指标对比表"""
    print(f"\n{'='*70}")
    print(f"  CV-Bench Comparison: Base vs Enhanced")
    print(f"{'='*70}")
    header = f"  {'Metric':<25s} {'Base':>8s} {'Enhanced':>10s} {'Delta':>8s}"
    print(header)
    print(f"  {'─'*55}")

    rows = [
        ("CV-Bench Overall", "cv_bench"),
        ("2D Accuracy", "acc_2d"),
        ("  ADE20K", "acc_ade"),
        ("  COCO", "acc_coco"),
        ("3D Accuracy", "acc_3d"),
        ("  Omni3D", "acc_omni"),
    ]
    for name, key in rows:
        b = base_metrics[key]
        e = enhanced_metrics[key]
        d = e - b
        sign = "+" if d > 0 else ""
        print(f"  {name:<25s} {b:8.2f} {e:10.2f} {sign}{d:8.2f}")

    # Per-task
    all_tasks = sorted(set(list(base_metrics["per_task"].keys()) +
                           list(enhanced_metrics["per_task"].keys())))
    print(f"  {'─'*55}")
    print(f"  Per-Task:")
    for t in all_tasks:
        b = base_metrics["per_task"].get(t, 0)
        e = enhanced_metrics["per_task"].get(t, 0)
        d = e - b
        sign = "+" if d > 0 else ""
        print(f"    {t:<23s} {b:8.2f} {e:10.2f} {sign}{d:8.2f}")
    print(f"{'='*70}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate on CV-Bench")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "both"],
                        help="Evaluation mode")
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/focus_ckpt/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None,
                        choices=["2D", "3D"],
                        help="Only evaluate on 2D or 3D subset")
    parser.add_argument("--save_dir", type=str, default="outputs/cvbench_results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for quick testing")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载 CV-Bench 数据集 ──────────────────────────────────────────────────
    print("Loading CV-Bench dataset...")
    if args.subset:
        cv_bench = load_dataset("nyu-visionx/CV-Bench", args.subset, split="test")
    else:
        cv_bench = load_dataset("nyu-visionx/CV-Bench", split="test")

    if args.max_samples:
        cv_bench = cv_bench.select(range(min(args.max_samples, len(cv_bench))))

    print(f"  Total samples: {len(cv_bench)}")

    # 统计任务分布
    task_counts = defaultdict(int)
    for item in cv_bench:
        task_counts[item["task"]] += 1
    for t, c in sorted(task_counts.items()):
        print(f"    {t}: {c}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print("\nLoading model...")
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
        predictor_ckpt=args.predictor_ckpt if args.mode != "base" else None,
        device=CFG.device,
    )

    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        print(f"Loading Qwen fine-tuned weights: {args.qwen_ckpt}")
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.base_model.load_state_dict(state, strict=False)
        print("Qwen weights loaded.")

    # ── 评估 ──────────────────────────────────────────────────────────────────
    base_metrics = None
    enhanced_metrics = None

    if args.mode in ["base", "both"]:
        base_results = evaluate_base(
            model, cv_bench,
            save_path=os.path.join(args.save_dir, "base_results.json"),
        )
        base_metrics = compute_metrics(base_results, label="Base Qwen")

    if args.mode in ["enhanced", "both"]:
        enhanced_results = evaluate_enhanced(
            model, cv_bench,
            save_path=os.path.join(args.save_dir, "enhanced_results.json"),
        )
        enhanced_metrics = compute_metrics(enhanced_results, label="Enhanced")

    if base_metrics and enhanced_metrics:
        print_comparison(base_metrics, enhanced_metrics)

    # 保存汇总
    summary = {}
    if base_metrics:
        summary["base"] = base_metrics
    if enhanced_metrics:
        summary["enhanced"] = enhanced_metrics
    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()