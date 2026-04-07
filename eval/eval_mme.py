"""
MME 评估脚本。

MME 包含 14 个子任务，评估感知和认知两大能力。
数据集: https://huggingface.co/datasets/lmms-lab/MME
论文: https://arxiv.org/abs/2306.13394

格式: Yes/No 是非题
  - 每张图配 2 个问题（一个答案 Yes，一个答案 No）
  - accuracy: 单问题准确率
  - accuracy+: 一张图的两个问题都答对才算对
  - score = accuracy + accuracy+，每个子任务满分 200

14 个子任务:
  Perception (10): existence, count, position, color, poster,
                   celebrity, scene, landmark, artwork, OCR
  Cognition (4):   commonsense_reasoning, numerical_calculation,
                   text_translation, code_reasoning

  Perception 满分 2000, Cognition 满分 800

用法：
    python eval/eval_mme.py --mode base
    python eval/eval_mme.py --mode both
    python eval/eval_mme.py --mode both --max_samples 200
    python eval/eval_mme.py --mode spatial --spatial_qwen_ckpt outputs/focus_v2_ckpt/qwen_best.pt
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


# ── 子任务定义 ────────────────────────────────────────────────────────────────

PERCEPTION_TASKS = [
    "existence", "count", "position", "color", "poster",
    "celebrity", "scene", "landmark", "artwork", "OCR",
]
COGNITION_TASKS = [
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
]
ALL_TASKS = PERCEPTION_TASKS + COGNITION_TASKS

# 与空间/距离相关
SPATIAL_TASKS = {"position", "count", "existence"}


# ── Prompt 构造 ───────────────────────────────────────────────────────────────

def build_mme_prompt(question: str) -> str:
    """MME 的 question 已经包含 'Please answer yes or no'，直接使用。"""
    # 如果问题已经包含指令就直接返回
    if "yes or no" in question.lower() or "answer yes" in question.lower():
        return question
    return f"{question}\nPlease answer yes or no."


# ── 答案匹配 ──────────────────────────────────────────────────────────────────

def extract_yes_no(response: str) -> str | None:
    """从模型输出中提取 Yes/No。"""
    text = response.strip().split("\n")[0].strip().lower()
    if not text:
        return None
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    # 包含匹配
    yes_pos = text.find("yes")
    no_pos = text.find("no")
    if yes_pos >= 0 and (no_pos < 0 or yes_pos < no_pos):
        return "Yes"
    if no_pos >= 0 and (yes_pos < 0 or no_pos < yes_pos):
        return "No"
    return None


def match_yes_no(response: str, answer: str) -> tuple[bool, str | None]:
    extracted = extract_yes_no(response)
    if extracted is not None:
        return extracted.lower() == answer.strip().lower(), extracted
    return False, None


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
        state = torch.load(qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"  Loaded {len(state)} weight tensors.")
    else:
        print("  WARNING: --qwen_ckpt not found, using vanilla model!")
    model.eval()
    return model, processor


def load_enhanced_model(args):
    from src.Model import QwenWithClusterPredictorAndSAE
    print("Loading enhanced model...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
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
        print("  Enhanced model weights loaded.")
    model.to(CFG.device); model.eval()
    return model


def load_spatial_model(args):
    from src.Model_v2 import QwenWithClusterPredictorAndSAE
    print("Loading spatial model (Model_v2)...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
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
    model.to(CFG.device); model.eval()
    return model


def load_enhanced_model_no_pcs(args):
    from ablation.Model_no_pcs import QwenWithClusterPredictorAndSAE
    print("Loading enhanced model (NO PCS ablation)...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
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
    model.to(CFG.device); model.eval()
    return model


# ── 评估循环 ──────────────────────────────────────────────────────────────────

def evaluate_base(base_model, processor, dataset, save_path=None, label="base"):
    from qwen_vl_utils import process_vision_info

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        category = item.get("category", "")
        question = item["question"]
        answer = item["answer"]
        question_id = item.get("question_id", "")
        prompt_text = build_mme_prompt(question)

        image = item.get("image")
        tmp_path = "/tmp/mme_tmp.png"
        if isinstance(image, Image.Image):
            image.convert("RGB").save(tmp_path)
        else:
            tmp_path = str(image) if image else ""

        response = ""
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{tmp_path}"},
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
                    **inputs, max_new_tokens=16, do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [error] qid={question_id}: {e}")

        correct, extracted = match_yes_no(response, answer)
        results.append({
            "question_id": question_id,
            "category": category,
            "question": question[:100],
            "answer": answer,
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
        category = item.get("category", "")
        question = item["question"]
        answer = item["answer"]
        question_id = item.get("question_id", "")
        prompt_text = build_mme_prompt(question)

        image = item.get("image")
        tmp_path = "/tmp/mme_tmp.png"
        if isinstance(image, Image.Image):
            image.convert("RGB").save(tmp_path)
        else:
            tmp_path = str(image) if image else ""

        response = ""
        cluster_ids = []
        try:
            result = model.generate(
                image_path=tmp_path, question=prompt_text, verbose=False,
            )
            response = result["final_answer"]
            cluster_ids = result.get("cluster_ids", [])
        except Exception as e:
            print(f"  [error] qid={question_id}: {e}")

        correct, extracted = match_yes_no(response, answer)
        results.append({
            "question_id": question_id,
            "category": category,
            "question": question[:100],
            "answer": answer,
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


# ── MME 官方计分 ──────────────────────────────────────────────────────────────

def compute_metrics(results, label=""):
    """
    MME 官方计分:
      - accuracy: 单问题准确率
      - accuracy+: 一张图的两个问题都对才算对 (按 question_id 前缀分组)
      - score = (accuracy + accuracy+) * 100, 满分 200/子任务
    """
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    n_total = len(results)
    n_unparsed = sum(1 for r in results if r.get("extracted") is None)

    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) "
              f"responses could not be parsed as Yes/No")
        for s in [r for r in results if r.get("extracted") is None][:3]:
            print(f"    example: cat={s['category']} ans={s['answer']}")
            print(f"      resp=\"{s['response'][:100]}\"")

    per_task = {}
    for cat, items in sorted(by_category.items()):
        n = len(items)
        # accuracy: 单问题
        n_correct = sum(1 for r in items if r["correct"])
        acc = n_correct / n if n else 0.0

        # accuracy+: 按图片分组 (question_id 去掉末尾的数字后缀做 key)
        # MME 的 question_id 格式通常是 "category_imgid_qid"
        # 同一张图的两个问题共享前缀
        by_image = defaultdict(list)
        for r in items:
            # 用 question_id 去掉最后一部分作为图片 key
            qid = r["question_id"]
            # 尝试多种分隔方式
            parts = qid.rsplit("_", 1) if "_" in qid else [qid, "0"]
            img_key = parts[0] if len(parts) > 1 else qid
            by_image[img_key].append(r)

        n_images = len(by_image)
        n_both_correct = sum(
            1 for img_items in by_image.values()
            if all(r["correct"] for r in img_items)
        )
        acc_plus = n_both_correct / n_images if n_images else 0.0

        score = (acc + acc_plus) * 100  # 满分 200

        per_task[cat] = {
            "accuracy": acc * 100,
            "accuracy_plus": acc_plus * 100,
            "score": score,
            "n_questions": n,
            "n_images": n_images,
        }

    # 汇总
    perception_score = sum(
        per_task[t]["score"] for t in PERCEPTION_TASKS if t in per_task
    )
    cognition_score = sum(
        per_task[t]["score"] for t in COGNITION_TASKS if t in per_task
    )
    total_score = perception_score + cognition_score

    print(f"\n{'='*60}")
    print(f"  MME Results: {label}")
    print(f"{'='*60}")
    print(f"  Total Score          : {total_score:.1f}")
    print(f"  Perception Score     : {perception_score:.1f}  (max 2000)")
    print(f"  Cognition Score      : {cognition_score:.1f}  (max 800)")
    print(f"{'─'*60}")
    print(f"  {'Task':<28s} {'Acc':>6s} {'Acc+':>6s} {'Score':>7s}  "
          f"{'n':>4s}")
    print(f"  {'─'*55}")

    print(f"  Perception:")
    for t in PERCEPTION_TASKS:
        if t in per_task:
            d = per_task[t]
            marker = " *" if t in SPATIAL_TASKS else ""
            print(f"    {t:<26s} {d['accuracy']:6.1f} "
                  f"{d['accuracy_plus']:6.1f} {d['score']:7.1f}  "
                  f"{d['n_questions']:>4d}{marker}")

    print(f"  Cognition:")
    for t in COGNITION_TASKS:
        if t in per_task:
            d = per_task[t]
            print(f"    {t:<26s} {d['accuracy']:6.1f} "
                  f"{d['accuracy_plus']:6.1f} {d['score']:7.1f}  "
                  f"{d['n_questions']:>4d}")

    print(f"  (* = spatial related)")
    print(f"  Unparsed: {n_unparsed}/{n_total}")
    print(f"{'='*60}")

    return {
        "total_score": total_score,
        "perception_score": perception_score,
        "cognition_score": cognition_score,
        "per_task": per_task,
        "n_unparsed": n_unparsed,
        "n_total": n_total,
    }


def print_comparison(metrics_dict):
    labels = list(metrics_dict.keys())
    if len(labels) < 2:
        return

    print(f"\n{'='*75}")
    print(f"  MME Comparison")
    print(f"{'='*75}")

    header = f"  {'Metric':<28s}"
    for label in labels:
        header += f" {label:>12s}"
    header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*70}")

    for name, key in [("Total Score", "total_score"),
                      ("Perception", "perception_score"),
                      ("Cognition", "cognition_score")]:
        line = f"  {name:<28s}"
        vals = [metrics_dict[l][key] for l in labels]
        for v in vals:
            line += f" {v:12.1f}"
        d = vals[-1] - vals[0]
        line += f" {'+'if d>0 else ''}{d:7.1f}"
        print(line)

    # Per-task scores
    all_tasks = []
    for t in PERCEPTION_TASKS + COGNITION_TASKS:
        if any(t in m["per_task"] for m in metrics_dict.values()):
            all_tasks.append(t)

    if all_tasks:
        print(f"  {'─'*70}")
        for t in all_tasks:
            marker = "*" if t in SPATIAL_TASKS else " "
            line = f"  {marker} {t:<26s}"
            vals = [metrics_dict[l]["per_task"].get(t, {}).get("score", 0)
                    for l in labels]
            for v in vals:
                line += f" {v:12.1f}"
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.1f}"
            print(line)

    print(f"{'='*75}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MME Benchmark Evaluation")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "spatial", "no_pcs",
                                 "finetune_baseline", "both", "all"])
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_0.75_64_5000/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--spatial_predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_spatial/predictor_best.pt")
    parser.add_argument("--spatial_qwen_ckpt", type=str, default=None)
    parser.add_argument("--no_pcs_predictor_ckpt", type=str,
                        default="outputs/ablation_no_pcs/predictor_best.pt")
    parser.add_argument("--no_pcs_qwen_ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str,
                        default="outputs/mme_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载数据集 ──
    print("Loading MME from HuggingFace (lmms-lab/MME)...")
    ds = load_dataset("lmms-lab/MME", split="test")
    print(f"  columns: {ds.column_names}")
    print(f"  Total samples: {len(ds)}")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"  Using first {len(ds)} samples")

    # 统计
    cat_counts = defaultdict(int)
    for item in ds:
        cat_counts[item.get("category", "unknown")] += 1
    for c, n in sorted(cat_counts.items()):
        task_type = "P" if c in PERCEPTION_TASKS else (
            "C" if c in COGNITION_TASKS else "?")
        marker = " *" if c in SPATIAL_TASKS else ""
        print(f"    [{task_type}] {c}: {n}{marker}")

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
            base_model, processor, ds,
            save_path=os.path.join(args.save_dir, "base_results.json"),
            label="base (vanilla)",
        )
        all_metrics["base"] = compute_metrics(res, label="Base Qwen2.5-VL")
        del base_model, processor; torch.cuda.empty_cache()

    if run_ft_base:
        ft_ckpt = args.baseline_ckpt or args.qwen_ckpt
        ft_model, ft_proc = load_finetune_baseline(ft_ckpt)
        res = evaluate_base(
            ft_model, ft_proc, ds,
            save_path=os.path.join(args.save_dir, "ft_baseline_results.json"),
            label="finetune_baseline",
        )
        all_metrics["ft_baseline"] = compute_metrics(
            res, label="Finetune Baseline")
        del ft_model, ft_proc; torch.cuda.empty_cache()

    if run_no_pcs:
        no_pcs_model = load_enhanced_model_no_pcs(args)
        res = evaluate_enhanced(
            no_pcs_model, ds,
            save_path=os.path.join(args.save_dir, "no_pcs_results.json"),
            label="enhanced (no PCS)",
        )
        all_metrics["no_pcs"] = compute_metrics(res, label="Enhanced (no PCS)")
        del no_pcs_model; torch.cuda.empty_cache()

    if run_enhanced:
        enhanced_model = load_enhanced_model(args)
        res = evaluate_enhanced(
            enhanced_model, ds,
            save_path=os.path.join(args.save_dir, "enhanced_results.json"),
            label="enhanced (v1)",
        )
        all_metrics["enhanced"] = compute_metrics(res, label="Enhanced (v1)")
        del enhanced_model; torch.cuda.empty_cache()

    if run_spatial:
        spatial_model = load_spatial_model(args)
        res = evaluate_enhanced(
            spatial_model, ds,
            save_path=os.path.join(args.save_dir, "spatial_results.json"),
            label="spatial (v2)",
        )
        all_metrics["spatial"] = compute_metrics(res, label="Spatial (v2)")
        del spatial_model; torch.cuda.empty_cache()

    if len(all_metrics) >= 2:
        print_comparison(all_metrics)

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nDone. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()