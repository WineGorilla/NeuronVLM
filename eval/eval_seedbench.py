"""
SEED-Bench 评估脚本。

SEED-Bench 包含 19K 多选题，覆盖 12 个评估维度（9 图像 + 3 视频）。
数据集: https://huggingface.co/datasets/lmms-lab/SEED-Bench
论文: https://arxiv.org/abs/2307.16125 (CVPR 2024)

12 个评估维度 (question_type_id):
  图像 (1-9):
    1: Scene Understanding       2: Instance Identity
    3: Instance Attributes       4: Instance Location  *
    5: Instances Counting         6: Spatial Relation   *
    7: Instance Interaction       8: Visual Reasoning
    9: Text Understanding
  视频 (10-12): 本脚本跳过视频题目

  (* = 与空间/距离相关)

⚠ 注意: 完整数据集约 27GB，首次下载较慢。
  可用 --max_samples 或 --spatial_only 快速测试。

用法：
    python eval/eval_seedbench.py --mode base
    python eval/eval_seedbench.py --mode both
    python eval/eval_seedbench.py --mode both --max_samples 500
    CUDA_VISIBLE_DEVICES=1 python eval/eval_seedbench.py --mode both --spatial_only
    python eval/eval_seedbench.py --mode spatial --spatial_qwen_ckpt outputs/focus_v2_ckpt/qwen_best.pt
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


# ── 维度定义 ──────────────────────────────────────────────────────────────────

DIMENSION_NAMES = {
    1: "Scene Understanding",
    2: "Instance Identity",
    3: "Instance Attributes",
    4: "Instance Location",
    5: "Instances Counting",
    6: "Spatial Relation",
    7: "Instance Interaction",
    8: "Visual Reasoning",
    9: "Text Understanding",
    10: "Action Recognition",
    11: "Action Prediction",
    12: "Procedure Understanding",
}

# 图像维度 (跳过视频 10-12)
IMAGE_DIMENSIONS = {1, 2, 3, 4, 5, 6, 7, 8, 9}

# 与空间/距离相关
SPATIAL_DIMENSIONS = {4, 6}  # Instance Location + Spatial Relation


# ── Prompt 构造 ───────────────────────────────────────────────────────────────

def build_seedbench_prompt(question: str, choices: dict[str, str]) -> str:
    instruction = (
        "Answer the following question.\n"
        "Select the correct option and output ONLY the letter.\n"
        "Do NOT output explanation.\n"
    )
    choice_lines = []
    for key in ["A", "B", "C", "D"]:
        if key in choices and choices[key]:
            choice_lines.append(f"{key}. {choices[key]}")

    return f"{instruction}\n{question}\n\nChoices:\n" + "\n".join(choice_lines) + "\n\nAnswer:"


# ── 答案匹配 ──────────────────────────────────────────────────────────────────

def extract_choice_letter(response: str) -> str | None:
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None
    m = re.search(r'\(([A-D])\)', text)
    if m:
        return m.group(1)
    m = re.search(r'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?([A-D])\)?', text)
    if m:
        return m.group(1)
    m = re.match(r'^([A-D])(?:[\s.,):]|$)', text)
    if m:
        return m.group(1)
    return None


def match_answer(response: str, answer: str) -> tuple[bool, str | None]:
    extracted = extract_choice_letter(response)
    if extracted is not None:
        return extracted == answer.strip().upper(), extracted
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
        dim_id = item["question_type_id"]
        question = item["question"]
        answer = item["answer"]
        choices = {
            "A": item.get("choice_a", ""),
            "B": item.get("choice_b", ""),
            "C": item.get("choice_c", ""),
            "D": item.get("choice_d", ""),
        }
        prompt_text = build_seedbench_prompt(question, choices)

        # image 字段是列表，取第一张
        images = item.get("image", [])
        if not images:
            results.append({
                "question_id": item.get("question_id", ""),
                "dimension": dim_id,
                "dimension_name": DIMENSION_NAMES.get(dim_id, ""),
                "answer": answer, "response": "",
                "extracted": None, "correct": False,
            })
            continue

        tmp_path = "/tmp/seedbench_tmp.png"
        img = images[0] if isinstance(images, list) else images
        if isinstance(img, Image.Image):
            img.convert("RGB").save(tmp_path)
        else:
            tmp_path = str(img)

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
                    **inputs, max_new_tokens=32, do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [error] qid={item.get('question_id','?')}: {e}")

        correct, extracted = match_answer(response, answer)
        results.append({
            "question_id": item.get("question_id", ""),
            "dimension": dim_id,
            "dimension_name": DIMENSION_NAMES.get(dim_id, ""),
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
        dim_id = item["question_type_id"]
        question = item["question"]
        answer = item["answer"]
        choices = {
            "A": item.get("choice_a", ""),
            "B": item.get("choice_b", ""),
            "C": item.get("choice_c", ""),
            "D": item.get("choice_d", ""),
        }
        prompt_text = build_seedbench_prompt(question, choices)

        images = item.get("image", [])
        if not images:
            results.append({
                "question_id": item.get("question_id", ""),
                "dimension": dim_id,
                "dimension_name": DIMENSION_NAMES.get(dim_id, ""),
                "answer": answer, "response": "",
                "extracted": None, "correct": False,
                "cluster_ids": [],
            })
            continue

        tmp_path = "/tmp/seedbench_tmp.png"
        img = images[0] if isinstance(images, list) else images
        if isinstance(img, Image.Image):
            img.convert("RGB").save(tmp_path)
        else:
            tmp_path = str(img)

        response = ""
        cluster_ids = []
        try:
            result = model.generate(
                image_path=tmp_path, question=prompt_text, verbose=False,
            )
            response = result["final_answer"]
            cluster_ids = result.get("cluster_ids", [])
        except Exception as e:
            print(f"  [error] qid={item.get('question_id','?')}: {e}")

        correct, extracted = match_answer(response, answer)
        results.append({
            "question_id": item.get("question_id", ""),
            "dimension": dim_id,
            "dimension_name": DIMENSION_NAMES.get(dim_id, ""),
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


# ── 统计 ──────────────────────────────────────────────────────────────────────

def compute_metrics(results, label=""):
    by_dim = defaultdict(list)
    for r in results:
        by_dim[r["dimension"]].append(r)

    def acc(items):
        if not items:
            return 0.0
        return sum(1 for i in items if i["correct"]) / len(items) * 100

    n_total = len(results)
    n_unparsed = sum(1 for r in results if r.get("extracted") is None)
    overall_acc = acc(results)

    per_dim = {}
    for dim_id in sorted(by_dim.keys()):
        per_dim[dim_id] = acc(by_dim[dim_id])

    # 空间相关
    spatial_items = []
    for d in SPATIAL_DIMENSIONS:
        spatial_items.extend(by_dim.get(d, []))
    acc_spatial = acc(spatial_items)

    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) "
              f"responses could not be parsed")

    print(f"\n{'='*60}")
    print(f"  SEED-Bench Results: {label}")
    print(f"{'='*60}")
    print(f"  Overall (image dims) : {overall_acc:.2f}  (n={n_total})")
    print(f"  Spatial Dimensions   : {acc_spatial:.2f}  "
          f"(n={len(spatial_items)})")
    print(f"{'─'*60}")
    print(f"  Per Dimension:")
    for dim_id in sorted(per_dim.keys()):
        name = DIMENSION_NAMES.get(dim_id, f"Dim_{dim_id}")
        n = len(by_dim[dim_id])
        marker = " *" if dim_id in SPATIAL_DIMENSIONS else ""
        print(f"    {dim_id:2d}. {name:25s}: {per_dim[dim_id]:6.2f}  "
              f"(n={n}){marker}")
    print(f"  (* = spatial related)")
    print(f"  Unparsed: {n_unparsed}/{n_total}")
    print(f"{'='*60}")

    return {
        "overall": overall_acc,
        "acc_spatial": acc_spatial,
        "per_dim": {str(k): v for k, v in per_dim.items()},
        "n_unparsed": n_unparsed,
        "n_total": n_total,
    }


def print_comparison(metrics_dict):
    labels = list(metrics_dict.keys())
    if len(labels) < 2:
        return

    print(f"\n{'='*70}")
    print(f"  SEED-Bench Comparison")
    print(f"{'='*70}")

    header = f"  {'Metric':<32s}"
    for label in labels:
        header += f" {label:>12s}"
    header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*65}")

    for name, key in [("Overall", "overall"),
                      ("Spatial Dimensions", "acc_spatial")]:
        line = f"  {name:<32s}"
        vals = [metrics_dict[l][key] for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        d = vals[-1] - vals[0]
        line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    print(f"  {'─'*65}")
    all_dims = sorted(set(
        k for m in metrics_dict.values() for k in m["per_dim"]
    ), key=lambda x: int(x))
    for dim_s in all_dims:
        dim_id = int(dim_s)
        name = DIMENSION_NAMES.get(dim_id, f"Dim_{dim_id}")
        marker = "*" if dim_id in SPATIAL_DIMENSIONS else " "
        line = f"  {marker} {dim_id:2d}. {name:<27s}"
        vals = [metrics_dict[l]["per_dim"].get(dim_s, 0) for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        d = vals[-1] - vals[0]
        line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    print(f"  {'─'*65}")
    line = f"  {'Unparsed':<32s}"
    for l in labels:
        line += f" {metrics_dict[l]['n_unparsed']:>12d}"
    print(line)
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SEED-Bench Evaluation")
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
    # ── SEED-Bench 特有参数 ──
    parser.add_argument("--spatial_only", action="store_true",
                        help="只评估 Instance Location + Spatial Relation")
    parser.add_argument("--dimensions", type=int, nargs="+", default=None,
                        help="指定维度 ID，如 --dimensions 4 6")
    parser.add_argument("--save_dir", type=str,
                        default="outputs/seedbench_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载数据集 ──
    print("Loading SEED-Bench from HuggingFace (lmms-lab/SEED-Bench)...")
    print("  ⚠ 完整数据集约 27GB，首次下载需要一些时间")
    ds = load_dataset("lmms-lab/SEED-Bench", split="test")
    print(f"  columns: {ds.column_names}")
    print(f"  Total samples: {len(ds)}")

    # 过滤: 只保留图像题目 (跳过视频 10-12)
    if args.spatial_only:
        target_dims = SPATIAL_DIMENSIONS
    elif args.dimensions:
        target_dims = set(args.dimensions)
    else:
        target_dims = IMAGE_DIMENSIONS

    ds = ds.filter(lambda x: x["question_type_id"] in target_dims)
    print(f"  After filtering (dims={sorted(target_dims)}): {len(ds)} samples")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"  Using first {len(ds)} samples")

    # 统计
    dim_counts = defaultdict(int)
    for item in ds:
        dim_counts[item["question_type_id"]] += 1
    for d, c in sorted(dim_counts.items()):
        name = DIMENSION_NAMES.get(d, "?")
        marker = " *" if d in SPATIAL_DIMENSIONS else ""
        print(f"    {d}: {name} ({c}){marker}")

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
        json.dump(all_metrics, f, indent=2)
    print(f"\nDone. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()