"""
CV-Bench 评估脚本。

用法：
    CUDA_VISIBLE_DEVICES=0 python eval/eval_cvbench.py --mode base
    CUDA_VISIBLE_DEVICES=0 python eval/eval_cvbench.py --mode enhanced
    python eval/eval_cvbench.py --mode spatial --qwen_ckpt outputs/focus_v2_ckpt/qwen_best.pt
    python eval/eval_cvbench.py --mode finetune_baseline --qwen_ckpt outputs/ablation_baseline/qwen_best.pt
    python eval/eval_cvbench.py --mode no_pcs --no_pcs_qwen_ckpt outputs/ablation_no_pcs/qwen_best.pt
    python eval/eval_cvbench.py --mode both
    python eval/eval_cvbench.py --mode all --qwen_ckpt outputs/ablation_baseline/qwen_best.pt
    python eval/eval_cvbench.py --mode both --max_samples 50
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


# ── Prompt 构造 ───────────────────────────────────────────────────────────────

def build_prompt(prompt: str, choices: list) -> str:
    choice_text = "\n".join(
        [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
    )
    instruction = (
        "Answer the following question.\n"
        "Select the correct option and output ONLY the letter.\n"
        "Do NOT output explanation.\n"
    ) #这是正式的

    return f"{instruction}\n{prompt}\n\nChoices:\n{choice_text}\n\nAnswer:"


# ── 答案匹配 ──────────────────────────────────────────────────────────────────

def extract_choice_letter(response: str) -> str | None:
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None
    m = re.search(r'\(([A-Z])\)', text)
    if m:
        return f"({m.group(1)})"
    m = re.search(r'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?([A-Z])\)?', text)
    if m:
        return f"({m.group(1)})"
    m = re.match(r'^([A-Z])(?:[\s.,):]|$)', text)
    if m:
        return f"({m.group(1)})"
    return None


def match_by_content(response: str, choices: list) -> str | None:
    resp = response.strip().split("\n")[0].strip().lower()
    for i, choice in enumerate(choices):
        if resp == str(choice).strip().lower():
            return f"({chr(ord('A') + i)})"
    return None


def match_answer(response: str, choices: list, answer: str) -> tuple[bool, str | None]:
    extracted = extract_choice_letter(response)
    if extracted is not None:
        return extracted == answer, extracted
    extracted = match_by_content(response, choices)
    if extracted is not None:
        return extracted == answer, extracted
    return False, None


# ── 加载模型 ──────────────────────────────────────────────────────────────────

def load_base_model():
    """原始 Qwen2.5-VL，无任何改动。"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print("Loading vanilla Qwen2.5-VL for baseline...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id,
        torch_dtype=torch.float16,
        device_map=CFG.device,
    )
    base_model.eval()
    return base_model, processor


def load_finetune_baseline(qwen_ckpt):
    """原始 Qwen2.5-VL + finetune 过的 top-8 层权重，无 hook / 额外模块。"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print("Loading finetune baseline (Qwen + finetuned layers, no hooks)...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id,
        torch_dtype=torch.float16,
        device_map=CFG.device,
    )

    if qwen_ckpt and os.path.exists(qwen_ckpt):
        print(f"  Loading finetuned weights: {qwen_ckpt}")
        state = torch.load(qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"  Loaded {len(state)} weight tensors.")
    else:
        print(f"  WARNING: --qwen_ckpt not provided or not found, using vanilla model!")

    model.eval()
    return model, processor


def load_enhanced_model(args):
    """带 ClusterPredictor + SAE + SemanticCompleter + PCS 的增强模型（无 spatial）。"""
    from src.Model import QwenWithClusterPredictorAndSAE

    print("Loading enhanced model (ClusterPredictor + SAE)...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
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
        print("  Enhanced model weights (Stage 2) loaded.")

    model.to(CFG.device)
    model.eval()
    return model


def load_spatial_model(args):
    """带 SpatialPatchInteraction 的 v2 增强模型。"""
    from src.Model_v2 import QwenWithClusterPredictorAndSAE

    print("Loading spatial model (Model_v2: enhanced + SpatialPatchInteraction)...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id,
        sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path,
        inject_layer=args.layer,
        latent_mult=CFG.latent_mult,
        topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.spatial_predictor_ckpt,
        device=CFG.device,
    )
    if args.spatial_qwen_ckpt and os.path.exists(args.spatial_qwen_ckpt):
        state = torch.load(args.spatial_qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("  Spatial model weights (Stage 2) loaded.")

    model.to(CFG.device)
    model.eval()
    return model


def load_enhanced_model_no_pcs(args):
    """消融版增强模型：无 PCA suppression。"""
    from ablation.Model_no_pcs import QwenWithClusterPredictorAndSAE

    print("Loading enhanced model (NO PCS ablation)...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id,
        sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path,
        inject_layer=args.layer,
        latent_mult=CFG.latent_mult,
        topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.no_pcs_predictor_ckpt,
        device=CFG.device,
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
    """用原始或 finetune 过的 Qwen2.5-VL 评估（无 hook）。"""
    from qwen_vl_utils import process_vision_info

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        image = item["image"]
        prompt = item["prompt"]
        answer = item["answer"]
        choices = item["choices"]

        prompt_text = build_prompt(prompt, choices)

        tmp_path = "/tmp/cvbench_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        response = ""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{tmp_path}"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(base_model.device)

            with torch.no_grad():
                output_ids = base_model.generate(
                    **inputs,
                    max_new_tokens=32, #basic是32
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({
            "idx": item["idx"],
            "task": item["task"],
            "source": item["source"],
            "type": item["type"],
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
    """用增强模型评估（enhanced 或 spatial 均可）。"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        image = item["image"]
        prompt = item["prompt"]
        answer = item["answer"]
        choices = item["choices"]

        prompt_text = build_prompt(prompt, choices)

        tmp_path = "/tmp/cvbench_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        response = ""
        cluster_ids = []
        try:
            result = model.generate(
                image_path=tmp_path,
                question=prompt_text,
                verbose=False,
            )
            response = result["final_answer"]
            cluster_ids = result.get("cluster_ids", [])
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({
            "idx": item["idx"],
            "task": item["task"],
            "source": item["source"],
            "type": item["type"],
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
    by_source = defaultdict(list)
    by_task   = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)
        by_task[r["task"]].append(r)

    def acc(items):
        if not items:
            return 0.0
        return sum(1 for i in items if i["correct"]) / len(items) * 100

    n_unparsed = sum(1 for r in results if r.get("extracted") is None)
    n_total = len(results)
    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) responses could not be parsed")
        unparsed_by_task = defaultdict(int)
        for r in results:
            if r.get("extracted") is None:
                unparsed_by_task[r["task"]] += 1
        for t, c in sorted(unparsed_by_task.items()):
            total_t = len(by_task[t])
            print(f"    {t}: {c}/{total_t} unparsed")
        samples = [r for r in results if r.get("extracted") is None][:3]
        for s in samples:
            print(f"    example: task={s['task']} ans={s['answer']}")
            print(f"      resp=\"{s['response'][:100]}\"")

    acc_ade  = acc(by_source.get("ADE20K", []))
    acc_coco = acc(by_source.get("COCO", []))
    acc_omni = acc(by_source.get("Omni3D", []))
    acc_2d   = (acc_ade + acc_coco) / 2
    acc_3d   = acc_omni
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
    print(f"  Per-Task:")
    for task_name in sorted(by_task.keys()):
        items = by_task[task_name]
        print(f"    {task_name:20s}: {acc(items):6.2f}  (n={len(items)})")
    print(f"  Unparsed: {n_unparsed}/{n_total}")
    print(f"{'='*60}")

    return {
        "cv_bench": cv_bench, "acc_2d": acc_2d, "acc_3d": acc_3d,
        "acc_ade": acc_ade, "acc_coco": acc_coco, "acc_omni": acc_omni,
        "per_task": {t: acc(items) for t, items in by_task.items()},
        "n_unparsed": n_unparsed, "n_total": n_total,
    }


def print_comparison(metrics_dict):
    """打印多个模型的对比表。"""
    labels = list(metrics_dict.keys())
    if len(labels) < 2:
        return

    print(f"\n{'='*70}")
    print(f"  CV-Bench Comparison")
    print(f"{'='*70}")

    header = f"  {'Metric':<25s}"
    for label in labels:
        header += f" {label:>12s}"
    if len(labels) >= 2:
        header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*60}")

    rows = [
        ("CV-Bench Overall", "cv_bench"),
        ("2D Accuracy", "acc_2d"),
        ("  ADE20K", "acc_ade"),
        ("  COCO", "acc_coco"),
        ("3D Accuracy", "acc_3d"),
        ("  Omni3D", "acc_omni"),
    ]
    for name, key in rows:
        line = f"  {name:<25s}"
        vals = [metrics_dict[l][key] for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    all_tasks = sorted(set(
        t for m in metrics_dict.values() for t in m["per_task"]
    ))
    print(f"  {'─'*60}")
    for t in all_tasks:
        line = f"    {t:<23s}"
        vals = [metrics_dict[l]["per_task"].get(t, 0) for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    print(f"  {'─'*60}")
    line = f"  {'Unparsed':<25s}"
    for l in labels:
        line += f" {metrics_dict[l]['n_unparsed']:>12d}"
    print(line)
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "spatial", "no_pcs",
                                 "finetune_baseline", "both", "all"])
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str, default="outputs/focus_ckpt_0.75_64_5000/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None,
                        help="For enhanced: Stage 2 weights. For finetune_baseline: ablation weights.")
    parser.add_argument("--baseline_ckpt", type=str, default=None,
                        help="Finetune baseline weights (used in --mode all)")
    # ── spatial 专用参数 ──
    parser.add_argument("--spatial_predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_spatial/predictor_best.pt",
                        help="Spatial model (Model_v2) predictor weights")
    parser.add_argument("--spatial_qwen_ckpt", type=str, default=None,
                        help="Spatial model (Model_v2) LM weights")
    # ── no_pcs 专用参数 ──
    parser.add_argument("--no_pcs_predictor_ckpt", type=str,
                        default="outputs/ablation_no_pcs/predictor_best.pt",
                        help="No-PCS ablation predictor weights")
    parser.add_argument("--no_pcs_qwen_ckpt", type=str, default=None,
                        help="No-PCS ablation LM weights")
    parser.add_argument("--subset", type=str, default=None, choices=["2D", "3D"])
    parser.add_argument("--save_dir", type=str, default="outputs/cvbench_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载数据集 ──
    print("Loading CV-Bench...")
    if args.subset:
        ds = load_dataset("nyu-visionx/CV-Bench", args.subset, split="test")
    else:
        ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    task_counts = defaultdict(int)
    for item in ds:
        task_counts[item["task"]] += 1
    for t, c in sorted(task_counts.items()):
        print(f"    {t}: {c}")

    # ── 评估 ──
    all_metrics = {}

    run_base     = args.mode in ["base", "both", "all"]
    run_ft_base  = args.mode in ["finetune_baseline", "all"]
    run_no_pcs   = args.mode in ["no_pcs", "all"]
    run_enhanced = args.mode in ["enhanced", "both", "all"]
    run_spatial  = args.mode in ["spatial", "all"]

    # 1. Base (vanilla Qwen)
    if run_base:
        base_model, processor = load_base_model()
        res = evaluate_base(
            base_model, processor, ds,
            save_path=os.path.join(args.save_dir, "base_results.json"),
            label="base (vanilla)",
        )
        all_metrics["base"] = compute_metrics(res, label="Base Qwen2.5-VL")
        del base_model, processor
        torch.cuda.empty_cache()

    # 2. Finetune baseline (same data, no hooks)
    if run_ft_base:
        ft_ckpt = args.baseline_ckpt or args.qwen_ckpt
        ft_model, ft_processor = load_finetune_baseline(ft_ckpt)
        res = evaluate_base(
            ft_model, ft_processor, ds,
            save_path=os.path.join(args.save_dir, "finetune_baseline_results.json"),
            label="finetune_baseline",
        )
        all_metrics["finetune_baseline"] = compute_metrics(res, label="Finetune Baseline (no hooks)")
        del ft_model, ft_processor
        torch.cuda.empty_cache()

    # 3. No-PCS ablation
    if run_no_pcs:
        no_pcs_model = load_enhanced_model_no_pcs(args)
        res = evaluate_enhanced(
            no_pcs_model, ds,
            save_path=os.path.join(args.save_dir, "no_pcs_results.json"),
            label="enhanced (no PCS)",
        )
        all_metrics["no_pcs"] = compute_metrics(res, label="Enhanced (no PCS)")
        del no_pcs_model
        torch.cuda.empty_cache()

    # 4. Enhanced (v1: full architecture, no spatial)
    if run_enhanced:
        enhanced_model = load_enhanced_model(args)
        res = evaluate_enhanced(
            enhanced_model, ds,
            save_path=os.path.join(args.save_dir, "enhanced_results.json"),
            label="enhanced (v1)",
        )
        all_metrics["enhanced"] = compute_metrics(res, label="Enhanced (v1)")
        del enhanced_model
        torch.cuda.empty_cache()

    # 5. Spatial (v2: full architecture + SpatialPatchInteraction)
    if run_spatial:
        spatial_model = load_spatial_model(args)
        res = evaluate_enhanced(
            spatial_model, ds,
            save_path=os.path.join(args.save_dir, "spatial_results.json"),
            label="spatial (v2)",
        )
        all_metrics["spatial"] = compute_metrics(res, label="Spatial (v2)")
        del spatial_model
        torch.cuda.empty_cache()

    # ── 对比 ──
    if len(all_metrics) >= 2:
        print_comparison(all_metrics)

    # ── 保存 ──
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nDone. Summary saved to {args.save_dir}/summary.json")


if __name__ == "__main__":
    main()