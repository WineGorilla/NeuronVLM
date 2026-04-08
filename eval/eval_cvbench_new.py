"""
CV-Bench 评估脚本。

用法：
    python eval/eval_cvbench.py --mode base
    python eval/eval_cvbench.py --mode enhanced
    python eval/eval_cvbench.py --mode random_sae
    CUDA_VISIBLE_DEVICES=0 python eval/eval_cvbench_new.py --mode pcs_only
    python eval/eval_cvbench.py --mode no_pcs
    python eval/eval_cvbench.py --mode spatial
    python eval/eval_cvbench.py --mode finetune_baseline --qwen_ckpt outputs/ablation_baseline/qwen_best.pt
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


# ── Prompt ────────────────────────────────────────────────────────────────────

def build_prompt(prompt, choices):
    choice_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
    instruction = (
        "Answer the following question.\n"
        "Select the correct option and output ONLY the letter.\n"
        "Do NOT output explanation.\n"
    )
    return f"{instruction}\n{prompt}\n\nChoices:\n{choice_text}\n\nAnswer:"


# ── 答案匹配 ──────────────────────────────────────────────────────────────────

def extract_choice_letter(response):
    text = response.strip().split("\n")[0].strip().upper()
    if not text: return None
    m = re.search(r'\(([A-Z])\)', text)
    if m: return f"({m.group(1)})"
    m = re.search(r'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?([A-Z])\)?', text)
    if m: return f"({m.group(1)})"
    m = re.match(r'^([A-Z])(?:[\s.,):]|$)', text)
    if m: return f"({m.group(1)})"
    return None

def match_by_content(response, choices):
    resp = response.strip().split("\n")[0].strip().lower()
    for i, choice in enumerate(choices):
        if resp == str(choice).strip().lower():
            return f"({chr(ord('A')+i)})"
    return None

def match_answer(response, choices, answer):
    extracted = extract_choice_letter(response)
    if extracted is not None:
        return extracted == answer, extracted
    extracted = match_by_content(response, choices)
    if extracted is not None:
        return extracted == answer, extracted
    return False, None


# ── 加载模型 ──────────────────────────────────────────────────────────────────

def load_base_model():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading vanilla Qwen2.5-VL for baseline...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16, device_map=CFG.device)
    model.eval()
    return model, processor

def load_finetune_baseline(qwen_ckpt):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading finetune baseline...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16, device_map=CFG.device)
    if qwen_ckpt and os.path.exists(qwen_ckpt):
        state = torch.load(qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"  Loaded {len(state)} weight tensors.")
    else:
        print("  WARNING: --qwen_ckpt not found, using vanilla model!")
    model.eval()
    return model, processor

def _load_enhanced(cls_import, args, predictor_ckpt, qwen_ckpt, label=""):
    print(f"Loading {label}...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    model = cls_import.from_pretrained(
        model_id=CFG.model_id, sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path, inject_layer=args.layer,
        latent_mult=CFG.latent_mult, topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=predictor_ckpt, device=CFG.device,
    )
    if qwen_ckpt and os.path.exists(qwen_ckpt):
        state = torch.load(qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"  {label} weights loaded.")
    model.to(CFG.device)
    model.eval()
    return model

def load_enhanced_model(args):
    from src.Model import QwenWithClusterPredictorAndSAE
    return _load_enhanced(QwenWithClusterPredictorAndSAE, args,
                          args.predictor_ckpt, args.qwen_ckpt, "Enhanced (v1)")

def load_spatial_model(args):
    from src.Model_v2 import QwenWithClusterPredictorAndSAE
    return _load_enhanced(QwenWithClusterPredictorAndSAE, args,
                          args.spatial_predictor_ckpt, args.spatial_qwen_ckpt, "Spatial (v2)")

def load_enhanced_model_no_pcs(args):
    from ablation.Model_no_pcs import QwenWithClusterPredictorAndSAE
    return _load_enhanced(QwenWithClusterPredictorAndSAE, args,
                          args.no_pcs_predictor_ckpt, args.no_pcs_qwen_ckpt, "Enhanced (no PCS)")

def load_pcs_only_model(args):
    from ablation.Model_pcs_only import QwenWithPCSOnly
    return _load_enhanced(QwenWithPCSOnly, args,
                          args.predictor_ckpt, args.qwen_ckpt, "PCS Only")

def load_random_sae_model(args):
    from ablation.Model_random_sae import QwenWithRandomSAE
    print("Loading Random SAE ablation model...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    model = QwenWithRandomSAE.from_pretrained(
        model_id=CFG.model_id, sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path, inject_layer=args.layer,
        latent_mult=CFG.latent_mult, topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.predictor_ckpt, device=CFG.device,
        random_seed=42,
    )
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("  Random SAE weights (Stage 2) loaded.")
    model.to(CFG.device)
    model.eval()
    return model


# ── 评估循环 ──────────────────────────────────────────────────────────────────

def evaluate_base(base_model, processor, dataset, save_path=None, label="base"):
    from qwen_vl_utils import process_vision_info
    print(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        image, prompt, answer, choices = item["image"], item["prompt"], item["answer"], item["choices"]
        prompt_text = build_prompt(prompt, choices)

        tmp_path = "/tmp/cvbench_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        response = ""
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{tmp_path}"},
                {"type": "text", "text": prompt_text},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(base_model.device)
            with torch.no_grad():
                output_ids = base_model.generate(**inputs, max_new_tokens=128, do_sample=False)
            response = processor.decode(output_ids[0, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({"idx": item["idx"], "task": item["task"], "source": item["source"],
                        "type": item["type"], "answer": answer, "response": response,
                        "extracted": extracted, "correct": correct})

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")
    return results


def evaluate_enhanced(model, dataset, save_path=None, label="enhanced"):
    print(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        image, prompt, answer, choices = item["image"], item["prompt"], item["answer"], item["choices"]
        prompt_text = build_prompt(prompt, choices)

        tmp_path = "/tmp/cvbench_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        response, cluster_ids = "", []
        try:
            result = model.generate(image_path=tmp_path, question=prompt_text, verbose=False)
            response = result["final_answer"]
            cluster_ids = result.get("cluster_ids", [])
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({"idx": item["idx"], "task": item["task"], "source": item["source"],
                        "type": item["type"], "answer": answer, "response": response,
                        "extracted": extracted, "correct": correct, "cluster_ids": cluster_ids})

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")
    return results


# ── 统计 ──────────────────────────────────────────────────────────────────────

def compute_metrics(results, label=""):
    by_source = defaultdict(list)
    by_task = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)
        by_task[r["task"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0.0

    n_unparsed = sum(1 for r in results if r.get("extracted") is None)
    n_total = len(results)
    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) unparsed")
        for s in [r for r in results if r.get("extracted") is None][:3]:
            print(f"    task={s['task']} ans={s['answer']} resp=\"{s['response'][:80]}\"")

    acc_ade = acc(by_source.get("ADE20K", []))
    acc_coco = acc(by_source.get("COCO", []))
    acc_omni = acc(by_source.get("Omni3D", []))
    acc_2d = (acc_ade + acc_coco) / 2
    acc_3d = acc_omni
    cv_bench = (acc_2d + acc_3d) / 2

    print(f"\n{'='*60}\n  CV-Bench Results: {label}\n{'='*60}")
    print(f"  CV-Bench Overall : {cv_bench:.2f}")
    print(f"  2D Accuracy      : {acc_2d:.2f}  (ADE={acc_ade:.2f}, COCO={acc_coco:.2f})")
    print(f"  3D Accuracy      : {acc_3d:.2f}  (Omni3D={acc_omni:.2f})")
    print(f"{'─'*60}")
    for t in sorted(by_task.keys()):
        print(f"    {t:20s}: {acc(by_task[t]):6.2f}  (n={len(by_task[t])})")
    print(f"  Unparsed: {n_unparsed}/{n_total}\n{'='*60}")

    return {"cv_bench": cv_bench, "acc_2d": acc_2d, "acc_3d": acc_3d,
            "acc_ade": acc_ade, "acc_coco": acc_coco, "acc_omni": acc_omni,
            "per_task": {t: acc(items) for t, items in by_task.items()},
            "n_unparsed": n_unparsed, "n_total": n_total}


def print_comparison(metrics_dict):
    labels = list(metrics_dict.keys())
    if len(labels) < 2: return
    print(f"\n{'='*70}\n  CV-Bench Comparison\n{'='*70}")
    header = f"  {'Metric':<25s}"
    for l in labels: header += f" {l:>12s}"
    if len(labels) >= 2: header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*60}")
    for name, key in [("CV-Bench Overall", "cv_bench"), ("2D Accuracy", "acc_2d"),
                      ("  ADE20K", "acc_ade"), ("  COCO", "acc_coco"),
                      ("3D Accuracy", "acc_3d"), ("  Omni3D", "acc_omni")]:
        line = f"  {name:<25s}"
        vals = [metrics_dict[l][key] for l in labels]
        for v in vals: line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)
    all_tasks = sorted(set(t for m in metrics_dict.values() for t in m["per_task"]))
    print(f"  {'─'*60}")
    for t in all_tasks:
        line = f"    {t:<23s}"
        vals = [metrics_dict[l]["per_task"].get(t, 0) for l in labels]
        for v in vals: line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)
    print(f"  {'─'*60}")
    line = f"  {'Unparsed':<25s}"
    for l in labels: line += f" {metrics_dict[l]['n_unparsed']:>12d}"
    print(line)
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "spatial", "no_pcs", "pcs_only",
                                 "finetune_baseline", "random_sae", "both", "all"])
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
    parser.add_argument("--subset", type=str, default=None, choices=["2D", "3D"])
    parser.add_argument("--save_dir", type=str, default="outputs/cvbench_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

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

    all_metrics = {}
    run_base       = args.mode in ["base", "both", "all"]
    run_ft_base    = args.mode in ["finetune_baseline", "all"]
    run_no_pcs     = args.mode in ["no_pcs", "all"]
    run_pcs_only   = args.mode in ["pcs_only", "all"]
    run_enhanced   = args.mode in ["enhanced", "both", "all"]
    run_spatial    = args.mode in ["spatial", "all"]
    run_random_sae = args.mode in ["random_sae", "all"]

    # 1. Base
    if run_base:
        m, p = load_base_model()
        res = evaluate_base(m, p, ds,
                            save_path=os.path.join(args.save_dir, "base_results.json"),
                            label="base (vanilla)")
        all_metrics["base"] = compute_metrics(res, label="Base Qwen2.5-VL")
        del m, p; torch.cuda.empty_cache()

    # 2. Finetune baseline
    if run_ft_base:
        ft_ckpt = args.baseline_ckpt or args.qwen_ckpt
        m, p = load_finetune_baseline(ft_ckpt)
        res = evaluate_base(m, p, ds,
                            save_path=os.path.join(args.save_dir, "ft_baseline_results.json"),
                            label="finetune_baseline")
        all_metrics["ft_baseline"] = compute_metrics(res, label="Finetune Baseline")
        del m, p; torch.cuda.empty_cache()

    # 3. No-PCS
    if run_no_pcs:
        m = load_enhanced_model_no_pcs(args)
        res = evaluate_enhanced(m, ds,
                                save_path=os.path.join(args.save_dir, "no_pcs_results.json"),
                                label="enhanced (no PCS)")
        all_metrics["no_pcs"] = compute_metrics(res, label="Enhanced (no PCS)")
        del m; torch.cuda.empty_cache()

    # 4. PCS-Only
    if run_pcs_only:
        m = load_pcs_only_model(args)
        res = evaluate_enhanced(m, ds,
                                save_path=os.path.join(args.save_dir, "pcs_only_results.json"),
                                label="pcs_only")
        all_metrics["pcs_only"] = compute_metrics(res, label="PCS Only")
        del m; torch.cuda.empty_cache()

    # 5. Enhanced (v1)
    if run_enhanced:
        m = load_enhanced_model(args)
        res = evaluate_enhanced(m, ds,
                                save_path=os.path.join(args.save_dir, "enhanced_results.json"),
                                label="enhanced (v1)")
        all_metrics["enhanced"] = compute_metrics(res, label="Enhanced (v1)")
        del m; torch.cuda.empty_cache()

    # 6. Spatial (v2)
    if run_spatial:
        m = load_spatial_model(args)
        res = evaluate_enhanced(m, ds,
                                save_path=os.path.join(args.save_dir, "spatial_results.json"),
                                label="spatial (v2)")
        all_metrics["spatial"] = compute_metrics(res, label="Spatial (v2)")
        del m; torch.cuda.empty_cache()

    # 7. Random SAE
    if run_random_sae:
        m = load_random_sae_model(args)
        res = evaluate_enhanced(m, ds,
                                save_path=os.path.join(args.save_dir, "random_sae_results.json"),
                                label="random_sae")
        all_metrics["random_sae"] = compute_metrics(res, label="Random SAE")
        del m; torch.cuda.empty_cache()

    if len(all_metrics) >= 2:
        print_comparison(all_metrics)

    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nDone. Summary saved to {args.save_dir}/summary.json")


if __name__ == "__main__":
    main()