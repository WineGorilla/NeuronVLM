"""
BLINK 评估脚本。

用法：
    python eval/eval_blink.py --mode base
    python eval/eval_blink.py --mode enhanced
    python eval/eval_blink.py --mode random_sae
    python eval/eval_blink.py --mode pcs_only
    CUDA_VISIBLE_DEVICES=0 python eval/eval_blink_new.py --mode no_pcs
    python eval/eval_blink.py --mode spatial
    python eval/eval_blink.py --mode finetune_baseline --qwen_ckpt outputs/ablation_baseline/qwen_best.pt
    CUDA_VISIBLE_DEVICES=1 python eval/eval_blink.py --mode both
    python eval/eval_blink.py --mode all --qwen_ckpt outputs/ablation_baseline/qwen_best.pt
    python eval/eval_blink.py --mode both --depth_spatial_only
    python eval/eval_blink.py --mode both --max_samples 50
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

ALL_SUBTASKS = [
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


# ── Prompt ────────────────────────────────────────────────────────────────────

def build_blink_prompt(prompt, choices, has_image_choices=False):
    if has_image_choices:
        choice_text = "\n".join([f"{chr(ord('A')+i)}. (Image {i+1})" for i in range(len(choices))])
    else:
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
    if not text:
        return None
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
        if isinstance(choice, str) and resp == choice.strip().lower():
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


# ── 图片工具 ──────────────────────────────────────────────────────────────────

def save_images_for_item(item, tmp_dir="/tmp/blink_eval"):
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    for key in ["image_1", "image_2", "image_3", "image_4"]:
        img = item.get(key)
        if img is not None and isinstance(img, Image.Image):
            p = os.path.join(tmp_dir, f"{key}.png")
            img.save(p)
            paths.append(p)
    return paths

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
    out = "/tmp/blink_eval/concat.png"
    concat.save(out)
    return out


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
    """通用的增强模型加载逻辑。"""
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
        subtask, prompt, answer, choices = item["subtask"], item["prompt"], item["answer"], item["choices"]
        has_image_choices = isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], Image.Image)
        prompt_text = build_blink_prompt(prompt, choices, has_image_choices)

        image_paths = save_images_for_item(item)
        if not image_paths:
            results.append({"idx": item.get("idx", len(results)), "subtask": subtask,
                            "answer": answer, "response": "", "extracted": None, "correct": False})
            continue

        concat_path = concat_images_horizontal(image_paths)
        response = ""
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{concat_path}"},
                {"type": "text", "text": prompt_text},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(base_model.device)
            with torch.no_grad():
                output_ids = base_model.generate(**inputs, max_new_tokens=32, do_sample=False)
            response = processor.decode(output_ids[0, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  [error] idx={item.get('idx', '?')}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({"idx": item.get("idx", len(results)), "subtask": subtask,
                        "answer": answer, "response": response, "extracted": extracted, "correct": correct})

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")
    return results


def evaluate_enhanced(model, dataset, save_path=None, label="enhanced"):
    print(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        subtask, prompt, answer, choices = item["subtask"], item["prompt"], item["answer"], item["choices"]
        has_image_choices = isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], Image.Image)
        prompt_text = build_blink_prompt(prompt, choices, has_image_choices)

        image_paths = save_images_for_item(item)
        if not image_paths:
            results.append({"idx": item.get("idx", len(results)), "subtask": subtask,
                            "answer": answer, "response": "", "extracted": None,
                            "correct": False, "cluster_ids": []})
            continue

        concat_path = concat_images_horizontal(image_paths)
        response, cluster_ids = "", []
        try:
            result = model.generate(image_path=concat_path, question=prompt_text, verbose=False)
            response = result["final_answer"]
            cluster_ids = result.get("cluster_ids", [])
        except Exception as e:
            print(f"  [error] idx={item.get('idx', '?')}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({"idx": item.get("idx", len(results)), "subtask": subtask,
                        "answer": answer, "response": response, "extracted": extracted,
                        "correct": correct, "cluster_ids": cluster_ids})

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")
    return results


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_blink_dataset(subtasks, split="val", max_samples=None):
    all_items = []
    for subtask in subtasks:
        print(f"  Loading {subtask}...")
        try:
            ds = load_dataset("BLINK-Benchmark/BLINK", subtask, split=split)
            for i, item in enumerate(ds):
                unified = {
                    "idx": f"{subtask}_{i}", "subtask": subtask,
                    "prompt": item.get("prompt", item.get("question", "")),
                    "answer": item.get("answer", ""), "choices": [],
                    "image_1": item.get("image_1", item.get("image", None)),
                    "image_2": item.get("image_2"), "image_3": item.get("image_3"),
                    "image_4": item.get("image_4"),
                }
                if "choices" in item and item["choices"] is not None:
                    unified["choices"] = item["choices"]
                elif "options" in item and item["options"] is not None:
                    unified["choices"] = item["options"]
                if isinstance(unified["choices"], str):
                    try: unified["choices"] = json.loads(unified["choices"])
                    except json.JSONDecodeError:
                        unified["choices"] = [c.strip() for c in unified["choices"].split(",")]
                all_items.append(unified)
            print(f"    {subtask}: {len(ds)} samples loaded")
        except Exception as e:
            print(f"    [error] Failed to load {subtask}: {e}")

    if max_samples and len(all_items) > max_samples:
        per_task = max(1, max_samples // len(subtasks))
        by_task = defaultdict(list)
        for item in all_items:
            by_task[item["subtask"]].append(item)
        sampled = []
        for task_items in by_task.values():
            sampled.extend(task_items[:per_task])
        all_items = sampled[:max_samples]

    print(f"  Total: {len(all_items)} samples")
    return all_items


# ── 统计 ──────────────────────────────────────────────────────────────────────

def compute_metrics(results, label=""):
    by_subtask = defaultdict(list)
    for r in results:
        by_subtask[r["subtask"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0.0

    n_unparsed = sum(1 for r in results if r.get("extracted") is None)
    n_total = len(results)
    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) unparsed")
        for s in [r for r in results if r.get("extracted") is None][:3]:
            print(f"    subtask={s['subtask']} ans={s['answer']} resp=\"{s['response'][:80]}\"")

    per_subtask = {t: acc(by_subtask[t]) for t in sorted(by_subtask.keys())}
    overall = sum(per_subtask.values()) / len(per_subtask) if per_subtask else 0.0

    ds_items = [r for t in DEPTH_SPATIAL_SUBTASKS for r in by_subtask.get(t, [])]
    other_items = [r for t in by_subtask if t not in DEPTH_SPATIAL_SUBTASKS for r in by_subtask[t]]

    print(f"\n{'='*60}\n  BLINK Results: {label}\n{'='*60}")
    print(f"  Overall (macro)       : {overall:.2f}")
    print(f"  Depth/Spatial         : {acc(ds_items):.2f}  (n={len(ds_items)})")
    print(f"  Other Tasks           : {acc(other_items):.2f}  (n={len(other_items)})")
    print(f"{'─'*60}")
    for t in sorted(by_subtask.keys()):
        marker = " *" if t in DEPTH_SPATIAL_SUBTASKS else ""
        print(f"    {t:30s}: {acc(by_subtask[t]):6.2f}  (n={len(by_subtask[t])}){marker}")
    print(f"  Unparsed: {n_unparsed}/{n_total}\n{'='*60}")

    return {"overall": overall, "acc_depth_spatial": acc(ds_items),
            "acc_other": acc(other_items), "per_subtask": per_subtask,
            "n_unparsed": n_unparsed, "n_total": n_total}


def print_comparison(metrics_dict):
    labels = list(metrics_dict.keys())
    if len(labels) < 2:
        return
    print(f"\n{'='*70}\n  BLINK Comparison\n{'='*70}")
    header = f"  {'Metric':<32s}"
    for l in labels: header += f" {l:>12s}"
    if len(labels) >= 2: header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*65}")
    for name, key in [("BLINK Overall", "overall"), ("Depth/Spatial", "acc_depth_spatial"),
                      ("Other Tasks", "acc_other")]:
        line = f"  {name:<32s}"
        vals = [metrics_dict[l][key] for l in labels]
        for v in vals: line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)
    all_subs = sorted(set(t for m in metrics_dict.values() for t in m["per_subtask"]))
    print(f"  {'─'*65}")
    for t in all_subs:
        marker = "*" if t in DEPTH_SPATIAL_SUBTASKS else " "
        line = f"  {marker} {t:<30s}"
        vals = [metrics_dict[l]["per_subtask"].get(t, 0) for l in labels]
        for v in vals: line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)
    print(f"  {'─'*65}")
    line = f"  {'Unparsed':<32s}"
    for l in labels: line += f" {metrics_dict[l]['n_unparsed']:>12d}"
    print(line)
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BLINK Benchmark Evaluation")
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
    parser.add_argument("--subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--save_dir", type=str, default="outputs/blink_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.subtasks:
        subtasks = args.subtasks
    elif args.depth_spatial_only:
        subtasks = DEPTH_SPATIAL_SUBTASKS
    else:
        subtasks = ALL_SUBTASKS

    print(f"Subtasks ({len(subtasks)}):")
    for t in subtasks:
        print(f"  - {t}{' (depth/spatial)' if t in DEPTH_SPATIAL_SUBTASKS else ''}")

    print("\nLoading BLINK...")
    dataset = load_blink_dataset(subtasks, args.split, args.max_samples)

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
        res = evaluate_base(m, p, dataset,
                            save_path=os.path.join(args.save_dir, "base_results.json"),
                            label="base (vanilla)")
        all_metrics["base"] = compute_metrics(res, label="Base Qwen2.5-VL")
        del m, p; torch.cuda.empty_cache()

    # 2. Finetune baseline
    if run_ft_base:
        ft_ckpt = args.baseline_ckpt or args.qwen_ckpt
        m, p = load_finetune_baseline(ft_ckpt)
        res = evaluate_base(m, p, dataset,
                            save_path=os.path.join(args.save_dir, "ft_baseline_results.json"),
                            label="finetune_baseline")
        all_metrics["ft_baseline"] = compute_metrics(res, label="Finetune Baseline")
        del m, p; torch.cuda.empty_cache()

    # 3. No-PCS
    if run_no_pcs:
        m = load_enhanced_model_no_pcs(args)
        res = evaluate_enhanced(m, dataset,
                                save_path=os.path.join(args.save_dir, "no_pcs_results.json"),
                                label="enhanced (no PCS)")
        all_metrics["no_pcs"] = compute_metrics(res, label="Enhanced (no PCS)")
        del m; torch.cuda.empty_cache()

    # 4. PCS-Only
    if run_pcs_only:
        m = load_pcs_only_model(args)
        res = evaluate_enhanced(m, dataset,
                                save_path=os.path.join(args.save_dir, "pcs_only_results.json"),
                                label="pcs_only")
        all_metrics["pcs_only"] = compute_metrics(res, label="PCS Only")
        del m; torch.cuda.empty_cache()

    # 5. Enhanced (v1)
    if run_enhanced:
        m = load_enhanced_model(args)
        res = evaluate_enhanced(m, dataset,
                                save_path=os.path.join(args.save_dir, "enhanced_results.json"),
                                label="enhanced (v1)")
        all_metrics["enhanced"] = compute_metrics(res, label="Enhanced (v1)")
        del m; torch.cuda.empty_cache()

    # 6. Spatial (v2)
    if run_spatial:
        m = load_spatial_model(args)
        res = evaluate_enhanced(m, dataset,
                                save_path=os.path.join(args.save_dir, "spatial_results.json"),
                                label="spatial (v2)")
        all_metrics["spatial"] = compute_metrics(res, label="Spatial (v2)")
        del m; torch.cuda.empty_cache()

    # 7. Random SAE
    if run_random_sae:
        m = load_random_sae_model(args)
        res = evaluate_enhanced(m, dataset,
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