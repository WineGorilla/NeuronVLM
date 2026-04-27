"""
Dense Cross-Attention Baseline 评估脚本。

用法：
    python eval/eval_dense_xattn.py --mode base
    python eval/eval_dense_xattn.py --mode enhanced --predictor_ckpt outputs/dense_xattn_ckpt/predictor_best.pt
    python eval/eval_dense_xattn.py --mode both --predictor_ckpt outputs/dense_xattn_ckpt/predictor_best.pt
    python eval/eval_dense_xattn.py --mode enhanced --predictor_ckpt outputs/dense_xattn_ckpt/predictor_best.pt --max_samples 50
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

try:
    from config import CFG
except ImportError:
    class CFG:
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        vis_layer = 8
        top_n_patches = 60
        device = "cuda"


# ── Prompt & 答案匹配 ────────────────────────────────────────────────────────

def build_prompt(prompt, choices):
    choice_text = "\n".join(
        [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
    )
    return (
        "Answer the following question.\n"
        "Select the correct option and output ONLY the letter.\n"
        "Do NOT output explanation.\n\n"
        f"{prompt}\n\nChoices:\n{choice_text}\n\nAnswer:"
    )


def extract_choice_letter(response):
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


def match_answer(response, choices, answer):
    extracted = extract_choice_letter(response)
    if extracted is not None:
        return extracted == answer, extracted
    resp = response.strip().split("\n")[0].strip().lower()
    for i, choice in enumerate(choices):
        if resp == str(choice).strip().lower():
            return f"({chr(ord('A') + i)})" == answer, f"({chr(ord('A') + i)})"
    return False, None


# ── 加载模型 ──────────────────────────────────────────────────────────────────

def load_enhanced_model(args):
    from Model_dense_xattn import QwenWithDenseCrossAttention

    print("Loading Dense Cross-Attention baseline...")
    model = QwenWithDenseCrossAttention.from_pretrained(
        model_id=args.model_id,
        inject_layer=args.layer,
        top_n_patches=args.top_n_patches,
        predictor_ckpt=args.predictor_ckpt,
        device=args.device,
    )
    model.to(args.device)
    model.eval()
    return model


# ── 评估 ──────────────────────────────────────────────────────────────────────

def evaluate_enhanced(model, dataset, save_path=None, label="dense_xattn"):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        image = item["image"]
        prompt_text = build_prompt(item["prompt"], item["choices"])
        answer = item["answer"]

        tmp_path = "/tmp/dense_xattn_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        response = ""
        try:
            result = model.generate(image_path=tmp_path, question=prompt_text, verbose=False)
            response = result["final_answer"]
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")

        correct, extracted = match_answer(response, item["choices"], answer)
        results.append({
            "idx": item["idx"], "task": item["task"],
            "source": item["source"], "type": item["type"],
            "answer": answer, "response": response,
            "extracted": extracted, "correct": correct,
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def compute_metrics(results, label=""):
    by_source = defaultdict(list)
    by_task = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)
        by_task[r["task"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    n_unparsed = sum(1 for r in results if r.get("extracted") is None)
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
    print(f"  2D Accuracy      : {acc_2d:.2f}  (ADE={acc_ade:.2f}, COCO={acc_coco:.2f})")
    print(f"  3D Accuracy      : {acc_3d:.2f}  (Omni3D={acc_omni:.2f})")
    print(f"  Per-Task:")
    for t in sorted(by_task.keys()):
        print(f"    {t:20s}: {acc(by_task[t]):6.2f}  (n={len(by_task[t])})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")
    print(f"{'='*60}")

    return {
        "cv_bench": cv_bench, "acc_2d": acc_2d, "acc_3d": acc_3d,
        "per_task": {t: acc(items) for t, items in by_task.items()},
        "n_unparsed": n_unparsed, "n_total": len(results),
    }


# ── BLINK 评估 ────────────────────────────────────────────────────────────────

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
        concat.paste(img, (x, 0)); x += img.width
    out = "/tmp/blink_eval_concat.png"
    concat.save(out)
    return out


def evaluate_blink(model, subtasks=None, max_samples=None, save_path=None):
    print(f"\n{'='*60}")
    print(f"  BLINK Evaluation (Dense XAttn)")
    print(f"{'='*60}")
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
                    except:
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
        has_image_choices = (isinstance(choices, list) and len(choices) > 0
                            and isinstance(choices[0], Image.Image))
        if has_image_choices:
            choice_text = "\n".join([f"{chr(ord('A')+i)}. (Image {i+1})" for i in range(len(choices))])
        else:
            choice_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
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
            results.append({"idx": item["idx"], "subtask": item["subtask"],
                            "answer": item["answer"], "response": "",
                            "extracted": None, "correct": False})
            continue
        concat_path = concat_images_horizontal(image_paths)
        try:
            result = model.generate(image_path=concat_path, question=prompt_text, verbose=False)
            response = result["final_answer"]
        except Exception as e:
            response = ""
        extracted = extract_choice_letter(response)
        answer = item["answer"]
        extracted_fmt = extracted if extracted else None
        # BLINK answer format: "(A)"
        correct = extracted_fmt == answer
        results.append({"idx": item["idx"], "subtask": item["subtask"],
                        "answer": answer, "response": response,
                        "extracted": extracted_fmt, "correct": correct})

    by_subtask = defaultdict(list)
    for r in results:
        by_subtask[r["subtask"]].append(r)
    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    per_subtask = {t: acc(by_subtask[t]) for t in sorted(by_subtask.keys())}
    overall = sum(per_subtask.values()) / len(per_subtask) if per_subtask else 0
    ds_items = [r for t in DEPTH_SPATIAL_SUBTASKS for r in by_subtask.get(t, [])]
    acc_ds = acc(ds_items)
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  BLINK Overall (macro) : {overall:.2f}")
    print(f"  Depth/Spatial         : {acc_ds:.2f}  (n={len(ds_items)})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return {"overall": overall, "acc_depth_spatial": acc_ds,
            "per_subtask": per_subtask, "n_unparsed": n_unparsed,
            "n_total": len(results)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="enhanced",
                        choices=["enhanced", "both"])
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["cvbench", "blink"],
                        choices=["cvbench", "blink", "mmstar", "realworldqa"])
    parser.add_argument("--model_id", type=str, default=CFG.model_id)
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--top_n_patches", type=int, default=CFG.top_n_patches)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/dense_xattn_ckpt/predictor_best.pt")
    parser.add_argument("--save_dir", type=str, default="outputs/dense_xattn_results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=CFG.device)
    parser.add_argument("--depth_spatial_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    all_metrics = {}

    # Dense Cross-Attention
    model = load_enhanced_model(args)

    if "cvbench" in args.benchmarks:
        print("\nLoading CV-Bench...")
        ds = load_dataset("nyu-visionx/CV-Bench", split="test")
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"  Samples: {len(ds)}")

        res = evaluate_enhanced(
            model, ds,
            save_path=os.path.join(args.save_dir, "cvbench_results.json"),
            label="Dense Cross-Attention",
        )
        all_metrics["cvbench"] = compute_metrics(res, label="Dense Cross-Attention")

    if "blink" in args.benchmarks:
        subtasks = DEPTH_SPATIAL_SUBTASKS if args.depth_spatial_only else None
        blink_metrics = evaluate_blink(
            model, subtasks=subtasks, max_samples=args.max_samples,
            save_path=os.path.join(args.save_dir, "blink_results.json"),
        )
        all_metrics["blink"] = blink_metrics

    # 保存汇总
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nDone. Results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()