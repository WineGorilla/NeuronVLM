"""
MMStar 评估脚本。

MMStar: 1500 道选择题，6 大能力维度，每题必须看图才能答对。
数据集字段：index, question, image, answer, category, l2_category

用法：
    python eval/eval_mmstar.py --mode base
    python eval/eval_mmstar.py --mode enhanced
    python eval/eval_mmstar.py --mode both
    python eval/eval_mmstar.py --mode all --qwen_ckpt outputs/ablation_baseline/qwen_best.pt
    python eval/eval_mmstar.py --mode both --max_samples 100
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

def build_prompt(question: str) -> str:
    """
    MMStar 的 question 字段已经包含了选项（A. xxx\nB. xxx\n...），
    只需要加上指令即可。
    """
    instruction = (
        "Answer the following question.\n"
        "Select the correct option and output ONLY the letter (A, B, C, or D).\n"
        "Do NOT output explanation.\n\n"
    )
    return f"{instruction}{question}\n\nAnswer:"


# ── 答案匹配 ──────────────────────────────────────────────────────────────────

def extract_choice_letter(response: str) -> str | None:
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None
    # 直接匹配单个字母
    m = re.match(r'^([A-D])(?:[\s.,):]|$)', text)
    if m:
        return m.group(1)
    # 匹配 (A) 格式
    m = re.search(r'\(([A-D])\)', text)
    if m:
        return m.group(1)
    # 匹配 "Answer is A" 格式
    m = re.search(r'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?([A-D])\)?', text)
    if m:
        return m.group(1)
    # 回退：找到第一个 A-D 字母
    m = re.search(r'([A-D])', text)
    if m:
        return m.group(1)
    return None


def match_answer(response: str, answer: str) -> tuple[bool, str | None]:
    """MMStar 的 answer 是单个字母如 'A', 'B', 'C', 'D'"""
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
        CFG.model_id,
        torch_dtype=torch.float16,
        device_map=CFG.device,
    )
    base_model.eval()
    return base_model, processor


def load_finetune_baseline(qwen_ckpt):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print("Loading finetune baseline...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id,
        torch_dtype=torch.float16,
        device_map=CFG.device,
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
        print("  Enhanced model weights loaded.")
    model.to(CFG.device)
    model.eval()
    return model


def load_enhanced_model_no_pcs(args):
    from ablation.Model_no_pcs import QwenWithClusterPredictorAndSAE

    print("Loading enhanced model (NO PCS)...")
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
    from qwen_vl_utils import process_vision_info

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        image = item["image"]
        question = item["question"]
        answer = item["answer"]
        category = item["category"]
        l2_category = item["l2_category"]

        prompt_text = build_prompt(question)

        tmp_path = "/tmp/mmstar_tmp.png"
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
                    max_new_tokens=32,
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [error] idx={item['index']}: {e}")

        correct, extracted = match_answer(response, answer)
        results.append({
            "index": item["index"],
            "category": category,
            "l2_category": l2_category,
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
        image = item["image"]
        question = item["question"]
        answer = item["answer"]
        category = item["category"]
        l2_category = item["l2_category"]

        prompt_text = build_prompt(question)

        tmp_path = "/tmp/mmstar_tmp.png"
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
            print(f"  [error] idx={item['index']}: {e}")

        correct, extracted = match_answer(response, answer)
        results.append({
            "index": item["index"],
            "category": category,
            "l2_category": l2_category,
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
    by_category    = defaultdict(list)
    by_l2_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)
        by_l2_category[r["l2_category"]].append(r)

    def acc(items):
        if not items:
            return 0.0
        return sum(1 for i in items if i["correct"]) / len(items) * 100

    n_unparsed = sum(1 for r in results if r.get("extracted") is None)
    n_total = len(results)
    overall_acc = acc(results)

    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) responses could not be parsed")
        samples = [r for r in results if r.get("extracted") is None][:3]
        for s in samples:
            print(f"    example: cat={s['category']} ans={s['answer']}")
            print(f"      resp=\"{s['response'][:100]}\"")

    print(f"\n{'='*60}")
    print(f"  MMStar Results: {label}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy : {overall_acc:.2f}  (n={n_total})")
    print(f"{'─'*60}")
    print(f"  Per Category (L1):")
    cat_accs = {}
    for cat_name in sorted(by_category.keys()):
        items = by_category[cat_name]
        a = acc(items)
        cat_accs[cat_name] = a
        print(f"    {cat_name:30s}: {a:6.2f}  (n={len(items)})")

    print(f"{'─'*60}")
    print(f"  Per Category (L2):")
    l2_accs = {}
    for cat_name in sorted(by_l2_category.keys()):
        items = by_l2_category[cat_name]
        a = acc(items)
        l2_accs[cat_name] = a
        print(f"    {cat_name:30s}: {a:6.2f}  (n={len(items)})")

    print(f"{'─'*60}")
    print(f"  Unparsed: {n_unparsed}/{n_total}")
    print(f"{'='*60}")

    return {
        "overall": overall_acc,
        "per_category": cat_accs,
        "per_l2_category": l2_accs,
        "n_unparsed": n_unparsed,
        "n_total": n_total,
    }


def print_comparison(metrics_dict):
    labels = list(metrics_dict.keys())
    if len(labels) < 2:
        return

    print(f"\n{'='*70}")
    print(f"  MMStar Comparison")
    print(f"{'='*70}")

    header = f"  {'Metric':<35s}"
    for label in labels:
        header += f" {label:>12s}"
    header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*65}")

    # Overall
    line = f"  {'Overall':35s}"
    vals = [metrics_dict[l]["overall"] for l in labels]
    for v in vals:
        line += f" {v:12.2f}"
    d = vals[-1] - vals[0]
    line += f" {'+'if d>0 else ''}{d:7.2f}"
    print(line)

    # L1 categories
    print(f"  {'─'*65}")
    all_cats = sorted(set(
        c for m in metrics_dict.values() for c in m["per_category"]
    ))
    for cat in all_cats:
        line = f"    {cat:33s}"
        vals = [metrics_dict[l]["per_category"].get(cat, 0) for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        d = vals[-1] - vals[0]
        line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    # Unparsed
    print(f"  {'─'*65}")
    line = f"  {'Unparsed':35s}"
    for l in labels:
        line += f" {metrics_dict[l]['n_unparsed']:>12d}"
    print(line)
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "no_pcs",
                                 "finetune_baseline", "both", "all"])
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_layer20/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--no_pcs_predictor_ckpt", type=str,
                        default="outputs/ablation_no_pcs/predictor_best.pt")
    parser.add_argument("--no_pcs_qwen_ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/mmstar_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载数据集 ──
    print("Loading MMStar...")
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    cat_counts = defaultdict(int)
    for item in ds:
        cat_counts[item["category"]] += 1
    for c, n in sorted(cat_counts.items()):
        print(f"    {c}: {n}")

    # ── 评估 ──
    all_metrics = {}

    run_base     = args.mode in ["base", "both", "all"]
    run_ft_base  = args.mode in ["finetune_baseline", "all"]
    run_no_pcs   = args.mode in ["no_pcs", "all"]
    run_enhanced = args.mode in ["enhanced", "both", "all"]

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

    if run_ft_base:
        ft_ckpt = args.baseline_ckpt or args.qwen_ckpt
        ft_model, ft_processor = load_finetune_baseline(ft_ckpt)
        res = evaluate_base(
            ft_model, ft_processor, ds,
            save_path=os.path.join(args.save_dir, "finetune_baseline_results.json"),
            label="finetune_baseline",
        )
        all_metrics["finetune_baseline"] = compute_metrics(res, label="Finetune Baseline")
        del ft_model, ft_processor
        torch.cuda.empty_cache()

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

    if run_enhanced:
        enhanced_model = load_enhanced_model(args)
        res = evaluate_enhanced(
            enhanced_model, ds,
            save_path=os.path.join(args.save_dir, "enhanced_results.json"),
            label="enhanced",
        )
        all_metrics["enhanced"] = compute_metrics(res, label="Enhanced")
        del enhanced_model
        torch.cuda.empty_cache()

    # ── 对比 ──
    if len(all_metrics) >= 2:
        print_comparison(all_metrics)

    # ── 保存 ──
    with open(os.path.join(args.save_dir, "summary_layer20.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nDone. Summary saved to {args.save_dir}/summary.json")


if __name__ == "__main__":
    main()