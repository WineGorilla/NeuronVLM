"""
评测脚本：对比原始 Qwen 和增强框架在测试集上的表现。

用法：
    python scripts/evaluate.py \
        --test_file data/test.jsonl \
        --predictor_ckpt outputs/focus_ckpt/predictor_best.pt \
        --qwen_ckpt outputs/focus_ckpt/qwen_best.pt \
        --max 200
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
from collections import defaultdict

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import CFG
from src.Model import QwenWithClusterPredictorAndSAE


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    return text.lower().split()


def bleu_n(pred: list, ref: list, n: int) -> float:
    if len(pred) < n:
        return 0.0
    pred_ngrams = [tuple(pred[i:i+n]) for i in range(len(pred)-n+1)]
    ref_ngrams  = [tuple(ref[i:i+n])  for i in range(len(ref)-n+1)]
    ref_set     = defaultdict(int)
    for g in ref_ngrams:
        ref_set[g] += 1
    match = 0
    for g in pred_ngrams:
        if ref_set[g] > 0:
            match += 1
            ref_set[g] -= 1
    return match / len(pred_ngrams) if pred_ngrams else 0.0


def rouge_l(pred: list, ref: list) -> float:
    m, n = len(pred), len(ref)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if pred[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs       = dp[m][n]
    precision = lcs / m
    recall    = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(preds: list, refs: list) -> dict:
    b1 = b4 = rl = 0.0
    n  = len(preds)
    for pred, ref in zip(preds, refs):
        p = tokenize(pred)
        r = tokenize(ref)
        b1 += bleu_n(p, r, 1)
        b4 += bleu_n(p, r, 4)
        rl += rouge_l(p, r)
    return {
        "BLEU-1":  round(b1 / n * 100, 2),
        "BLEU-4":  round(b4 / n * 100, 2),
        "ROUGE-L": round(rl / n * 100, 2),
        "count":   n,
    }


# ── 原始 Qwen 推理 ────────────────────────────────────────────────────────────

def run_base_qwen(samples: list, processor, model, max_tokens: int = 128) -> list:
    preds = []
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  [base] {i+1}/{len(samples)}")
        try:
            messages = [[{
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text",  "text":  sample["question"]},
                ],
            }]]
            texts = [
                processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=texts, images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(CFG.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            pred = processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
            preds.append(pred)
        except Exception as e:
            print(f"  [skip] {sample['image']}: {e}")
            preds.append("")
    return preds


# ── 增强框架推理 ──────────────────────────────────────────────────────────────

def run_enhanced(samples: list, model: QwenWithClusterPredictorAndSAE,
                 max_tokens: int = 128) -> list:
    preds = []
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  [enhanced] {i+1}/{len(samples)}")
        try:
            result = model.generate(
                image_path = sample["image"],
                question   = sample["question"],
                verbose    = False,
            )
            preds.append(result["final_answer"])
        except Exception as e:
            print(f"  [skip] {sample['image']}: {e}")
            preds.append("")
    return preds


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file",      type=str, default="data/test.jsonl")
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/focus_ckpt/predictor_best.pt")
    parser.add_argument("--qwen_ckpt",      type=str, default=None)
    parser.add_argument("--max",            type=int, default=500)
    parser.add_argument("--output",         type=str,
                        default="outputs/eval_results.json")
    parser.add_argument("--layer",          type=int, default=CFG.vis_layer)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 加载测试数据
    samples = []
    with open(args.test_file) as f:
        for line in f:
            samples.append(json.loads(line))
    samples = [s for s in samples if os.path.exists(s["image"])]
    if args.max:
        samples = samples[:args.max]
    print(f"Test samples: {len(samples)}")

    refs = [s["answer"] for s in samples]

    # ── 评测原始 Qwen ──────────────────────────────────────────────────────────
    print("\n[1/2] Evaluating base Qwen...")
    processor  = AutoProcessor.from_pretrained(CFG.model_id)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.bfloat16,
    ).to(CFG.device)
    base_model.eval()

    base_preds   = run_base_qwen(samples, processor, base_model)
    base_metrics = compute_metrics(base_preds, refs)
    print(f"  Base Qwen   : {base_metrics}")

    del base_model
    torch.cuda.empty_cache()

    # ── 评测增强框架 ───────────────────────────────────────────────────────────
    print("\n[2/2] Evaluating enhanced model...")
    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
    enhanced_model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id          = CFG.model_id,
        sae_ckpt_dir      = CFG.save_dir,
        cluster_path      = cluster_path,
        inject_layer      = args.layer,
        latent_mult       = CFG.latent_mult,
        topk              = CFG.topk,
        top_n_patches     = CFG.top_n_patches,
        predictor_ckpt    = args.predictor_ckpt,
        device            = CFG.device,
    )

    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        print(f"  Loading Qwen fine-tuned weights: {args.qwen_ckpt}")
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        enhanced_model.base_model.load_state_dict(state, strict=False)

    enhanced_preds   = run_enhanced(samples, enhanced_model)
    enhanced_metrics = compute_metrics(enhanced_preds, refs)
    print(f"  Enhanced    : {enhanced_metrics}")

    # ── 结果对比 ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"{'Metric':<12} {'Base Qwen':>12} {'Enhanced':>12} {'Delta':>10}")
    print(f"{'-'*50}")
    for key in ["BLEU-1", "BLEU-4", "ROUGE-L"]:
        base  = base_metrics[key]
        enh   = enhanced_metrics[key]
        delta = enh - base
        sign  = "+" if delta >= 0 else ""
        print(f"{key:<12} {base:>12.2f} {enh:>12.2f} {sign}{delta:>9.2f}")
    print(f"{'='*50}")

    # 保存详细结果
    results = {
        "base_metrics":     base_metrics,
        "enhanced_metrics": enhanced_metrics,
        "samples": [
            {
                "image":    s["image"],
                "question": s["question"],
                "ref":      s["answer"],
                "base":     base_preds[i],
                "enhanced": enhanced_preds[i],
            }
            for i, s in enumerate(samples)
        ]
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved: {args.output}")


if __name__ == "__main__":
    main()