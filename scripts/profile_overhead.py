"""
开销分析脚本：测量 Base / +SAE+Predictor / +SAE+Predictor+PCS 的延迟、内存、参数量。

使用 CV-Bench 数据进行 profiling。

用法：
    CUDA_VISIBLE_DEVICES=0 python scripts/profile_overhead.py
    python eval/profile_overhead.py --n_samples 20
    python eval/profile_overhead.py --n_samples 50 --warmup 5
    python eval/profile_overhead.py --layer 8 --predictor_ckpt outputs/qwen_layer8_old/focus_ckpt_0.75_64_5000/predictor_best.pt

输出：
    终端打印对比表
    outputs/profile_results/overhead_report.json
"""
import os
import sys
import json
import time
import argparse
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import CFG


# ══════════════════════════════════════════════════════════════════════════════
# GPU 工具
# ══════════════════════════════════════════════════════════════════════════════

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def gpu_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0

def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载 (CV-Bench)
# ══════════════════════════════════════════════════════════════════════════════

def load_test_data(n_samples=20):
    print(f"Loading CV-Bench (first {n_samples} samples for profiling)...")
    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))
    print(f"  Loaded {len(ds)} samples")
    return ds


def save_image(item):
    tmp_path = "/tmp/profile_tmp.png"
    img = item["image"]
    if isinstance(img, Image.Image):
        img.save(tmp_path)
    else:
        tmp_path = img
    return tmp_path


def build_prompt(item):
    choices = item["choices"]
    choice_text = "\n".join(
        [f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)]
    )
    return (
        "Answer the following question.\n"
        "Select the correct option and output ONLY the letter.\n"
        "Do NOT output explanation.\n\n"
        f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Profile: Base (vanilla Qwen2.5-VL)
# ══════════════════════════════════════════════════════════════════════════════

def profile_base(ds, warmup=3):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"\n{'='*60}")
    print(f"  Profiling: Base (vanilla Qwen2.5-VL)")
    print(f"{'='*60}")

    clean_gpu()
    reset_peak()

    mem_before = gpu_mem_mb()
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16, device_map=CFG.device,
    )
    model.eval()
    mem_after = gpu_mem_mb()
    model_mem = mem_after - mem_before
    print(f"  Model memory: {model_mem:.1f} MB")

    def run_one(item):
        tmp_path = save_image(item)
        prompt = build_prompt(item)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{tmp_path}"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(
            text=[text], images=img_in, videos=vid_in,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=32, do_sample=False)

    # Warmup
    print(f"  Warming up ({warmup} samples)...")
    for i in range(min(warmup, len(ds))):
        run_one(ds[i])

    # Profile
    reset_peak()
    latencies = []
    for item in tqdm(ds, desc="  Base"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one(item)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    peak_mem = gpu_peak_mb()
    result = _build_result(model_mem, peak_mem, latencies)
    print(f"  Avg: {result['avg_latency_s']:.4f}s  Peak: {result['peak_memory_mb']:.1f} MB")

    del model, processor
    clean_gpu()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Profile: Enhanced (SAE + Predictor, 无 PCS)
# ══════════════════════════════════════════════════════════════════════════════

def profile_enhanced_no_pcs(ds, args, warmup=3):
    from src.Model import QwenWithClusterPredictorAndSAE

    print(f"\n{'='*60}")
    print(f"  Profiling: Enhanced (SAE + Predictor, NO PCS)")
    print(f"{'='*60}")

    clean_gpu()
    reset_peak()

    mem_before = gpu_mem_mb()
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
        print("  Qwen Stage 2 weights loaded.")
    model.to(CFG.device)
    model.eval()

    # 禁用 PCS
    with torch.no_grad():
        model.pc_suppressor.alpha_param.fill_(-100.0)
    print("  PCS disabled (alpha → 0)")

    mem_after = gpu_mem_mb()
    model_mem = mem_after - mem_before
    print(f"  Model memory: {model_mem:.1f} MB")

    def run_one(item):
        tmp_path = save_image(item)
        prompt = build_prompt(item)
        model.generate(image_path=tmp_path, question=prompt,
                       max_new_tokens=32, verbose=False)

    # Warmup
    print(f"  Warming up ({warmup} samples)...")
    for i in range(min(warmup, len(ds))):
        run_one(ds[i])

    # Profile
    reset_peak()
    latencies = []
    for item in tqdm(ds, desc="  Enhanced (no PCS)"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one(item)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    peak_mem = gpu_peak_mb()
    result = _build_result(model_mem, peak_mem, latencies)
    print(f"  Avg: {result['avg_latency_s']:.4f}s  Peak: {result['peak_memory_mb']:.1f} MB")

    del model
    clean_gpu()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Profile: Enhanced (SAE + Predictor + PCS)
# ══════════════════════════════════════════════════════════════════════════════

def profile_enhanced_with_pcs(ds, args, warmup=3):
    from src.Model import QwenWithClusterPredictorAndSAE

    print(f"\n{'='*60}")
    print(f"  Profiling: Enhanced (SAE + Predictor + PCS)")
    print(f"{'='*60}")

    clean_gpu()
    reset_peak()

    mem_before = gpu_mem_mb()
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
        print("  Qwen Stage 2 weights loaded.")
    model.to(CFG.device)
    model.eval()
    print(f"  PCS enabled (alpha_param={model.pc_suppressor.alpha_param.item():.4f})")

    mem_after = gpu_mem_mb()
    model_mem = mem_after - mem_before
    print(f"  Model memory: {model_mem:.1f} MB")

    def run_one(item):
        tmp_path = save_image(item)
        prompt = build_prompt(item)
        model.generate(image_path=tmp_path, question=prompt,
                       max_new_tokens=32, verbose=False)

    # Warmup
    print(f"  Warming up ({warmup} samples)...")
    for i in range(min(warmup, len(ds))):
        run_one(ds[i])

    # Profile
    reset_peak()
    latencies = []
    for item in tqdm(ds, desc="  Enhanced (with PCS)"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one(item)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    peak_mem = gpu_peak_mb()
    result = _build_result(model_mem, peak_mem, latencies)
    print(f"  Avg: {result['avg_latency_s']:.4f}s  Peak: {result['peak_memory_mb']:.1f} MB")

    del model
    clean_gpu()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 参数量统计
# ══════════════════════════════════════════════════════════════════════════════

def count_params(args):
    """统计各模块参数量。"""
    from src.Model import (
        ClusterPredictor, ImageClusterScorer, ExtraProjector,
        SemanticCrossAttention, PrincipalComponentSuppressor,
        SemanticCompleter,
    )
    from src.SAE import SAE

    # 获取 dim
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(CFG.model_id)
    dim = config.text_config.hidden_size
    latent_dim = dim * CFG.latent_mult

    # 获取 n_clusters
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    with open(cluster_path) as f:
        cluster_info = json.load(f)
    n_clusters = len(cluster_info["clusters"])

    modules = {
        "SAE":                          SAE(dim, latent_dim, CFG.topk),
        "ClusterPredictor":             ClusterPredictor(dim, n_clusters),
        "ImageClusterScorer":           ImageClusterScorer(dim, n_clusters),
        "ExtraProjector":               ExtraProjector(dim),
        "SemanticCrossAttention":       SemanticCrossAttention(dim),
        "SemanticCompleter":            SemanticCompleter(dim, latent_dim),
        "PCS (PrincipalComponentSupp.)": PrincipalComponentSuppressor(),
    }

    counts = {}
    for name, mod in modules.items():
        counts[name] = sum(p.numel() for p in mod.parameters())

    # 汇总
    sae_predictor_keys = [
        "SAE", "ClusterPredictor", "ImageClusterScorer",
        "ExtraProjector", "SemanticCrossAttention", "SemanticCompleter",
    ]
    counts["Total (SAE+Predictor)"] = sum(counts[k] for k in sae_predictor_keys)
    counts["Total (SAE+Predictor+PCS)"] = (
        counts["Total (SAE+Predictor)"]
        + counts["PCS (PrincipalComponentSupp.)"]
    )

    # SAE 是否可训练
    trainable_info = {
        "SAE":                          "No (frozen)",
        "ClusterPredictor":             "Yes",
        "ImageClusterScorer":           "Yes",
        "ExtraProjector":               "Yes",
        "SemanticCrossAttention":       "Yes",
        "SemanticCompleter":            "Yes",
        "PCS (PrincipalComponentSupp.)": "Yes",
        "Total (SAE+Predictor)":        "",
        "Total (SAE+Predictor+PCS)":    "",
    }

    return counts, trainable_info


# ══════════════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════════════

def _build_result(model_mem, peak_mem, latencies):
    import math
    lat = sorted(latencies)
    n = len(lat)
    avg = sum(lat) / n
    variance = sum((x - avg) ** 2 for x in lat) / n
    std = math.sqrt(variance)
    return {
        "model_memory_mb": round(model_mem, 1),
        "peak_memory_mb":  round(peak_mem, 1),
        "avg_latency_s":   round(avg, 4),
        "std_latency_s":   round(std, 4),
        "median_latency_s": round(lat[n // 2], 4),
        "min_latency_s":   round(lat[0], 4),
        "max_latency_s":   round(lat[-1], 4),
        "n_samples":       n,
    }


def fmt_params(n):
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def delta_str(val, unit=""):
    """格式化 delta 值，正数加 +。"""
    if val > 0:
        return f"+{val}{unit}"
    return f"{val}{unit}"


def pct_str(enhanced_val, base_val):
    """计算并格式化百分比变化。"""
    if base_val == 0:
        return "N/A"
    pct = (enhanced_val - base_val) / base_val * 100
    return f"{'+'if pct>0 else ''}{pct:.2f}%"


# ══════════════════════════════════════════════════════════════════════════════
# 打印报告
# ══════════════════════════════════════════════════════════════════════════════

def print_report(base_r, no_pcs_r, pcs_r, param_counts, trainable_info, n_samples):

    W = 90  # 表宽

    print(f"\n{'═'*W}")
    print(f"  Overhead Profiling Report ({n_samples} samples, CV-Bench)")
    print(f"{'═'*W}")

    # ── 表 1: 绝对值 ──────────────────────────────────────────────────────────
    print(f"\n  ┌─ Latency & Memory {'─'*(W-22)}┐")
    header = (
        f"  │ {'Configuration':<26s}│ {'Avg (s)':>9s} │ {'Std':>7s} │ {'Median':>8s} │"
        f" {'Min':>7s} │ {'Max':>7s} │ {'Model (MB)':>11s} │ {'Peak (MB)':>11s} │"
    )
    print(header)
    print(f"  ├{'─'*27}┼{'─'*11}┼{'─'*9}┼{'─'*10}┼{'─'*9}┼{'─'*9}┼{'─'*13}┼{'─'*13}┤")

    for label, r in [
        ("Base (Qwen2.5-VL)",      base_r),
        ("+ SAE + Predictor",      no_pcs_r),
        ("+ SAE + Predictor + PCS", pcs_r),
    ]:
        print(
            f"  │ {label:<26s}│ {r['avg_latency_s']:>9.4f} │ {r['std_latency_s']:>7.4f} │ {r['median_latency_s']:>8.4f} │"
            f" {r['min_latency_s']:>7.4f} │ {r['max_latency_s']:>7.4f} │"
            f" {r['model_memory_mb']:>10.1f} │ {r['peak_memory_mb']:>10.1f}  │"
        )

    print(f"  └{'─'*27}┴{'─'*11}┴{'─'*9}┴{'─'*10}┴{'─'*9}┴{'─'*9}┴{'─'*13}┴{'─'*13}┘")

    # ── 表 2: Overhead ────────────────────────────────────────────────────────
    print(f"\n  ┌─ Overhead vs Base {'─'*(W-22)}┐")
    header2 = (
        f"  │ {'Configuration':<28s}│ {'Latency Δ (s)':>14s} │"
        f" {'Latency Δ (%)':>14s} │ {'Memory Δ (MB)':>14s} │ {'Memory Δ (%)':>14s} │"
    )
    print(header2)
    print(f"  ├{'─'*29}┼{'─'*16}┼{'─'*16}┼{'─'*16}┼{'─'*16}┤")

    for label, r in [
        ("+ SAE + Predictor",       no_pcs_r),
        ("+ SAE + Predictor + PCS", pcs_r),
    ]:
        lat_d = round(r["avg_latency_s"] - base_r["avg_latency_s"], 4)
        mem_d = round(r["peak_memory_mb"] - base_r["peak_memory_mb"], 1)
        lat_p = pct_str(r["avg_latency_s"], base_r["avg_latency_s"])
        mem_p = pct_str(r["peak_memory_mb"], base_r["peak_memory_mb"])
        print(
            f"  │ {label:<28s}│ {delta_str(lat_d, 's'):>14s} │"
            f" {lat_p:>14s} │ {delta_str(mem_d, ''):>13s}  │ {mem_p:>14s} │"
        )

    # PCS marginal
    pcs_lat_d = round(pcs_r["avg_latency_s"] - no_pcs_r["avg_latency_s"], 4)
    pcs_mem_d = round(pcs_r["peak_memory_mb"] - no_pcs_r["peak_memory_mb"], 1)
    pcs_lat_p = pct_str(pcs_r["avg_latency_s"], no_pcs_r["avg_latency_s"])
    pcs_mem_p = pct_str(pcs_r["peak_memory_mb"], no_pcs_r["peak_memory_mb"])
    print(f"  ├{'─'*29}┼{'─'*16}┼{'─'*16}┼{'─'*16}┼{'─'*16}┤")
    print(
        f"  │ {'PCS only (marginal)':<28s}│ {delta_str(pcs_lat_d, 's'):>14s} │"
        f" {pcs_lat_p:>14s} │ {delta_str(pcs_mem_d, ''):>13s}  │ {pcs_mem_p:>14s} │"
    )
    print(f"  └{'─'*29}┴{'─'*16}┴{'─'*16}┴{'─'*16}┴{'─'*16}┘")

    # ── 表 3: 参数量 ──────────────────────────────────────────────────────────
    BASE_PARAMS = 7_615_616_000  # Qwen2.5-VL-7B approx

    print(f"\n  ┌─ Module Parameters {'─'*(W-23)}┐")
    header3 = (
        f"  │ {'Module':<34s}│ {'Params':>12s} │ {'% of Base (7.6B)':>17s} │ {'Trainable':>10s} │"
    )
    print(header3)
    print(f"  ├{'─'*35}┼{'─'*14}┼{'─'*19}┼{'─'*12}┤")

    for name, count in param_counts.items():
        pct = f"{count / BASE_PARAMS * 100:.4f}%"
        train = trainable_info.get(name, "")
        if name.startswith("Total"):
            print(f"  ├{'─'*35}┼{'─'*14}┼{'─'*19}┼{'─'*12}┤")
            print(
                f"  │ {name:<34s}│ {fmt_params(count):>12s} │ {pct:>17s} │ {'':>10s} │"
            )
        else:
            print(
                f"  │ {name:<34s}│ {fmt_params(count):>12s} │ {pct:>17s} │ {train:>10s} │"
            )

    print(f"  └{'─'*35}┴{'─'*14}┴{'─'*19}┴{'─'*12}┘")
    print(f"{'═'*W}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Overhead Profiling: Base vs Enhanced vs +PCS")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of CV-Bench samples to profile")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup samples (excluded from timing)")
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/qwen_layer8_old/focus_ckpt_0.75_64_5000/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/profile_results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载数据 ──
    ds = load_test_data(args.n_samples)

    # ── 参数量统计（不需要 GPU） ──
    print("\nCounting module parameters...")
    param_counts, trainable_info = count_params(args)

    # ── Profile 三个配置 ──
    base_r    = profile_base(ds, warmup=args.warmup)
    no_pcs_r  = profile_enhanced_no_pcs(ds, args, warmup=args.warmup)
    pcs_r     = profile_enhanced_with_pcs(ds, args, warmup=args.warmup)

    # ── 打印报告 ──
    print_report(base_r, no_pcs_r, pcs_r, param_counts, trainable_info, args.n_samples)

    # ── 保存 JSON ──
    report = {
        "n_samples": args.n_samples,
        "layer": args.layer,
        "profiles": {
            "base": base_r,
            "enhanced_no_pcs": no_pcs_r,
            "enhanced_with_pcs": pcs_r,
        },
        "overhead_vs_base": {
            "sae_predictor": {
                "latency_delta_s": round(no_pcs_r["avg_latency_s"] - base_r["avg_latency_s"], 4),
                "latency_pct": round((no_pcs_r["avg_latency_s"] - base_r["avg_latency_s"]) / base_r["avg_latency_s"] * 100, 2),
                "memory_delta_mb": round(no_pcs_r["peak_memory_mb"] - base_r["peak_memory_mb"], 1),
                "memory_pct": round((no_pcs_r["peak_memory_mb"] - base_r["peak_memory_mb"]) / base_r["peak_memory_mb"] * 100, 2),
            },
            "sae_predictor_pcs": {
                "latency_delta_s": round(pcs_r["avg_latency_s"] - base_r["avg_latency_s"], 4),
                "latency_pct": round((pcs_r["avg_latency_s"] - base_r["avg_latency_s"]) / base_r["avg_latency_s"] * 100, 2),
                "memory_delta_mb": round(pcs_r["peak_memory_mb"] - base_r["peak_memory_mb"], 1),
                "memory_pct": round((pcs_r["peak_memory_mb"] - base_r["peak_memory_mb"]) / base_r["peak_memory_mb"] * 100, 2),
            },
            "pcs_marginal": {
                "latency_delta_s": round(pcs_r["avg_latency_s"] - no_pcs_r["avg_latency_s"], 4),
                "memory_delta_mb": round(pcs_r["peak_memory_mb"] - no_pcs_r["peak_memory_mb"], 1),
            },
        },
        "param_counts": param_counts,
    }
    json_path = os.path.join(args.save_dir, "overhead_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()