"""
开销分析脚本：测量 Base / +SAE+Predictor / +SAE+Predictor+PCS 的延迟、内存、参数量。

使用 CV-Bench 数据进行 profiling。

修复要点 (vs 旧版):
  1a. processor 与 model 分开测量，base 与 enhanced 口径一致
  1b. 记录加载过程中的 peak（load_peak_mb），捕捉中间态双倍占用
  1c. 每个配置在独立子进程中跑，杜绝 CUDA context 交叉污染
  1d. 加载前做完整的 CUDA 状态重置（gc + empty_cache + sync + reset_peak）
  1e. model memory 测量拆分为:
       - model_mem_mb: 加载完成后模型常驻显存 (memory_allocated delta)
       - load_peak_mb: 加载过程中的峰值显存 (max_memory_allocated delta)
       - processor_mem_mb: processor 单独的显存占用

用法：
    CUDA_VISIBLE_DEVICES=0 python profile_overhead.py --layer 8 --n_samples 20
    python eval/profile_overhead.py --n_samples 20
    python eval/profile_overhead.py --n_samples 50 --warmup 5
    python eval/profile_overhead.py --layer 8 --predictor_ckpt outputs/qwen_layer8_old/focus_ckpt_0.75_64_5000/predictor_best.pt
    # 跳过某个配置 (调试用)
    python eval/profile_overhead.py --skip base

输出：
    终端打印对比表
    outputs/profile_results/overhead_report.json
"""

import os
import sys
import json
import time
import math
import argparse
import gc
import multiprocessing as mp
from typing import Dict, Any, Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import CFG


# ══════════════════════════════════════════════════════════════════════════════
#  GPU 工具（改进版）
# ══════════════════════════════════════════════════════════════════════════════

def _has_cuda() -> bool:
    return torch.cuda.is_available()


def gpu_mem_mb() -> float:
    """当前 GPU 已分配显存 (MB)。"""
    if _has_cuda():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def gpu_peak_mb() -> float:
    """自上次 reset 以来的峰值显存 (MB)。"""
    if _has_cuda():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def full_gpu_reset():
    """
    完整的 GPU 状态清理。
    在每次测量前调用，确保：
      - Python 侧无残留引用 (gc.collect)
      - PyTorch caching allocator 释放所有 cached block
      - CUDA 同步完成
      - peak 统计归零
    """
    gc.collect()
    if _has_cuda():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


# ══════════════════════════════════════════════════════════════════════════════
#  数据加载 (CV-Bench)
# ══════════════════════════════════════════════════════════════════════════════

def load_test_data(n_samples: int = 20):
    print(f"Loading CV-Bench (first {n_samples} samples for profiling)...")
    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))
    print(f"  Loaded {len(ds)} samples")
    return ds


def save_image(item) -> str:
    tmp_path = "/tmp/profile_tmp.png"
    img = item["image"]
    if isinstance(img, Image.Image):
        img.save(tmp_path)
    else:
        tmp_path = img
    return tmp_path


def build_prompt(item) -> str:
    choices = item["choices"]
    choice_text = "\n".join(
        [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
    )
    return (
        "Answer the following question.\n"
        "Select the correct option and output ONLY the letter.\n"
        "Do NOT output explanation.\n\n"
        f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  统计工具
# ══════════════════════════════════════════════════════════════════════════════

def _build_result(
    model_mem: float,
    load_peak: float,
    processor_mem: float,
    inference_peak: float,
    latencies: List[float],
) -> Dict[str, Any]:
    """
    构建单个配置的 profiling 结果。

    Args:
        model_mem:       加载完成后模型常驻显存 (MB)
        load_peak:       加载过程中峰值显存 (MB)，包含临时分配
        processor_mem:   processor 占用的显存 (MB)
        inference_peak:  推理阶段峰值显存 (MB)
        latencies:       每个样本的延迟列表 (秒)
    """
    lat = sorted(latencies)
    n = len(lat)
    avg = sum(lat) / n
    variance = sum((x - avg) ** 2 for x in lat) / n
    std = math.sqrt(variance)
    return {
        # ── 显存指标 ──
        "model_memory_mb":     round(model_mem, 1),
        "load_peak_mb":        round(load_peak, 1),
        "processor_memory_mb": round(processor_mem, 1),
        "peak_memory_mb":      round(inference_peak, 1),
        # ── 延迟指标 ──
        "avg_latency_s":       round(avg, 4),
        "std_latency_s":       round(std, 4),
        "median_latency_s":    round(lat[n // 2], 4),
        "min_latency_s":       round(lat[0], 4),
        "max_latency_s":       round(lat[-1], 4),
        "n_samples":           n,
    }


def fmt_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def delta_str(val, unit: str = "") -> str:
    if val > 0:
        return f"+{val}{unit}"
    return f"{val}{unit}"


def pct_str(enhanced_val: float, base_val: float) -> str:
    if base_val == 0:
        return "N/A"
    pct = (enhanced_val - base_val) / base_val * 100
    return f"{'+'if pct > 0 else ''}{pct:.2f}%"


# ══════════════════════════════════════════════════════════════════════════════
#  Profile: Base (vanilla Qwen2.5-VL)
# ══════════════════════════════════════════════════════════════════════════════

def profile_base(ds, warmup: int = 3) -> Dict[str, Any]:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"\n{'=' * 60}")
    print(f"  Profiling: Base (vanilla Qwen2.5-VL)")
    print(f"{'=' * 60}")

    # ── 第一步：测量 processor 显存 ──
    full_gpu_reset()
    mem_0 = gpu_mem_mb()
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    mem_after_proc = gpu_mem_mb()
    processor_mem = mem_after_proc - mem_0
    print(f"  Processor memory: {processor_mem:.1f} MB")

    # ── 第二步：测量 model 显存（与 processor 分开） ──
    full_gpu_reset()
    # 注意：reset 后 processor 可能仍在 CPU，不影响 GPU 计量
    # 但 processor 对象本身还在，只是它的 GPU tensor（如果有）被 empty_cache 了
    # 为安全起见，我们重新记录基线
    mem_before_model = gpu_mem_mb()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id,
        torch_dtype=torch.float16,
        device_map=CFG.device,
    )
    model.eval()

    torch.cuda.synchronize()
    mem_after_model = gpu_mem_mb()
    model_mem = mem_after_model - mem_before_model

    # 加载过程中的峰值
    load_peak = gpu_peak_mb() - mem_before_model

    print(f"  Model memory (resident): {model_mem:.1f} MB")
    print(f"  Model load peak:         {load_peak:.1f} MB")

    # ── 第三步：构建推理函数 ──
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

    # ── Warmup ──
    print(f"  Warming up ({warmup} samples)...")
    for i in range(min(warmup, len(ds))):
        run_one(ds[i])

    # ── 推理 Profile ──
    # reset peak 只针对推理阶段
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    for item in tqdm(ds, desc="  Base"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one(item)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    inference_peak = gpu_peak_mb()

    result = _build_result(model_mem, load_peak, processor_mem,
                           inference_peak, latencies)
    print(f"  Avg latency:    {result['avg_latency_s']:.4f}s")
    print(f"  Inference peak: {result['peak_memory_mb']:.1f} MB")

    del model, processor
    full_gpu_reset()
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Profile: Enhanced (SAE + Predictor, 无 PCS)
# ══════════════════════════════════════════════════════════════════════════════

def profile_enhanced_no_pcs(ds, args, warmup: int = 3) -> Dict[str, Any]:
    from src.Model import QwenWithClusterPredictorAndSAE

    print(f"\n{'=' * 60}")
    print(f"  Profiling: Enhanced (SAE + Predictor, NO PCS)")
    print(f"{'=' * 60}")

    # ── 测量 model 显存 ──
    full_gpu_reset()
    mem_before = gpu_mem_mb()

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
        predictor_ckpt=args.predictor_ckpt,
        device=CFG.device,
    )
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        del state
        gc.collect()
        print("  Qwen Stage 2 weights loaded.")

    model.to(CFG.device)
    model.eval()

    # 禁用 PCS
    with torch.no_grad():
        model.pc_suppressor.alpha_param.fill_(-100.0)
    print("  PCS disabled (alpha → 0)")

    torch.cuda.synchronize()
    mem_after = gpu_mem_mb()
    model_mem = mem_after - mem_before
    load_peak = gpu_peak_mb() - mem_before

    # Enhanced model 内部封装了 processor，这里单独估算
    # processor 显存通常很小，但为一致性仍然记录
    processor_mem = 0.0  # 封装在 model 内部，无法单独拆分；标记为 0 表示已含在 model_mem 中

    print(f"  Model memory (resident): {model_mem:.1f} MB")
    print(f"  Model load peak:         {load_peak:.1f} MB")

    # ── 推理函数 ──
    def run_one(item):
        tmp_path = save_image(item)
        prompt = build_prompt(item)
        model.generate(
            image_path=tmp_path, question=prompt,
            max_new_tokens=32, verbose=False,
        )

    # ── Warmup ──
    print(f"  Warming up ({warmup} samples)...")
    for i in range(min(warmup, len(ds))):
        run_one(ds[i])

    # ── 推理 Profile ──
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    for item in tqdm(ds, desc="  Enhanced (no PCS)"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one(item)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    inference_peak = gpu_peak_mb()

    result = _build_result(model_mem, load_peak, processor_mem,
                           inference_peak, latencies)
    print(f"  Avg latency:    {result['avg_latency_s']:.4f}s")
    print(f"  Inference peak: {result['peak_memory_mb']:.1f} MB")

    del model
    full_gpu_reset()
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Profile: Enhanced (SAE + Predictor + PCS)
# ══════════════════════════════════════════════════════════════════════════════

def profile_enhanced_with_pcs(ds, args, warmup: int = 3) -> Dict[str, Any]:
    from src.Model import QwenWithClusterPredictorAndSAE

    print(f"\n{'=' * 60}")
    print(f"  Profiling: Enhanced (SAE + Predictor + PCS)")
    print(f"{'=' * 60}")

    # ── 测量 model 显存 ──
    full_gpu_reset()
    mem_before = gpu_mem_mb()

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
        predictor_ckpt=args.predictor_ckpt,
        device=CFG.device,
    )
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        del state
        gc.collect()
        print("  Qwen Stage 2 weights loaded.")

    model.to(CFG.device)
    model.eval()
    print(f"  PCS enabled (alpha_param={model.pc_suppressor.alpha_param.item():.4f})")

    torch.cuda.synchronize()
    mem_after = gpu_mem_mb()
    model_mem = mem_after - mem_before
    load_peak = gpu_peak_mb() - mem_before

    processor_mem = 0.0

    print(f"  Model memory (resident): {model_mem:.1f} MB")
    print(f"  Model load peak:         {load_peak:.1f} MB")

    # ── 推理函数 ──
    def run_one(item):
        tmp_path = save_image(item)
        prompt = build_prompt(item)
        model.generate(
            image_path=tmp_path, question=prompt,
            max_new_tokens=32, verbose=False,
        )

    # ── Warmup ──
    print(f"  Warming up ({warmup} samples)...")
    for i in range(min(warmup, len(ds))):
        run_one(ds[i])

    # ── 推理 Profile ──
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    for item in tqdm(ds, desc="  Enhanced (with PCS)"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_one(item)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    inference_peak = gpu_peak_mb()

    result = _build_result(model_mem, load_peak, processor_mem,
                           inference_peak, latencies)
    print(f"  Avg latency:    {result['avg_latency_s']:.4f}s")
    print(f"  Inference peak: {result['peak_memory_mb']:.1f} MB")

    del model
    full_gpu_reset()
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  参数量统计
# ══════════════════════════════════════════════════════════════════════════════

def count_params(args) -> Tuple[Dict[str, int], Dict[str, str]]:
    """统计各模块参数量（纯 CPU，不需要 GPU）。"""
    from src.Model import (
        ClusterPredictor, ImageClusterScorer, ExtraProjector,
        SemanticCrossAttention, PrincipalComponentSuppressor,
        SemanticCompleter,
    )
    from src.SAE import SAE
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(CFG.model_id)
    dim = config.text_config.hidden_size
    latent_dim = dim * CFG.latent_mult

    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
    with open(cluster_path) as f:
        cluster_info = json.load(f)
    n_clusters = len(cluster_info["clusters"])

    modules = {
        "SAE":                           SAE(dim, latent_dim, CFG.topk),
        "ClusterPredictor":              ClusterPredictor(dim, n_clusters),
        "ImageClusterScorer":            ImageClusterScorer(dim, n_clusters),
        "ExtraProjector":                ExtraProjector(dim),
        "SemanticCrossAttention":        SemanticCrossAttention(dim),
        "SemanticCompleter":             SemanticCompleter(dim, latent_dim),
        "PCS (PrincipalComponentSupp.)": PrincipalComponentSuppressor(),
    }

    counts = {}
    for name, mod in modules.items():
        counts[name] = sum(p.numel() for p in mod.parameters())

    sae_predictor_keys = [
        "SAE", "ClusterPredictor", "ImageClusterScorer",
        "ExtraProjector", "SemanticCrossAttention", "SemanticCompleter",
    ]
    counts["Total (SAE+Predictor)"] = sum(counts[k] for k in sae_predictor_keys)
    counts["Total (SAE+Predictor+PCS)"] = (
        counts["Total (SAE+Predictor)"]
        + counts["PCS (PrincipalComponentSupp.)"]
    )

    trainable_info = {
        "SAE":                           "No (frozen)",
        "ClusterPredictor":              "Yes",
        "ImageClusterScorer":            "Yes",
        "ExtraProjector":                "Yes",
        "SemanticCrossAttention":        "Yes",
        "SemanticCompleter":             "Yes",
        "PCS (PrincipalComponentSupp.)": "Yes",
        "Total (SAE+Predictor)":         "",
        "Total (SAE+Predictor+PCS)":     "",
    }

    return counts, trainable_info


# ══════════════════════════════════════════════════════════════════════════════
#  子进程隔离运行
# ══════════════════════════════════════════════════════════════════════════════

def _run_in_subprocess(target_fn, args_tuple, result_queue: mp.Queue):
    """
    在独立子进程中执行 target_fn(*args_tuple)，
    将返回值放入 result_queue。

    这样每个配置拥有独立的 CUDA context，
    彻底避免前一个配置的 caching allocator / cuBLAS workspace 残留污染。
    """
    try:
        result = target_fn(*args_tuple)
        result_queue.put(("ok", result))
    except Exception as e:
        import traceback
        result_queue.put(("error", traceback.format_exc()))


def run_isolated(target_fn, *args):
    """
    在独立子进程中运行 profiling 函数并返回结果。
    如果子进程出错，打印 traceback 并返回 None。
    """
    ctx = mp.get_context("spawn")  # spawn 确保全新的 CUDA context
    q = ctx.Queue()
    p = ctx.Process(target=_run_in_subprocess, args=(target_fn, args, q))
    p.start()
    p.join()

    if q.empty():
        print("  [ERROR] Subprocess returned no result (possibly crashed).")
        return None

    status, payload = q.get()
    if status == "ok":
        return payload
    else:
        print(f"  [ERROR] Subprocess failed:\n{payload}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  子进程入口（顶层函数，可被 pickle）
# ══════════════════════════════════════════════════════════════════════════════

def _subprocess_profile_base(n_samples: int, warmup: int):
    ds = load_test_data(n_samples)
    return profile_base(ds, warmup=warmup)


def _subprocess_profile_no_pcs(n_samples: int, warmup: int,
                                layer: int, predictor_ckpt: str,
                                qwen_ckpt: Optional[str]):
    ds = load_test_data(n_samples)
    args = argparse.Namespace(
        layer=layer, predictor_ckpt=predictor_ckpt, qwen_ckpt=qwen_ckpt,
    )
    return profile_enhanced_no_pcs(ds, args, warmup=warmup)


def _subprocess_profile_with_pcs(n_samples: int, warmup: int,
                                  layer: int, predictor_ckpt: str,
                                  qwen_ckpt: Optional[str]):
    ds = load_test_data(n_samples)
    args = argparse.Namespace(
        layer=layer, predictor_ckpt=predictor_ckpt, qwen_ckpt=qwen_ckpt,
    )
    return profile_enhanced_with_pcs(ds, args, warmup=warmup)


# ══════════════════════════════════════════════════════════════════════════════
#  打印报告
# ══════════════════════════════════════════════════════════════════════════════

def print_report(
    base_r: Dict, no_pcs_r: Dict, pcs_r: Dict,
    param_counts: Dict, trainable_info: Dict, n_samples: int,
):
    W = 100

    print(f"\n{'═' * W}")
    print(f"  Overhead Profiling Report ({n_samples} samples, CV-Bench)")
    print(f"{'═' * W}")

    # ── 表 1: 绝对值 ──
    print(f"\n  ┌─ Latency & Memory {'─' * (W - 22)}┐")
    header = (
        f"  │ {'Configuration':<28s}│ {'Avg (s)':>9s} │ {'Std':>7s} │ {'Median':>8s} │"
        f" {'Min':>7s} │ {'Max':>7s} │"
        f" {'Model':>8s} │ {'LoadPk':>8s} │ {'InfPeak':>9s} │"
    )
    print(header)
    sep = (
        f"  ├{'─' * 29}┼{'─' * 11}┼{'─' * 9}┼{'─' * 10}┼"
        f"{'─' * 9}┼{'─' * 9}┼"
        f"{'─' * 10}┼{'─' * 10}┼{'─' * 11}┤"
    )
    print(sep)

    rows = [
        ("Base (Qwen2.5-VL)",       base_r),
        ("+ SAE + Predictor",       no_pcs_r),
        ("+ SAE + Predictor + PCS", pcs_r),
    ]
    for label, r in rows:
        if r is None:
            print(f"  │ {label:<28s}│ {'(skipped)':>9s} │ {'':>7s} │ {'':>8s} │"
                  f" {'':>7s} │ {'':>7s} │ {'':>8s} │ {'':>8s} │ {'':>9s} │")
            continue
        print(
            f"  │ {label:<28s}│ {r['avg_latency_s']:>9.4f} │ {r['std_latency_s']:>7.4f} │"
            f" {r['median_latency_s']:>8.4f} │ {r['min_latency_s']:>7.4f} │"
            f" {r['max_latency_s']:>7.4f} │"
            f" {r['model_memory_mb']:>7.1f}M │ {r['load_peak_mb']:>7.1f}M │"
            f" {r['peak_memory_mb']:>8.1f}M │"
        )

    bot = (
        f"  └{'─' * 29}┴{'─' * 11}┴{'─' * 9}┴{'─' * 10}┴"
        f"{'─' * 9}┴{'─' * 9}┴"
        f"{'─' * 10}┴{'─' * 10}┴{'─' * 11}┘"
    )
    print(bot)

    # 说明
    print("    Model = 加载后常驻显存 | LoadPk = 加载过程中峰值 | InfPeak = 推理阶段峰值")
    if base_r and base_r.get("processor_memory_mb", 0) > 0.1:
        print(f"    Base processor memory (单独测量): {base_r['processor_memory_mb']:.1f} MB")

    # ── 表 2: Overhead ──
    if base_r:
        print(f"\n  ┌─ Overhead vs Base {'─' * (W - 22)}┐")
        header2 = (
            f"  │ {'Configuration':<30s}│ {'Lat Δ (s)':>12s} │ {'Lat Δ (%)':>12s} │"
            f" {'Mem Δ (MB)':>12s} │ {'Mem Δ (%)':>12s} │"
        )
        print(header2)
        print(f"  ├{'─' * 31}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┤")

        for label, r in [
            ("+ SAE + Predictor",       no_pcs_r),
            ("+ SAE + Predictor + PCS", pcs_r),
        ]:
            if r is None:
                print(f"  │ {label:<30s}│ {'(skipped)':>12s} │ {'':>12s} │ {'':>12s} │ {'':>12s} │")
                continue
            lat_d = round(r["avg_latency_s"] - base_r["avg_latency_s"], 4)
            mem_d = round(r["peak_memory_mb"] - base_r["peak_memory_mb"], 1)
            lat_p = pct_str(r["avg_latency_s"], base_r["avg_latency_s"])
            mem_p = pct_str(r["peak_memory_mb"], base_r["peak_memory_mb"])
            print(
                f"  │ {label:<30s}│ {delta_str(lat_d, 's'):>12s} │ {lat_p:>12s} │"
                f" {delta_str(mem_d, ''):>12s} │ {mem_p:>12s} │"
            )

        # PCS marginal
        if no_pcs_r and pcs_r:
            pcs_lat_d = round(pcs_r["avg_latency_s"] - no_pcs_r["avg_latency_s"], 4)
            pcs_mem_d = round(pcs_r["peak_memory_mb"] - no_pcs_r["peak_memory_mb"], 1)
            pcs_lat_p = pct_str(pcs_r["avg_latency_s"], no_pcs_r["avg_latency_s"])
            pcs_mem_p = pct_str(pcs_r["peak_memory_mb"], no_pcs_r["peak_memory_mb"])
            print(f"  ├{'─' * 31}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┤")
            print(
                f"  │ {'PCS only (marginal)':<30s}│ {delta_str(pcs_lat_d, 's'):>12s} │"
                f" {pcs_lat_p:>12s} │ {delta_str(pcs_mem_d, ''):>12s} │ {pcs_mem_p:>12s} │"
            )

        print(f"  └{'─' * 31}┴{'─' * 14}┴{'─' * 14}┴{'─' * 14}┴{'─' * 14}┘")

    # ── 表 3: 参数量 ──
    BASE_PARAMS = 7_615_616_000

    print(f"\n  ┌─ Module Parameters {'─' * (W - 23)}┐")
    header3 = (
        f"  │ {'Module':<36s}│ {'Params':>12s} │ {'% of Base (7.6B)':>17s} │ {'Trainable':>10s} │"
    )
    print(header3)
    print(f"  ├{'─' * 37}┼{'─' * 14}┼{'─' * 19}┼{'─' * 12}┤")

    for name, count in param_counts.items():
        pct = f"{count / BASE_PARAMS * 100:.4f}%"
        train = trainable_info.get(name, "")
        if name.startswith("Total"):
            print(f"  ├{'─' * 37}┼{'─' * 14}┼{'─' * 19}┼{'─' * 12}┤")
            print(
                f"  │ {name:<36s}│ {fmt_params(count):>12s} │ {pct:>17s} │ {'':>10s} │"
            )
        else:
            print(
                f"  │ {name:<36s}│ {fmt_params(count):>12s} │ {pct:>17s} │ {train:>10s} │"
            )

    print(f"  └{'─' * 37}┴{'─' * 14}┴{'─' * 19}┴{'─' * 12}┘")
    print(f"{'═' * W}")


# ══════════════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Overhead Profiling: Base vs Enhanced vs +PCS"
    )
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of CV-Bench samples to profile")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup samples (excluded from timing)")
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/qwen_layer8_old/focus_ckpt_0.75_64_5000/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/profile_results")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        choices=["base", "no_pcs", "pcs"],
                        help="Skip certain configurations (for debugging)")
    parser.add_argument("--no_isolation", action="store_true",
                        help="Run all configs in same process (faster but less accurate)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 参数量统计（不需要 GPU）──
    print("\nCounting module parameters...")
    param_counts, trainable_info = count_params(args)

    # ── Profile 三个配置 ──
    base_r = no_pcs_r = pcs_r = None

    if args.no_isolation:
        # ── 同进程模式（快速但有交叉污染风险）──
        print("\n[WARNING] Running in same process (--no_isolation). "
              "Memory numbers may have cross-contamination.")
        ds = load_test_data(args.n_samples)

        if "base" not in args.skip:
            base_r = profile_base(ds, warmup=args.warmup)
        if "no_pcs" not in args.skip:
            no_pcs_r = profile_enhanced_no_pcs(ds, args, warmup=args.warmup)
        if "pcs" not in args.skip:
            pcs_r = profile_enhanced_with_pcs(ds, args, warmup=args.warmup)
    else:
        # ── 子进程隔离模式（推荐）──
        print("\nUsing subprocess isolation (spawn) for clean CUDA context.")

        if "base" not in args.skip:
            print("\n[Subprocess] Launching base profiling...")
            base_r = run_isolated(
                _subprocess_profile_base,
                args.n_samples, args.warmup,
            )

        if "no_pcs" not in args.skip:
            print("\n[Subprocess] Launching enhanced (no PCS) profiling...")
            no_pcs_r = run_isolated(
                _subprocess_profile_no_pcs,
                args.n_samples, args.warmup,
                args.layer, args.predictor_ckpt, args.qwen_ckpt,
            )

        if "pcs" not in args.skip:
            print("\n[Subprocess] Launching enhanced (with PCS) profiling...")
            pcs_r = run_isolated(
                _subprocess_profile_with_pcs,
                args.n_samples, args.warmup,
                args.layer, args.predictor_ckpt, args.qwen_ckpt,
            )

    # ── 打印报告 ──
    print_report(base_r, no_pcs_r, pcs_r, param_counts, trainable_info, args.n_samples)

    # ── 保存 JSON ──
    report: Dict[str, Any] = {
        "n_samples": args.n_samples,
        "layer": args.layer,
        "isolation_mode": "subprocess" if not args.no_isolation else "same_process",
        "profiles": {
            "base":             base_r,
            "enhanced_no_pcs":  no_pcs_r,
            "enhanced_with_pcs": pcs_r,
        },
        "param_counts": param_counts,
    }

    # 计算 overhead（仅当两个配置都有结果时）
    overhead: Dict[str, Any] = {}
    if base_r and no_pcs_r:
        overhead["sae_predictor"] = {
            "latency_delta_s": round(
                no_pcs_r["avg_latency_s"] - base_r["avg_latency_s"], 4
            ),
            "latency_pct": round(
                (no_pcs_r["avg_latency_s"] - base_r["avg_latency_s"])
                / base_r["avg_latency_s"] * 100, 2
            ),
            "memory_delta_mb": round(
                no_pcs_r["peak_memory_mb"] - base_r["peak_memory_mb"], 1
            ),
            "memory_pct": round(
                (no_pcs_r["peak_memory_mb"] - base_r["peak_memory_mb"])
                / base_r["peak_memory_mb"] * 100, 2
            ),
        }
    if base_r and pcs_r:
        overhead["sae_predictor_pcs"] = {
            "latency_delta_s": round(
                pcs_r["avg_latency_s"] - base_r["avg_latency_s"], 4
            ),
            "latency_pct": round(
                (pcs_r["avg_latency_s"] - base_r["avg_latency_s"])
                / base_r["avg_latency_s"] * 100, 2
            ),
            "memory_delta_mb": round(
                pcs_r["peak_memory_mb"] - base_r["peak_memory_mb"], 1
            ),
            "memory_pct": round(
                (pcs_r["peak_memory_mb"] - base_r["peak_memory_mb"])
                / base_r["peak_memory_mb"] * 100, 2
            ),
        }
    if no_pcs_r and pcs_r:
        overhead["pcs_marginal"] = {
            "latency_delta_s": round(
                pcs_r["avg_latency_s"] - no_pcs_r["avg_latency_s"], 4
            ),
            "memory_delta_mb": round(
                pcs_r["peak_memory_mb"] - no_pcs_r["peak_memory_mb"], 1
            ),
        }
    report["overhead_vs_base"] = overhead

    json_path = os.path.join(args.save_dir, "overhead_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()