"""
把 cache_layer{N}.pkl 转换为 feature_index_layer{N}.pkl — LLaVA-OneVision 版。
GPU 加速版本（使用 PyTorch），支持中间 checkpoint 断点续跑 + 定期裁剪防 OOM。

用法：
    python llava_next/build_feature_index_llava_gpu.py --layer 8 --min_images 16
    python llava/build_feature_index_llava_gpu.py --layer 8 --batch_size 512
    python llava/build_feature_index_llava_gpu.py --layer 8 --checkpoint_every 10000
    python llava/build_feature_index_llava_gpu.py --layer 8 --no_resume
    CUDA_VISIBLE_DEVICES=0 python llava/build_feature_index_llava_gpu.py --layer 8 --prune_every 2000 --prune_keep 15 --batch_size 256 --checkpoint_every 2500
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gc
import pickle
import argparse
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch

from llava_next.config_llava import CFG


def get_dense_z(item):
    z = item["z"]
    if sp.issparse(z):
        return z.toarray()
    return z


def process_batch_gpu(batch_items, top_n_patches, device):
    """在 GPU 上批量处理多张图片。"""
    results = []

    for item in batch_items:
        z = get_dense_z(item)  # (n_patches, n_features)
        image_path = item["image_path"]
        H_tok = item["H_tok"]
        W_tok = item["W_tok"]

        z_gpu = torch.from_numpy(z).to(device, dtype=torch.float32)
        pos_z = torch.clamp(z_gpu, min=0)

        max_per_feature, _ = pos_z.max(dim=0)
        active_mask = max_per_feature > 0
        active_fids = torch.where(active_mask)[0]

        if active_fids.numel() == 0:
            results.append([])
            continue

        pos_z_active = pos_z[:, active_fids]

        k = min(top_n_patches, pos_z_active.shape[0])
        topk_values, topk_indices = torch.topk(pos_z_active, k=k, dim=0)

        scores = topk_values.mean(dim=0)

        valid_mask = scores > 0
        valid_scores = scores[valid_mask]
        valid_fids = active_fids[valid_mask]
        valid_topk_indices = topk_indices[:, valid_mask]

        valid_fids_cpu = valid_fids.cpu().numpy()
        valid_scores_cpu = valid_scores.cpu().numpy()
        valid_topk_cpu = valid_topk_indices.t().cpu().numpy()

        image_results = []
        for j in range(len(valid_fids_cpu)):
            image_results.append((
                int(valid_fids_cpu[j]),
                float(valid_scores_cpu[j]),
                image_path,
                H_tok,
                W_tok,
                valid_topk_cpu[j].tolist()
            ))
        results.append(image_results)

    return results


# ── 内存优化：定期裁剪 ───────────────────────────────────────────────────────

def prune_feature_scores(feature_scores, keep_n):
    """
    对每个 feature，只保留 score 最高的 keep_n 条记录。
    防止 feature_scores 无限增长导致 RAM OOM。
    """
    pruned_count = 0
    for fid in feature_scores:
        entries = feature_scores[fid]
        if len(entries) > keep_n:
            entries.sort(key=lambda x: x[0], reverse=True)
            pruned_count += len(entries) - keep_n
            feature_scores[fid] = entries[:keep_n]
    return pruned_count


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(feature_scores, processed_count, checkpoint_path):
    """保存中间结果（先写临时文件再重命名，防止损坏）。"""
    ckpt = {
        "feature_scores": dict(feature_scores),
        "processed_count": processed_count,
    }
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp_path, checkpoint_path)
    size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
    print(f"  [checkpoint] saved at {processed_count} images ({size_mb:.1f} MB)")


def load_checkpoint(checkpoint_path):
    """加载 checkpoint，返回 (feature_scores, processed_count) 或 None。"""
    if not os.path.exists(checkpoint_path):
        return None
    print(f"  [checkpoint] found: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    feature_scores = defaultdict(list, ckpt["feature_scores"])
    processed_count = ckpt["processed_count"]
    print(f"  [checkpoint] resuming from image {processed_count}, "
          f"{len(feature_scores)} features so far")
    return feature_scores, processed_count


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",      type=int, default=CFG.vis_layer)
    parser.add_argument("--top_k",      type=int, default=10)
    parser.add_argument("--min_images", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="每批处理的图片数（控制 GPU 显存占用）")
    parser.add_argument("--device",     type=str, default="cuda",
                        help="设备：cuda / cuda:0 / cpu")
    parser.add_argument("--checkpoint_every", type=int, default=10000,
                        help="每处理 N 张图片保存一次 checkpoint（0 = 不保存）")
    parser.add_argument("--prune_every", type=int, default=2500,
                        help="每处理 N 张图片裁剪一次内存（0 = 不裁剪）")
    parser.add_argument("--prune_keep",  type=int, default=None,
                        help="裁剪时每个 feature 保留多少条（默认 = top_k * 3）")
    parser.add_argument("--no_resume",  action="store_true",
                        help="忽略已有 checkpoint，从头开始")
    args = parser.parse_args()

    prune_keep = args.prune_keep or args.top_k * 3

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_path = os.path.join(CFG.llava_cache_dir, f"cache_layer{args.layer}.pkl")
    index_path = os.path.join(CFG.llava_cache_dir, f"feature_index_layer{args.layer}.pkl")
    checkpoint_path = os.path.join(CFG.llava_cache_dir, f"feature_index_layer{args.layer}.ckpt.pkl")

    print(f"Loading cache: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache)} images")

    # ---- 尝试恢复 checkpoint ----
    feature_scores = defaultdict(list)
    start_from = 0

    if not args.no_resume:
        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            feature_scores, start_from = ckpt

    # ---- GPU 批量处理 ----
    total = len(cache)
    batch_size = args.batch_size
    last_checkpoint_count = start_from
    last_prune_count = start_from

    print(f"Building feature index (batch_size={batch_size}, "
          f"checkpoint_every={args.checkpoint_every}, "
          f"prune_every={args.prune_every}, prune_keep={prune_keep})...")

    if start_from > 0:
        print(f"  Skipping first {start_from} images (already processed)")

    for start in range(start_from, total, batch_size):
        end = min(start + batch_size, total)
        batch = cache[start:end]

        batch_results = process_batch_gpu(batch, CFG.top_n_patches, device)

        for image_results in batch_results:
            for (fid, score, image_path, H_tok, W_tok, top_patch_idx) in image_results:
                feature_scores[fid].append(
                    (score, image_path, H_tok, W_tok, top_patch_idx)
                )

        # 进度日志
        if end % 5000 < batch_size or end == total:
            print(f"  [{end}/{total}]  (features: {len(feature_scores)})")

        # ---- 定期裁剪内存 ----
        if (args.prune_every > 0
                and (end - last_prune_count) >= args.prune_every):
            pruned = prune_feature_scores(feature_scores, prune_keep)
            if pruned > 0:
                gc.collect()
                print(f"  [prune] removed {pruned} low-score entries, "
                      f"keeping top-{prune_keep} per feature")
            last_prune_count = end

        # ---- 中间保存 checkpoint ----
        if (args.checkpoint_every > 0
                and (end - last_checkpoint_count) >= args.checkpoint_every):
            save_checkpoint(feature_scores, end, checkpoint_path)
            last_checkpoint_count = end

    print(f"Found {len(feature_scores)} active features (before filtering)")

    # ---- 去重、过滤、取 top-k ----
    print(f"Deduplicating, filtering (min_images={args.min_images}), "
          f"keeping top-{args.top_k}...")

    feature_index = {}
    skipped = 0
    for fid, entries in feature_scores.items():
        entries.sort(key=lambda x: x[0], reverse=True)

        seen = set()
        deduped = []
        for entry in entries:
            image_path = entry[1]
            if image_path not in seen:
                seen.add(image_path)
                deduped.append(entry)

        if len(deduped) < args.min_images:
            skipped += 1
            continue

        feature_index[fid] = deduped[:args.top_k]

    print(f"Skipped {skipped} features with < {args.min_images} images")

    with open(index_path, "wb") as f:
        pickle.dump(feature_index, f)

    print(f"Saved feature index: {index_path}")
    print(f"  Features: {len(feature_index)}")
    print(f"  Size: {os.path.getsize(index_path) / 1024 / 1024:.1f} MB")

    # ---- 清理 checkpoint ----
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  Cleaned up checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()