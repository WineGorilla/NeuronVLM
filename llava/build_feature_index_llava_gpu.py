"""
把 cache_layer{N}.pkl 转换为 feature_index_layer{N}.pkl — LLaVA-OneVision 版。
GPU 加速版本（使用 PyTorch）。

用法：
    python llava/build_feature_index_llava_gpu.py --layer 8 --min_images 3
    python llava/build_feature_index_llava_gpu.py --layer 8 --top_k 10 --min_images 3
    CUDA_VISIBLE_DEVICES=0 python llava/build_feature_index_llava_gpu.py --layer 8 --batch_size 512
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import argparse
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch

from config import CFG


def get_dense_z(item):
    z = item["z"]
    if sp.issparse(z):
        return z.toarray()
    return z


def process_batch_gpu(batch_items, top_n_patches, device):
    """
    在 GPU 上批量处理多张图片，返回每张图的 (fid, score, top_patch_idx) 结果。
    """
    results = []

    for item in batch_items:
        z = get_dense_z(item)  # (n_patches, n_features)
        image_path = item["image_path"]
        H_tok = item["H_tok"]
        W_tok = item["W_tok"]

        # 转到 GPU
        z_gpu = torch.from_numpy(z).to(device, dtype=torch.float32)

        # pos_z = ReLU(z)
        pos_z = torch.clamp(z_gpu, min=0)

        # 找到 active features：每个 feature 在所有 patch 上的最大值 > 0
        max_per_feature, _ = pos_z.max(dim=0)  # (n_features,)
        active_mask = max_per_feature > 0
        active_fids = torch.where(active_mask)[0]  # GPU tensor of feature indices

        if active_fids.numel() == 0:
            results.append([])
            continue

        # 只取 active features 的列，减少计算量
        pos_z_active = pos_z[:, active_fids]  # (n_patches, n_active)

        # 对每个 active feature，找 top_n_patches 个 patch
        k = min(top_n_patches, pos_z_active.shape[0])
        # topk 沿 patch 维度（dim=0）
        topk_values, topk_indices = torch.topk(pos_z_active, k=k, dim=0)  # (k, n_active)

        # 计算每个 feature 的平均 score
        scores = topk_values.mean(dim=0)  # (n_active,)

        # 过滤 score > 0
        valid_mask = scores > 0
        valid_scores = scores[valid_mask]
        valid_fids = active_fids[valid_mask]
        valid_topk_indices = topk_indices[:, valid_mask]

        # 搬回 CPU 收集结果
        valid_fids_cpu = valid_fids.cpu().numpy()
        valid_scores_cpu = valid_scores.cpu().numpy()
        valid_topk_cpu = valid_topk_indices.t().cpu().numpy()  # (n_valid, k)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",      type=int, default=CFG.vis_layer)
    parser.add_argument("--top_k",      type=int, default=10)
    parser.add_argument("--min_images", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="每批处理的图片数（控制 GPU 显存占用）")
    parser.add_argument("--device",     type=str, default="cuda",
                        help="设备：cuda / cuda:0 / cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_path = os.path.join(CFG.llava_cache_dir, f"cache_layer{args.layer}.pkl")
    index_path = os.path.join(CFG.llava_cache_dir, f"feature_index_layer{args.layer}.pkl")

    print(f"Loading cache: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache)} images")

    # ---- GPU 批量处理 ----
    feature_scores = defaultdict(list)
    total = len(cache)
    batch_size = args.batch_size

    print(f"Building feature index (batch_size={batch_size})...")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = cache[start:end]

        batch_results = process_batch_gpu(batch, CFG.top_n_patches, device)

        for image_results in batch_results:
            for (fid, score, image_path, H_tok, W_tok, top_patch_idx) in image_results:
                feature_scores[fid].append(
                    (score, image_path, H_tok, W_tok, top_patch_idx)
                )

        if end % 5000 < batch_size or end == total:
            print(f"  [{end}/{total}]")

    print(f"Found {len(feature_scores)} active features (before filtering)")

    # ---- 去重、过滤、取 top-k（CPU 即可） ----
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


if __name__ == "__main__":
    main()