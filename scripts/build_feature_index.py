"""
把 cache_layer{N}.pkl 转换为 feature_index_layer{N}.pkl。

feature_index 格式：
    {
        feature_id: [
            (image_score, image_path, H_tok, W_tok, top_patch_indices),
            ...
        ]
    }
    每个 feature 只保留正激活最强的 top-K 张图（同一张图去重，至少 min_images 张）。
    image_score 用 top patch 的平均激活值计算，代表该 feature 在这张图里最典型区域的强度。

用法：
    python scripts/build_feature_index.py --layer 4 --min_images 3
    python scripts/build_feature_index.py --layer 8 --top_k 10 --min_images 3
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import argparse
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from config import CFG


def get_dense_z(item):
    z = item["z"]
    if sp.issparse(z):
        return z.toarray()
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",      type=int, default=CFG.vis_layer)
    parser.add_argument("--top_k",      type=int, default=10,
                        help="每个 feature 保留激活最强的 top-K 张图")
    parser.add_argument("--min_images", type=int, default=3,
                        help="每个 feature 至少需要 N 张不同图片才保留")
    args = parser.parse_args()

    cache_path = os.path.join(CFG.cache_dir, f"cache_layer{args.layer}.pkl")
    index_path = os.path.join(CFG.cache_dir, f"feature_index_layer{args.layer}.pkl")

    print(f"Loading cache: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache)} images")

    # feature_id -> list of (score, image_path, H_tok, W_tok, top_patch_indices)
    feature_scores = defaultdict(list)

    print("Building feature index...")
    for i, item in enumerate(cache):
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(cache)}]")

        z          = get_dense_z(item)
        image_path = item["image_path"]
        H_tok      = item["H_tok"]
        W_tok      = item["W_tok"]

        # 只保留正激活
        pos_z = np.maximum(z, 0)

        # 找有正激活的 feature
        active_features = np.where(pos_z.max(axis=0) > 0)[0]

        for fid in active_features:
            patch_scores = pos_z[:, fid]   # (num_tokens,) 每个 patch 对该 feature 的激活

            # 先找激活最强的 top_n_patches 个 patch
            top_patch_idx = patch_scores.argsort()[::-1][:CFG.top_n_patches]

            # 用 top patch 的平均激活作为这张图的代表分数
            score = float(patch_scores[top_patch_idx].mean())

            if score <= 0:
                continue

            feature_scores[int(fid)].append(
                (score, image_path, H_tok, W_tok, top_patch_idx.tolist())
            )

    print(f"Found {len(feature_scores)} active features (before filtering)")

    print(f"Deduplicating, filtering (min_images={args.min_images}), "
          f"keeping top-{args.top_k}...")

    feature_index = {}
    skipped = 0

    for fid, entries in feature_scores.items():
        # 按 top patch 平均激活分数降序排序
        entries.sort(key=lambda x: x[0], reverse=True)

        # 同一张图只保留激活最强的那条
        seen    = set()
        deduped = []
        for entry in entries:
            image_path = entry[1]
            if image_path not in seen:
                seen.add(image_path)
                deduped.append(entry)

        # 图片数不足则跳过
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