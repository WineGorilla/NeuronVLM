"""
把 cache_layer{N}.pkl 转换为 feature_index_layer{N}.pkl — LLaVA-OneVision 版。

用法：
    python llava/build_feature_index_llava.py --layer 8
    python llava/build_feature_index_llava.py --layer 8 --top_k 10 --min_images 3
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
    parser.add_argument("--top_k",      type=int, default=10)
    parser.add_argument("--min_images", type=int, default=5)
    args = parser.parse_args()

    cache_path = os.path.join(CFG.llava_cache_dir, f"cache_layer{args.layer}.pkl")
    index_path = os.path.join(CFG.llava_cache_dir, f"feature_index_layer{args.layer}.pkl")

    print(f"Loading cache: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache)} images")

    feature_scores = defaultdict(list)

    print("Building feature index...")
    for i, item in enumerate(cache):
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(cache)}]")

        z          = get_dense_z(item)
        image_path = item["image_path"]
        H_tok      = item["H_tok"]
        W_tok      = item["W_tok"]

        pos_z = np.maximum(z, 0)
        active_features = np.where(pos_z.max(axis=0) > 0)[0]

        for fid in active_features:
            patch_scores = pos_z[:, fid]
            top_patch_idx = patch_scores.argsort()[::-1][:CFG.top_n_patches]
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
        entries.sort(key=lambda x: x[0], reverse=True)
        seen    = set()
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