"""
调试神经元

用法：
    python scripts/explore.py --layer 8                        # 查看 feature 统计
    python scripts/explore.py --layer 8 --feature 27416        # 调试单个 feature
    python scripts/explore.py --layer 8 --show_labels          # 查看已有标注
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import random
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import CFG
from src.utils import make_masked_image


def load_cache(layer: int):
    cache_path = os.path.join(CFG.cache_dir, f"cache_layer{layer}.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    print(f"loaded {len(cache)} images from {cache_path}")
    return cache


def show_feature_stats(all_z: np.ndarray, layer: int):
    active_ids = np.where(all_z.max(axis=0) > 0)[0]
    print(f"active features: {len(active_ids)} / {all_z.shape[1]}")

    freq = (all_z > 0).mean(axis=0)
    plt.hist(freq[freq > 0], bins=50)
    plt.xlabel("activation frequency")
    plt.ylabel("num features")
    plt.title(f"Layer {layer} — feature activation frequency")
    plt.tight_layout()
    plt.show()


def debug_feature(cache: list, feature_id: int, layer: int):
    results = []
    for item in cache:
        masked_img, image_score = make_masked_image(
            item["image_path"], item["z"], feature_id,
            item["H_tok"], item["W_tok"], top_n=CFG.top_n_patches,
        )
        results.append({"image_path": item["image_path"],
                        "masked_img": masked_img, "image_score": image_score})

    results.sort(key=lambda x: x["image_score"], reverse=True)
    top5 = results[:CFG.top_n_images]

    fig, axes = plt.subplots(1, len(top5), figsize=(5 * len(top5), 5))
    for i, res in enumerate(top5):
        axes[i].imshow(cv2.cvtColor(res["masked_img"], cv2.COLOR_BGR2RGB))
        axes[i].axis("off")
        axes[i].set_title(
            f"{os.path.basename(res['image_path'])}\nscore={res['image_score']:.3f}",
            fontsize=9,
        )
    plt.suptitle(f"Feature {feature_id} (layer {layer})", fontsize=12)
    plt.tight_layout()
    plt.show()


def show_labels(layer: int, sample_n: int = 10):
    label_path = os.path.join(CFG.label_dir, f"feature_labels_layer{layer}.json")
    with open(label_path) as f:
        labels = json.load(f)
    print(f"{len(labels)} features labeled")
    for fid, label in random.sample(list(labels.items()), min(sample_n, len(labels))):
        print(f"  {fid:>6s} -> {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--feature", type=int, default=None, help="调试单个 feature id")
    parser.add_argument("--show_labels", action="store_true", help="查看已有 Claude 标注")
    args = parser.parse_args()

    cache = load_cache(args.layer)
    all_z = np.concatenate([item["z"] for item in cache], axis=0)

    if args.show_labels:
        show_labels(args.layer)
    elif args.feature is not None:
        debug_feature(cache, args.feature, args.layer)
    else:
        show_feature_stats(all_z, args.layer)


if __name__ == "__main__":
    main()