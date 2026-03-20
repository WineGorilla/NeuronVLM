"""
推理脚本。

用法：
    python scripts/inference.py \
        --image data/images/train2014/xxx.jpg \
        --question "What color is the dog?"

    python scripts/inference.py \
        --image xxx.jpg \
        --question "..." \
        --focus_ckpt outputs/focus_ckpt/model_best.pt \
        --inject_scale 0.3
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from config import CFG
from src.Model import QwenWithFocusAndSAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",        type=str, required=True)
    parser.add_argument("--question",     type=str, required=True)
    parser.add_argument("--layer",        type=int, default=CFG.vis_layer)
    parser.add_argument("--inject_scale", type=float, default=0.3)
    parser.add_argument("--focus_ckpt",   type=str,
                        default="outputs/focus_ckpt/model_best.pt")
    args = parser.parse_args()

    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )

    model = QwenWithFocusAndSAE.from_pretrained(
        model_id     = CFG.model_id,
        sae_ckpt_dir = CFG.save_dir,
        cluster_path = cluster_path,
        layer        = args.layer,
        latent_mult  = CFG.latent_mult,
        topk         = CFG.topk,
        inject_scale = args.inject_scale,
        focus_ckpt   = args.focus_ckpt,
        device       = CFG.device,
    )

    result = model.generate(
        image_path = args.image,
        question   = args.question,
        verbose    = True,
    )

    print(f"\n{'='*50}")
    print(f"Think        : {result['think_output']}")
    print(f"Clusters     : {result['cluster_ids']} → {result['cluster_names']}")
    print(f"Base answer  : {result['base_answer']}")
    print(f"Final answer : {result['final_answer']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()