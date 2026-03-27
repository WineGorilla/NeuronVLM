"""
推理脚本 — Single-Forward Architecture

用法：
    python src/inference.py \
        --image data/images/train2014/COCO_train2014_000000094420.jpg \
        --question "What is the current action of the bus?"

    python src/inference.py \
        --image xxx.jpg \
        --question "..." \
        --predictor_ckpt outputs/focus_ckpt/predictor_best.pt \
        --qwen_ckpt outputs/focus_ckpt/qwen_best.pt
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
from config import CFG
from src.Model import QwenWithClusterPredictorAndSAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",          type=str, required=True)
    parser.add_argument("--question",       type=str, required=True)
    parser.add_argument("--layer",          type=int, default=CFG.vis_layer)
    parser.add_argument("--suppress_layer", type=int, default=-8,
                        help="Layer for PCA suppression (negative = from end)")
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/focus_ckpt/predictor_best.pt")
    parser.add_argument("--qwen_ckpt",      type=str, default=None)
    parser.add_argument("--max_tokens",     type=int, default=128)
    parser.add_argument("--top_k_clusters", type=int, default=10)
    args = parser.parse_args()

    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )

    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id          = CFG.model_id,
        sae_ckpt_dir      = CFG.save_dir,
        cluster_path      = cluster_path,
        inject_layer      = args.layer,
        suppress_layer    = args.suppress_layer,
        latent_mult       = CFG.latent_mult,
        topk              = CFG.topk,
        top_n_patches     = CFG.top_n_patches,
        top_k_clusters    = args.top_k_clusters,
        predictor_ckpt    = args.predictor_ckpt,
        device            = CFG.device,
    )

    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        print(f"Loading Qwen fine-tuned weights: {args.qwen_ckpt}")
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.base_model.load_state_dict(state, strict=False)
        print("  Qwen weights loaded.")

    result = model.generate(
        image_path     = args.image,
        question       = args.question,
        max_new_tokens = args.max_tokens,
        verbose        = True,
    )

    print(f"\n{'='*50}")
    print(f"Clusters : {result['cluster_ids']} → {result['cluster_names']}")
    print(f"Answer   : {result['final_answer']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()