"""
单张图像的 SAE feature 可视化。

用法：
    python scripts/visualize.py --image wide.png --layer 8
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import CFG
from src.SAE import SAE
from src.utils import get_top_features, get_peak_blocks


# ── backbone + SAE 加载 ───────────────────────────────────────────────────────

def load_model_and_sae(layer: int):
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16
    ).to(CFG.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.text_config.hidden_size
    latent_dim = hidden_dim * CFG.latent_mult

    ckpt_path = os.path.join(CFG.save_dir, f"sae_layer{layer}.pt")
    sae = SAE(hidden_dim, latent_dim, CFG.topk).float().to(CFG.device)
    sae.load_state_dict(torch.load(ckpt_path, map_location=CFG.device))
    sae.eval()

    return processor, model, sae


# ── 推理：提取 image token 的 SAE 编码 ───────────────────────────────────────

def extract_latents(image_path: str, layer: int, processor, model):
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": "Describe the image"},
        ],
    }]]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts, images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(CFG.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    h           = outputs.hidden_states[layer + 1]
    image_grid  = inputs["image_grid_thw"]
    H_grid      = image_grid[0, 1].item()
    W_grid      = image_grid[0, 2].item()
    spatial_merge = model.config.vision_config.spatial_merge_size
    num_img_tokens = int(H_grid * W_grid / (spatial_merge ** 2))
    H_tok       = int(H_grid // spatial_merge)
    W_tok       = int(W_grid  // spatial_merge)

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    input_ids   = inputs["input_ids"][0]
    vision_pos  = (input_ids == vision_start_id).nonzero()[0].item() + 1

    img_tokens  = h[:, vision_pos : vision_pos + num_img_tokens, :]
    tokens      = img_tokens.reshape(-1, img_tokens.shape[-1]).float()

    return tokens, H_tok, W_tok


def encode_with_sae(tokens, sae):
    with torch.no_grad():
        z = sae.encode(tokens)
    return z.cpu().numpy()


# ── 可视化：在图上标注 top feature 的激活峰值位置 ─────────────────────────────

def visualize_top_features(image_path: str, z: np.ndarray,
                            H_tok: int, W_tok: int,
                            top_n: int = 64, layer: int = 8):
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    canvas = img.copy()

    top_feature_ids, _ = get_top_features(z, top_n=top_n, mode="mean")
    blocks = get_peak_blocks(z, top_feature_ids, H_tok, W_tok, orig_h, orig_w)

    colors = plt.cm.tab20(np.linspace(0, 1, len(blocks)))

    for i, block in enumerate(blocks):
        color = tuple((np.array(colors[i][:3]) * 255).astype(int).tolist())
        cv2.rectangle(
            canvas,
            (block["px_x"], block["px_y"]),
            (block["px_x"] + block["block_w"], block["px_y"] + block["block_h"]),
            color, 2,
        )
        cv2.putText(
            canvas, str(block["fid"]),
            (block["px_x"] + 2, block["px_y"] + block["block_h"] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Top {top_n} features, peak blocks (layer {layer})", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(CFG.cache_dir, f"vis_layer{layer}_top{top_n}.png")
    plt.savefig(save_path, dpi=150)
    print(f"saved: {save_path}")
    plt.show()


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="wide.png")
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--top_n", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading model & SAE (layer {args.layer})...")
    processor, model, sae = load_model_and_sae(args.layer)

    print(f"Extracting latents for: {args.image}")
    tokens, H_tok, W_tok = extract_latents(args.image, args.layer, processor, model)
    z = encode_with_sae(tokens, sae)

    print(f"z shape: {z.shape}, token grid: {H_tok}×{W_tok}")
    visualize_top_features(args.image, z, H_tok, W_tok,
                           top_n=args.top_n, layer=args.layer)


if __name__ == "__main__":
    main()