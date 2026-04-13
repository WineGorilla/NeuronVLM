"""
只构建 SAE latent 缓存，不做标注。
使用数据集中的真实问题构建 prompt，与训练时保持一致。
不会去提取重复的图片，一张照片只提取一次。

用法：
    CUDA_VISIBLE_DEVICES=2 python scripts/build_cache_new.py --layer 8 && CUDA_VISIBLE_DEVICES=2 python scripts/build_feature_index_gpu.py --layer 8 --min_images 3 --batch_size 512
    python scripts/build_cache.py --layer 24
    python scripts/build_cache.py --all    # 构建所有层
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import argparse

import scipy.sparse as sp
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import CFG
from src.SAE import SAE


def load_sae_model(layer: int):
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


def forward_single(image_path: str, question: str, layer: int, processor, model, sae):
    """
    使用真实问题构建 prompt，与训练时保持一致。
    """
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image",  "image": image_path},
            {"type": "text",   "text":  question},
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

    h             = outputs.hidden_states[layer + 1]
    image_grid    = inputs["image_grid_thw"]
    H_grid        = image_grid[0, 1].item()
    W_grid        = image_grid[0, 2].item()
    spatial_merge = model.config.vision_config.spatial_merge_size
    num_img_tokens = int(H_grid * W_grid / (spatial_merge ** 2))
    H_tok         = int(H_grid // spatial_merge)
    W_tok         = int(W_grid  // spatial_merge)

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    input_ids   = inputs["input_ids"][0]
    vision_pos  = (input_ids == vision_start_id).nonzero()[0].item() + 1

    img_tokens = h[:, vision_pos : vision_pos + num_img_tokens, :]
    tokens     = img_tokens.reshape(-1, img_tokens.shape[-1]).float()

    with torch.no_grad():
        z = sae.encode(tokens)
        z = z.cpu().numpy()

    return sp.csr_matrix(z), H_tok, W_tok


def build_cache(layer: int):
    cache_path = os.path.join(CFG.cache_dir, f"cache_layer{layer}.pkl")

    if os.path.exists(cache_path):
        print(f"[layer {layer}] cache already exists: {cache_path}, skipping.")
        return

    print(f"[layer {layer}] loading model & SAE...")
    processor, model, sae = load_sae_model(layer)

    samples = []
    with open(CFG.train_file) as f:
        for line in f:
            samples.append(json.loads(line))

    unique_images = set(s["image"] for s in samples)
    print(f"[layer {layer}] total samples: {len(samples)}, unique images: {len(unique_images)}")

    cache = []
    seen_images = set()
    for i, sample in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(samples)}] unique={len(seen_images)}, cached={len(cache)}")

        image_path = sample["image"]

        # 按图片去重，每张图只用第一个 question
        if image_path in seen_images:
            continue
        seen_images.add(image_path)

        question = sample.get("question") or "Describe the image."

        try:
            z_sparse, H_tok, W_tok = forward_single(
                image_path, question, layer, processor, model, sae
            )
            cache.append({
                "image_path": image_path,
                "question":   question,
                "z":          z_sparse,
                "H_tok":      H_tok,
                "W_tok":      W_tok,
            })
        except Exception as e:
            print(f"  [skip] {image_path}: {e}")

    print(f"[layer {layer}] total samples: {len(samples)}, unique images: {len(seen_images)}, cached: {len(cache)}")

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[layer {layer}] saved: {cache_path} ({len(cache)} items)")

    del model, sae
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--layer", type=int, help="构建单层缓存")
    group.add_argument("--all",   action="store_true", help="构建所有层缓存")
    args = parser.parse_args()

    os.makedirs(CFG.cache_dir, exist_ok=True)

    layers = CFG.layers if args.all else [args.layer]
    for layer in layers:
        build_cache(layer)

    print("\nAll done. Next steps:")
    for layer in layers:
        print(f"  python scripts/build_feature_index.py --layer {layer}")
    for layer in layers:
        print(f"  python scripts/interpret.py --layer {layer}")


if __name__ == "__main__":
    main()