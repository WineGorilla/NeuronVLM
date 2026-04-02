"""
不会去提取重复的图片，一张照片只提取一次

只构建 SAE latent 缓存，不做标注。
使用数据集中的真实问题构建 prompt，与训练时保持一致。

优化：用 hook 提取目标层后提前终止 forward，省掉后续层的计算。

用法：
    python scripts/build_cache.py --layer 8
    python scripts/build_cache.py --all
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import argparse

import scipy.sparse as sp
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import CFG
from src.SAE import SAE


def load_sae_model(layer: int):
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        CFG.model_id, min_pixels=min_pixels, max_pixels=max_pixels
    )
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


def _find_layers(model):
    for path_fn in [
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
    ]:
        try:
            layers = path_fn(model)
            if hasattr(layers, '__len__') and len(layers) > 0:
                return layers
        except AttributeError:
            pass
    raise AttributeError("Cannot find transformer layers")


def forward_single(image_path: str, question: str, layer: int, processor, model, sae, lm_layers):
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

    # ── 用 hook 提取目标层，提前终止 forward ──────────────────
    captured = {}

    def hook_fn(module, input, output):
        captured["hidden"] = output[0].detach()
        raise StopIteration

    handle = lm_layers[layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**inputs)
    except StopIteration:
        pass
    finally:
        handle.remove()

    h = captured["hidden"]

    input_ids      = inputs["input_ids"][0]
    image_token_id = model.config.image_token_id
    vision_indices = (input_ids == image_token_id).nonzero(as_tuple=False).squeeze(-1)

    if len(vision_indices) == 0:
        raise ValueError("No vision tokens found")

    img_tokens = h[0, vision_indices, :]
    flat = img_tokens.float()

    image_grid    = inputs["image_grid_thw"]
    H_grid        = image_grid[0, 1].item()
    W_grid        = image_grid[0, 2].item()
    spatial_merge = model.config.vision_config.spatial_merge_size
    H_tok         = int(H_grid // spatial_merge)
    W_tok         = int(W_grid // spatial_merge)

    with torch.no_grad():
        z = sae.encode(flat)
        z = z.cpu().numpy()

    return sp.csr_matrix(z), H_tok, W_tok


def build_cache(layer: int):
    cache_path = os.path.join(CFG.cache_dir, f"cache_layer{layer}.pkl")

    if os.path.exists(cache_path):
        print(f"[layer {layer}] cache already exists: {cache_path}, skipping.")
        return

    print(f"[layer {layer}] loading model & SAE...")
    processor, model, sae = load_sae_model(layer)
    lm_layers = _find_layers(model)
    print(f"[layer {layer}] found {len(lm_layers)} transformer layers, early stop at layer {layer}")

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

        if image_path in seen_images:
            continue
        seen_images.add(image_path)

        question = sample.get("question") or "Describe the image."

        try:
            z_sparse, H_tok, W_tok = forward_single(
                image_path, question, layer, processor, model, sae, lm_layers
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