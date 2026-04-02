"""
SAE latent 缓存构建 — LLaVA-OneVision 版。

不会去提取重复的图片，一张照片只提取一次。
只构建 SAE latent 缓存，不做标注。
使用数据集中的真实问题构建 prompt，与训练时保持一致。

用法：
    python llava/build_cache_llava.py --layer 8
    python llava/build_cache_llava.py --all
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import argparse

import scipy.sparse as sp
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import transformers

from config import CFG
from src.SAE import SAE

transformers.logging.set_verbosity_error()


def load_sae_model(layer: int):
    processor = AutoProcessor.from_pretrained(CFG.llava_model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        CFG.llava_model_id, torch_dtype=torch.float16
    ).to(CFG.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.text_config.hidden_size
    latent_dim = hidden_dim * CFG.latent_mult

    ckpt_path = os.path.join(CFG.save_llava_dir, f"sae_layer{layer}.pt")
    sae = SAE(hidden_dim, latent_dim, CFG.topk).float().to(CFG.device)
    sae.load_state_dict(torch.load(ckpt_path, map_location=CFG.device))
    sae.eval()

    return processor, model, sae


def forward_single(image_path: str, question: str, layer: int, processor, model, sae):
    """
    LLaVA-OV 版本的单图前向。

    与 Qwen 版的区别：
      - 用 PIL Image 加载图片
      - conversation 格式：{"type": "image"} 而非 {"type": "image", "image": path}
      - vision token 定位：找 image_token_index 而非 <|vision_start|>
      - 无需 process_vision_info()
    """
    image = Image.open(image_path).convert("RGB")

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=False
    )
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(CFG.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    h = outputs.hidden_states[layer + 1]

    # ── vision token 定位 ─────────────────────────────────────
    input_ids = inputs["input_ids"][0]
    image_token_id = getattr(
        model.config, "image_token_index",
        processor.tokenizer.convert_tokens_to_ids("<image>"),
    )
    vision_indices = (input_ids == image_token_id).nonzero(as_tuple=False).squeeze(-1)

    if len(vision_indices) == 0:
        raise ValueError("No vision tokens found")

    img_tokens = h[0, vision_indices, :]  # (num_vision, hidden_dim)
    flat = img_tokens.float()

    # ── 计算 H_tok, W_tok 用于后续可视化 ─────────────────────
    # LLaVA-OV 的 vision token 数量取决于图片分辨率和 anyres 策略
    # 这里用 sqrt 近似，如果有 image_sizes 可以更精确
    n_tokens = len(vision_indices)
    H_tok = W_tok = int(n_tokens ** 0.5)
    # 尝试从 image_sizes 推算更精确的 H/W
    if "image_sizes" in inputs:
        img_sizes = inputs["image_sizes"]
        if img_sizes is not None and len(img_sizes) > 0:
            orig_h, orig_w = img_sizes[0].tolist()
            aspect = orig_w / orig_h
            H_tok = max(1, int((n_tokens / aspect) ** 0.5))
            W_tok = max(1, n_tokens // H_tok)

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

    samples = []
    with open(CFG.train_file) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"[layer {layer}] caching {len(samples)} images...")
    cache = []
    seen_images = set()
    for i, sample in enumerate(samples):
        image_path = sample["image"]

        if image_path in seen_images:
            continue
        seen_images.add(image_path)

        question = sample.get("question") or "Describe the image."

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(samples)}] unique={len(seen_images)}")

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