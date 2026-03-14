"""
批量 SAE feature 语义标注流程：
  1. 缓存所有训练图像的 SAE latent（cache_layer{N}.pkl）
  2. 找所有有激活的 feature
  3. 对每个 feature，找 top-5 图像，生成 patch-masked 图
  4. 调用 Claude API 解读视觉语义，输出简短标签
  5. 断点续跑，结果保存到 feature_labels_layer{N}.json

用法：
    python scripts/interpret.py --layer 8
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import anthropic
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import CFG
from src.SAE import SAE
from src.utils import make_masked_image, image_to_base64


# ── 加载模型 ──────────────────────────────────────────────────────────────────

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


# ── 单图前向：返回 SAE 编码 + token 网格尺寸 ─────────────────────────────────

def forward_single(image_path: str, layer: int, processor, model, sae):
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
        z = sae.encode(tokens).cpu().numpy()

    return z, H_tok, W_tok


# ── Step 1：缓存所有图的 latent ───────────────────────────────────────────────

def build_cache(layer: int, processor, model, sae) -> list:
    cache_path = os.path.join(CFG.cache_dir, f"cache_layer{layer}.pkl")

    if os.path.exists(cache_path):
        print(f"loading cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"loaded {len(cache)} items.")
        return cache

    samples = []
    with open(CFG.train_file) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"caching {len(samples)} images...")
    cache = []
    for i, sample in enumerate(samples):
        image_path = sample["image"]
        print(f"  [{i+1}/{len(samples)}] {image_path}")
        z, H_tok, W_tok = forward_single(image_path, layer, processor, model, sae)
        cache.append({"image_path": image_path, "z": z, "H_tok": H_tok, "W_tok": W_tok})

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"saved: {cache_path}")
    return cache


# ── Step 2：调用 Claude 解读 feature ─────────────────────────────────────────

def interpret_feature_with_claude(top5_results: list, feature_id: int, layer: int) -> str:
    client  = anthropic.Anthropic()
    content = []
    for i, res in enumerate(top5_results):
        content.append({"type": "text", "text": f"Image {i+1} (score={res['image_score']:.3f}):"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": image_to_base64(res["masked_img"]),
        }})
    content.append({"type": "text", "text": (
        f"These are the top-5 images most strongly activating SAE feature {feature_id} "
        f"(layer {layer}) of a vision-language model. "
        "In each image, only the patches with the strongest activation are visible; the rest are blacked out. "
        "Based on the visible regions across all 5 images, what visual concept or semantic pattern does this feature likely represent? "
        "Please give a concise label (1-5 words) only, no explanation."
    )})
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=32,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text.strip()


# ── 调试用：单 feature 可视化 ─────────────────────────────────────────────────

def visualize_feature(feature_id: int, top5: list, layer: int):
    n = len(top5)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for i, res in enumerate(top5):
        axes[i].imshow(cv2.cvtColor(res["masked_img"], cv2.COLOR_BGR2RGB))
        axes[i].axis("off")
        axes[i].set_title(
            f"{os.path.basename(res['image_path'])}\nscore={res['image_score']:.3f}",
            fontsize=9,
        )
    plt.suptitle(f"Feature {feature_id} — Top {n} images (layer {layer})", fontsize=12)
    plt.tight_layout()
    plt.show()

    # 保存拼接图
    target_h    = 400
    concat_imgs = []
    for res in top5:
        img     = res["masked_img"]
        scale   = target_h / img.shape[0]
        resized = cv2.resize(img, (int(img.shape[1] * scale), target_h))

        # 添加白色边框
        bordered = cv2.copyMakeBorder(
            resized,
            5, 5, 5, 5,
            cv2.BORDER_CONSTANT,
            value=(255,255,255)
        )

        concat_imgs.append(bordered)
    combined  = np.concatenate(concat_imgs, axis=1)
    save_path = os.path.join(CFG.cache_dir, f"feature_{feature_id}_layer{layer}_top{n}.png")
    cv2.imwrite(save_path, combined)
    print(f"saved: {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--debug_feature", type=int, default=None,
                        help="只可视化单个 feature（不调用 Claude）")
    args = parser.parse_args()

    os.makedirs(CFG.label_dir, exist_ok=True)
    os.makedirs(CFG.cache_dir, exist_ok=True)

    print(f"Loading model & SAE (layer {args.layer})...")
    processor, model, sae = load_model_and_sae(args.layer)

    # Step 1：缓存
    cache = build_cache(args.layer, processor, model, sae)

    # 调试模式：只看单个 feature
    if args.debug_feature is not None:
        fid = args.debug_feature
        results = []
        for item in cache:
            masked_img, image_score = make_masked_image(
                item["image_path"], item["z"], fid,
                item["H_tok"], item["W_tok"], top_n=CFG.top_n_patches,
            )
            results.append({"image_path": item["image_path"],
                            "masked_img": masked_img, "image_score": image_score})
        results.sort(key=lambda x: x["image_score"], reverse=True)
        visualize_feature(fid, results[:CFG.top_n_images], args.layer)
        return

    # Step 2：遍历所有 active feature
    all_z              = np.concatenate([item["z"] for item in cache], axis=0)
    active_feature_ids = np.where(all_z.max(axis=0) > 0)[0]
    print(f"active features: {len(active_feature_ids)} / {all_z.shape[1]}")

    label_path = os.path.join(CFG.label_dir, f"feature_labels_layer{args.layer}.json")
    if os.path.exists(label_path):
        with open(label_path) as f:
            feature_label_dict = {int(k): v for k, v in json.load(f).items()}
        print(f"resuming from {len(feature_label_dict)} done features")
    else:
        feature_label_dict = {}

    for target_feature_id in active_feature_ids:
        if int(target_feature_id) in feature_label_dict:
            continue

        results = []
        for item in cache:
            masked_img, image_score = make_masked_image(
                item["image_path"], item["z"], target_feature_id,
                item["H_tok"], item["W_tok"], top_n=CFG.top_n_patches,
            )
            results.append({"image_path": item["image_path"],
                            "masked_img": masked_img, "image_score": image_score})

        results.sort(key=lambda x: x["image_score"], reverse=True)
        top5 = results[:CFG.top_n_images]

        if top5[0]["image_score"] == 0:
            continue

        label = interpret_feature_with_claude(top5, target_feature_id, args.layer)
        feature_label_dict[int(target_feature_id)] = label
        print(f"  feature {target_feature_id:6d} -> {label}")

        # 每 10 个保存一次（断点续跑）
        if len(feature_label_dict) % 10 == 0:
            with open(label_path, "w") as f:
                json.dump(feature_label_dict, f, indent=2, ensure_ascii=False)

    with open(label_path, "w") as f:
        json.dump(feature_label_dict, f, indent=2, ensure_ascii=False)
    print(f"saved {len(feature_label_dict)} features -> {label_path}")


if __name__ == "__main__":
    main()