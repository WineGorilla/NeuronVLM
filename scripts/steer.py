"""
通过 SAE latent space 进行 Feature Steering。

流程：
  1. 用问题关键词匹配 feature_labels，找到相关 feature
  2. 正常前向到第 N 层，得到 image token 的 hidden state h
  3. SAE 编码得到 z，增强相关 feature 的激活值
  4. 用残差叠加：steered_h = h + (decoder(z_steered) - decoder(z))
  5. 通过 hook 把 steered_h 注入模型，继续推理生成回答
  6. 对比原始输出 vs steered 输出

用法：
    python scripts/steer.py \
        --image  data/images/train2014/COCO_train2014_000000000009.jpg \
        --question "What animals are in the image?" \
        --layer 8 \
        --alpha 3.0

    # 指定特定 feature id（跳过自动匹配）
    python scripts/steer.py \
        --image  data/images/train2014/COCO_train2014_000000000009.jpg \
        --question "Describe the image" \
        --layer 8 \
        --feature_ids 1024 2731 \
        --alpha 3.0

    # 抑制某个概念（alpha < 0）
    python scripts/steer.py \
        --image  data/images/train2014/COCO_train2014_000000000009.jpg \
        --question "Describe the image" \
        --layer 8 \
        --feature_ids 1024 \
        --alpha -2.0
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import CFG
from src.model import SAE


# ── 模型加载 ──────────────────────────────────────────────────────────────────

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


# ── Feature 匹配：用问题关键词匹配 feature label ──────────────────────────────

def find_related_features(question: str, label_path: str,
                          top_n: int = 5) -> List[Tuple[int, str]]:
    """
    用问题里的关键词去匹配 feature_labels，返回最相关的 top_n 个 feature。

    Returns:
        list of (feature_id, label)
    """
    if not os.path.exists(label_path):
        print(f"[warn] label file not found: {label_path}")
        return []

    with open(label_path) as f:
        labels = json.load(f)

    question_words = set(question.lower().split())
    # 去掉常见停用词
    stopwords = {"what", "is", "are", "the", "a", "an", "in", "on", "of",
                 "and", "or", "this", "that", "there", "do", "does", "how",
                 "why", "where", "which", "describe", "image", "picture"}
    keywords = question_words - stopwords

    scored = []
    for fid, label in labels.items():
        label_words = set(label.lower().split())
        score = len(keywords & label_words)
        if score > 0:
            scored.append((int(fid), label, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    results = [(fid, label) for fid, label, _ in scored[:top_n]]

    return results


# ── 构建模型输入 ──────────────────────────────────────────────────────────────

def build_inputs(image_path: str, question: str, processor):
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": question},
        ],
    }]]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts, images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(CFG.device)
    return inputs


# ── 获取 image token 的位置信息 ───────────────────────────────────────────────

def get_image_token_info(inputs, model, processor):
    image_grid    = inputs["image_grid_thw"]
    H_grid        = image_grid[0, 1].item()
    W_grid        = image_grid[0, 2].item()
    spatial_merge = model.config.vision_config.spatial_merge_size
    num_img_tokens = int(H_grid * W_grid / (spatial_merge ** 2))

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    input_ids   = inputs["input_ids"][0]
    vision_pos  = (input_ids == vision_start_id).nonzero()[0].item() + 1

    return vision_pos, num_img_tokens


# ── 核心：残差叠加 steering ───────────────────────────────────────────────────

def make_steering_hook(sae, vision_pos: int, num_img_tokens: int,
                       feature_ids: List[int], alpha: float, layer: int):
    """
    返回一个 forward hook，在指定层的 hidden state 上做残差 steering。

    steered_h = h + (decoder(z_steered) - decoder(z_original))
    """
    hook_fired = [False]

    def hook(module, input, output):
        if hook_fired[0]:
            return output

        # output 可能是 tuple，取 hidden state
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        # 只操作 image tokens
        img_h = h[:, vision_pos : vision_pos + num_img_tokens, :]
        flat  = img_h.reshape(-1, img_h.shape[-1]).float()

        with torch.no_grad():
            # 原始编码
            z_original = sae.encode(flat)

            # 增强指定 feature
            z_steered = z_original.clone()
            for fid in feature_ids:
                if fid < z_steered.shape[-1]:
                    z_steered[:, fid] *= alpha

            # 残差：只叠加被改变的部分
            original_recon = sae.decoder(z_original)
            steered_recon  = sae.decoder(z_steered)
            delta = (steered_recon - original_recon).to(h.dtype)

        # 叠加 delta 到原始 hidden state
        h_new = h.clone()
        h_new[:, vision_pos : vision_pos + num_img_tokens, :] += delta

        hook_fired[0] = True

        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


# ── 生成回答 ──────────────────────────────────────────────────────────────────

def generate_answer(model, processor, inputs, max_new_tokens: int = 256) -> str:
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # 只解码新生成的部分
    input_len = inputs["input_ids"].shape[1]
    new_ids   = output_ids[:, input_len:]
    return processor.decode(new_ids[0], skip_special_tokens=True)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True,  help="图片路径")
    parser.add_argument("--question",    required=True,  help="输入问题")
    parser.add_argument("--layer",       type=int, default=CFG.vis_layer)
    parser.add_argument("--alpha",       type=float, default=3.0,
                        help="steering 强度，>1 增强，<0 抑制")
    parser.add_argument("--feature_ids", type=int, nargs="+", default=None,
                        help="手动指定 feature id，不指定则自动从问题匹配")
    parser.add_argument("--top_n",       type=int, default=5,
                        help="自动匹配时取 top_n 个 feature")
    parser.add_argument("--max_tokens",  type=int, default=256)
    args = parser.parse_args()

    print(f"Loading model & SAE (layer {args.layer})...")
    processor, model, sae = load_model_and_sae(args.layer)

    # ── 确定要 steer 的 feature ids ──────────────────────────────
    if args.feature_ids:
        feature_ids = args.feature_ids
        print(f"Using specified feature ids: {feature_ids}")
    else:
        label_path = os.path.join(
            CFG.label_dir, f"feature_labels_layer{args.layer}.json"
        )
        matched = find_related_features(args.question, label_path, top_n=args.top_n)
        if not matched:
            print("[warn] No matching features found, running without steering.")
            feature_ids = []
        else:
            feature_ids = [fid for fid, _ in matched]
            print(f"Matched features:")
            for fid, label in matched:
                print(f"  feature {fid:6d} -> {label}")

    # ── 构建输入 ──────────────────────────────────────────────────
    inputs = build_inputs(args.image, args.question, processor)
    vision_pos, num_img_tokens = get_image_token_info(inputs, model, processor)

    # ── 1. 原始输出（无 steering）────────────────────────────────
    print("\n" + "="*50)
    print("Original output:")
    original_answer = generate_answer(model, processor, inputs, args.max_tokens)
    print(original_answer)

    # ── 2. Steered 输出 ───────────────────────────────────────────
    if feature_ids:
        # 注册 hook 到指定层
        # Qwen2.5-VL 的层结构：model.model.layers[i]
        target_layer = model.model.layers[args.layer]
        hook = make_steering_hook(
            sae, vision_pos, num_img_tokens,
            feature_ids, args.alpha, args.layer
        )
        handle = target_layer.register_forward_hook(hook)

        print("\n" + "="*50)
        print(f"Steered output (alpha={args.alpha}, features={feature_ids}):")
        steered_answer = generate_answer(model, processor, inputs, args.max_tokens)
        print(steered_answer)

        handle.remove()

        # ── 对比 ──────────────────────────────────────────────────
        print("\n" + "="*50)
        print("Diff summary:")
        print(f"  Layer:    {args.layer}")
        print(f"  Features: {feature_ids}")
        print(f"  Alpha:    {args.alpha}")
        print(f"  Question: {args.question}")
        print(f"\n  [Original] {original_answer}")
        print(f"\n  [Steered]  {steered_answer}")


if __name__ == "__main__":
    main()