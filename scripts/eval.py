"""
测试 Router 效果：对比原始 Qwen 和加了 Router 的 Qwen 的回答。

用法：
    python scripts/eval.py \
        --image data/images/train2014/COCO_train2014_000000000009.jpg \
        --question "What animals are in the image?"
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import CFG
from src.Model import QwenWithSAERouter
from src.train_router import get_token_positions


def build_inputs(image_path, question, processor, device):
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
    ).to(device)
    return inputs


def generate_answer(model, inputs, processor, max_new_tokens=256):
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    new_ids   = output_ids[:, input_len:]
    return processor.decode(new_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--question",   required=True)
    parser.add_argument("--router_tag", default="best", help="router checkpoint tag")
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    router_save_dir = os.path.join(CFG.save_dir, "routers")

    print("Loading Qwen2.5-VL...")
    processor  = AutoProcessor.from_pretrained(CFG.model_id)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16,
    ).to(CFG.device)

    # ── 1. 原始 Qwen 回答 ──────────────────────────────────────
    inputs = build_inputs(args.image, args.question, processor, CFG.device)
    print("\n" + "="*50)
    print("Original Qwen:")
    original_answer = generate_answer(base_model, inputs, processor, args.max_tokens)
    print(original_answer)

    # ── 2. 加载 Router ─────────────────────────────────────────
    print("\nBuilding QwenWithSAERouter...")
    model = QwenWithSAERouter(
        base_model   = base_model,
        layers       = CFG.layers,
        sae_ckpt_dir = CFG.save_dir,
        latent_mult  = CFG.latent_mult,
        topk         = CFG.topk,
        topk_route   = 64,
        max_alpha    = 3.0,
    ).to(CFG.device)
    model.load_routers(router_save_dir, tag=args.router_tag)

    # ── 3. Router 增强后的回答 ─────────────────────────────────
    inputs = build_inputs(args.image, args.question, processor, CFG.device)
    vision_pos, num_img_tokens, text_positions = get_token_positions(
        inputs, model, processor
    )
    model.set_context(vision_pos, num_img_tokens, text_positions)

    print("\n" + "="*50)
    print(f"Router-enhanced Qwen (tag={args.router_tag}):")
    router_answer = generate_answer(model, inputs, processor, args.max_tokens)
    print(router_answer)
    model.clear_context()

    # ── 4. 对比 ────────────────────────────────────────────────
    print("\n" + "="*50)
    print("Question :", args.question)
    print("Original :", original_answer)
    print("Router   :", router_answer)


if __name__ == "__main__":
    main()