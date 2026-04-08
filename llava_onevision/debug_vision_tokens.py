"""
调试脚本：检查 LLaVA-OV 的 vision token 定位是否正确。
用法：python llava/debug_vision_tokens.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from config import CFG


def main():
    print("Loading LLaVA-OneVision...")
    processor = AutoProcessor.from_pretrained(CFG.llava_model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        CFG.llava_model_id, torch_dtype=torch.float16,
    ).to(CFG.device)
    model.eval()

    # ── 随便找一张图 ──
    test_image_path = "data/images/train2014/COCO_train2014_000000393223.jpg"
    for d in ["data", "images", "."]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    test_image_path = os.path.join(d, f)
                    break
        if test_image_path:
            break

    if test_image_path is None:
        print("找不到测试图片，请手动指定路径")
        return

    print(f"Test image: {test_image_path}")
    image = Image.open(test_image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # ── 构建输入 ──
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What is in this image?"},
        ]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(CFG.device)

    input_ids = inputs["input_ids"][0]
    print(f"\n{'='*60}")
    print(f"input_ids length: {len(input_ids)}")

    # ── 找 image token ──
    image_token_id = getattr(
        model.config, "image_token_index",
        processor.tokenizer.convert_tokens_to_ids("<image>"),
    )
    print(f"image_token_id: {image_token_id}")

    n_placeholder = (input_ids == image_token_id).sum().item()
    placeholder_positions = (input_ids == image_token_id).nonzero(as_tuple=False).squeeze(-1)
    v_pos = placeholder_positions[0].item() if len(placeholder_positions) > 0 else -1
    print(f"<image> placeholder 数量: {n_placeholder}")
    print(f"<image> 起始位置 (v_pos): {v_pos}")

    # ── Forward，拿 hidden_states ──
    print(f"\nRunning forward...")
    with torch.no_grad():
        outputs = model(
            **{k: v for k, v in inputs.items() if torch.is_tensor(v)},
            output_hidden_states=True,
            return_dict=True,
        )

    hs_seq_len = outputs.hidden_states[0].shape[1]
    print(f"\n{'='*60}")
    print(f"input_ids length  : {len(input_ids)}")
    print(f"hidden_states seq : {hs_seq_len}")
    print(f"差值 (seq - ids)  : {hs_seq_len - len(input_ids)}")
    print(f"placeholder 数量  : {n_placeholder}")
    print(f"{'='*60}")

    if hs_seq_len != len(input_ids):
        print(f"\n★★★ 问题确认 ★★★")
        print(f"hidden_states 的 seq_len ({hs_seq_len}) != input_ids 长度 ({len(input_ids)})")
        print(f"说明 LLaVA-OV 在 forward 中把 {n_placeholder} 个 <image> placeholder")
        actual_n_img = hs_seq_len - len(input_ids) + n_placeholder
        print(f"展开成了 {actual_n_img} 个 vision tokens")
        print(f"\n当前 SAE 训练代码用 n_img={n_placeholder} 去切 hidden_states，")
        print(f"实际应该用 n_img={actual_n_img}，切到的内容完全错位！")
    else:
        print(f"\n✓ seq_len 一致，vision token 定位应该没问题")

    # ── 打印各层 hidden_states shape ──
    print(f"\n各层 hidden_states shape:")
    for i, hs in enumerate(outputs.hidden_states[:3]):
        print(f"  layer {i}: {hs.shape}")
    print(f"  ...")
    print(f"  layer {len(outputs.hidden_states)-1}: {outputs.hidden_states[-1].shape}")


if __name__ == "__main__":
    main()