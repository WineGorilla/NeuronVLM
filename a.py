import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images

model_id = "deepseek-ai/deepseek-vl2-small"

# 加载
processor = DeepseekVLV2Processor.from_pretrained(model_id)
tokenizer = processor.tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model = model.to(torch.bfloat16).cuda().eval()

# 修复 rope_scaling：清除 transformers 4.48 注入的假配置
config = model.language.config
if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
    print(f"  [fix] Clearing injected rope_scaling: {config.rope_scaling}")
    config.rope_scaling = None
    # 重新初始化所有 attention 层的 RoPE
    for layer in model.language.model.layers:
        attn = layer.self_attn
        attn.rotary_emb = type(attn.rotary_emb)(
            dim=attn.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
    print("  [fix] RoPE re-initialized.")

print(f"Model type: {type(model)}")

# 创建一张纯色测试图
img = Image.new("RGB", (256, 256), color=(128, 64, 200))
img.save("/tmp/test_img.png")

conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\nWhat color is this image?",
        "images": ["/tmp/test_img.png"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

pil_images = load_pil_images(conversation)
inputs = processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt="",
).to(model.device)

print(f"Input keys: {list(inputs.keys())}")
print(f"images_seq_mask shape: {inputs.images_seq_mask.shape}")
print(f"n_image_tokens: {inputs.images_seq_mask.sum().item()}")

embeds = model.prepare_inputs_embeds(**inputs)
print(f"embeds shape: {embeds.shape}")

outputs = model.language.generate(
    inputs_embeds=embeds,
    attention_mask=inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=32,
    do_sample=False,
    use_cache=True,
)
print(f"Response: {tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)}")