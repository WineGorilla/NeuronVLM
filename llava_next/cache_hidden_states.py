"""
第一阶段：缓存 LLaVA-NeXT hidden states。
跑一遍数据集，把每张图在指定 layer 的 vision token hidden states
存为 .pt 文件，后续训练 SAE 时直接加载，跳过 LLaVA 前向。

用法：
    python -m llava_next.cache_hidden_states
    CUDA_VISIBLE_DEVICES=0 python -m llava_next.cache_hidden_states
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import transformers

from llava_next.config_llava import CFG
from llava_next.dataset_llava_next import VisionTextDataset, build_collate

transformers.logging.set_verbosity_error()

LLAVA_NEXT_MODEL_ID = getattr(CFG, "llava_next_model_id", "llava-hf/llama3-llava-next-8b-hf")
CACHE_DIR           = getattr(CFG, "llava_cache_dir", "outputs/llava_next")


class UniqueImageDataset(VisionTextDataset):
    """只保留每张图片的第一个 sample，去重。"""
    def __init__(self, path: str):
        super().__init__(path)
        seen = set()
        deduped = []
        for s in self.samples:
            if s["image"] not in seen:
                seen.add(s["image"])
                deduped.append(s)
        print(f"  Dataset dedup: {len(self.samples)} samples -> {len(deduped)} unique images")
        self.samples = deduped


def extract_vision_hidden(hidden_state, input_ids, image_token_id):
    """提取所有 vision token 的 hidden states。"""
    parts = []
    for b in range(hidden_state.shape[0]):
        mask = (input_ids[b] == image_token_id)
        if mask.any():
            parts.append(hidden_state[b][mask])
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def cache():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Loading LLaVA-NeXT: {LLAVA_NEXT_MODEL_ID}")
    processor = LlavaNextProcessor.from_pretrained(LLAVA_NEXT_MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_NEXT_MODEL_ID,
        torch_dtype=torch.float16,
    ).to(CFG.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    image_token_id = getattr(
        model.config, "image_token_index",
        processor.tokenizer.convert_tokens_to_ids("<image>"),
    )

    dataset = UniqueImageDataset(CFG.train_file)
    loader  = DataLoader(
        dataset,
        batch_size = 1,          # 缓存阶段 bs=1 最稳妥
        shuffle    = False,
        collate_fn = build_collate(processor),
    )

    print(f"Dataset size   : {len(dataset)} unique images")
    print(f"Layers to cache: {CFG.layers}")
    print(f"Cache dir      : {CACHE_DIR}")

    # 检查已有的缓存，支持断点续跑
    existing = set()
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".pt"):
            existing.add(f)
    print(f"Already cached : {len(existing)} files, will skip them")

    skipped = 0
    saved   = 0

    for idx, batch in enumerate(tqdm(loader, desc="Caching")):
        # 文件名用 idx 命名
        # 先检查是否所有 layer 都已缓存
        all_cached = all(
            f"hidden_layer{l}_{idx:06d}.pt" in existing
            for l in CFG.layers
        )
        if all_cached:
            skipped += 1
            continue

        batch = {
            k: v.to(CFG.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            outputs = model(
                **{k: v for k, v in batch.items() if torch.is_tensor(v)},
                output_hidden_states=True,
                return_dict=True,
            )

        input_ids = batch["input_ids"]

        for l in CFG.layers:
            fname = f"hidden_layer{l}_{idx:06d}.pt"
            if fname in existing:
                continue

            h = outputs.hidden_states[l + 1]
            img_tokens = extract_vision_hidden(h, input_ids, image_token_id)

            if img_tokens is None or img_tokens.shape[0] == 0:
                continue

            # 存为 float16 节省空间（训练时转 float32）
            save_path = os.path.join(CACHE_DIR, fname)
            torch.save(img_tokens.cpu().half(), save_path)

        saved += 1

    print(f"\nDone! saved={saved}, skipped(already cached)={skipped}")
    print(f"Cache dir: {CACHE_DIR}")


if __name__ == "__main__":
    cache()