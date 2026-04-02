"""
LLaVA-OneVision 版 Dataset + collate_fn。

与 Qwen 版的区别：
  - 不依赖 qwen_vl_utils.process_vision_info()
  - 用 PIL.Image 加载图片
  - conversation 格式：{"type": "image"} 不带 image 路径
  - processor(images=, text=) 而非 processor(images=, videos=, text=)
"""
import json
from PIL import Image
from torch.utils.data import Dataset


class VisionTextDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_collate(processor):
    """
    返回 DataLoader 用的 collate_fn。
    统一单轮对话：user(图片+问题) → assistant(答案)

    注意：LLaVA-OV 的 processor 只支持单图 batch_size=1，
    多图需要逐个处理。这里保持和 Qwen 版一样的 batch_size=1 约定。
    """
    def collate(batch):
        # batch_size=1 时 batch 只有一个元素
        x = batch[0]
        question = x["question"]
        answer   = x.get("answer", "")
        image_path = x["image"]

        # 加载图片
        image = Image.open(image_path).convert("RGB")

        # 构建 conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        if answer:
            conversation.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            })

        has_answer = bool(answer)
        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=not has_answer,
        )

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        return inputs

    return collate