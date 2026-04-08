"""
LLaVA-NeXT-LLaMA3 版 Dataset + collate_fn。

与 LLaVA-OV 版完全相同，接口一致：
  - processor(images=, text=) 格式相同
  - apply_chat_template 接口相同
  - 唯一区别是 processor 类型从 AutoProcessor 变为 LlavaNextProcessor，
    但 build_collate 不关心具体类型，所以代码完全一样。
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
    batch_size=1。
    """
    def collate(batch):
        x = batch[0]
        question   = x["question"]
        answer     = x.get("answer", "")
        image_path = x["image"]

        image = Image.open(image_path).convert("RGB")

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