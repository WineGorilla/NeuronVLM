import json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


class VisionTextDataset(Dataset):
    """从 JSONL 文件加载图文对，每行格式：{"image": "...", "question": "...", "answer": "..."}"""

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
    """返回 DataLoader 用的 collate_fn。

    包含 answer 时构建完整对话（user + assistant），用于训练。
    没有 answer 时只构建 user 部分，用于推理。
    """

    def collate(batch):
        messages = []
        for x in batch:
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": x["image"]},
                        {"type": "text",  "text": x["question"]},
                    ],
                }
            ]
            # 有 answer 则拼入 assistant 回答（训练时需要）
            if x.get("answer"):
                msg.append({
                    "role": "assistant",
                    "content": x["answer"],
                })
            messages.append(msg)

        # 训练时 add_generation_prompt=False（answer 已在 messages 里）
        # 推理时 add_generation_prompt=True（让模型续写）
        has_answer = all(x.get("answer") for x in batch)
        texts = [
            processor.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=not has_answer,
            )
            for m in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    return collate