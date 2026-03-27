
import json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


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
    """
    def collate(batch):
        messages = []
        for x in batch:
            question = x["question"]
            answer   = x.get("answer", "")
            image    = x["image"]

            msg = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  question},
                ],
            }]
            if answer:
                msg.append({
                    "role":    "assistant",
                    "content": answer,
                })

            messages.append(msg)

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
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        return inputs

    return collate