"""
数据集 + collate_fn。

支持两种数据格式：

1. 普通格式（train.jsonl）：
   {"image", "question", "answer"}
   user:      图片 + 问题
   assistant: 答案

2. cluster 格式（train_cluster.jsonl）：
   {"image", "question", "answer", "focus_clusters": [2, 9]}
   训练时构建两轮对话：
     第一轮 user:      只有问题文字
     第一轮 assistant: <think>[2,9]</think>
     第二轮 user:      图片 + 问题
     第二轮 assistant: 答案
"""
import json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

THINK_START = "<think>"
THINK_END   = "</think>"


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
    cluster 格式构建两轮对话，普通格式构建单轮对话。
    """
    def collate(batch):
        messages = []
        for x in batch:
            focus_clusters = x.get("focus_clusters", None)
            question       = x["question"]
            answer         = x.get("answer", "")
            image          = x["image"]

            if focus_clusters:
                # 两轮对话
                # 第一轮：只有问题文字 → <think>[ids]</think>
                ids_str  = ",".join(str(c) for c in focus_clusters)
                think_str = f"{THINK_START}[{ids_str}]{THINK_END}"

                msg = [
                    {
                        "role":    "user",
                        "content": question,   # 纯文字，无图片
                    },
                    {
                        "role":    "assistant",
                        "content": think_str,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text",  "text":  question},
                        ],
                    },
                ]
                if answer:
                    msg.append({
                        "role":    "assistant",
                        "content": answer,
                    })
            else:
                # 单轮对话
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


def parse_think(text: str) -> list:
    """
    从生成文本里解析 cluster ids。
    输入：  "<think>[2,9]</think>"
    输出：  [2, 9]
    """
    if THINK_START not in text or THINK_END not in text:
        return []
    try:
        start   = text.index(THINK_START) + len(THINK_START)
        end     = text.index(THINK_END)
        content = text[start:end].strip().strip("[]")
        ids     = [int(x.strip()) for x in content.split(",") if x.strip()]
        return ids
    except Exception:
        return []


def strip_think(text: str) -> str:
    """
    去掉 <think>...</think>，只保留答案。
    输入：  "<think>[2,9]</think>\nThe dog is brown."
    输出：  "The dog is brown."
    """
    if THINK_START not in text:
        return text
    try:
        end    = text.index(THINK_END) + len(THINK_END)
        answer = text[end:].strip()
        return answer
    except Exception:
        return text