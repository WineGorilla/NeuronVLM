# import json
# from torch.utils.data import Dataset
# from qwen_vl_utils import process_vision_info


# class VisionTextDataset(Dataset):
#     """从 JSONL 文件加载图文对，每行格式：{"image": "...", "question": "...", "answer": "..."}"""

#     def __init__(self, path: str):
#         self.samples = []
#         with open(path) as f:
#             for line in f:
#                 self.samples.append(json.loads(line))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]


# def build_collate(processor):
#     """返回 DataLoader 用的 collate_fn。

#     包含 answer 时构建完整对话（user + assistant），用于训练。
#     没有 answer 时只构建 user 部分，用于推理。
#     """

#     def collate(batch):
#         messages = []
#         for x in batch:
#             msg = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": x["image"]},
#                         {"type": "text",  "text": x["question"]},
#                     ],
#                 }
#             ]
#             # 有 answer 则拼入 assistant 回答（训练时需要）
#             if x.get("answer"):
#                 msg.append({
#                     "role": "assistant",
#                     "content": x["answer"],
#                 })
#             messages.append(msg)

#         # 训练时 add_generation_prompt=False（answer 已在 messages 里）
#         # 推理时 add_generation_prompt=True（让模型续写）
#         has_answer = all(x.get("answer") for x in batch)
#         texts = [
#             processor.apply_chat_template(
#                 m,
#                 tokenize=False,
#                 add_generation_prompt=not has_answer,
#             )
#             for m in messages
#         ]

#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=texts,
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         return inputs

#     return collate





"""
数据集 + collate_fn。

train.jsonl 格式：
    {"image": "...", "question": "...", "answer": "..."}

train_supervised.jsonl 格式（含 target_features）：
    {"image": "...", "question": "...", "answer": "...", "target_features": {"8": [...], "24": [...]}}
"""
import json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


class VisionTextDataset(Dataset):
    """从 JSONL 文件加载图文对。"""

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

    - 有 answer 时构建完整对话（user + assistant），用于训练
    - 没有 answer 时只构建 user 部分，用于推理
    - target_features 字段单独提取，不传给 processor
    """

    def collate(batch):
        # 单独提取 target_features，避免传给 processor 报错
        target_features_list = []
        for x in batch:
            tf = x.pop("target_features", None)
            # 兼容旧格式（列表而不是字典）
            if isinstance(tf, list):
                from config import CFG
                tf = {str(CFG.vis_layer): tf}
            target_features_list.append(tf)

        messages = []
        for x in batch:
            msg = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": x["image"]},
                    {"type": "text",  "text": x["question"]},
                ],
            }]
            if x.get("answer"):
                msg.append({
                    "role": "assistant",
                    "content": x["answer"],
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
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 把 target_features 挂回 inputs（只有监督数据才有）
        if any(t is not None for t in target_features_list):
            inputs["target_features"] = target_features_list

        return inputs

    return collate