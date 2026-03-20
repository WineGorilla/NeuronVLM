"""
构建含有 target_features 的监督学习数据集。

使用 feature_index 反转查询，速度比 cache 快很多。

输出格式：
    {
        "image":    "...",
        "question": "...",
        "answer":   "...",
        "target_features": {
            "8":  [1024, 3821]
        }
    }

用法：
    python data/build_feature_data.py --layer 8 --max 100
    python data/build_feature_data.py --layer 8
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import argparse
import base64
import time
from collections import defaultdict

import cv2
import anthropic

from config import CFG


# ── 加载函数 ──────────────────────────────────────────────────────────────────

def load_feature_index(layer: int) -> dict:
    index_path = os.path.join(CFG.cache_dir, f"feature_index_layer{layer}.pkl")
    assert os.path.exists(index_path), \
        f"Feature index not found: {index_path}\n" \
        f"Run `python scripts/build_feature_index.py --layer {layer}` first."
    with open(index_path, "rb") as f:
        feature_index = pickle.load(f)
    print(f"  [layer {layer}] feature index loaded: {len(feature_index)} features")
    return feature_index


def load_feature_labels(layer: int) -> dict:
    label_path = os.path.join(CFG.label_dir, f"feature_labels_layer{layer}.json")
    assert os.path.exists(label_path), \
        f"Feature labels not found: {label_path}\n" \
        f"Run `python scripts/interpret.py --layer {layer}` first."
    with open(label_path) as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    print(f"  [layer {layer}] feature labels loaded: {len(labels)}")
    return labels


def build_image_to_features(feature_index: dict, feature_labels: dict) -> dict:
    """
    从 feature_index 反转，构建 image_path -> [(fid, score), ...] 的映射。
    只保留有语义标签的 feature。
    """
    image_to_features = defaultdict(list)
    for fid, entries in feature_index.items():
        if fid not in feature_labels:
            continue
        for entry in entries:
            score, image_path = entry[0], entry[1]
            image_to_features[image_path].append((fid, score))

    # 每张图的 feature 按分数降序排序
    for image_path in image_to_features:
        image_to_features[image_path].sort(key=lambda x: x[1], reverse=True)

    print(f"  image_to_features built: {len(image_to_features)} images")
    return dict(image_to_features)


def load_train_data(path: str) -> list:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Train data loaded: {len(samples)} samples from {path}")
    return samples


# ── 图片工具 ──────────────────────────────────────────────────────────────────

def image_to_base64(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


# ── Claude 选 feature ─────────────────────────────────────────────────────────

def select_features_with_claude(
    question:        str,
    answer:          str,
    candidates:      list,   # [(fid, label), ...]
    layer:           int,
    image_path:      str,
    client:          anthropic.Anthropic,
    retries:         int = 3,
) -> list:
    candidate_str = "\n".join([
        f"  {fid}: {label}"
        for fid, label in candidates
    ])

    prompt = (
        f"You are looking at an image. "
        f"A visual question answering model needs to answer this question:\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"The model's layer {layer} SAE has detected the following visual features "
        f"in this image (feature_id: semantic_label):\n\n"
        f"{candidate_str}\n\n"
        f"Looking at the image and the question, select which features the model "
        f"should focus on to answer correctly.\n\n"
        f"Requirements:\n"
        f"- Select 3 to 8 feature IDs most relevant to answering the question\n"
        f"- Use the image content AND the semantic labels to decide\n"
        f"- A feature is relevant if its concept appears in the image AND relates to the question\n"
        f"- Return ONLY a JSON array of integers, no explanation\n\n"
        f"Example: [1024, 3821, 7291]"
    )

    img_b64 = image_to_base64(image_path)

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 128,
                messages   = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type":       "base64",
                                "media_type": "image/jpeg",
                                "data":       img_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

            text = response.content[0].text.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            feature_ids   = json.loads(text)
            candidate_ids = {fid for fid, _ in candidates}
            feature_ids   = [
                int(fid) for fid in feature_ids
                if int(fid) in candidate_ids
            ]

            if len(feature_ids) > 0:
                return feature_ids

        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"    Claude error layer{layer} (attempt {attempt+1}/{retries}): {e}")
            time.sleep(3)

    return []


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",          type=int, required=True)
    parser.add_argument("--max_candidates", type=int, default=100)
    parser.add_argument("--max",            type=int, default=None)
    parser.add_argument("--input",          type=str, default=CFG.train_file)
    parser.add_argument("--output",         type=str,
                        default="data/train_supervised.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading data for layer: {args.layer}")
    feature_index  = load_feature_index(args.layer)
    feature_labels = load_feature_labels(args.layer)

    # 反转 index：image_path -> [(fid, score), ...]
    image_to_features = build_image_to_features(feature_index, feature_labels)

    samples = load_train_data(args.input)
    if args.max:
        samples = samples[:args.max]

    client = anthropic.Anthropic()

    # 断点续跑
    existing = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                item = json.loads(line)
                existing.add(item["image"])
        print(f"Resuming: {len(existing)} already done")

    count   = 0
    skipped = 0

    with open(args.output, "a") as out:
        for i, sample in enumerate(samples):
            image_path = sample["image"]
            question   = sample.get("question", "")
            answer     = sample.get("answer",   "")

            if image_path in existing:
                continue
            if not os.path.exists(image_path):
                skipped += 1
                continue
            if not question or not answer:
                skipped += 1
                continue

            # 直接从反转 index 查这张图的候选 feature
            if image_path not in image_to_features:
                skipped += 1
                continue

            fid_scores = image_to_features[image_path][:args.max_candidates]
            candidates = [
                (fid, feature_labels[fid])
                for fid, _ in fid_scores
                if fid in feature_labels
            ]

            if not candidates:
                skipped += 1
                continue

            selected = select_features_with_claude(
                question, answer, candidates, args.layer, image_path, client
            )
            if not selected:
                skipped += 1
                continue

            result = {
                "image":    image_path,
                "question": question,
                "answer":   answer,
                "target_features": {
                    str(args.layer): selected
                },
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

            if count % 10 == 0:
                out.flush()

            if (i + 1) % 50 == 0:
                labels = [feature_labels.get(fid, "?") for fid in selected]
                print(f"  [{i+1}/{len(samples)}] done={count} skipped={skipped}")
                print(f"    Q: {question}")
                print(f"    layer {args.layer}: {list(zip(selected, labels))}")

    print(f"\nDone. wrote {count} samples -> {args.output}")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()