"""
构建含有 target_features 的监督学习数据集。

候选 feature 策略：
    所有有语义标签且在该图片中有激活（> 0）的 feature 全部作为候选，
    结合图片内容，让 Claude 像人一样根据问题语义判断哪些 feature 有助于回答问题。

输出格式：
    {
        "image":    "...",
        "question": "...",
        "answer":   "...",
        "target_features": {
            "8":  [1024, 3821],
            "24": [2731, 5012]
        }
    }

用法：
    python scripts/build_feature_data.py --max 100
    python scripts/build_feature_data.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import argparse
import base64
import time

import cv2
import numpy as np
import anthropic

from config import CFG


# ── 加载函数 ──────────────────────────────────────────────────────────────────

def load_cache(layer: int) -> dict:
    cache_path = os.path.join(CFG.cache_dir, f"cache_layer{layer}.pkl")
    assert os.path.exists(cache_path), \
        f"Cache not found: {cache_path}\n" \
        f"Run `python scripts/build_cache.py --layer {layer}` first."
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    import scipy.sparse as sp
    result = {item["image_path"]: item["z"].toarray() if sp.issparse(item["z"]) else item["z"] for item in cache}
    print(f"  [layer {layer}] cache loaded: {len(result)} images")
    return result


def load_feature_labels(layer: int) -> dict:
    label_path = os.path.join(CFG.label_dir, f"feature_labels_layer{layer}.json")
    assert os.path.exists(label_path), \
        f"Feature labels not found: {label_path}\n" \
        f"Run `python scripts/interpret.py --layer {layer}` first."
    with open(label_path) as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    print(f"  [layer {layer}] feature labels loaded: {len(labels)}")
    return labels


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


# ── Feature 候选 ──────────────────────────────────────────────────────────────

def get_candidates(
    z:              np.ndarray,
    feature_labels: dict,
    max_candidates: int = 100,
) -> list:
    """
    返回所有有语义标签且在该图片中有激活（> 0）的 feature。
    超过 max_candidates 时取激活最强的前 max_candidates 个。

    Returns:
        list of (feature_id, label)，按激活值降序排列
    """
    mean_acts = z.mean(axis=0)

    results = []
    for fid, label in feature_labels.items():
        if mean_acts[fid] > 0:
            results.append((int(fid), label, float(mean_acts[fid])))

    results.sort(key=lambda x: x[2], reverse=True)
    results = results[:max_candidates]

    return [(fid, label) for fid, label, _ in results]


# ── Claude 选 feature ─────────────────────────────────────────────────────────

def select_features_with_claude(
    question:   str,
    answer:     str,
    candidates: list,
    layer:      int,
    image_path: str,
    client:     anthropic.Anthropic,
    retries:    int = 3,
) -> list:
    """
    给 Claude 看图片 + 问题 + 候选 feature，
    让它像人一样判断哪些 feature 有助于回答这个问题。

    Returns:
        list of feature_id 或 []
    """
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
    parser.add_argument("--max_candidates", type=int, default=100,
                        help="传给 Claude 的最大候选 feature 数量")
    parser.add_argument("--max",            type=int, default=None,
                        help="最多处理的样本数量")
    parser.add_argument("--input",          type=str, default=CFG.train_file)
    parser.add_argument("--output",         type=str,
                        default="data/train_supervised.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading data for layers: {CFG.layers}")
    caches             = {}
    feature_labels_all = {}
    for l in CFG.layers:
        caches[l]             = load_cache(l)
        feature_labels_all[l] = load_feature_labels(l)

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

            # 对每一层分别找 target_features
            target_features = {}

            for l in CFG.layers:
                if image_path not in caches[l]:
                    continue

                candidates = get_candidates(
                    caches[l][image_path],
                    feature_labels_all[l],
                    max_candidates = args.max_candidates,
                )
                if not candidates:
                    continue

                selected = select_features_with_claude(
                    question, answer, candidates, l, image_path, client
                )
                if selected:
                    target_features[str(l)] = selected

            if not target_features:
                skipped += 1
                continue

            result = {
                "image":           image_path,
                "question":        question,
                "answer":          answer,
                "target_features": target_features,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

            if count % 10 == 0:
                out.flush()

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(samples)}] done={count} skipped={skipped}")
                print(f"    Q: {question}")
                for l, fids in target_features.items():
                    labels = [
                        feature_labels_all[int(l)].get(fid, "?")
                        for fid in fids
                    ]
                    print(f"    layer {l}: {list(zip(fids, labels))}")

    print(f"\nDone. wrote {count} samples -> {args.output}")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()