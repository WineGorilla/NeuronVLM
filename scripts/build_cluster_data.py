"""
构建含有 focus_clusters 的训练数据。
只用问题文本让 Claude 选择需要关注的 cluster，不传图片节省费用。

输出格式：
    {
        "image": "...",
        "question": "...",
        "answer": "...",
        "focus_clusters": [3, 7],
    }

用法：
    CUDA_VISIBLE_DEVICES=1 python scripts/build_cluster_data.py --layer 8 --max 4410 && CUDA_VISIBLE_DEVICES=1 python -m src.train_focus --stage 1 && CUDA_VISIBLE_DEVICES=1 python -m src.train_focus --stage 2 --resume best && CUDA_VISIBLE_DEVICES=1 python scripts/eval_all.py --mode enhanced --layer 4
    python data/build_cluster_data.py --layer 8
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
import time

import anthropic

from config import CFG


# ── 加载函数 ──────────────────────────────────────────────────────────────────

def load_cluster_info(layer: int) -> dict:
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{layer}.json")
    assert os.path.exists(cluster_path), \
        f"Cluster file not found: {cluster_path}\n" \
        f"Run `python scripts/cluster_features.py --layer {layer}` first."
    with open(cluster_path) as f:
        data = json.load(f)
    clusters = {int(k): v for k, v in data["clusters"].items()}
    print(f"  [layer {layer}] clusters loaded: {len(clusters)}")
    return clusters


def load_train_data(path: str) -> list:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Train data loaded: {len(samples)} samples")
    return samples


# ── Claude 选 cluster ─────────────────────────────────────────────────────────

def select_clusters_with_claude(
    question:           str,
    candidate_clusters: list,   # [(cid, name), ...]
    client:             anthropic.Anthropic,
    retries:            int = 3,
) -> list:
    """只传问题文本，让 Claude 选出最相关的 cluster。"""
    candidate_str = "\n".join([
        f"  {cid}: {name}"
        for cid, name in candidate_clusters
    ])

    prompt = (
        f"A visual question answering model needs to answer this question:\n\n"
        f"Question: {question}\n\n"
        f"The following visual concept clusters are available:\n\n"
        f"{candidate_str}\n\n"
        f"Which clusters are most relevant to answering this question?\n\n"
        f"Requirements:\n"
        f"- Select 1 to 3 cluster IDs most relevant to the question\n"
        f"- Think about what visual concepts the question is asking about\n"
        f"- Return ONLY a JSON array of integers, no explanation\n\n"
        f"Example: [3, 7]"
    )

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 32,
                messages   = [{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            cluster_ids   = json.loads(text)
            candidate_ids = {cid for cid, _ in candidate_clusters}
            cluster_ids   = [
                int(cid) for cid in cluster_ids
                if int(cid) in candidate_ids
            ]

            if len(cluster_ids) > 0:
                return cluster_ids

        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"    Claude error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(3)

    return []


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",  type=int, required=True)
    parser.add_argument("--max",    type=int, default=None)
    parser.add_argument("--input",  type=str, default=CFG.train_file)
    parser.add_argument("--output", type=str,
                        default="data/train_cluster.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading data for layer: {args.layer}")
    clusters = load_cluster_info(args.layer)

    # 所有 cluster 的列表，传给 Claude
    all_candidates = [
        (cid, info["name"])
        for cid, info in sorted(clusters.items())
    ]

    samples = load_train_data(args.input)
    if args.max:
        samples = samples[:args.max]

    client = anthropic.Anthropic()

    # 断点续跑（用 image + question 作为唯一 key）
    existing = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                item = json.loads(line)
                existing.add(item["image"] + item["question"])
        print(f"Resuming: {len(existing)} already done")

    count   = 0
    skipped = 0

    with open(args.output, "a") as out:
        for i, sample in enumerate(samples):
            image_path = sample["image"]
            question   = sample.get("question", "")
            answer     = sample.get("answer",   "")

            key = image_path + question
            if key in existing:
                continue
            if not os.path.exists(image_path):
                skipped += 1
                continue
            if not question or not answer:
                skipped += 1
                continue

            selected = select_clusters_with_claude(
                question, all_candidates, client
            )
            if not selected:
                skipped += 1
                continue

            result = {
                "image":          image_path,
                "question":       question,
                "answer":         answer,
                "focus_clusters": selected,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

            if count % 10 == 0:
                out.flush()

            if (i + 1) % 100 == 0:
                cluster_names = [clusters[cid]["name"] for cid in selected]
                print(f"  [{i+1}/{len(samples)}] done={count} skipped={skipped}")
                print(f"    Q: {question}")
                print(f"    focus: {list(zip(selected, cluster_names))}")

    print(f"\nDone. wrote {count} samples -> {args.output}")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()