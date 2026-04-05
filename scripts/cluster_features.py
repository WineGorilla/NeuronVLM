"""
对 SAE feature 标签做聚类，生成 feature -> 类别 的映射。

流程：
    1. 加载 feature_labels_layer{N}.json
    2. 用 sentence-transformers 把标签编码成向量
    3. K-Means 聚类
    4. 用 GPT/Claude 给每个类别起名
    5. 保存 feature_clusters_layer{N}.json

输出格式：
    {
        "clusters": {
            "0": {"name": "animals", "features": [1024, 2731, ...]},
            "1": {"name": "color texture", "features": [3821, ...]},
            ...
        },
        "feature_to_cluster": {
            "1024": 0,
            "2731": 0,
            "3821": 1,
            ...
        }
    }

用法：
    python scripts/cluster_features.py --layer 8 --n_clusters 128
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
import numpy as np
import anthropic
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

from config import CFG


def load_feature_labels(layer: int) -> dict:
    label_path = os.path.join(CFG.label_dir, f"feature_labels_layer{layer}.json")
    with open(label_path) as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    print(f"Loaded {len(labels)} feature labels")
    return labels


def encode_labels(labels: dict, model_name: str = "all-MiniLM-L6-v2"):
    """用 sentence-transformers 把标签编码成向量。"""
    print(f"Encoding labels with {model_name}...")
    encoder   = SentenceTransformer(model_name)
    fids      = list(labels.keys())
    texts     = [labels[fid] for fid in fids]
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = normalize(embeddings)   # L2 归一化
    print(f"Embeddings shape: {embeddings.shape}")
    return fids, embeddings


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, seed: int = 42):
    """K-Means 聚类。"""
    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    print(f"Clustering done.")
    return labels


def name_cluster_with_claude(cluster_labels: list, cluster_id: int) -> str:
    """
    给每个 cluster 里的 feature 标签列表，让 Claude 起一个简短的类别名。
    """
    client = anthropic.Anthropic()

    # 取最多 20 个标签展示给 Claude
    sample = cluster_labels[:20]
    prompt = (
        f"These are semantic labels of visual features detected by a SAE "
        f"(Sparse Autoencoder) in a vision-language model:\n\n"
        f"{chr(10).join(f'  - {l}' for l in sample)}\n\n"
        f"What is the common visual concept or category these features share?\n"
        f"Give a concise category name of 1-3 words only, no explanation."
    )

    try:
        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 16,
            messages   = [{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  [warn] Claude failed for cluster {cluster_id}: {e}")
        return f"cluster_{cluster_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",      type=int, default=CFG.vis_layer)
    parser.add_argument("--n_clusters", type=int, default=128,
                        help="聚类数量")
    parser.add_argument("--encoder",    type=str,
                        default="all-MiniLM-L6-v2",
                        help="sentence-transformers 模型名")
    parser.add_argument("--no_name",    action="store_true",
                        help="跳过 Claude 命名，直接用 cluster_N")
    args = parser.parse_args()

    os.makedirs(CFG.label_dir, exist_ok=True)

    # 加载标签
    feature_labels = load_feature_labels(args.layer)

    # 编码
    fids, embeddings = encode_labels(feature_labels, args.encoder)

    # 聚类
    cluster_assignments = cluster_embeddings(embeddings, args.n_clusters)

    # 整理结果：cluster_id -> [fid, ...]
    cluster_to_features = {}
    for fid, cluster_id in zip(fids, cluster_assignments):
        cid = int(cluster_id)
        if cid not in cluster_to_features:
            cluster_to_features[cid] = []
        cluster_to_features[cid].append(fid)

    # 给每个 cluster 命名
    print(f"\nNaming {args.n_clusters} clusters...")
    cluster_names = {}
    for cid in sorted(cluster_to_features.keys()):
        fid_list    = cluster_to_features[cid]
        label_list  = [feature_labels[fid] for fid in fid_list]

        # 按频率排序，常见的标签排前面
        label_list.sort()

        if args.no_name:
            name = f"cluster_{cid}"
        else:
            name = name_cluster_with_claude(label_list, cid)

        cluster_names[cid] = name
        print(f"  cluster {cid:2d} ({len(fid_list):4d} features): {name}")
        print(f"    sample: {label_list[:5]}")

    # 构建输出
    clusters = {}
    for cid in sorted(cluster_to_features.keys()):
        clusters[str(cid)] = {
            "name":     cluster_names[cid],
            "features": cluster_to_features[cid],
        }

    feature_to_cluster = {}
    for fid, cluster_id in zip(fids, cluster_assignments):
        feature_to_cluster[str(fid)] = int(cluster_id)

    result = {
        "n_clusters":        args.n_clusters,
        "layer":             args.layer,
        "clusters":          clusters,
        "feature_to_cluster": feature_to_cluster,
    }

    # 保存
    save_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {save_path}")
    print(f"  Total features clustered: {len(feature_to_cluster)}")
    print(f"  Clusters: {args.n_clusters}")
    print(f"\nCluster summary:")
    for cid in sorted(cluster_to_features.keys()):
        name = cluster_names[cid]
        size = len(cluster_to_features[cid])
        print(f"  [{cid:2d}] {name:<30} ({size} features)")


if __name__ == "__main__":
    main()