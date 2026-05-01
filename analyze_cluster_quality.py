"""
聚类质量分析：计算不同 K 下的 Silhouette Score。
用于验证 NeuronEye SNS 中 K=64 的聚类质量最优。

用法：
    python analyze_cluster_quality.py \
        --labels feature_labels_layer8.json \
        --clusters assets/feature_clusters_layer8_cluster32.json \
                    assets/feature_clusters_layer8_cluster64.json \
                    assets/feature_clusters_layer8_cluster128.json \
        --encoder all-MiniLM-L6-v2

输出：
    1. 终端打印 Silhouette Score 对比表
    2. 保存 LaTeX 表格片段 (cluster_silhouette.tex)
    3. 保存 bar chart (cluster_silhouette.pdf)
"""

import json
import argparse
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_labels(path: str) -> dict:
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_clusters(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # 返回 feature_to_cluster 映射 (int -> int)
    return {int(k): int(v) for k, v in data["feature_to_cluster"].items()}, data["n_clusters"]


def encode_labels(labels: dict, model_name: str):
    encoder = SentenceTransformer(model_name)
    fids = sorted(labels.keys())
    texts = [labels[fid] for fid in fids]
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = normalize(embeddings)  # L2 归一化，与聚类时一致
    return fids, embeddings


def compute_silhouette(fids, embeddings, f2c):
    """
    对齐 fids 和 cluster assignments，计算 silhouette score。
    用 cosine 距离，与聚类时的归一化嵌入一致。
    """
    # 只保留有 cluster assignment 的 features
    valid_idx = [i for i, fid in enumerate(fids) if fid in f2c]
    valid_emb = embeddings[valid_idx]
    valid_labels = np.array([f2c[fids[i]] for i in valid_idx])

    score = silhouette_score(valid_emb, valid_labels, metric="cosine")
    sample_scores = silhouette_samples(valid_emb, valid_labels, metric="cosine")

    return score, sample_scores, valid_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=True,
                        help="feature_labels_layer8.json 路径")
    parser.add_argument("--clusters", type=str, nargs="+", required=True,
                        help="cluster 文件路径列表，如 *_cluster32.json *_cluster64.json *_cluster128.json")
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    # 1. 加载并编码标签（只需做一次）
    print("Loading feature labels...")
    labels = load_labels(args.labels)
    print(f"  {len(labels)} features loaded")

    print("Encoding labels...")
    fids, embeddings = encode_labels(labels, args.encoder)
    print(f"  Embeddings shape: {embeddings.shape}")

    # 2. 对每个 K 计算 Silhouette Score
    results = []
    for cluster_path in args.clusters:
        print(f"\nProcessing: {cluster_path}")
        f2c, n_clusters = load_clusters(cluster_path)
        score, sample_scores, valid_labels = compute_silhouette(fids, embeddings, f2c)

        # 每个 cluster 的平均 silhouette
        per_cluster = {}
        for cid in range(n_clusters):
            mask = valid_labels == cid
            if mask.sum() > 0:
                per_cluster[cid] = sample_scores[mask].mean()

        cluster_sizes = np.bincount(valid_labels, minlength=n_clusters)

        results.append({
            "K": n_clusters,
            "silhouette": score,
            "per_cluster": per_cluster,
            "cluster_sizes": cluster_sizes,
            "n_features": len(valid_labels),
        })

        print(f"  K={n_clusters}: Silhouette Score = {score:.4f}")
        print(f"  Cluster size: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
              f"mean={cluster_sizes.mean():.1f}, std={cluster_sizes.std():.1f}")

    # 3. 打印对比表
    print("\n" + "=" * 50)
    print("Silhouette Score Comparison")
    print("=" * 50)
    print(f"{'K':>6}  {'Silhouette':>12}  {'#Features':>10}  {'Size min':>9}  {'Size max':>9}  {'Size std':>9}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["K"]):
        print(f"{r['K']:>6}  {r['silhouette']:>12.4f}  {r['n_features']:>10}  "
              f"{r['cluster_sizes'].min():>9}  {r['cluster_sizes'].max():>9}  "
              f"{r['cluster_sizes'].std():>9.1f}")

    # 4. 生成 LaTeX 表格片段
    best_k = max(results, key=lambda x: x["silhouette"])["K"]
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Silhouette score for different numbers of neuron clusters.}")
    latex_lines.append(r"\label{tab:silhouette}")
    latex_lines.append(r"\begin{tabular}{cccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Clusters $K$ & Silhouette Score & Cluster Size (mean $\pm$ std) & \#Features \\")
    latex_lines.append(r"\midrule")
    for r in sorted(results, key=lambda x: x["K"]):
        k = r["K"]
        s = r["silhouette"]
        mean_sz = r["cluster_sizes"].mean()
        std_sz = r["cluster_sizes"].std()
        n = r["n_features"]
        bold = r"\textbf{" if k == best_k else ""
        bold_end = "}" if k == best_k else ""
        latex_lines.append(
            f"  {k} & {bold}{s:.4f}{bold_end} & "
            f"{mean_sz:.1f} $\\pm$ {std_sz:.1f} & {n} \\\\"
        )
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_str = "\n".join(latex_lines)
    tex_path = f"{args.output_dir}/cluster_silhouette.tex"
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"\nLaTeX table saved: {tex_path}")
    print(latex_str)

    # 5. 生成 bar chart
    fig, ax = plt.subplots(figsize=(4, 3))
    ks = [r["K"] for r in sorted(results, key=lambda x: x["K"])]
    scores = [r["silhouette"] for r in sorted(results, key=lambda x: x["K"])]
    colors = ["#4A90D9" if k != best_k else "#E74C3C" for k in ks]

    bars = ax.bar([str(k) for k in ks], scores, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Number of Clusters $K$", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Cluster Quality (Cosine Distance)", fontsize=12)

    # 在 bar 上标数值
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{score:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(scores) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fig_path = f"{args.output_dir}/cluster_silhouette.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()