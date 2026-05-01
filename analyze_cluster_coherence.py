"""
聚类语义一致性分析：计算每个 cluster 内部的平均余弦相似度。

验证每个 neuron cluster 是否真的对应一个连贯的视觉概念。

用法：
    python analyze_cluster_coherence.py \
        --labels feature_labels_layer8.json \
        --clusters feature_clusters_layer8_cluster32.json \
                    feature_clusters_layer8_cluster64.json \
                    feature_clusters_layer8_cluster128.json

输出：
    1. 终端打印每个 K 的簇内平均余弦相似度
    2. 展示 top/bottom cluster 的名称和内部标签
    3. 保存 LaTeX 表格 (cluster_coherence.tex)
    4. 保存 box plot (cluster_coherence.pdf)
"""

import json
import argparse
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_labels(path: str) -> dict:
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_clusters(path: str):
    with open(path) as f:
        data = json.load(f)
    n_clusters = data["n_clusters"]
    clusters = {}
    for cid, info in data["clusters"].items():
        clusters[int(cid)] = {
            "name": info["name"],
            "features": [int(f) for f in info["features"]],
        }
    return clusters, n_clusters


def encode_labels(labels: dict, model_name: str):
    encoder = SentenceTransformer(model_name)
    fids = sorted(labels.keys())
    texts = [labels[fid] for fid in fids]
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = normalize(embeddings)
    fid_to_emb = {fid: embeddings[i] for i, fid in enumerate(fids)}
    return fid_to_emb


def compute_intra_cluster_similarity(clusters: dict, fid_to_emb: dict):
    """计算每个 cluster 内部的平均 pairwise 余弦相似度。"""
    results = []
    for cid in sorted(clusters.keys()):
        fids = [f for f in clusters[cid]["features"] if f in fid_to_emb]
        name = clusters[cid]["name"]

        if len(fids) < 2:
            results.append({
                "cid": cid, "name": name, "size": len(fids),
                "mean_sim": float("nan"), "std_sim": float("nan"),
            })
            continue

        embs = np.stack([fid_to_emb[f] for f in fids])
        sim_matrix = cosine_similarity(embs)

        # 取上三角（排除对角线）
        triu_idx = np.triu_indices(len(fids), k=1)
        pairwise_sims = sim_matrix[triu_idx]

        results.append({
            "cid": cid,
            "name": name,
            "size": len(fids),
            "mean_sim": pairwise_sims.mean(),
            "std_sim": pairwise_sims.std(),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--clusters", type=str, nargs="+", required=True)
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--top_n", type=int, default=5,
                        help="展示 top/bottom N 个 cluster 的详细标签")
    args = parser.parse_args()

    # 1. 编码标签（只做一次）
    print("Loading and encoding feature labels...")
    labels = load_labels(args.labels)
    fid_to_emb = encode_labels(labels, args.encoder)
    print(f"  {len(fid_to_emb)} features encoded\n")

    # 2. 对每个 K 分析
    all_results = []
    for cluster_path in args.clusters:
        print(f"{'=' * 60}")
        print(f"Processing: {cluster_path}")
        print(f"{'=' * 60}")
        clusters, n_clusters = load_clusters(cluster_path)
        per_cluster = compute_intra_cluster_similarity(clusters, fid_to_emb)

        valid = [r for r in per_cluster if not np.isnan(r["mean_sim"])]
        sims = [r["mean_sim"] for r in valid]
        overall_mean = np.mean(sims)
        overall_std = np.std(sims)

        print(f"\n  K={n_clusters}: Intra-cluster cosine similarity")
        print(f"    Overall: {overall_mean:.4f} ± {overall_std:.4f}")
        print(f"    Min cluster sim: {min(sims):.4f}")
        print(f"    Max cluster sim: {max(sims):.4f}")

        # Top N 最一致的 cluster
        sorted_by_sim = sorted(valid, key=lambda x: x["mean_sim"], reverse=True)
        print(f"\n  Top-{args.top_n} most coherent clusters:")
        for r in sorted_by_sim[:args.top_n]:
            cid = r["cid"]
            cluster_fids = clusters[cid]["features"]
            cluster_labels = [labels[f] for f in cluster_fids if f in labels][:8]
            print(f"    [{cid:2d}] {r['name']:<30} sim={r['mean_sim']:.4f}  "
                  f"size={r['size']}")
            print(f"         labels: {cluster_labels}")

        # Bottom N 最不一致的 cluster
        print(f"\n  Bottom-{args.top_n} least coherent clusters:")
        for r in sorted_by_sim[-args.top_n:]:
            cid = r["cid"]
            cluster_fids = clusters[cid]["features"]
            cluster_labels = [labels[f] for f in cluster_fids if f in labels][:8]
            print(f"    [{cid:2d}] {r['name']:<30} sim={r['mean_sim']:.4f}  "
                  f"size={r['size']}")
            print(f"         labels: {cluster_labels}")

        all_results.append({
            "K": n_clusters,
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "per_cluster": per_cluster,
            "sims": sims,
        })
        print()

    # 3. 打印对比总结
    print(f"\n{'=' * 60}")
    print("Summary: Intra-Cluster Semantic Coherence")
    print(f"{'=' * 60}")
    print(f"{'K':>6}  {'Mean Cosine Sim':>16}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: x["K"]):
        print(f"{r['K']:>6}  {r['overall_mean']:>16.4f}  {r['overall_std']:>8.4f}  "
              f"{min(r['sims']):>8.4f}  {max(r['sims']):>8.4f}")

    # 4. LaTeX 表格
    best_k = max(all_results, key=lambda x: x["overall_mean"])["K"]
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Intra-cluster semantic coherence measured by average pairwise cosine similarity of neuron label embeddings.}")
    latex_lines.append(r"\label{tab:coherence}")
    latex_lines.append(r"\begin{tabular}{ccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Clusters $K$ & Mean Intra-Cluster Similarity & Std \\")
    latex_lines.append(r"\midrule")
    for r in sorted(all_results, key=lambda x: x["K"]):
        k = r["K"]
        bold = r"\textbf{" if k == best_k else ""
        bold_end = "}" if k == best_k else ""
        latex_lines.append(
            f"  {k} & {bold}{r['overall_mean']:.4f}{bold_end} & {r['overall_std']:.4f} \\\\"
        )
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_str = "\n".join(latex_lines)
    tex_path = f"{args.output_dir}/cluster_coherence.tex"
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"\nLaTeX table saved: {tex_path}")
    print(latex_str)

    # 5. Box plot: 每个 K 的簇内相似度分布
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    data_for_box = [r["sims"] for r in sorted(all_results, key=lambda x: x["K"])]
    k_labels = [f"K={r['K']}" for r in sorted(all_results, key=lambda x: x["K"])]

    bp = ax.boxplot(data_for_box, labels=k_labels, patch_artist=True,
                    widths=0.4, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=5))

    colors = ["#7FB3D8", "#4A90D9", "#2C5F8A"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Intra-Cluster Cosine Similarity", fontsize=11)
    ax.set_title("Semantic Coherence by Cluster Granularity", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fig_path = f"{args.output_dir}/cluster_coherence.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()