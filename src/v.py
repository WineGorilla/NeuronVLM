"""
Vision Token 热图可视化 — 对比原始模型 vs 增强模型在 inject 层的聚焦区域。

生成两类热图：
  1. SAE Cluster Activation: 哪些 vision patches 被语义注入激活（per-cluster 热图）
  2. Vanilla vs Enhanced: 原始 Qwen 和增强后模型在 inject 层的 vision token 能量对比

用法：
    # 看 SAE cluster 激活热图
    python src/v.py \
        --image data/images/train2014/COCO_train2014_000000394476.jpg \
        --question "Is the bathroom clean?" \
        --predictor_ckpt outputs/focus_ckpt_0.75_64_5000/predictor_best.pt \
        --vis combined

    # 看原始 vs 增强对比
    python src/visualize_heatmap.py --image xxx.jpg --question "..." --vis compare

    # 都看
    python src/visualize_heatmap.py --image xxx.jpg --question "..." --vis both

    # 合并成一张大图
    python src/visualize_heatmap.py --image xxx.jpg --question "..." --vis combined
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import CFG
from src.Model import QwenWithClusterPredictorAndSAE, _find_layers


class HeatmapVisualizer:
    """对比原始模型 vs 增强模型在 inject 层的聚焦区域。"""

    def __init__(self, model):
        self.model = model
        self.sae_layer_data = {}
        self.compare_data = {}

    def _get_vision_grid(self, inputs):
        grid = inputs["image_grid_thw"]
        merge = self.model.base_model.config.vision_config.spatial_merge_size
        H = int(grid[0, 1].item() // merge)
        W = int(grid[0, 2].item() // merge)
        return H, W

    def _reshape_to_grid(self, data):
        """把 1D patch 数据 reshape 成 (H, W) 热图。"""
        H, W = self.grid_H, self.grid_W
        n = min(len(data), H * W)
        if n == H * W:
            return data[:n].reshape(H, W)
        return np.pad(data, (0, H * W - len(data))).reshape(H, W)

    def _overlay_heatmap(self, ax, img, heatmap, title, cmap="jet", alpha=0.5):
        """在原图上叠加热图。"""
        H, W = self.grid_H, self.grid_W
        ax.imshow(img)
        hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        hm_resized = np.array(Image.fromarray(
            (hm_norm * 255).astype(np.uint8)
        ).resize((W * 14, H * 14), Image.BILINEAR))
        ax.imshow(hm_resized, cmap=cmap, alpha=alpha)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    @torch.no_grad()
    def run(self, image_path, question, max_new_tokens=64):
        """
        两次 forward:
          Pass 1: 增强模型（开启所有 hook）→ 采集 inject 层 vision hidden + cluster 信息
          Pass 2: 原始模型（关闭所有 hook）→ 采集 inject 层 vision hidden（vanilla baseline）
        """
        inputs = self.model._build_inputs(image_path, question)
        v_pos, n_img, text_pos = self.model._get_token_positions(inputs)
        self.model._cached_positions = (v_pos, n_img, text_pos)

        H, W = self._get_vision_grid(inputs)
        self.grid_H, self.grid_W = H, W
        self.n_img = n_img

        layers = self.model._layers
        inject_idx = self.model.inject_layer

        # ══════════════════════════════════════════════════════════════════════
        # Pass 1: 增强模型（开启 hook），采集 inject 层 vision hidden
        # ══════════════════════════════════════════════════════════════════════
        captured = {"enhanced": None}

        def hook_enhanced(module, input, output):
            hs = output[0]
            if hs.shape[1] > 1:
                captured["enhanced"] = hs[0, v_pos:v_pos+n_img, :].detach().float().clone()

        h_cap = layers[inject_idx].register_forward_hook(hook_enhanced)
        handles = self.model._activate_hook(self.model.HOOK_READ_WRITE)
        try:
            input_len = inputs["input_ids"].shape[1]
            out_ids = self.model.base_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        finally:
            self.model._deactivate_hook(handles)
            h_cap.remove()

        answer = self.model.processor.decode(
            out_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        cluster_ids = list(self.model._last_cluster_ids)
        cluster_names = [
            self.model.clusters[c]["name"]
            for c in cluster_ids if c in self.model.clusters
        ]

        # 处理 SAE cluster 激活数据
        if captured["enhanced"] is not None:
            self._process_sae_layer(captured["enhanced"])

        # ══════════════════════════════════════════════════════════════════════
        # Pass 2: 原始模型（不开任何 hook），采集 inject 层 vision hidden
        # ══════════════════════════════════════════════════════════════════════
        captured_vanilla = {"vanilla": None}

        def hook_vanilla(module, input, output):
            hs = output[0]
            if hs.shape[1] > 1:
                captured_vanilla["vanilla"] = hs[0, v_pos:v_pos+n_img, :].detach().float().clone()

        h_van = layers[inject_idx].register_forward_hook(hook_vanilla)
        try:
            self.model.base_model(**inputs, return_dict=True)
        finally:
            h_van.remove()

        # 处理对比数据
        if captured["enhanced"] is not None and captured_vanilla["vanilla"] is not None:
            self._process_comparison(
                captured_vanilla["vanilla"],
                captured["enhanced"],
            )

        return {
            "answer": answer,
            "cluster_ids": cluster_ids,
            "cluster_names": cluster_names,
        }

    def _process_sae_layer(self, vision_h):
        """计算每个 vision patch 的 cluster 激活强度。"""
        sae = self.model.sae
        n_patches = vision_h.shape[0]

        z = sae.encode(vision_h)

        cluster_ids = self.model._last_cluster_ids
        if not cluster_ids:
            self.sae_layer_data["activation_map"] = z.abs().sum(dim=-1).cpu().numpy()
            self.sae_layer_data["per_cluster"] = {}
            return

        total_activation = torch.zeros(n_patches, device=vision_h.device)
        per_cluster = {}

        for cid in cluster_ids:
            fids = [f for f in self.model.cluster_to_features.get(cid, [])
                    if f < z.shape[-1]]
            if not fids:
                continue

            acts = z[:, fids].sum(dim=-1)
            total_activation += F.relu(acts)

            cname = self.model.clusters.get(cid, {}).get("name", f"cluster_{cid}")
            per_cluster[cid] = {
                "name": cname,
                "activation": acts.cpu().numpy(),
            }

        self.sae_layer_data["activation_map"] = total_activation.cpu().numpy()
        self.sae_layer_data["per_cluster"] = per_cluster

    def _process_comparison(self, vanilla_h, enhanced_h):
        """对比 vanilla vs enhanced 在 inject 层的 vision hidden 能量。"""
        self.compare_data["vanilla_energy"] = vanilla_h.norm(dim=-1).cpu().numpy()
        self.compare_data["enhanced_energy"] = enhanced_h.norm(dim=-1).cpu().numpy()

    # ── 绘图 ─────────────────────────────────────────────────────────────────

    def plot_sae_heatmap(self, original_image, save_path=None):
        """绘制 SAE inject 层的 cluster 激活热图。"""
        if "activation_map" not in self.sae_layer_data:
            print("  [warning] No SAE layer data to plot.")
            return

        activation = self.sae_layer_data["activation_map"]
        per_cluster = self.sae_layer_data.get("per_cluster", {})
        H, W = self.grid_H, self.grid_W

        n_clusters = len(per_cluster)
        n_plots = 1 + n_clusters
        n_cols = min(4, n_plots)
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 5 * n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        img = np.array(original_image.resize((W * 14, H * 14)))

        heatmap = self._reshape_to_grid(activation)
        self._overlay_heatmap(axes[0], img, heatmap,
                              f"SAE Layer {self.model.inject_layer}\nAll Clusters")

        for i, (cid, cdata) in enumerate(sorted(per_cluster.items())):
            hm = self._reshape_to_grid(np.clip(cdata["activation"], 0, None))
            self._overlay_heatmap(axes[1 + i], img, hm,
                                  f"Cluster {cid}\n{cdata['name']}")

        for j in range(1 + n_clusters, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  SAE heatmap saved: {save_path}")
        plt.show()
        plt.close()

    def plot_comparison(self, original_image, save_path=None):
        """绘制 Vanilla vs Enhanced 在 inject 层的能量对比热图。"""
        if "vanilla_energy" not in self.compare_data:
            print("  [warning] No comparison data to plot.")
            return

        H, W = self.grid_H, self.grid_W
        img = np.array(original_image.resize((W * 14, H * 14)))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        v_energy = self.compare_data["vanilla_energy"]
        e_energy = self.compare_data["enhanced_energy"]
        vmin = min(v_energy.min(), e_energy.min())
        vmax = max(v_energy.max(), e_energy.max())

        for ax, data, title in [
            (axes[0], v_energy, "Vanilla Qwen"),
            (axes[1], e_energy, "Enhanced (Ours)"),
        ]:
            hm = self._reshape_to_grid(data)
            ax.imshow(img)
            hm_norm = (hm - vmin) / (vmax - vmin + 1e-8)
            hm_r = np.array(Image.fromarray(
                (hm_norm * 255).astype(np.uint8)
            ).resize((W * 14, H * 14), Image.BILINEAR))
            ax.imshow(hm_r, cmap="jet", alpha=0.5)
            ax.set_title(f"Layer {self.model.inject_layer}\n{title}", fontsize=12)
            ax.axis("off")

        fig.suptitle("Vision Token Energy Distribution", fontsize=14, y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Comparison heatmap saved: {save_path}")
        plt.show()
        plt.close()

    def plot_combined(self, original_image, save_path=None):
        """合并: 上排 Vanilla vs Enhanced 对比，下排 SAE cluster 激活。"""
        if "activation_map" not in self.sae_layer_data:
            print("  [warning] No SAE layer data.")
            return
        if "vanilla_energy" not in self.compare_data:
            print("  [warning] No comparison data.")
            return

        H, W = self.grid_H, self.grid_W
        img = np.array(original_image.resize((W * 14, H * 14)))

        per_cluster = self.sae_layer_data.get("per_cluster", {})
        sorted_clusters = sorted(per_cluster.items(),
                                 key=lambda x: x[1]["activation"].max(),
                                 reverse=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # ── 上排: Vanilla vs Enhanced ──
        v_energy = self.compare_data["vanilla_energy"]
        e_energy = self.compare_data["enhanced_energy"]
        vmin = min(v_energy.min(), e_energy.min())
        vmax = max(v_energy.max(), e_energy.max())

        for j, (data, title) in enumerate([
            (v_energy, "Vanilla Qwen"),
            (e_energy, "Enhanced (Ours)"),
        ]):
            ax = axes[0, j]
            hm = self._reshape_to_grid(data)
            ax.imshow(img)
            hm_norm = (hm - vmin) / (vmax - vmin + 1e-8)
            hm_r = np.array(Image.fromarray(
                (hm_norm * 255).astype(np.uint8)
            ).resize((W * 14, H * 14), Image.BILINEAR))
            ax.imshow(hm_r, cmap="jet", alpha=0.5)
            ax.set_title(f"Layer {self.model.inject_layer} | {title}", fontsize=11)
            ax.axis("off")

        axes[0, 2].axis("off")

        # ── 下排: SAE cluster 激活 ──
        activation = self.sae_layer_data["activation_map"]
        hm_all = self._reshape_to_grid(activation)
        self._overlay_heatmap(axes[1, 0], img, hm_all, "All Clusters")

        for i, (cid, cdata) in enumerate(sorted_clusters[:2]):
            hm_c = self._reshape_to_grid(np.clip(cdata["activation"], 0, None))
            self._overlay_heatmap(axes[1, 1 + i], img, hm_c,
                                  f"Cluster {cid}: {cdata['name']}")

        if len(sorted_clusters) < 2:
            for i in range(len(sorted_clusters), 2):
                axes[1, 1 + i].axis("off")

        lam = F.softplus(self.model.semantic_cross_attn.lambda_param).item()
        fig.suptitle(f"Vision Focus Analysis  |  λ = {lam:.3f}",
                     fontsize=14, y=1.01)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Combined heatmap saved: {save_path}")
        plt.show()
        plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize vanilla vs enhanced vision token focus"
    )
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--suppress_layer", type=int, default=-8)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/focus_ckpt/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--top_k_clusters", type=int, default=10)
    parser.add_argument("--vis", type=str, default="both",
                        choices=["sae", "compare", "both", "combined"],
                        help="sae=cluster激活, compare=vanilla vs enhanced, "
                             "both=分开画, combined=合并一张")
    parser.add_argument("--save_dir", type=str, default="outputs/heatmaps")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载模型 ──
    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id,
        sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path,
        inject_layer=args.layer,
        suppress_layer=args.suppress_layer,
        latent_mult=CFG.latent_mult,
        topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        top_k_clusters=args.top_k_clusters,
        predictor_ckpt=args.predictor_ckpt,
        device=CFG.device,
    )

    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        print(f"Loading Qwen fine-tuned weights: {args.qwen_ckpt}")
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.base_model.load_state_dict(state, strict=False)
        print("  Qwen weights loaded.")

    # ── 运行可视化 ──
    original_image = Image.open(args.image).convert("RGB")

    viz = HeatmapVisualizer(model)
    result = viz.run(args.image, args.question, max_new_tokens=args.max_tokens)

    print(f"\n{'='*50}")
    print(f"Question : {args.question}")
    print(f"Answer   : {result['answer']}")
    print(f"Clusters : {result['cluster_ids']} → {result['cluster_names']}")
    print(f"Grid     : {viz.grid_H} x {viz.grid_W} = {viz.n_img} patches")
    print(f"{'='*50}")

    img_name = os.path.splitext(os.path.basename(args.image))[0]

    if args.vis in ("sae", "both"):
        viz.plot_sae_heatmap(
            original_image,
            save_path=os.path.join(args.save_dir, f"{img_name}_sae.png"),
        )
    if args.vis in ("compare", "both"):
        viz.plot_comparison(
            original_image,
            save_path=os.path.join(args.save_dir, f"{img_name}_compare.png"),
        )
    if args.vis == "combined":
        viz.plot_combined(
            original_image,
            save_path=os.path.join(args.save_dir, f"{img_name}_combined.png"),
        )


if __name__ == "__main__":
    main()