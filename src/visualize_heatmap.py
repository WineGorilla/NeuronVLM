"""
Vision Token 热图可视化 — 查看模型在 SAE inject 层和 PCS suppress 层的聚焦区域。

生成两类热图：
  1. SAE Inject 层: 哪些 vision patches 被语义注入激活（cluster activation 强度）
  2. PCS Suppress 层: 主成分抑制前后 vision tokens 的能量分布变化

用法：
    python src/visualize_heatmap.py \
        --image data/images/train2014/COCO_train2014_000000415275.jpg \
        --question "Is there organic food in this store?" \
        --predictor_ckpt outputs/focus_ckpt_0.75_64_5000/predictor_best.pt \
        --save_dir outputs/heatmaps

    # 只看 SAE 层
    python src/visualize_heatmap.py --image xxx.jpg --question "..." --vis sae

    # 只看 PCS 层
    python src/visualize_heatmap.py --image xxx.jpg --question "..." --vis pcs

    # 都看
    python src/visualize_heatmap.py --image xxx.jpg --question "..." --vis both
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
import matplotlib.cm as cm
from PIL import Image

from config import CFG
from src.Model import QwenWithClusterPredictorAndSAE, _find_layers


class HeatmapVisualizer:
    """在 SAE inject 层和 PCS suppress 层采集 vision token 信息并生成热图。"""

    def __init__(self, model):
        self.model = model
        self.sae_layer_data = {}    # SAE inject 层的数据
        self.pcs_layer_data = {}    # PCS suppress 层的数据

    def _get_vision_grid(self, inputs):
        """获取 vision token 的空间网格尺寸 (H, W)。"""
        grid = inputs["image_grid_thw"]
        merge = self.model.base_model.config.vision_config.spatial_merge_size
        H = int(grid[0, 1].item() // merge)
        W = int(grid[0, 2].item() // merge)
        return H, W

    @torch.no_grad()
    def run(self, image_path, question, max_new_tokens=64):
        """
        执行一次 forward，同时在 SAE 层和 PCS 层捕获 vision hidden states。

        SAE 层捕获：
          - 每个 vision patch 的 cluster 激活强度
          - 被 top-k patch 选中的位置

        PCS 层捕获：
          - 抑制前的 vision hidden states
          - 抑制后的 vision hidden states（通过 pc_suppressor）
        """
        inputs = self.model._build_inputs(image_path, question)
        v_pos, n_img, text_pos = self.model._get_token_positions(inputs)
        self.model._cached_positions = (v_pos, n_img, text_pos)

        H, W = self._get_vision_grid(inputs)
        self.grid_H, self.grid_W = H, W
        self.n_img = n_img

        layers = self.model._layers
        inject_idx = self.model.inject_layer
        suppress_idx = self.model.suppress_layer

        # ── 注册临时 hooks 采集数据 ──────────────────────────────────────────
        captured = {
            "sae_pre": None,     # inject 层输出的 vision hidden（注入前）
            "pcs_pre": None,     # suppress 层输出的 vision hidden（抑制前）
        }

        def hook_sae(module, input, output):
            hs = output[0]
            if hs.shape[1] <= 1:
                return
            captured["sae_pre"] = hs[0, v_pos:v_pos+n_img, :].detach().float().clone()

        def hook_pcs(module, input, output):
            hs = output[0]
            if hs.shape[1] <= 1:
                return
            captured["pcs_pre"] = hs[0, v_pos:v_pos+n_img, :].detach().float().clone()

        h1 = layers[inject_idx].register_forward_hook(hook_sae)
        h2 = layers[suppress_idx].register_forward_hook(hook_pcs)

        # 同时用模型自带的 hook 做正常推理
        handles = self.model._activate_hook(self.model.HOOK_READ_WRITE)
        try:
            input_len = inputs["input_ids"].shape[1]
            out_ids = self.model.base_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        finally:
            self.model._deactivate_hook(handles)
            h1.remove()
            h2.remove()

        answer = self.model.processor.decode(
            out_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        # ── 处理 SAE 层数据 ─────────────────────────────────────────────────
        if captured["sae_pre"] is not None:
            self._process_sae_layer(captured["sae_pre"], text_pos, inputs)

        # ── 处理 PCS 层数据 ─────────────────────────────────────────────────
        if captured["pcs_pre"] is not None:
            self._process_pcs_layer(captured["pcs_pre"])

        return {
            "answer": answer,
            "cluster_ids": self.model._last_cluster_ids,
            "cluster_names": [
                self.model.clusters[c]["name"]
                for c in self.model._last_cluster_ids
                if c in self.model.clusters
            ],
        }

    def _process_sae_layer(self, vision_h, text_pos, inputs):
        """计算每个 vision patch 的 cluster 激活强度。"""
        sae = self.model.sae
        n_patches = vision_h.shape[0]

        # SAE encode
        z = sae.encode(vision_h)  # (n_patches, latent_dim)

        # 获取激活的 cluster ids
        cluster_ids = self.model._last_cluster_ids
        if not cluster_ids:
            # 如果没有激活的 cluster，用所有 z 的 norm 作为热图
            self.sae_layer_data["activation_map"] = z.abs().sum(dim=-1).cpu().numpy()
            self.sae_layer_data["per_cluster"] = {}
            return

        # 总体激活强度（所有激活 cluster 的贡献叠加）
        total_activation = torch.zeros(n_patches, device=vision_h.device)

        per_cluster = {}
        for cid in cluster_ids:
            fids = [f for f in self.model.cluster_to_features.get(cid, [])
                    if f < z.shape[-1]]
            if not fids:
                continue

            # 该 cluster 在每个 patch 上的激活强度
            acts = z[:, fids].sum(dim=-1)  # (n_patches,)
            total_activation += F.relu(acts)

            cname = self.model.clusters.get(cid, {}).get("name", f"cluster_{cid}")
            per_cluster[cid] = {
                "name": cname,
                "activation": acts.cpu().numpy(),
            }

        self.sae_layer_data["activation_map"] = total_activation.cpu().numpy()
        self.sae_layer_data["per_cluster"] = per_cluster

    def _process_pcs_layer(self, vision_h):
        """计算 PCS 抑制前后的能量分布。"""
        # 抑制前：每个 patch 的 L2 norm（能量）
        energy_before = vision_h.norm(dim=-1).cpu().numpy()

        # 手动执行 PCS 计算抑制后的结果
        suppressed = self.model.pc_suppressor(vision_h)
        energy_after = suppressed.norm(dim=-1).cpu().numpy()

        # 变化量
        energy_diff = energy_before - energy_after  # 正值 = 被抑制的量

        self.pcs_layer_data["energy_before"] = energy_before
        self.pcs_layer_data["energy_after"] = energy_after
        self.pcs_layer_data["energy_diff"] = energy_diff

    # ── 绘图 ─────────────────────────────────────────────────────────────────

    def plot_sae_heatmap(self, original_image, save_path=None):
        """绘制 SAE inject 层的 cluster 激活热图。"""
        if "activation_map" not in self.sae_layer_data:
            print("  [warning] No SAE layer data to plot.")
            return

        activation = self.sae_layer_data["activation_map"]
        per_cluster = self.sae_layer_data.get("per_cluster", {})
        H, W = self.grid_H, self.grid_W

        # 总数量可能与 H*W 不完全匹配（padding等），截取或填充
        n = min(len(activation), H * W)
        heatmap = activation[:n].reshape(H, W) if n == H * W else \
                  np.pad(activation, (0, H*W - len(activation))).reshape(H, W)

        # 计算需要多少子图：1 总体 + 每个 cluster 一个
        n_clusters = len(per_cluster)
        n_plots = 1 + n_clusters
        n_cols = min(4, n_plots)
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 5 * n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # ── 总体热图 ──
        ax = axes[0]
        img = np.array(original_image.resize((W * 14, H * 14)))
        ax.imshow(img)
        hm_resized = np.array(Image.fromarray(
            (heatmap / (heatmap.max() + 1e-8) * 255).astype(np.uint8)
        ).resize((W * 14, H * 14), Image.BILINEAR))
        ax.imshow(hm_resized, cmap="jet", alpha=0.5)
        ax.set_title(f"SAE Layer {self.model.inject_layer}\nAll Clusters Combined",
                     fontsize=11)
        ax.axis("off")

        # ── 每个 cluster 的热图 ──
        for i, (cid, cdata) in enumerate(sorted(per_cluster.items())):
            ax = axes[1 + i]
            acts = cdata["activation"]
            n_c = min(len(acts), H * W)
            hm = acts[:n_c].reshape(H, W) if n_c == H * W else \
                 np.pad(acts, (0, H*W - len(acts))).reshape(H, W)
            hm = np.clip(hm, 0, None)  # ReLU

            ax.imshow(img)
            hm_r = np.array(Image.fromarray(
                (hm / (hm.max() + 1e-8) * 255).astype(np.uint8)
            ).resize((W * 14, H * 14), Image.BILINEAR))
            ax.imshow(hm_r, cmap="jet", alpha=0.5)
            ax.set_title(f"Cluster {cid}\n{cdata['name']}", fontsize=10)
            ax.axis("off")

        # 隐藏多余子图
        for j in range(1 + n_clusters, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  SAE heatmap saved: {save_path}")
        plt.show()
        plt.close()

    def plot_pcs_heatmap(self, original_image, save_path=None):
        """绘制 PCS suppress 层的能量分布热图（抑制前 / 抑制后 / 差异）。"""
        if "energy_before" not in self.pcs_layer_data:
            print("  [warning] No PCS layer data to plot.")
            return

        H, W = self.grid_H, self.grid_W

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        img = np.array(original_image.resize((W * 14, H * 14)))

        titles = ["Before PCS\n(Energy per patch)",
                   "After PCS\n(Energy per patch)",
                   "Suppressed Amount\n(Before - After)"]
        keys = ["energy_before", "energy_after", "energy_diff"]
        cmaps = ["hot", "hot", "coolwarm"]

        for ax, title, key, cmap_name in zip(axes, titles, keys, cmaps):
            data = self.pcs_layer_data[key]
            n = min(len(data), H * W)
            hm = data[:n].reshape(H, W) if n == H * W else \
                 np.pad(data, (0, H*W - len(data))).reshape(H, W)

            ax.imshow(img)
            hm_r = np.array(Image.fromarray(
                ((hm - hm.min()) / (hm.max() - hm.min() + 1e-8) * 255
                 ).astype(np.uint8)
            ).resize((W * 14, H * 14), Image.BILINEAR))
            ax.imshow(hm_r, cmap=cmap_name, alpha=0.5)
            ax.set_title(f"PCS Layer {self.model.suppress_layer}\n{title}",
                         fontsize=11)
            ax.axis("off")

        # 添加 alpha 信息
        alpha = F.softplus(self.model.pc_suppressor.alpha_param).item()
        fig.suptitle(f"Principal Component Suppression (α = {alpha:.4f})",
                     fontsize=13, y=1.02)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  PCS heatmap saved: {save_path}")
        plt.show()
        plt.close()

    def plot_combined(self, original_image, save_path=None):
        """合并绘制：上排 SAE，下排 PCS。"""
        if "activation_map" not in self.sae_layer_data:
            print("  [warning] No SAE layer data.")
            return
        if "energy_before" not in self.pcs_layer_data:
            print("  [warning] No PCS layer data.")
            return

        H, W = self.grid_H, self.grid_W
        img = np.array(original_image.resize((W * 14, H * 14)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # ── 上排: SAE ──
        # 总体
        activation = self.sae_layer_data["activation_map"]
        n = min(len(activation), H * W)
        hm = activation[:n].reshape(H, W) if n == H * W else \
             np.pad(activation, (0, H*W - len(activation))).reshape(H, W)

        ax = axes[0, 0]
        ax.imshow(img)
        hm_r = np.array(Image.fromarray(
            (hm / (hm.max() + 1e-8) * 255).astype(np.uint8)
        ).resize((W * 14, H * 14), Image.BILINEAR))
        ax.imshow(hm_r, cmap="jet", alpha=0.5)
        ax.set_title(f"SAE Layer {self.model.inject_layer}\nAll Clusters", fontsize=11)
        ax.axis("off")

        # top-2 clusters
        per_cluster = self.sae_layer_data.get("per_cluster", {})
        sorted_clusters = sorted(per_cluster.items(),
                                 key=lambda x: x[1]["activation"].max(),
                                 reverse=True)
        for i, (cid, cdata) in enumerate(sorted_clusters[:2]):
            ax = axes[0, 1 + i]
            acts = cdata["activation"]
            n_c = min(len(acts), H * W)
            hm_c = np.clip(acts[:n_c], 0, None)
            hm_c = hm_c.reshape(H, W) if n_c == H * W else \
                   np.pad(hm_c, (0, H*W - len(hm_c))).reshape(H, W)
            ax.imshow(img)
            hm_r = np.array(Image.fromarray(
                (hm_c / (hm_c.max() + 1e-8) * 255).astype(np.uint8)
            ).resize((W * 14, H * 14), Image.BILINEAR))
            ax.imshow(hm_r, cmap="jet", alpha=0.5)
            ax.set_title(f"Cluster {cid}: {cdata['name']}", fontsize=10)
            ax.axis("off")

        if len(sorted_clusters) < 2:
            for i in range(len(sorted_clusters), 2):
                axes[0, 1 + i].axis("off")

        # ── 下排: PCS ──
        titles = ["Before PCS", "After PCS", "Suppressed"]
        keys = ["energy_before", "energy_after", "energy_diff"]
        cmaps = ["hot", "hot", "coolwarm"]

        for j, (title, key, cmap_name) in enumerate(zip(titles, keys, cmaps)):
            ax = axes[1, j]
            data = self.pcs_layer_data[key]
            n = min(len(data), H * W)
            hm = data[:n].reshape(H, W) if n == H * W else \
                 np.pad(data, (0, H*W - len(data))).reshape(H, W)
            ax.imshow(img)
            hm_r = np.array(Image.fromarray(
                ((hm - hm.min()) / (hm.max() - hm.min() + 1e-8) * 255
                 ).astype(np.uint8)
            ).resize((W * 14, H * 14), Image.BILINEAR))
            ax.imshow(hm_r, cmap=cmap_name, alpha=0.5)
            ax.set_title(f"PCS Layer {self.model.suppress_layer}\n{title}",
                         fontsize=11)
            ax.axis("off")

        lam = F.softplus(self.model.semantic_cross_attn.lambda_param).item()
        alpha = F.softplus(self.model.pc_suppressor.alpha_param).item()
        fig.suptitle(f"Vision Token Heatmaps  |  λ_inject={lam:.3f}  "
                     f"α_suppress={alpha:.4f}",
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
        description="Visualize SAE inject and PCS suppress layer heatmaps"
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
                        choices=["sae", "pcs", "both", "combined"],
                        help="sae=SAE层, pcs=PCS层, both=分开画, combined=合并画")
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

    # 文件名前缀
    img_name = os.path.splitext(os.path.basename(args.image))[0]

    if args.vis == "sae":
        viz.plot_sae_heatmap(
            original_image,
            save_path=os.path.join(args.save_dir, f"{img_name}_sae.png"),
        )
    elif args.vis == "pcs":
        viz.plot_pcs_heatmap(
            original_image,
            save_path=os.path.join(args.save_dir, f"{img_name}_pcs.png"),
        )
    elif args.vis == "both":
        viz.plot_sae_heatmap(
            original_image,
            save_path=os.path.join(args.save_dir, f"{img_name}_sae.png"),
        )
        viz.plot_pcs_heatmap(
            original_image,
            save_path=os.path.join(args.save_dir, f"{img_name}_pcs.png"),
        )
    elif args.vis == "combined":
        viz.plot_combined(
            original_image,
            save_path=os.path.join(args.save_dir, f"{img_name}_combined.png"),
        )


if __name__ == "__main__":
    main()