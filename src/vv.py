"""
极简论文热图 — 原图 + Overall Focus + Neuron Cluster 1/2/3...

经典 jet 风格：低激活=蓝色，高激活=红色，全图半透明覆盖。
低于阈值的区域统一压成最低蓝色。

用法：
    python src/vv.py \
        --image data/images/train2014/COCO_train2014_000000524522.jpg \
        --question "Are all the players wearing black shirts?" \
        --predictor_ckpt outputs/focus_ckpt_0.75_64_5000/predictor_best.pt --n_clusters 3


    python src/vv.py \
        --image data/images/train2014/COCO_train2014_000000132310.jpg \
        --question "How many people are wearing a red shirt?" \
        --predictor_ckpt outputs/qwen_layer8_old/focus_ckpt_0.75_64_5000/predictor_best.pt \
        --save_dir outputs/heatmaps --n_clusters 4

    python src/visualize_heatmap.py --image xxx.jpg --question "..." --n_clusters 3
    python src/visualize_heatmap.py --image xxx.jpg --question "..." --threshold 0.3
    python src/visualize_heatmap.py --image xxx.jpg --question "..." --alpha 0.5
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import CFG
from src.Model import QwenWithClusterPredictorAndSAE


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
})


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _reshape_to_grid(data, H, W):
    n = min(len(data), H * W)
    if n == H * W:
        return data[:n].reshape(H, W)
    padded = np.zeros(H * W)
    padded[:n] = data[:n]
    return padded.reshape(H, W)


def _render_overlay(img_array, heatmap_2d, threshold=0.3, alpha=0.45):
    """
    经典 jet 热图叠加：
      1. 归一化 heatmap 到 [0, 1]
      2. 低于 threshold 的统一设为 0（显示为蓝色）
      3. 用 jet colormap 映射成 RGB
      4. 半透明叠加到原图上
    """
    H_px, W_px = img_array.shape[:2]

    # 归一化
    hm = heatmap_2d.copy().astype(np.float64)
    hm_max = hm.max()
    if hm_max > 0:
        hm = hm / hm_max
    else:
        hm = np.zeros_like(hm)

    # 低于阈值的压成 0（jet colormap 中 0 = 深蓝）
    hm = np.where(hm < threshold, 0.0, hm)

    # 上采样到图片尺寸（双三次插值，平滑）
    hm_pil = Image.fromarray(hm.astype(np.float32), mode="F")
    hm_resized = np.array(hm_pil.resize((W_px, H_px), Image.BICUBIC))
    hm_resized = np.clip(hm_resized, 0, 1)

    # jet colormap 映射
    cmap = plt.cm.jet
    hm_colored = cmap(hm_resized)[:, :, :3]  # (H, W, 3) float [0,1]

    # 半透明叠加
    img_float = img_array.astype(np.float64) / 255.0
    blended = img_float * (1 - alpha) + hm_colored * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return blended


# ── Visualizer ────────────────────────────────────────────────────────────────

class HeatmapVisualizer:
    def __init__(self, model):
        self.model = model
        self.sae_layer_data = {}

    def _get_vision_grid(self, inputs):
        grid = inputs["image_grid_thw"]
        merge = self.model.base_model.config.vision_config.spatial_merge_size
        H = int(grid[0, 1].item() // merge)
        W = int(grid[0, 2].item() // merge)
        return H, W

    @torch.no_grad()
    def run(self, image_path, question, max_new_tokens=64):
        inputs = self.model._build_inputs(image_path, question)
        v_pos, n_img, text_pos = self.model._get_token_positions(inputs)
        self.model._cached_positions = (v_pos, n_img, text_pos)
        H, W = self._get_vision_grid(inputs)
        self.grid_H, self.grid_W = H, W
        self.n_img = n_img

        layers = self.model._layers
        captured = {}

        def hook_sae(module, input, output):
            hs = output[0]
            if hs.shape[1] <= 1: return
            captured["sae_pre"] = hs[0, v_pos:v_pos+n_img, :].detach().float().clone()

        h1 = layers[self.model.inject_layer].register_forward_hook(hook_sae)
        handles = self.model._activate_hook(self.model.HOOK_READ_WRITE)
        try:
            input_len = inputs["input_ids"].shape[1]
            out_ids = self.model.base_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            self.model._deactivate_hook(handles)
            h1.remove()

        answer = self.model.processor.decode(
            out_ids[0, input_len:], skip_special_tokens=True).strip()

        if "sae_pre" in captured:
            self._process_sae(captured["sae_pre"])

        return {
            "answer": answer,
            "cluster_ids": self.model._last_cluster_ids,
        }

    def _process_sae(self, vision_h):
        z = self.model.sae.encode(vision_h)
        cluster_ids = self.model._last_cluster_ids
        n_patches = vision_h.shape[0]

        total = torch.zeros(n_patches, device=vision_h.device)
        per_cluster = {}

        for cid in cluster_ids:
            fids = [f for f in self.model.cluster_to_features.get(cid, [])
                    if f < z.shape[-1]]
            if not fids: continue
            acts = z[:, fids].sum(dim=-1)
            total += F.relu(acts)
            per_cluster[cid] = acts.cpu().numpy()

        self.sae_layer_data["total"] = total.cpu().numpy()
        self.sae_layer_data["per_cluster"] = per_cluster

    def plot(self, original_image, n_clusters=3, threshold=0.3, alpha=0.45,
             save_path=None):
        """
        原图 | Overall Focus | Neuron Cluster 1 | Neuron Cluster 2 | ...
        """
        if "total" not in self.sae_layer_data:
            print("  [warning] No data."); return

        H, W = self.grid_H, self.grid_W
        img = np.array(original_image.resize((W * 14, H * 14)))

        per_cluster = self.sae_layer_data["per_cluster"]
        sorted_clusters = sorted(per_cluster.items(),
                                 key=lambda x: x[1].max(), reverse=True)
        n_show = min(n_clusters, len(sorted_clusters))
        n_cols = 2 + n_show

        # 图尺寸
        panel_w = 3.2
        panel_h = panel_w * H / W
        fig_w = panel_w * n_cols + 0.12 * (n_cols - 1)
        fig_h = panel_h + 0.5

        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
        plt.subplots_adjust(wspace=0.04, left=0.005, right=0.995,
                            top=0.95, bottom=0.08)

        # ── 原图（干净，不做处理）──
        axes[0].imshow(img)
        axes[0].set_xlabel("Input Image", fontsize=9, labelpad=4)
        axes[0].set_xticks([]); axes[0].set_yticks([])
        for s in axes[0].spines.values():
            s.set_linewidth(0.4); s.set_color("#999999")

        # ── Overall Focus ──
        total = self.sae_layer_data["total"]
        hm = _reshape_to_grid(total, H, W)
        overlay = _render_overlay(img, hm, threshold=threshold, alpha=alpha)
        axes[1].imshow(overlay)
        axes[1].set_xlabel("Overall Focus", fontsize=9, labelpad=4)
        axes[1].set_xticks([]); axes[1].set_yticks([])
        for s in axes[1].spines.values():
            s.set_linewidth(0.4); s.set_color("#999999")

        # ── Neuron Cluster 1, 2, 3... ──
        for i, (cid, acts) in enumerate(sorted_clusters[:n_show]):
            ax = axes[2 + i]
            acts_clip = np.clip(acts, 0, None)
            hm_c = _reshape_to_grid(acts_clip, H, W)
            overlay_c = _render_overlay(img, hm_c, threshold=threshold, alpha=alpha)
            ax.imshow(overlay_c)
            ax.set_xlabel(f"Neuron Cluster {cid}", fontsize=9, labelpad=4)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("#999999")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.03,
                        facecolor="white", edgecolor="none")
            print(f"  Saved: {save_path}")
            pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03,
                        facecolor="white", edgecolor="none")
            print(f"  PDF:   {pdf_path}")
        plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/qwen_layer8_old/focus_ckpt_0.75_64_5000/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="展示 top-N 个 neuron cluster")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="低于 max*threshold 的区域显示为蓝色 (0~1)")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="热图叠加透明度 (0=纯原图, 1=纯热图)")
    parser.add_argument("--save_dir", type=str, default="outputs/heatmaps")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id, sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path, inject_layer=args.layer,
        latent_mult=CFG.latent_mult, topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.predictor_ckpt, device=CFG.device,
    )
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.base_model.load_state_dict(state, strict=False)

    original_image = Image.open(args.image).convert("RGB")
    viz = HeatmapVisualizer(model)
    result = viz.run(args.image, args.question, max_new_tokens=args.max_tokens)

    print(f"Q: {args.question}")
    print(f"A: {result['answer']}")
    print(f"Clusters: {len(result['cluster_ids'])} activated")

    img_name = os.path.splitext(os.path.basename(args.image))[0]
    viz.plot(
        original_image,
        n_clusters=args.n_clusters,
        threshold=args.threshold,
        alpha=args.alpha,
        save_path=os.path.join(args.save_dir, f"{img_name}_focus.png"),
    )


if __name__ == "__main__":
    main()