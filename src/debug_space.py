"""
快速诊断：检查 embedding vs extra token 的数值空间是否匹配。

用法：
    python src/debug_space.py \
        --image data/images/train2014/COCO_train2014_000000416767.jpg \
        --question "Is there any human?"
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from config import CFG
from src.Model import QwenWithClusterPredictorAndSAE


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{CFG.vis_layer}.json"
    )

    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id       = CFG.model_id,
        sae_ckpt_dir   = CFG.save_dir,
        cluster_path   = cluster_path,
        inject_layer   = CFG.vis_layer,
        latent_mult    = CFG.latent_mult,
        topk           = CFG.topk,
        top_n_patches  = CFG.top_n_patches,
        predictor_ckpt = "outputs/focus_ckpt/predictor_best.pt",
        device         = CFG.device,
    )

    with torch.no_grad():
        inputs = model._build_inputs(args.image, args.question, for_generation=True)
        vision_pos, num_img_tokens, text_positions = model._get_token_positions(inputs)
        h = model._get_hidden_at_layer(inputs, model.inject_layer)

        # 1. Embedding 空间
        embeds = model.base_model.model.language_model.embed_tokens(inputs["input_ids"])
        img_embeds = embeds[0, vision_pos:vision_pos + num_img_tokens, :]
        txt_embeds = embeds[0, text_positions, :]

        # 2. Layer 8 hidden state 空间
        img_hidden = h[0, vision_pos:vision_pos + num_img_tokens, :]
        txt_hidden = h[0, text_positions, :]

        # 3. Predictor → extra tokens
        _, cluster_ids, cluster_probs = model._predict_clusters(h, text_positions)
        extra = model._get_extra_tokens(h, cluster_ids, vision_pos, num_img_tokens, cluster_probs)

        print("=" * 65)
        print("  Space Diagnosis: Embedding vs Hidden State vs Extra Tokens")
        print("=" * 65)

        def stats(name, t):
            t = t.float()
            print(f"  {name:30s} | mean={t.mean():+10.4f} | "
                  f"abs_mean={t.abs().mean():8.4f} | "
                  f"std={t.std():8.4f} | "
                  f"norm={t.norm(dim=-1).mean():8.2f}")

        print("\n[Embedding layer output]")
        stats("Image embeddings", img_embeds)
        stats("Text embeddings", txt_embeds)

        print(f"\n[Layer {CFG.vis_layer} hidden state]")
        stats("Image hidden", img_hidden)
        stats("Text hidden", txt_hidden)

        if extra is not None:
            print(f"\n[Extra tokens (injected)]")
            stats("Extra tokens", extra)

            # 4. 比值
            embed_scale = img_embeds.float().abs().mean().item()
            extra_scale = extra.float().abs().mean().item()
            ratio = extra_scale / embed_scale if embed_scale > 0 else float('inf')

            print(f"\n[Scale ratio]")
            print(f"  extra / embedding = {ratio:.2f}x")

            if ratio > 3.0:
                print(f"\n  ⚠️  Extra tokens 比 embedding 大 {ratio:.1f} 倍！")
                print(f"  → 模型会把 extra tokens 当噪声忽略或产生异常输出。")
                print(f"  → 需要加投影层 (LayerNorm + Linear) 对齐空间。")
            elif ratio < 0.3:
                print(f"\n  ⚠️  Extra tokens 比 embedding 小 {ratio:.1f} 倍！")
                print(f"  → Extra tokens 信号太弱，模型看不到。")
                print(f"  → 需要放大或投影。")
            else:
                print(f"\n  ✅ 数值范围基本匹配 ({ratio:.2f}x)。")
                print(f"  → 问题可能在其他地方（训练数据、cluster 质量等）。")
        else:
            print("\n  ⚠️  没有产生 extra tokens（没有 cluster 被激活）。")

        # 5. Cosine similarity
        if extra is not None:
            print(f"\n[Cosine similarity]")
            # extra vs image embeds
            cos_embed = torch.nn.functional.cosine_similarity(
                extra[:min(10, len(extra))].float().unsqueeze(1),
                img_embeds[:min(10, len(img_embeds))].float().unsqueeze(0),
                dim=-1
            ).mean().item()
            # extra vs image hidden
            cos_hidden = torch.nn.functional.cosine_similarity(
                extra[:min(10, len(extra))].float().unsqueeze(1),
                img_hidden[:min(10, len(img_hidden))].float().unsqueeze(0),
                dim=-1
            ).mean().item()
            print(f"  extra ↔ image_embedding : {cos_embed:+.4f}")
            print(f"  extra ↔ image_hidden    : {cos_hidden:+.4f}")
            print(f"  (extra 来自 hidden space，和 hidden 相似度应该高)")

        print(f"\n  Clusters: {cluster_ids}")
        print("=" * 65)


if __name__ == "__main__":
    main()