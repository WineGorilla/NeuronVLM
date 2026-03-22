"""
两阶段训练 - Latent Semantic Injection v2

阶段1：python -m src.train_focus --stage 1
    冻结 Qwen + SAE + ExtraProjector
    训练 ClusterPredictor + ImageClusterScorer
    loss = BCE + λ_align * alignment
    → 学会选 cluster，且 question↔image 一致

阶段2：python -m src.train_focus --stage 2 --resume best
    冻结 SAE
    联合训练所有可学习模块（分组 LR）：
        - ExtraProjector + Qwen 后8层：LR = 1e-5（主力）
        - ClusterPredictor + ImageClusterScorer：LR = 1e-5（持续微调）
    loss = LM + λ_bce * BCE + λ_align * alignment
    → 端到端：cluster 选择 + 投影 + 生成 全部联合优化
"""
import os
import sys
import signal
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import random
import torch
import torch.nn.functional as F

from config import CFG
from src.Model import QwenWithClusterPredictorAndSAE
from src.dataset import VisionTextDataset


LR_STAGE1      = 1e-4
LR_STAGE2_MAIN = 1e-5   # Qwen 后8层 + ExtraProjector
LR_STAGE2_AUX  = 1e-5   # ClusterPredictor + ImageClusterScorer（持续微调）
EPOCHS         = 3
GRAD_ACCUM     = 8
LOG_EVERY      = 10
SAVE_EVERY     = 500
SAVE_DIR       = "outputs/focus_ckpt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",  type=int, default=1, choices=[1, 2])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data",   type=str, default="data/train_cluster.jsonl")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    cluster_path   = os.path.join(CFG.label_dir, f"feature_clusters_layer{CFG.vis_layer}.json")
    predictor_ckpt = os.path.join(SAVE_DIR, f"predictor_{args.resume}.pt") if args.resume else None

    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id          = CFG.model_id,
        sae_ckpt_dir      = CFG.save_dir,
        cluster_path      = cluster_path,
        inject_layer      = CFG.vis_layer,
        latent_mult       = CFG.latent_mult,
        topk              = CFG.topk,
        top_n_patches     = CFG.top_n_patches,
        cluster_threshold = 0.5,
        bce_lambda        = 0.5,
        align_lambda      = 0.3,
        predictor_ckpt    = predictor_ckpt,
        device            = CFG.device,
    )

    dataset = VisionTextDataset(args.data)

    if args.stage == 1:
        # ── Stage 1: ClusterPredictor + ImageClusterScorer ────
        print("Stage 1: Training ClusterPredictor + ImageClusterScorer")
        for p in model.base_model.parameters():
            p.requires_grad = False
        for p in model.extra_projector.parameters():
            p.requires_grad = False
        for p in model.cluster_predictor.parameters():
            p.requires_grad = True
        for p in model.image_cluster_scorer.parameters():
            p.requires_grad = True

        params = (list(model.cluster_predictor.parameters()) +
                  list(model.image_cluster_scorer.parameters()))
        optimizer = torch.optim.AdamW(params, lr=LR_STAGE1, weight_decay=1e-4)
        trainable_layer_ids = []

    else:
        # ── Stage 2: 联合训练，分组 LR ───────────────────────
        print("Stage 2: Joint training (LM + BCE + alignment)")

        # 冻结 Qwen 全部，再选择性解冻
        for p in model.base_model.parameters():
            p.requires_grad = False

        layers = model.base_model.model.language_model.layers
        trainable_layer_ids = list(range(len(layers) - 8, len(layers)))
        for i in trainable_layer_ids:
            for p in layers[i].parameters():
                p.requires_grad = True
        for p in model.base_model.lm_head.parameters():
            p.requires_grad = True

        # 解冻所有可学习模块
        for p in model.extra_projector.parameters():
            p.requires_grad = True
        for p in model.cluster_predictor.parameters():
            p.requires_grad = True
        for p in model.image_cluster_scorer.parameters():
            p.requires_grad = True

        # 分组 LR
        param_groups = [
            {
                "params": [p for p in model.base_model.parameters() if p.requires_grad] +
                          list(model.extra_projector.parameters()),
                "lr": LR_STAGE2_MAIN,
                "name": "main",
            },
            {
                "params": list(model.cluster_predictor.parameters()) +
                          list(model.image_cluster_scorer.parameters()),
                "lr": LR_STAGE2_AUX,
                "name": "aux",
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        params = [p for g in param_groups for p in g["params"]]

    trainable = sum(p.numel() for p in params)
    print(f"Dataset   : {len(dataset)} samples")
    print(f"Trainable : {trainable:,} params")

    # ── 保存函数 ──────────────────────────────────────────────
    def save(tag: str):
        if args.stage == 1:
            path = os.path.join(SAVE_DIR, f"predictor_{tag}.pt")
            model.save_predictor(path)
        else:
            # 保存 Qwen 微调层
            qwen_path = os.path.join(SAVE_DIR, f"qwen_{tag}.pt")
            state = {
                k: v for k, v in model.base_model.state_dict().items()
                if any(
                    k.startswith(f"model.language_model.layers.{i}.")
                    for i in trainable_layer_ids
                )
                or k.startswith("lm_head.")
            }
            torch.save(state, qwen_path)
            print(f"  saved: {qwen_path}")

            # 保存 predictor + scorer + projector
            pred_path = os.path.join(SAVE_DIR, f"predictor_{tag}.pt")
            model.save_predictor(pred_path)

    def handle_sigint(sig, frame):
        print("\n[interrupted] saving...")
        save("interrupted")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # ── 训练循环 ──────────────────────────────────────────────
    global_step = 0
    best_loss   = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        print(f"\n{'='*50}\nEpoch {epoch+1}/{EPOCHS}\n{'='*50}")

        epoch_loss  = 0.0
        valid_steps = 0
        accum_loss  = 0.0
        micro_count = 0

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for step, idx in enumerate(indices):
            item = dataset[idx]
            try:
                if args.stage == 1:
                    # ── 阶段1：BCE + alignment ───────────────────────
                    inputs = model._build_inputs(
                        item["image"], item["question"], for_generation=False
                    )
                    vision_pos, num_img_tokens, text_positions = \
                        model._get_token_positions(inputs)
                    h = model._get_hidden_at_layer(inputs, model.inject_layer)

                    logits = model.cluster_predictor(h, text_positions)
                    target = torch.zeros(1, model.n_clusters, device=model.device)
                    for cid in item.get("focus_clusters", []):
                        if 0 <= cid < model.n_clusters:
                            target[0, cid] = 1.0
                    bce_loss = F.binary_cross_entropy_with_logits(logits, target)

                    h_vision = h[0, vision_pos:vision_pos + num_img_tokens, :].float()
                    align_loss = model._compute_alignment_loss(h_vision, logits)

                    total_loss = bce_loss + model.align_lambda * align_loss

                    if step % LOG_EVERY == 0:
                        print(f"  step {step:5d} | bce={bce_loss.item():.4f} "
                              f"align={align_loss.item():.4f}")

                else:
                    # ── 阶段2：LM + BCE + alignment ──────────────────
                    total_loss, lm_loss, bce_loss_val = model.compute_loss(
                        image_path     = item["image"],
                        question       = item["question"],
                        answer         = item["answer"],
                        focus_clusters = item.get("focus_clusters", []),
                        include_bce    = True,
                    )

                    if step % LOG_EVERY == 0:
                        print(f"  step {step:5d} | total={total_loss.item():.4f} "
                              f"lm={lm_loss:.4f} bce={bce_loss_val:.4f}")

                loss = total_loss / GRAD_ACCUM

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"  [skip] nan/inf at step {step}")
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    micro_count = 0
                    continue

                loss.backward()
                accum_loss  += total_loss.item()
                micro_count += 1

            except Exception as e:
                print(f"  [skip] step {step}: {e}")
                optimizer.zero_grad(set_to_none=True)
                accum_loss  = 0.0
                micro_count = 0
                continue

            if micro_count == GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                eff_loss     = accum_loss / GRAD_ACCUM
                epoch_loss  += eff_loss
                valid_steps += 1
                global_step += 1

                if global_step % SAVE_EVERY == 0:
                    save(f"step{global_step}")

                accum_loss  = 0.0
                micro_count = 0

        # flush
        if micro_count > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss  += accum_loss / micro_count
            valid_steps += 1

        if valid_steps == 0:
            print("  No valid steps.")
            continue

        avg_loss = epoch_loss / valid_steps
        print(f"\n  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        save(f"epoch{epoch+1}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save("best")
            print(f"  New best: {best_loss:.4f}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()