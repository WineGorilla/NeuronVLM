"""
两阶段训练 — Single-Forward Architecture

Stage 1: 学 what to focus
  - 训练 ClusterPredictor + ImageClusterScorer
  - Hook 只读：预测 cluster，不修改 hidden
  - Loss = BCE + alignment
  - 命令: python -m src.train_focus --stage 1

Stage 2: 学 how to inject
  - 训练 SemanticCrossAttention + ExtraProjector + top LM layers
  - Hook 读写：预测 cluster + 注入 extra tokens
  - Loss = LM + BCE + alignment
  - 命令: python -m src.train_focus --stage 2 --resume best
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


# ── 超参数 ────────────────────────────────────────────────────────────────────

LR_STAGE1      = 1e-4   # predictor + scorer
LR_STAGE2_MAIN = 1e-5   # LM layers + cross-attn + projector
LR_STAGE2_AUX  = 1e-5   # predictor + scorer（小 lr 微调或冻结）
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
    parser.add_argument("--freeze_predictor", action="store_true",
                        help="Stage 2: freeze predictor (use Stage 1 weights as-is)")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    cluster_path   = os.path.join(CFG.label_dir, f"feature_clusters_layer{CFG.vis_layer}.json")
    predictor_ckpt = (
        os.path.join(SAVE_DIR, f"predictor_{args.resume}.pt")
        if args.resume else None
    )

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

    # ── 配置 requires_grad + optimizer ────────────────────────────────────────

    if args.stage == 1:
        print("=" * 60)
        print("Stage 1: Learn what to focus (ClusterPredictor + Scorer)")
        print("=" * 60)

        # 冻结一切
        for p in model.base_model.parameters():
            p.requires_grad = False
        for p in model.extra_projector.parameters():
            p.requires_grad = False
        for p in model.semantic_cross_attn.parameters():
            p.requires_grad = False

        # 只训 predictor + scorer
        for p in model.cluster_predictor.parameters():
            p.requires_grad = True
        for p in model.image_cluster_scorer.parameters():
            p.requires_grad = True

        params = (
            list(model.cluster_predictor.parameters())
            + list(model.image_cluster_scorer.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=LR_STAGE1, weight_decay=1e-4)
        trainable_layer_ids = []

    else:
        print("=" * 60)
        print("Stage 2: Learn how to inject (CrossAttention + LM layers)")
        print("=" * 60)

        if not args.resume:
            print("  WARNING: Stage 2 without --resume, predictor is random!")

        # 冻结 base model
        for p in model.base_model.parameters():
            p.requires_grad = False

        # 解冻 top-8 LM layers + lm_head
        layers = model.base_model.model.language_model.layers
        trainable_layer_ids = list(range(len(layers) - 8, len(layers)))
        for i in trainable_layer_ids:
            for p in layers[i].parameters():
                p.requires_grad = True
        for p in model.base_model.lm_head.parameters():
            p.requires_grad = True

        # 注入模块：必须训练
        for p in model.extra_projector.parameters():
            p.requires_grad = True
        for p in model.semantic_cross_attn.parameters():
            p.requires_grad = True
        for p in model.pc_suppressor.parameters():
            p.requires_grad = True
        for p in model.semantic_completer.parameters():
            p.requires_grad = True

        # Predictor: 可冻结或小 lr 微调
        predictor_trainable = not args.freeze_predictor
        for p in model.cluster_predictor.parameters():
            p.requires_grad = predictor_trainable
        for p in model.image_cluster_scorer.parameters():
            p.requires_grad = predictor_trainable

        if args.freeze_predictor:
            print("  Predictor: FROZEN (using Stage 1 weights)")
        else:
            print(f"  Predictor: fine-tuning at lr={LR_STAGE2_AUX}")

        # 参数组
        main_params = (
            [p for p in model.base_model.parameters() if p.requires_grad]
            + list(model.extra_projector.parameters())
            + list(model.semantic_cross_attn.parameters())
            + list(model.pc_suppressor.parameters())
            + list(model.semantic_completer.parameters())
        )
        param_groups = [{"params": main_params, "lr": LR_STAGE2_MAIN}]

        if predictor_trainable:
            aux_params = (
                list(model.cluster_predictor.parameters())
                + list(model.image_cluster_scorer.parameters())
            )
            param_groups.append({"params": aux_params, "lr": LR_STAGE2_AUX})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        params = [p for g in param_groups for p in g["params"]]

    trainable = sum(p.numel() for p in params)
    print(f"Dataset   : {len(dataset)} samples")
    print(f"Trainable : {trainable:,} params")
    print()

    # ── 保存逻辑 ─────────────────────────────────────────────────────────────

    def save(tag: str):
        # 始终保存 predictor + 注入模块
        pred_path = os.path.join(SAVE_DIR, f"predictor_{tag}.pt")
        model.save_predictor(pred_path)

        # Stage 2 额外保存解冻的 LM 层
        if args.stage == 2:
            qwen_path = os.path.join(SAVE_DIR, f"qwen_{tag}.pt")
            state = {
                k: v
                for k, v in model.base_model.state_dict().items()
                if any(
                    k.startswith(f"model.language_model.layers.{i}.")
                    for i in trainable_layer_ids
                )
                or k.startswith("lm_head.")
            }
            torch.save(state, qwen_path)
            print(f"  Saved LM layers: {qwen_path}")

    def handle_sigint(sig, frame):
        print("\n[interrupted] saving...")
        save("interrupted")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # ── 训练循环 ─────────────────────────────────────────────────────────────

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
                # ── 统一调用 compute_loss ─────────────────────────────────────
                total_loss, metric_a, metric_b = model.compute_loss(
                    image_path     = item["image"],
                    question       = item["question"],
                    answer         = item.get("answer", ""),
                    focus_clusters = item.get("focus_clusters", []),
                    stage          = args.stage,
                    include_bce    = True,
                )

                # ── 日志 ─────────────────────────────────────────────────────
                if step % LOG_EVERY == 0:
                    if args.stage == 1:
                        print(f"  step {step:5d} | bce={metric_a:.4f} "
                              f"align={metric_b:.4f} "
                              f"total={total_loss.item():.4f}")
                    else:
                        lam = F.softplus(model.semantic_cross_attn.lambda_param).item()
                        print(f"  step {step:5d} | total={total_loss.item():.4f} "
                              f"lm={metric_a:.4f} bce={metric_b:.4f} λ={lam:.4f}")

                # ── NaN/Inf 检查 ─────────────────────────────────────────────
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"  [skip] nan/inf at step {step}")
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss  = 0.0
                    micro_count = 0
                    continue

                # ── 梯度累积 ─────────────────────────────────────────────────
                loss = total_loss / GRAD_ACCUM
                loss.backward()
                accum_loss  += total_loss.item()
                micro_count += 1

            except Exception as e:
                print(f"  [skip] step {step}: {e}")
                optimizer.zero_grad(set_to_none=True)
                accum_loss  = 0.0
                micro_count = 0
                continue

            # ── 累积满 GRAD_ACCUM 步后更新 ───────────────────────────────────
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

        # ── epoch 末尾: 处理剩余累积 ─────────────────────────────────────────
        if micro_count > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss  += accum_loss / micro_count
            valid_steps += 1

        if valid_steps == 0:
            print("  No valid steps this epoch.")
            continue

        avg_loss = epoch_loss / valid_steps
        print(f"\n  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        save(f"epoch{epoch+1}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save("best")
            print(f"  ★ New best: {best_loss:.4f}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()