"""
Dense Cross-Attention Baseline 训练脚本。

与 NeuronEye 完全对齐的训练设置：
  - 5K VQA 训练样本
  - Frozen backbone
  - 同样的 learning rate, gradient accumulation
  - 同样的 Layer 8 inject + Layer 20 PCS

区别仅在于：不使用 SAE / neuron clusters，直接在 dense space 做 cross-attention。

用法：
    python src_ablation/train_dense_xattn.py
    python train_dense_xattn.py --lr 1e-4 --epochs 3
    python train_dense_xattn.py --save_dir outputs/dense_xattn_ckpt
"""
import os
import sys
import json
import argparse
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

try:
    from config import CFG
except ImportError:
    class CFG:
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        train_file = "data/train_cluster_qwen5k_5k_pld.json"
        vis_layer = 8
        device = "cuda"
        top_n_patches = 60

from src.dataset import VisionTextDataset, build_collate


def train():
    parser = argparse.ArgumentParser(description="Train Dense Cross-Attention Baseline")
    parser.add_argument("--model_id", type=str, default=CFG.model_id)
    parser.add_argument("--train_file", type=str, default="data/train_cluster_qwen5k_5k_pld.jsonl")
    parser.add_argument("--inject_layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--top_n_patches", type=int, default=CFG.top_n_patches)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=6)
    parser.add_argument("--save_dir", type=str, default="outputs/dense_xattn_ckpt")
    parser.add_argument("--device", type=str, default=CFG.device)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载模型 ──────────────────────────────────────────────────────────────

    # 延迟导入，确保 Model_dense_xattn.py 在 path 中
    from Model_dense_xattn import QwenWithDenseCrossAttention

    model = QwenWithDenseCrossAttention.from_pretrained(
        model_id=args.model_id,
        inject_layer=args.inject_layer,
        top_n_patches=args.top_n_patches,
        device=args.device,
    )

    # ── 训练参数 ──────────────────────────────────────────────────────────────

    trainable_params = []
    for m in [model.text_query_proj, model.dense_cross_attn, model.pc_suppressor]:
        trainable_params.extend(p for p in m.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in trainable_params)
    print(f"  Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # ── 数据 ──────────────────────────────────────────────────────────────────

    dataset = VisionTextDataset(args.train_file)
    print(f"  Training samples: {len(dataset)}")

    # ── 保存函数 ──────────────────────────────────────────────────────────────

    best_loss = float('inf')

    def save_checkpoint(tag="latest"):
        path = os.path.join(args.save_dir, f"predictor_{tag}.pt")
        model.save_predictor(path)

    # Ctrl+C 自动保存
    def handle_sigint(sig, frame):
        print("\n[interrupted] saving checkpoint...")
        save_checkpoint("interrupted")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # ── 训练循环 ──────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  Dense Cross-Attention Baseline Training")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Grad Accum: {args.grad_accum}")
    print(f"  Inject Layer: {args.inject_layer}, Top-N Patches: {args.top_n_patches}")
    print(f"{'='*60}\n")

    model.train()
    # 确保 base_model 仍然 eval
    model.base_model.eval()

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_samples = 0

        for i, sample in enumerate(dataset):
            try:
                image_path = sample["image"]
                question = sample["question"]
                answer = sample["answer"]

                loss, lm_loss_val, _ = model.compute_loss(
                    image_path, question, answer, stage=2
                )

                loss = loss / args.grad_accum
                loss.backward()

                if (i + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += lm_loss_val
                n_samples += 1

                if (i + 1) % 50 == 0:
                    avg = epoch_loss / n_samples
                    lam = torch.nn.functional.softplus(
                        model.dense_cross_attn.lambda_param
                    ).item()
                    print(f"  Epoch {epoch} | Step {i+1}/{len(dataset)} | "
                          f"Loss: {avg:.4f} | λ: {lam:.4f}")

            except Exception as e:
                print(f"  [error] sample {i}: {e}")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

        # 残余梯度
        if n_samples % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_samples, 1)
        print(f"\n  Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Samples: {n_samples}")

        save_checkpoint(f"epoch{epoch}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint("best")
            print(f"  New best! Loss: {best_loss:.4f}")

    save_checkpoint("final")

    # 保存训练配置
    config = {
        "method": "Dense Cross-Attention Baseline",
        "model_id": args.model_id,
        "inject_layer": args.inject_layer,
        "top_n_patches": args.top_n_patches,
        "lr": args.lr,
        "epochs": args.epochs,
        "grad_accum": args.grad_accum,
        "trainable_params": total_params,
        "best_loss": best_loss,
    }
    with open(os.path.join(args.save_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Checkpoints saved to {args.save_dir}/")


if __name__ == "__main__":
    train()