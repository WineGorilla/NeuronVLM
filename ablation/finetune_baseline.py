"""
Ablation Baseline: 纯 finetune，无架构改动

用途：
  验证 CV-Bench 的提升到底来自架构（SAE routing + semantic completion + PCA suppression）
  还是来自 finetune 数据本身。

做法：
  用完全相同的数据、相同的超参数（lr, epochs, grad_accum, weight_decay），
  只 finetune Qwen 的 top-8 层 + lm_head，不加任何 hook / 额外模块。
  然后跑 CV-Bench 对比。

  如果 baseline 也涨到 ~78，说明提升来自数据。
  如果 baseline 停在 ~77，说明架构贡献了那 1+ 个点。

用法：
  python -m ablation.finetune_baseline \
      --data data/train_cluster.jsonl \
      --output_dir outputs/ablation_baseline
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import random
import signal
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import CFG
from src.dataset import VisionTextDataset


# ── 超参数（与 train_focus Stage 2 完全一致）─────────────────────────────────
LR          = 1e-5
EPOCHS      = 3
GRAD_ACCUM  = 8
LOG_EVERY   = 10
SAVE_EVERY  = 500
N_UNFREEZE  = 8     # 解冻最后 8 层


def build_inputs(processor, image_path, question, answer, device):
    """构建训练输入（与 Model._build_inputs 逻辑一致）"""
    from qwen_vl_utils import process_vision_info

    msg = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text":  question},
        ],
    }]
    msg.append({"role": "assistant", "content": answer})
    messages = [msg]

    texts = [processor.apply_chat_template(
        m, tokenize=False, add_generation_prompt=False
    ) for m in messages]
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(
        text=texts, images=img_in, videos=vid_in,
        padding=True, return_tensors="pt",
    )
    return inputs.to(device)


def build_labels(input_ids, processor, device):
    """只对 assistant 回复部分计算 LM loss（与 Model._build_labels 一致）"""
    labels = input_ids.clone()
    im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    ast_ids  = processor.tokenizer.encode("assistant", add_special_tokens=False)

    for b in range(input_ids.shape[0]):
        ids = input_ids[b].tolist()
        start = None
        for i in range(len(ids) - 2, -1, -1):
            if ids[i] == im_start and ids[i+1:i+1+len(ast_ids)] == ast_ids:
                start = i + 1 + len(ast_ids) + 1
                break
        labels[b, :start if start else len(ids)] = -100

    return labels.to(device)


def main():
    parser = argparse.ArgumentParser(description="Ablation: finetune baseline (no hooks)")
    parser.add_argument("--data",       type=str, default="data/train_cluster.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation_baseline")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--n_unfreeze", type=int, default=N_UNFREEZE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = CFG.device

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    print(f"Loading Qwen: {CFG.model_id}...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.bfloat16,
    ).to(device)

    # ── 冻结 / 解冻 ─────────────────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad = False

    layers = model.model.language_model.layers
    n_layers = len(layers)
    trainable_layer_ids = list(range(n_layers - args.n_unfreeze, n_layers))

    for i in trainable_layer_ids:
        for p in layers[i].parameters():
            p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)

    print(f"  Unfrozen layers: {trainable_layer_ids}")
    print(f"  Trainable: {trainable:,} params")

    # ── 数据 ─────────────────────────────────────────────────────────────────
    dataset = VisionTextDataset(args.data)
    print(f"  Dataset: {len(dataset)} samples")

    # ── 优化器（与 train_focus Stage 2 完全一致）──────────────────────────────
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # ── 保存 ─────────────────────────────────────────────────────────────────
    def save(tag):
        path = os.path.join(args.output_dir, f"qwen_{tag}.pt")
        state = {
            k: v for k, v in model.state_dict().items()
            if any(k.startswith(f"model.language_model.layers.{i}.") for i in trainable_layer_ids)
            or k.startswith("lm_head.")
        }
        torch.save(state, path)
        print(f"  Saved: {path}")

    def handle_sigint(sig, frame):
        print("\n[interrupted] saving...")
        save("interrupted")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    global_step = 0
    best_loss   = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        print(f"\n{'='*50}\nEpoch {epoch+1}/{args.epochs}\n{'='*50}")

        epoch_loss  = 0.0
        valid_steps = 0
        accum_loss  = 0.0
        micro_count = 0

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for step, idx in enumerate(indices):
            item = dataset[idx]
            try:
                inputs = build_inputs(
                    processor, item["image"], item["question"],
                    item.get("answer", ""), device,
                )
                labels = build_labels(inputs["input_ids"], processor, device)

                out = model(**inputs, labels=labels, return_dict=True)
                total_loss = out.loss

                if step % LOG_EVERY == 0:
                    print(f"  step {step:5d} | lm_loss={total_loss.item():.4f}")

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"  [skip] nan/inf at step {step}")
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss  = 0.0
                    micro_count = 0
                    continue

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
    print(f"Weights saved to: {args.output_dir}/")
    print(f"\nTo evaluate on CV-Bench, load with:")
    print(f"  --qwen_ckpt {args.output_dir}/qwen_best.pt")


if __name__ == "__main__":
    main()