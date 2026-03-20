"""
训练 QwenWithFocusAndSAE。

训练逻辑：
    对每条数据 {image, question, answer, focus_clusters}：
    1. 第一阶段（冻结，只训练 <think> 输出）：
       只用问题文字 → <think>[ids]</think>
       LM loss 对 <think> 部分计算
    2. 第二阶段（联合训练）：
       用 focus_clusters 找最强 patch → 注入 → 计算 loss_inject
       不注入 → 计算 loss_base
       total = loss_inject + aux_lambda * relu(loss_inject - loss_base + margin)

用法：
    python -m src.train_focus                    # 阶段1：只训练 <think>
    python -m src.train_focus --joint            # 阶段2：联合训练
    python -m src.train_focus --joint --resume best
"""
import os
import sys
import signal
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import CFG
from src.SAE import SAE
from src.Model import QwenWithFocusAndSAE
from src.dataset import VisionTextDataset


LR         = 1e-5
EPOCHS     = 3
GRAD_ACCUM = 8
LOG_EVERY  = 10
SAVE_EVERY = 500
SAVE_DIR   = "outputs/focus_ckpt"


def build_labels_think(input_ids: torch.Tensor, processor) -> torch.Tensor:
    """
    只对 <think>...</think> 部分计算 loss。
    用于阶段1训练。
    """
    from src.dataset import THINK_START, THINK_END
    labels    = torch.full_like(input_ids, -100)
    think_start_ids = processor.tokenizer.encode(THINK_START, add_special_tokens=False)
    think_end_ids   = processor.tokenizer.encode(THINK_END,   add_special_tokens=False)

    for b in range(input_ids.shape[0]):
        ids = input_ids[b].tolist()
        # 找 <think> 和 </think> 的位置
        start_pos = None
        for i in range(len(ids) - len(think_start_ids)):
            if ids[i:i+len(think_start_ids)] == think_start_ids:
                start_pos = i
                break
        if start_pos is None:
            continue
        end_pos = None
        for i in range(start_pos, len(ids) - len(think_end_ids)):
            if ids[i:i+len(think_end_ids)] == think_end_ids:
                end_pos = i + len(think_end_ids)
                break
        if end_pos is None:
            continue
        labels[b, start_pos:end_pos] = input_ids[b, start_pos:end_pos]

    return labels


def build_labels_answer(input_ids: torch.Tensor, processor) -> torch.Tensor:
    """只对最后一个 assistant 答案部分计算 loss。"""
    labels        = input_ids.clone()
    im_start_id   = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_ids = processor.tokenizer.encode("assistant", add_special_tokens=False)

    for b in range(input_ids.shape[0]):
        ids          = input_ids[b].tolist()
        answer_start = None
        for i in range(len(ids) - 2, -1, -1):
            if ids[i] == im_start_id:
                if ids[i+1:i+1+len(assistant_ids)] == assistant_ids:
                    answer_start = i + 1 + len(assistant_ids) + 1
                    break
        if answer_start is None:
            labels[b, :] = -100
        else:
            labels[b, :answer_start] = -100

    return labels


def collate_think_only(processor):
    """
    阶段1的 collate：只用问题文字构建输入，输出 <think>。
    """
    from src.dataset import THINK_START, THINK_END

    def collate(batch):
        messages = []
        for x in batch:
            focus_clusters = x.get("focus_clusters", [])
            question       = x["question"]
            ids_str        = ",".join(str(c) for c in focus_clusters)
            think_str      = f"{THINK_START}[{ids_str}]{THINK_END}"

            messages.append([
                {"role": "user",      "content": question},
                {"role": "assistant", "content": think_str},
            ])

        texts  = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages
        ]
        inputs = processor(text=texts, padding=True, return_tensors="pt")
        return inputs

    return collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint",  action="store_true", help="联合训练（阶段2）")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data",   type=str, default="data/train_cluster.jsonl")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    if args.joint:
        # 阶段2：加载完整模型
        cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{CFG.vis_layer}.json")
        focus_ckpt   = os.path.join(SAVE_DIR, f"model_{args.resume}.pt") if args.resume else None

        model = QwenWithFocusAndSAE.from_pretrained(
            model_id     = CFG.model_id,
            sae_ckpt_dir = CFG.save_dir,
            cluster_path = cluster_path,
            layer        = CFG.vis_layer,
            latent_mult  = CFG.latent_mult,
            topk         = CFG.topk,
            inject_scale = 0.3,
            aux_lambda   = 0.1,
            aux_margin   = 0.1,
            focus_ckpt   = focus_ckpt,
            device       = CFG.device,
        )
        processor = model.processor

        # 只训练最后 8 层 + lm_head
        for p in model.base_model.parameters():
            p.requires_grad = False
        layers = model.base_model.model.language_model.layers
        trainable_layer_ids = list(range(len(layers) - 8, len(layers)))
        for i in trainable_layer_ids:
            for p in layers[i].parameters():
                p.requires_grad = True
        for p in model.base_model.lm_head.parameters():
            p.requires_grad = True

        dataset    = VisionTextDataset(args.data)
        use_joint  = True

    else:
        # 阶段1：只训练 <think> 输出
        print("Loading Qwen2.5-VL...")
        processor = AutoProcessor.from_pretrained(CFG.model_id)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            CFG.model_id, torch_dtype=torch.float16,
        ).to(CFG.device)

        for p in base_model.parameters():
            p.requires_grad = False
        layers = base_model.model.language_model.layers
        trainable_layer_ids = list(range(len(layers) - 8, len(layers)))
        for i in trainable_layer_ids:
            for p in layers[i].parameters():
                p.requires_grad = True
        for p in base_model.lm_head.parameters():
            p.requires_grad = True

        if args.resume:
            ckpt = os.path.join(SAVE_DIR, f"model_{args.resume}.pt")
            if os.path.exists(ckpt):
                base_model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
                print(f"Resumed: {ckpt}")

        model     = base_model
        dataset   = VisionTextDataset(args.data)
        use_joint = False

    trainable = sum(p.numel() for p in (model.base_model if use_joint else model).parameters()
                    if p.requires_grad)
    total     = sum(p.numel() for p in (model.base_model if use_joint else model).parameters())
    print(f"Mode      : {'joint' if use_joint else 'think-only'}")
    print(f"Dataset   : {len(dataset)} samples")
    print(f"Trainable : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    params    = [p for p in (model.base_model if use_joint else model).parameters()
                 if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    if not use_joint:
        loader = DataLoader(
            dataset, batch_size=1, shuffle=True,
            collate_fn=collate_think_only(processor),
        )

    # ── 保存函数 ──────────────────────────────────────────────
    def save_model(tag: str):
        path  = os.path.join(SAVE_DIR, f"model_{tag}.pt")
        m     = model.base_model if use_joint else model
        state = {
            k: v for k, v in m.state_dict().items()
            if any(k.startswith(f"model.language_model.layers.{i}.") for i in trainable_layer_ids)
            or k.startswith("lm_head.")
        }
        torch.save(state, path)
        print(f"  saved: {path}")

    def handle_sigint(sig, frame):
        print("\n[interrupted] saving...")
        save_model("interrupted")
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

        if use_joint:
            # 联合训练：手动遍历数据
            import random
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            iterator = (dataset[i] for i in indices)
        else:
            iterator = iter(loader)

        for step, item in enumerate(iterator):
            try:
                if use_joint:
                    # 联合训练：直接调用 compute_loss
                    total_loss, l_inj, l_base, l_aux = model.compute_loss(
                        image_path     = item["image"],
                        question       = item["question"],
                        answer         = item["answer"],
                        focus_clusters = item.get("focus_clusters", []),
                    )
                    loss = total_loss / GRAD_ACCUM

                    if step % LOG_EVERY == 0:
                        print(f"  step {step} | "
                              f"total={total_loss.item():.4f} "
                              f"inject={l_inj:.4f} "
                              f"base={l_base:.4f} "
                              f"aux={l_aux:.4f}")
                else:
                    # 阶段1：只训练 <think>
                    batch  = item
                    batch  = {k: v.to(CFG.device) if torch.is_tensor(v) else v
                               for k, v in batch.items()}
                    labels = build_labels_think(batch["input_ids"], processor).to(CFG.device)
                    if (labels != -100).sum() == 0:
                        continue
                    outputs = model(
                        **{k: v for k, v in batch.items() if torch.is_tensor(v)},
                        labels=labels, return_dict=True,
                    )
                    total_loss = outputs.loss
                    loss       = total_loss / GRAD_ACCUM

                    if step % LOG_EVERY == 0:
                        print(f"  step {step} | loss={total_loss.item():.4f}")

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"  [skip] nan/inf at step {step}")
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0; micro_count = 0
                    continue

                loss.backward()
                accum_loss  += total_loss.item()
                micro_count += 1

            except Exception as e:
                print(f"  [skip] step {step}: {e}")
                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0; micro_count = 0
                continue

            if micro_count == GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                eff_loss    = accum_loss / GRAD_ACCUM
                epoch_loss += eff_loss
                valid_steps += 1
                global_step += 1

                if global_step % SAVE_EVERY == 0:
                    save_model(f"step{global_step}")

                accum_loss = 0.0; micro_count = 0

        # flush
        if micro_count > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += accum_loss / micro_count
            valid_steps += 1

        if valid_steps == 0:
            print("  No valid steps.")
            continue

        avg_loss = epoch_loss / valid_steps
        print(f"\n  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        save_model(f"epoch{epoch+1}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model("best")
            print(f"  New best: {best_loss:.4f}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()