"""
训练 FeatureRouter。

冻结：Qwen 所有参数 + SAE 所有参数
训练：各层 FeatureRouter（参数量极小，收敛快）

损失：标准 Causal LM loss（只对 assistant 回答部分计算）
      Router 选对了 feature → 模型回答更准 → loss 更低 → Router 得到正向梯度

用法：
    python -m src.train_router
    python -m src.train_router --resume best    # 从已有 checkpoint 继续
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import CFG
from src.dataset import VisionTextDataset, build_collate
from src.model_with_sae import QwenWithSAERouter


# ── 训练超参 ──────────────────────────────────────────────────────────────────

LR         = 1e-4
EPOCHS     = 3
LOG_EVERY  = 10
SAVE_EVERY = 200
GRAD_ACCUM = 8   # 实际 batch_size=1，有效 batch_size = 8


# ── 获取 image / text token 位置 ─────────────────────────────────────────────

def get_token_positions(inputs, model, processor):
    """
    当前实现假设 batch_size = 1。

    返回：
        vision_pos:     image token 起始位置
        num_img_tokens: image token 数量
        text_positions: text token 的位置索引 (Tensor)

    注意：
        当前 text_positions 是“除去图像区域和 special token 后的 token”。
        如果后续你想只保留 user question，可以再进一步精确截取 user 段。
    """
    image_grid = inputs["image_grid_thw"]   # (1, 3) -> [T, H, W]
    H_grid = image_grid[0, 1].item()
    W_grid = image_grid[0, 2].item()

    spatial_merge = model.base_model.config.vision_config.spatial_merge_size
    num_img_tokens = int(H_grid * W_grid / (spatial_merge ** 2))

    input_ids = inputs["input_ids"][0]

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_pos = (input_ids == vision_start_id).nonzero(as_tuple=False)[0].item() + 1

    special_ids = {
        x for x in [
            processor.tokenizer.convert_tokens_to_ids("<|im_start|>"),
            processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            processor.tokenizer.pad_token_id,
        ]
        if x is not None
    }

    text_mask = torch.ones(len(input_ids), dtype=torch.bool)
    text_mask[vision_pos: vision_pos + num_img_tokens] = False

    for sid in special_ids:
        text_mask = text_mask & (input_ids != sid)

    text_positions = text_mask.nonzero(as_tuple=False).squeeze(-1)

    return vision_pos, num_img_tokens, text_positions


# ── 构建 labels：只对 assistant 回答计算 loss ─────────────────────────────────

def build_labels(input_ids: torch.Tensor, processor) -> torch.Tensor:
    """
    Qwen chat template 结构大致为：
        <|im_start|>system\n...<|im_end|>
        <|im_start|>user\n...<|im_end|>
        <|im_start|>assistant\n {answer} <|im_end|>

    只对最后一个 assistant 段计算 loss，其余全部 mask 为 -100。
    """
    labels = input_ids.clone()

    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_ids = processor.tokenizer.encode("assistant", add_special_tokens=False)

    for b in range(input_ids.shape[0]):
        ids = input_ids[b].tolist()
        answer_start = None

        # 从后往前找最后一个 <|im_start|> assistant
        for i in range(len(ids) - 2, 0, -1):
            if ids[i] == im_start_id:
                if ids[i + 1: i + 1 + len(assistant_ids)] == assistant_ids:
                    # +1: im_start
                    # +len(assistant_ids): "assistant"
                    # +1: 通常后面会有一个换行 token
                    answer_start = i + 1 + len(assistant_ids) + 1
                    break

        if answer_start is None:
            labels[b, :] = -100
        else:
            labels[b, :answer_start] = -100

    return labels


# ── 单步前向，返回 loss / GRAD_ACCUM ─────────────────────────────────────────

def forward_step(model, batch, processor):
    """
    只做 forward，返回已经除以 GRAD_ACCUM 的 loss。
    这样外层做 gradient accumulation 时，梯度 scale 保持正确。
    """
    input_ids = batch["input_ids"]

    vision_pos, num_img_tokens, text_positions = get_token_positions(
        batch, model, processor
    )

    model.set_context(
        vision_pos=vision_pos,
        num_img_tokens=num_img_tokens,
        text_positions=text_positions.to(CFG.device),
    )

    labels = build_labels(input_ids, processor).to(CFG.device)

    try:
        outputs = model(
            **{k: v for k, v in batch.items() if torch.is_tensor(v)},
            labels=labels,
            return_dict=True,
        )
        return outputs.loss / GRAD_ACCUM
    finally:
        # 无论 forward 是否报错，都清掉 side-channel context
        model.clear_context()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="加载已有 router checkpoint 的 tag，如 'best'"
    )
    args = parser.parse_args()

    router_save_dir = os.path.join(CFG.save_dir, "routers")
    os.makedirs(router_save_dir, exist_ok=True)

    # ── 加载 Qwen ──────────────────────────────────────────────
    print("Loading Qwen2.5-VL...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id,
        torch_dtype=torch.float16,
    ).to(CFG.device)

    # ── 构建完整模型 ───────────────────────────────────────────
    print("\nBuilding QwenWithSAERouter...")
    model = QwenWithSAERouter(
        base_model=base_model,
        layers=CFG.layers,
        sae_ckpt_dir=CFG.save_dir,
        latent_mult=CFG.latent_mult,
        topk=CFG.topk,
        topk_route=64,
        max_alpha=3.0,
    ).to(CFG.device)

    # 可选：从已有 Router checkpoint 恢复
    if args.resume:
        model.load_routers(router_save_dir, tag=args.resume)

    # ── 优化器（只优化 Router）────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.router_parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    # ── 数据（固定 batch_size=1）──────────────────────────────
    dataset = VisionTextDataset(CFG.train_file)
    loader = DataLoader(
        dataset,
        batch_size=1,   # 当前 token 位置逻辑默认 batch_size=1
        shuffle=True,
        collate_fn=build_collate(processor),
    )

    print(f"\nDataset   : {len(dataset)} samples")
    print(f"Grad accum: {GRAD_ACCUM}  (effective batch size = {GRAD_ACCUM})")
    print("Start training...\n")

    # ── 训练循环 ───────────────────────────────────────────────
    global_step = 0
    best_loss = float("inf")

    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        print(f"{'=' * 50}")
        print(f"Epoch {epoch + 1} / {EPOCHS}")
        print(f"{'=' * 50}")

        epoch_loss = 0.0
        valid_steps = 0
        accum_loss = 0.0
        micro_count = 0

        for step, batch in enumerate(loader):
            batch = {
                k: v.to(CFG.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            try:
                loss = forward_step(model, batch, processor)
                loss.backward()
                accum_loss += loss.item()
                micro_count += 1
            except Exception as e:
                print(f"  [skip] step {step}: {e}")
                optimizer.zero_grad()
                accum_loss = 0.0
                micro_count = 0
                continue

            # 每 GRAD_ACCUM 个 micro-steps 更新一次
            if micro_count == GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(
                    model.router_parameters(),
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

                # 由于每个 micro loss 都已经 / GRAD_ACCUM，
                # accum_loss 约等于一个 optimizer step 对应的平均 loss
                effective_loss = accum_loss

                epoch_loss += effective_loss
                valid_steps += 1
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    print(f"  global_step {global_step:5d} | loss {effective_loss:.4f}")

                if global_step % SAVE_EVERY == 0:
                    model.save_routers(router_save_dir, tag=f"step{global_step}")

                accum_loss = 0.0
                micro_count = 0

        # ── epoch 结束，flush 剩余未满 GRAD_ACCUM 的梯度 ─────────
        if micro_count > 0:
            torch.nn.utils.clip_grad_norm_(
                model.router_parameters(),
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

            effective_loss = accum_loss

            epoch_loss += effective_loss
            valid_steps += 1
            global_step += 1

            if global_step % LOG_EVERY == 0:
                print(f"  global_step {global_step:5d} | loss {effective_loss:.4f}")

            if global_step % SAVE_EVERY == 0:
                model.save_routers(router_save_dir, tag=f"step{global_step}")

            accum_loss = 0.0
            micro_count = 0

        if valid_steps == 0:
            print("  No valid optimization steps in this epoch.")
            continue

        avg_loss = epoch_loss / valid_steps
        print(f"\n  Epoch {epoch + 1} avg loss : {avg_loss:.4f}")

        # 每个 epoch 保存一次
        model.save_routers(router_save_dir, tag=f"epoch{epoch + 1}")

        # 保存最优
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_routers(router_save_dir, tag="best")
            print(f"  New best loss: {best_loss:.4f}")

    print("\nTraining complete.")
    print(f"Best loss    : {best_loss:.4f}")
    print(f"Router saved : {router_save_dir}")


if __name__ == "__main__":
    main()