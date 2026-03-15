"""
训练 FeatureRouter。

冻结：Qwen 所有参数 + SAE 所有参数
训练：各层 FeatureRouter（参数量极小，收敛快）

损失：
    普通模式：total_loss = lm_loss
    监督模式：total_loss = lm_loss + λ * router_loss
              router_loss = 鼓励 Router 对 target_features 输出更高的 alpha

用法：
    python -m src.train_router                       # 普通模式
    python -m src.train_router --supervised          # 监督模式
    python -m src.train_router --resume best         # 断点续跑
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import CFG
from src.dataset import VisionTextDataset, build_collate
from src.Model import QwenWithSAERouter


# ── 超参 ─────────────────────────────────────────────────────

LR            = 1e-4
EPOCHS        = 3
LOG_EVERY     = 10
SAVE_EVERY    = 200
GRAD_ACCUM    = 8
LAMBDA_ROUTER = 0.1    # Router 监督损失权重


# ── 工具函数 ─────────────────────────────────────────────────

def get_model_device(model):
    return next(model.parameters()).device


def get_token_positions(inputs, model, processor):
    """batch_size=1，返回 vision_pos、num_img_tokens、text_positions。"""
    device = get_model_device(model)

    if "image_grid_thw" not in inputs:
        raise KeyError("inputs missing `image_grid_thw`")

    image_grid = inputs["image_grid_thw"]
    if image_grid.ndim != 2 or image_grid.shape[0] != 1:
        raise ValueError(
            f"Expects batch_size=1, got shape={tuple(image_grid.shape)}"
        )

    H_grid         = image_grid[0, 1].item()
    W_grid         = image_grid[0, 2].item()
    spatial_merge  = model.base_model.config.vision_config.spatial_merge_size
    num_img_tokens = int(H_grid * W_grid / (spatial_merge ** 2))

    input_ids       = inputs["input_ids"][0]
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")

    vision_positions = (input_ids == vision_start_id).nonzero(as_tuple=False)
    if vision_positions.numel() == 0:
        raise ValueError("No <|vision_start|> token found in input_ids")
    vision_pos = vision_positions[0].item() + 1

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

    text_mask = torch.ones(len(input_ids), dtype=torch.bool, device=input_ids.device)
    text_mask[vision_pos : vision_pos + num_img_tokens] = False
    for sid in special_ids:
        text_mask = text_mask & (input_ids != sid)

    text_positions = text_mask.nonzero(as_tuple=False).squeeze(-1).to(device)
    return vision_pos, num_img_tokens, text_positions


def build_labels(input_ids: torch.Tensor, processor) -> torch.Tensor:
    """只对最后一个 assistant 段计算 loss，其余置为 -100。"""
    labels        = input_ids.clone()
    im_start_id   = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_ids = processor.tokenizer.encode("assistant", add_special_tokens=False)

    for b in range(input_ids.shape[0]):
        ids          = input_ids[b].tolist()
        answer_start = None
        for i in range(len(ids) - 2, -1, -1):
            if ids[i] == im_start_id:
                if ids[i + 1 : i + 1 + len(assistant_ids)] == assistant_ids:
                    answer_start = i + 1 + len(assistant_ids) + 1
                    break
        if answer_start is None:
            labels[b, :] = -100
        else:
            labels[b, :answer_start] = -100

    return labels


def compute_router_loss(model, target_features_dict: dict, device: str) -> torch.Tensor:
    """
    对每一层的 Router，计算监督损失。
    鼓励 Router 对 target_features 输出更高的 alpha。

    target_features_dict: {"8": [1024, 3821], "24": [2731, 5012]}
    """
    router_loss = torch.tensor(0.0, device=device)
    count       = 0

    for layer_str, layer in model.wrapped_layers.items():
        if layer_str not in target_features_dict:
            continue

        target_ids = target_features_dict[layer_str]
        if not target_ids:
            continue

        if not hasattr(layer, "last_alpha") or layer.last_alpha is None:
            continue

        alpha      = layer.last_alpha                # (num_tokens, latent_dim)
        target_ids = torch.tensor(
            [t for t in target_ids if t < alpha.shape[-1]],
            dtype=torch.long,
            device=device,
        )

        if len(target_ids) == 0:
            continue

        # 最大化 target feature 的平均 alpha
        target_alpha = alpha[:, target_ids].mean()
        router_loss  = router_loss - target_alpha    # 负号：最大化 = 最小化负值
        count       += 1

    if count > 0:
        router_loss = router_loss / count

    return router_loss


# ── 单步前向 ─────────────────────────────────────────────────

def forward_step(
    model,
    batch,
    processor,
    target_features_dict: dict = None,
) -> torch.Tensor:
    device = get_model_device(model)

    vision_pos, num_img_tokens, text_positions = get_token_positions(
        batch, model, processor
    )
    model.set_context(
        vision_pos     = vision_pos,
        num_img_tokens = num_img_tokens,
        text_positions = text_positions,
    )

    labels = build_labels(batch["input_ids"], processor).to(device)

    try:
        tensor_batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        outputs  = model(**tensor_batch, labels=labels, return_dict=True)
        lm_loss  = outputs.loss

        # Router 监督损失（有 target_features 时才计算）
        if target_features_dict:
            router_loss = compute_router_loss(model, target_features_dict, device)
            total_loss  = lm_loss + LAMBDA_ROUTER * router_loss
        else:
            total_loss  = lm_loss

        return total_loss / GRAD_ACCUM

    finally:
        model.clear_context()


# ── 主流程 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",     type=str,  default=None,
                        help="加载已有 router checkpoint 的 tag")
    parser.add_argument("--supervised", action="store_true",
                        help="使用 train_supervised.jsonl（含 target_features）")
    parser.add_argument("--sup_file",   type=str,
                        default="data/train_supervised.jsonl",
                        help="监督数据集路径")
    args = parser.parse_args()

    sae_ckpt_dir    = CFG.save_dir
    router_save_dir = os.path.join(CFG.save_dir, "routers")
    os.makedirs(router_save_dir, exist_ok=True)

    print("Loading Qwen2.5-VL...")
    processor  = AutoProcessor.from_pretrained(CFG.model_id)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16,
    ).to(CFG.device)

    print("\nBuilding QwenWithSAERouter...")
    model = QwenWithSAERouter(
        base_model   = base_model,
        layers       = CFG.layers,
        sae_ckpt_dir = sae_ckpt_dir,
        latent_mult  = CFG.latent_mult,
        topk         = CFG.topk,
        topk_route   = 64,
        max_alpha    = 3.0,
    ).to(CFG.device)

    if args.resume:
        model.load_routers(router_save_dir, tag=args.resume)

    router_params = model.router_parameters()
    if len(router_params) == 0:
        raise RuntimeError("No router parameters found.")

    optimizer = torch.optim.AdamW(
        router_params, lr=LR, weight_decay=1e-4,
    )

    # 选择数据集
    if args.supervised and os.path.exists(args.sup_file):
        print(f"Using supervised dataset: {args.sup_file}")
        dataset = VisionTextDataset(args.sup_file)
    else:
        if args.supervised:
            print(f"[warn] supervised file not found: {args.sup_file}")
            print(f"Falling back to: {CFG.train_file}")
        dataset = VisionTextDataset(CFG.train_file)

    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle    = True,
        collate_fn = build_collate(processor),
    )

    print(f"\nDataset   : {len(dataset)} samples")
    print(f"Grad accum: {GRAD_ACCUM}  (effective batch size = {GRAD_ACCUM})")
    print(f"Mode      : {'supervised' if args.supervised else 'standard'}")
    print("Start training...\n")

    global_step = 0
    best_loss   = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        print("=" * 50)
        print(f"Epoch {epoch + 1} / {EPOCHS}")
        print("=" * 50)

        epoch_loss  = 0.0
        valid_steps = 0
        accum_loss  = 0.0
        micro_count = 0

        for step, batch in enumerate(loader):
            # 提取 target_features（监督模式下 collate 会附加）
            target_features_dict = None
            if args.supervised and "target_features" in batch:
                raw = batch.pop("target_features")
                # raw 是 list（batch 维度），取第一个（batch_size=1）
                tf = raw[0] if isinstance(raw, list) else raw
                if tf is not None:
                    # 确保 key 是 str
                    target_features_dict = {str(k): v for k, v in tf.items()}

            batch = {
                k: v.to(CFG.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            try:
                loss = forward_step(model, batch, processor, target_features_dict)
                loss.backward()
                accum_loss  += loss.item()
                micro_count += 1
            except Exception as e:
                print(f"  [skip] step {step}: {e}")
                optimizer.zero_grad(set_to_none=True)
                accum_loss  = 0.0
                micro_count = 0
                continue

            if micro_count == GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(router_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                effective_loss = accum_loss
                epoch_loss    += effective_loss
                valid_steps   += 1
                global_step   += 1

                if global_step % LOG_EVERY == 0:
                    print(f"  global_step {global_step:5d} | loss {effective_loss:.4f}")
                if global_step % SAVE_EVERY == 0:
                    model.save_routers(router_save_dir, tag=f"step{global_step}")

                accum_loss  = 0.0
                micro_count = 0

        # flush 剩余梯度
        if micro_count > 0:
            torch.nn.utils.clip_grad_norm_(router_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            effective_loss = accum_loss
            epoch_loss    += effective_loss
            valid_steps   += 1
            global_step   += 1

            if global_step % LOG_EVERY == 0:
                print(f"  global_step {global_step:5d} | loss {effective_loss:.4f}")
            if global_step % SAVE_EVERY == 0:
                model.save_routers(router_save_dir, tag=f"step{global_step}")

            accum_loss  = 0.0
            micro_count = 0

        if valid_steps == 0:
            print("  No valid optimization steps in this epoch.")
            continue

        avg_loss = epoch_loss / valid_steps
        print(f"\n  Epoch {epoch + 1} avg loss : {avg_loss:.4f}")

        model.save_routers(router_save_dir, tag=f"epoch{epoch + 1}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_routers(router_save_dir, tag="best")
            print(f"  New best loss: {best_loss:.4f}")

    print("\nTraining complete.")
    print(f"Best loss    : {best_loss:.4f}")
    print(f"Router saved : {router_save_dir}")


if __name__ == "__main__":
    main()