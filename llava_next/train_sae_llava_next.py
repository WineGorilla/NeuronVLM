"""
SAE 训练入口 — LLaVA-NeXT-LLaMA3 版（修正版）。
修正点：
  1. SAE 加入 b1/b2/b3 偏置项（论文公式）
  2. 加入 auxiliary loss 防止 dead features
  3. 加入 gradient accumulation（论文用 8*4=32 effective batch）
  4. vision token 提取改用 mask 方式（兼容 anyres）
  5. TopK SAE 不再加额外 L1 sparsity

用法：
    python -m llava_next.train_sae_llava_next --max_steps 10000
    CUDA_VISIBLE_DEVICES=0 python -m llava_next.train_sae_llava_next


    CUDA_VISIBLE_DEVICES=2 python scripts/build_cache_new.py --layer 20
"""
import os
import sys
import signal
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import transformers

from llava_next.config_llava import CFG
from llava_next.SAE_llava import SAE
from llava_next.dataset_llava_next import VisionTextDataset, build_collate

transformers.logging.set_verbosity_error()

# ── 路径配置 ──────────────────────────────────────────────────
LLAVA_NEXT_MODEL_ID = getattr(CFG, "llava_next_model_id", "llava-hf/llama3-llava-next-8b-hf")
SAVE_DIR            = getattr(CFG, "save_llava_next_dir", "outputs/sae_llava_next")

# ── 训练超参数（可在 config 中覆盖）─────────────────────────────
GRAD_ACCUM   = getattr(CFG, "grad_accum_steps", 4)   # 论文: 8 batch * 4 accum = 32
AUX_COEF     = getattr(CFG, "aux_coef", 1/32)        # auxiliary loss 系数
DEAD_THRESH  = getattr(CFG, "dead_threshold", 40)     # 多少 batch 没激活算 dead


class UniqueImageDataset(VisionTextDataset):
    """只保留每张图片的第一个 sample，去重。"""
    def __init__(self, path: str):
        super().__init__(path)
        seen = set()
        deduped = []
        for s in self.samples:
            if s["image"] not in seen:
                seen.add(s["image"])
                deduped.append(s)
        print(f"  Dataset dedup: {len(self.samples)} samples -> {len(deduped)} unique images")
        self.samples = deduped


def extract_vision_hidden(hidden_state, input_ids, image_token_id):
    """
    从 hidden_state 中提取所有 vision token。
    用 mask 方式，兼容 anyres 下 vision tokens 不连续的情况。
    返回 (N_total_vision_tokens, hidden_dim) 或 None。
    """
    parts = []
    for b in range(hidden_state.shape[0]):
        mask = (input_ids[b] == image_token_id)
        if mask.any():
            parts.append(hidden_state[b][mask])
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最多训练 N 个 optimizer step 后停止")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 加载 LLaVA-NeXT-LLaMA3 ───────────────────────────────
    print(f"Loading LLaVA-NeXT-LLaMA3: {LLAVA_NEXT_MODEL_ID}")
    processor = LlavaNextProcessor.from_pretrained(LLAVA_NEXT_MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_NEXT_MODEL_ID,
        torch_dtype=torch.float16,
    ).to(CFG.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── hidden_dim & SAE ──────────────────────────────────────
    hidden_dim = model.config.text_config.hidden_size
    latent_dim = hidden_dim * CFG.latent_mult
    print(f"hidden_dim={hidden_dim}, latent_dim={latent_dim}")

    saes = {
        l: SAE(hidden_dim, latent_dim, CFG.topk).float().to(CFG.device)
        for l in CFG.layers
    }

    sae_params = [p for sae in saes.values() for p in sae.parameters()]
    optimizer  = torch.optim.AdamW(sae_params, lr=CFG.lr)

    dataset = UniqueImageDataset(CFG.train_file)
    loader  = DataLoader(
        dataset,
        batch_size = CFG.batch_size,
        shuffle    = True,
        collate_fn = build_collate(processor),
    )

    image_token_id = getattr(
        model.config, "image_token_index",
        processor.tokenizer.convert_tokens_to_ids("<image>"),
    )
    save_every = getattr(CFG, "save_every", 5000)

    print(f"Dataset size       : {len(dataset)} (unique images)")
    print(f"Image token ID     : {image_token_id}")
    print(f"Layers             : {CFG.layers}")
    print(f"Grad accumulation  : {GRAD_ACCUM}")
    print(f"Effective batch    : {CFG.batch_size * GRAD_ACCUM}")
    print(f"Aux loss coef      : {AUX_COEF}")
    print(f"Save every         : {save_every} steps")

    def save_checkpoints(tag="latest"):
        for l in CFG.layers:
            path = os.path.join(SAVE_DIR, f"sae_layer{l}_{tag}.pt")
            torch.save(saes[l].state_dict(), path)
            print(f"  saved {path}")
            latest_path = os.path.join(SAVE_DIR, f"sae_layer{l}.pt")
            torch.save(saes[l].state_dict(), latest_path)

    def handle_sigint(sig, frame):
        print("\n[interrupted] saving checkpoints before exit...")
        save_checkpoints("interrupted")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    optimizer_step = 0
    micro_step     = 0
    accum_loss     = 0.0
    optimizer.zero_grad()

    for epoch in range(CFG.sae_epochs):
        print(f"\nEpoch {epoch}")

        for step, batch in enumerate(loader):
            batch = {
                k: v.to(CFG.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                outputs = model(
                    **{k: v for k, v in batch.items() if torch.is_tensor(v)},
                    output_hidden_states=True,
                    return_dict=True,
                )

            input_ids = batch["input_ids"]

            loss_total = 0.0
            skip       = False

            for l in CFG.layers:
                h = outputs.hidden_states[l + 1]

                # ── 提取 vision tokens（mask 方式，兼容 anyres）──
                img_tokens = extract_vision_hidden(h, input_ids, image_token_id)

                if img_tokens is None or img_tokens.shape[0] == 0:
                    print(f"  [skip] step {step}: no vision tokens found")
                    skip = True
                    break

                img_tokens = img_tokens.float()

                if torch.isnan(img_tokens).any() or torch.isinf(img_tokens).any():
                    print(f"  [skip] step {step} layer {l}: nan/inf")
                    skip = True
                    break

                sae = saes[l]
                recon, z = sae(img_tokens)

                # ── 论文 loss: MSE + auxiliary（不加额外 L1）────
                recon_loss = F.mse_loss(recon, img_tokens)
                aux_loss   = sae.auxiliary_loss(img_tokens, recon, DEAD_THRESH)
                loss       = recon_loss + AUX_COEF * aux_loss

                # 更新 dead feature 统计
                sae.update_fired_stats(z)

                loss_total += loss

            if skip:
                continue

            loss_total /= len(CFG.layers)

            # ── gradient accumulation ─────────────────────────
            (loss_total / GRAD_ACCUM).backward()
            accum_loss += loss_total.item()
            micro_step += 1

            if micro_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(sae_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

                for sae in saes.values():
                    sae.normalize_decoder()

                if optimizer_step % save_every == 0:
                    save_checkpoints(f"step{optimizer_step}")

                if optimizer_step % 10 == 0:
                    avg = accum_loss / GRAD_ACCUM
                    n_dead = sum(
                        (sae.num_batches_since_fired >= DEAD_THRESH).sum().item()
                        for sae in saes.values()
                    )
                    print(f"  step {optimizer_step:6d} | loss {avg:.6f} | dead {n_dead}")
                    accum_loss = 0.0

                if args.max_steps and optimizer_step >= args.max_steps:
                    print(f"\n  Reached max_steps={args.max_steps}, stopping.")
                    save_checkpoints(f"step{optimizer_step}")
                    print("\nTraining complete.")
                    return

        save_checkpoints(f"epoch{epoch}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()