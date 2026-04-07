"""
SAE 训练入口 — LLaVA-NeXT-LLaMA3 版。
用法：
    python -m llava_next.train_sae_llava_next --max_steps 5000
    python -m llava_next.train_sae_llava_next --max_steps 25000
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

from config import CFG
from src.SAE import SAE
from llava_next.dataset_llava_next import VisionTextDataset, build_collate

transformers.logging.set_verbosity_error()

# ── 路径配置（需要在 config.py 中添加，或使用 fallback）────────────────────────
LLAVA_NEXT_MODEL_ID = getattr(CFG, "llava_next_model_id", "llava-hf/llama3-llava-next-8b-hf")
SAVE_DIR            = getattr(CFG, "save_llava_next_dir", "outputs/sae_llava_next")


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


def _find_vision_token_range(input_ids, image_token_id):
    image_mask = (input_ids == image_token_id)
    image_positions = image_mask.nonzero(as_tuple=False).squeeze(-1)

    if image_positions.numel() == 0:
        return None, 0

    v_pos = image_positions[0].item()
    n_img = image_positions.numel()
    return v_pos, n_img


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最多训练 N 个 optimizer step 后停止")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 加载 LLaVA-NeXT-LLaMA3 ───────────────────────────────────────────────
    print(f"Loading LLaVA-NeXT-LLaMA3: {LLAVA_NEXT_MODEL_ID}")
    processor = LlavaNextProcessor.from_pretrained(LLAVA_NEXT_MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_NEXT_MODEL_ID,
        torch_dtype=torch.float16,
    ).to(CFG.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── LLaMA3-8B: hidden_size=4096 ──────────────────────────────────────────
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

    # ── image_token_index: LLaMA3 版是 128256 ────────────────────────────────
    image_token_id = getattr(
        model.config, "image_token_index",
        processor.tokenizer.convert_tokens_to_ids("<image>"),
    )
    save_every = getattr(CFG, "save_every", 5000)

    print(f"Dataset size       : {len(dataset)} (unique images)")
    print(f"Image token ID     : {image_token_id}")
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

            input_ids = batch["input_ids"][0]
            v_pos, n_img = _find_vision_token_range(input_ids, image_token_id)

            if v_pos is None or n_img == 0:
                print(f"  [skip] step {step}: no vision tokens found")
                continue

            loss_total = 0.0
            skip       = False

            for l in CFG.layers:
                h = outputs.hidden_states[l + 1]

                img_tokens = h[:, v_pos:v_pos + n_img, :]
                flat       = img_tokens.reshape(-1, img_tokens.shape[-1]).float()

                if torch.isnan(flat).any() or torch.isinf(flat).any():
                    print(f"  [skip] step {step} layer {l}: nan/inf")
                    skip = True
                    break

                sae       = saes[l]
                recon, z  = sae(flat)

                loss = F.mse_loss(recon, flat) + CFG.sparsity_coef * z.abs().mean()
                loss_total += loss

            if skip:
                optimizer.zero_grad()
                continue

            loss_total /= len(CFG.layers)

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(sae_params, max_norm=1.0)
            optimizer.step()
            optimizer_step += 1

            for sae in saes.values():
                sae.normalize_decoder()

            if optimizer_step % save_every == 0:
                save_checkpoints(f"step{optimizer_step}")

            if step % 10 == 0:
                print(f"  step {step:6d} | loss {loss_total.item():.6f}")

            if args.max_steps and optimizer_step >= args.max_steps:
                print(f"\n  Reached max_steps={args.max_steps}, stopping.")
                save_checkpoints(f"step{optimizer_step}")
                print("\nTraining complete.")
                return

        save_checkpoints(f"epoch{epoch}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()