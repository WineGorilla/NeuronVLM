"""
SAE 训练入口。
用法：python -m src.train
"""
import os
import sys
import signal
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from src.SAE import SAE
from src.dataset import VisionTextDataset, build_collate


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


def train():
    os.makedirs(CFG.save_dir, exist_ok=True)

    print("Loading model:", CFG.model_id)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id,
        torch_dtype=torch.float16,
    ).to(CFG.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

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

    spatial_merge   = model.config.vision_config.spatial_merge_size
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    save_every      = getattr(CFG, "save_every", 5000)

    print(f"Dataset size       : {len(dataset)} (unique images)")
    print(f"Save every         : {save_every} steps")

    # ── 保存函数 ───────────────────────────────────────────────
    def save_checkpoints(tag="latest"):
        for l in CFG.layers:
            path = os.path.join(CFG.save_dir, f"sae_layer{l}_{tag}.pt")
            torch.save(saes[l].state_dict(), path)
            print(f"  saved {path}")
            latest_path = os.path.join(CFG.save_dir, f"sae_layer{l}.pt")
            torch.save(saes[l].state_dict(), latest_path)

    # ── Ctrl+C 自动保存 ────────────────────────────────────────
    def handle_sigint(sig, frame):
        print("\n[interrupted] saving checkpoints before exit...")
        save_checkpoints("interrupted")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # ── 训练循环 ───────────────────────────────────────────────
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
                    output_hidden_states = True,
                    return_dict          = True,
                )

            loss_total = 0.0
            skip       = False

            for l in CFG.layers:
                h = outputs.hidden_states[l + 1]

                image_grid     = batch["image_grid_thw"]
                num_img_tokens = int(
                    image_grid[0, 1] * image_grid[0, 2] / (spatial_merge ** 2)
                )
                input_ids  = batch["input_ids"][0]
                vision_pos = (input_ids == vision_start_id).nonzero()[0].item() + 1

                img_tokens = h[:, vision_pos : vision_pos + num_img_tokens, :]
                flat       = img_tokens.reshape(-1, img_tokens.shape[-1]).float()

                # nan 检测
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

        save_checkpoints(f"epoch{epoch}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()