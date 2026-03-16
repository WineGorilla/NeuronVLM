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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from config import CFG
from src.SAE import SAE
from src.dataset import VisionTextDataset, build_collate


def train():
    os.makedirs(CFG.save_dir, exist_ok=True)

    print("Loading model:", CFG.model_id)
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
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = CFG.epochs * 100000,
        eta_min = CFG.lr * 0.1,
    )

    # ── 保存函数 ───────────────────────────────────────────────
    def save_checkpoints(tag: str = "latest"):
        for l in CFG.layers:
            path = os.path.join(CFG.save_dir, f"sae_layer{l}_{tag}.pt")
            torch.save(saes[l].state_dict(), path)
            print(f"  saved {path}")

    # ── 注册 Ctrl+C 信号处理，中断时自动保存 ──────────────────
    def handle_sigint(sig, frame):
        print("\n[interrupted] saving checkpoints before exit...")
        save_checkpoints(tag="interrupted")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    dataset = VisionTextDataset(CFG.train_file)
    loader  = DataLoader(
        dataset,
        batch_size = CFG.batch_size,
        shuffle    = True,
        collate_fn = build_collate(processor),
    )

    spatial_merge   = model.config.vision_config.spatial_merge_size
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    grad_accum      = getattr(CFG, "grad_accum", 8)
    save_every      = getattr(CFG, "save_every", 5000)   # 每 N 个 optimizer step 保存一次

    print(f"Dataset    : {len(dataset)} samples")
    print(f"Grad accum : {grad_accum}  (effective batch size = {grad_accum})")
    print(f"Save every : {save_every} optimizer steps")

    optimizer.zero_grad()
    optimizer_step = 0

    for epoch in range(CFG.epochs):
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

            loss_total   = torch.tensor(0.0, device=CFG.device)
            recon_total  = torch.tensor(0.0, device=CFG.device)
            sparse_total = torch.tensor(0.0, device=CFG.device)
            skip         = False

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

                if torch.isnan(flat).any() or torch.isinf(flat).any():
                    print(f"  [skip] step {step} layer {l}: nan/inf in hidden state")
                    skip = True
                    break

                flat = F.layer_norm(flat, [flat.shape[-1]])

                sae      = saes[l]
                recon, z = sae(flat)

                if step % 100 == 0 and l == CFG.layers[0]:
                    print(
                        f"    z stats: mean={z.mean():.4f} "
                        f"std={z.std():.4f} "
                        f"abs_mean={z.abs().mean():.4f} "
                        f"nonzero={(z != 0).float().mean():.4f}"
                    )

                if torch.isnan(recon).any() or torch.isnan(z).any():
                    print(f"  [skip] step {step} layer {l}: nan in SAE output")
                    skip = True
                    break

                recon_loss  = F.mse_loss(recon, flat)
                sparse_loss = CFG.sparsity_coef * z.abs().mean()
                loss        = recon_loss + sparse_loss

                loss_total   += loss
                recon_total  += recon_loss
                sparse_total += sparse_loss

            if skip:
                optimizer.zero_grad()
                continue

            loss_total   /= len(CFG.layers)
            recon_total  /= len(CFG.layers)
            sparse_total /= len(CFG.layers)

            (loss_total / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(sae_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                for sae in saes.values():
                    sae.normalize_decoder()

                # 每 save_every 步保存一次
                if optimizer_step % save_every == 0:
                    save_checkpoints(tag=f"step{optimizer_step}")

            if step % 10 == 0:
                print(
                    f"  step {step:6d} | "
                    f"total={loss_total.item():.4f} "
                    f"recon={recon_total.item():.4f} "
                    f"sparse={sparse_total.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # 每 epoch 保存
        save_checkpoints(tag=f"epoch{epoch}")
        # 同时覆盖保存 latest（供后续流程直接使用）
        for l in CFG.layers:
            path = os.path.join(CFG.save_dir, f"sae_layer{l}.pt")
            torch.save(saes[l].state_dict(), path)
        print(f"  saved latest checkpoints")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()