"""
SAE 训练入口 — 从缓存的 hidden states 训练（极快版）。
不再需要加载 LLaVA 模型，直接读取预缓存的 vision token hidden states。

用法：
    # 先跑缓存（只需一次）：
    python -m llava_next.cache_hidden_states

    # 再跑训练（可以反复跑、调参）：
    python -m llava_next.train_sae_cached --max_steps 50000
"""
import os
import sys
import signal
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader

from llava_next.config_llava import CFG
from llava_next.SAE_llava import SAE

# ── 路径配置 ──────────────────────────────────────────────────
CACHE_DIR   = getattr(CFG, "llava_cache_dir", "outputs/llava_next")
SAVE_DIR    = getattr(CFG, "save_llava_next_dir", "outputs/sae_llava_next")
GRAD_ACCUM  = getattr(CFG, "grad_accum", 8)
AUX_COEF    = getattr(CFG, "aux_coef", 1/32)
DEAD_THRESH = getattr(CFG, "dead_threshold", 40)


# ── 缓存数据集 ────────────────────────────────────────────────
class CachedHiddenDataset(Dataset):
    """从磁盘加载预缓存的 vision token hidden states。"""

    def __init__(self, cache_dir: str, layer: int):
        pattern = os.path.join(cache_dir, f"hidden_layer{layer}_*.pt")
        self.files = sorted(glob.glob(pattern))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No cached files found at {pattern}\n"
                f"请先运行: python -m llava_next.cache_hidden_states"
            )
        print(f"  CachedHiddenDataset: {len(self.files)} files for layer {layer}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 返回 (N_vision_tokens, hidden_dim)，float16
        return torch.load(self.files[idx], map_location="cpu")


def collate_cached(batch):
    """
    把多个样本的 vision tokens 拼在一起。
    每个样本的 token 数可能不同（anyres），所以直接 cat。
    """
    return torch.cat(batch, dim=0)  # (total_tokens, hidden_dim)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最多训练 N 个 optimizer step 后停止")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 目前只支持单层（你的 CFG.layers 通常就一个）────────────────
    assert len(CFG.layers) == 1, "缓存训练模式目前只支持单层"
    layer = CFG.layers[0]

    # ── 从一个缓存文件读取 hidden_dim ─────────────────────────
    dataset = CachedHiddenDataset(CACHE_DIR, layer)
    sample  = dataset[0]
    hidden_dim = sample.shape[-1]
    latent_dim = hidden_dim * CFG.latent_mult
    print(f"hidden_dim={hidden_dim}, latent_dim={latent_dim}, topk={CFG.topk}")

    # ── SAE ───────────────────────────────────────────────────
    sae = SAE(hidden_dim, latent_dim, CFG.topk).float().to(CFG.device)
    sae_params = list(sae.parameters())
    optimizer  = torch.optim.AdamW(sae_params, lr=CFG.lr)

    # ── DataLoader（可以用大 batch，因为只是读 tensor）──────────
    # batch_size 这里是"几张图拼一起"，实际 token 数会更多
    loader = DataLoader(
        dataset,
        batch_size = CFG.batch_size,
        shuffle    = True,
        collate_fn = collate_cached,
        num_workers = 4,
        pin_memory  = True,
    )

    save_every = getattr(CFG, "save_every", 5000)

    print(f"Dataset size       : {len(dataset)} cached samples")
    print(f"Layer              : {layer}")
    print(f"Batch size (images): {CFG.batch_size}")
    print(f"Grad accumulation  : {GRAD_ACCUM}")
    print(f"Aux loss coef      : {AUX_COEF}")
    print(f"Save every         : {save_every} steps")

    def save_checkpoints(tag="latest"):
        path = os.path.join(SAVE_DIR, f"sae_layer{layer}_{tag}.pt")
        torch.save(sae.state_dict(), path)
        print(f"  saved {path}")
        latest = os.path.join(SAVE_DIR, f"sae_layer{layer}.pt")
        torch.save(sae.state_dict(), latest)

    def handle_sigint(sig, frame):
        print("\n[interrupted] saving checkpoint...")
        save_checkpoints("interrupted")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    optimizer_step = 0
    micro_step     = 0
    accum_loss     = 0.0
    optimizer.zero_grad()

    for epoch in range(CFG.sae_epochs):
        print(f"\nEpoch {epoch}")

        for step, img_tokens in enumerate(loader):
            # img_tokens: (total_tokens, hidden_dim), float16
            img_tokens = img_tokens.float().to(CFG.device)

            if torch.isnan(img_tokens).any() or torch.isinf(img_tokens).any():
                print(f"  [skip] step {step}: nan/inf")
                continue

            recon, z = sae(img_tokens)

            recon_loss = F.mse_loss(recon, img_tokens)
            aux_loss   = sae.auxiliary_loss(img_tokens, recon, DEAD_THRESH)
            loss       = recon_loss + AUX_COEF * aux_loss

            sae.update_fired_stats(z)

            # ── gradient accumulation ─────────────────────────
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(sae_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

                sae.normalize_decoder()

                if optimizer_step % save_every == 0:
                    save_checkpoints(f"step{optimizer_step}")

                if optimizer_step % 10 == 0:
                    avg = accum_loss / GRAD_ACCUM
                    n_dead = (sae.num_batches_since_fired >= DEAD_THRESH).sum().item()
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