"""
SAE 训练入口。
用法：python -m src.train
"""
import os
import sys
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

    # ── 加载 backbone
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

    # 构建 SAE
    saes = {
        l: SAE(hidden_dim, latent_dim, CFG.topk).float().to(CFG.device)
        for l in CFG.layers
    }

    optimizer = torch.optim.AdamW(
        [p for sae in saes.values() for p in sae.parameters()],
        lr=CFG.lr,
    )

    # 数据
    dataset = VisionTextDataset(CFG.train_file)
    loader  = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=build_collate(processor),
    )

    spatial_merge = model.config.vision_config.spatial_merge_size
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")

    # 训练循环
    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch}")

        for step, batch in enumerate(loader):
            batch = {k: v.to(CFG.device) for k, v in batch.items()}

            # backbone
            with torch.no_grad():
                outputs = model(
                    **batch,
                    output_hidden_states=True,
                    return_dict=True,
                )

            loss_total = 0.0

            for l in CFG.layers:
                h = outputs.hidden_states[l + 1]   # (1, seq_len, hidden_dim)

                # 定位 image tokens
                image_grid     = batch["image_grid_thw"]
                num_img_tokens = int(
                    image_grid[0, 1] * image_grid[0, 2] / (spatial_merge ** 2)
                )
                input_ids  = batch["input_ids"][0]
                vision_pos = (input_ids == vision_start_id).nonzero()[0].item() + 1

                img_tokens = h[:, vision_pos : vision_pos + num_img_tokens, :]
                flat       = img_tokens.reshape(-1, img_tokens.shape[-1]).float()

                sae = saes[l]
                recon, z = sae(flat)

                # 重建损失 + 辅助稀疏惩罚
                loss = F.mse_loss(recon, flat) + CFG.sparsity_coef * z.abs().mean()
                loss_total += loss

            loss_total /= len(CFG.layers)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # decoder 列归一化
            for sae in saes.values():
                sae.normalize_decoder()

            if step % 10 == 0:
                print(f"  step {step:5d} | loss {loss_total.item():.6f}")

        # 每 epoch 保存权重
        for l in CFG.layers:
            path = os.path.join(CFG.save_dir, f"sae_layer{l}.pt")
            torch.save(saes[l].state_dict(), path)
            print(f"  saved {path}")


if __name__ == "__main__":
    train()