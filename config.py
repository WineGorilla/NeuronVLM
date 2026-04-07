from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Config:
    # 模型
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    llava_model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

    llava_next_model_id = "llava-hf/llama3-llava-next-8b-hf"
    save_llava_next_dir = "outputs/sae_llava_next"

    # 数据

    train_file: str = "data/train_vqa2.jsonl"

    # ── SAE 层索引（内部会 +1 以跳过 embedding 层）──────────────
    layers: List[int] = field(default_factory=lambda: [8])

    # ── SAE 超参 ──────────────────────────────────────────────
    latent_mult: int = 32      # latent_dim = hidden_dim * latent_mult
    topk: int = 32           

    # ── 训练超参 ──────────────────────────────────────────────
    batch_size:    int   = 1
    sae_epochs:    int   = 1
    epochs:        int   = 2
    lr:            float = 1e-4   # 归一化去掉后可以稍大
    sparsity_coef: float = 0.05
    grad_accum:    int   = 8
    save_every: int = 5000   # 每 N 个 optimizer step 保存一次



    # ── 路径 ──────────────────────────────────────────────────
    save_llava_dir: str = "outputs/llava/sae_ckpt"
    save_dir: str = "outputs/qwen/sae_ckpt"

    llava_cache_dir: str = "outputs/llava"
    cache_dir: str = "outputs/qwen"
    label_dir: str = "assets"

    # ── 推理 / 可视化 
    vis_layer: int = 8           # 可视化时使用的层
    vis_feature_id: int = 120    # 默认查看的 feature id
    top_n_patches: int = 120     # 每张图保留激活最强的 patch 数 注意llava和qwen是不一样个数的 qwen60 llava480 4倍
    top_n_images: int = 5        # 每个 feature 展示的 top-N 图片数 注意

    # ── 设备 ──────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# 全局单例，所有模块直接 from config import CFG
CFG = Config()