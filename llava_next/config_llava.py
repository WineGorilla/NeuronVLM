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
    train_file: str = "data/test.jsonl"#正式版："data/train_vqa2.jsonl"
    # ── SAE 层索引（内部会 +1 以跳过 embedding 层）──────────────
    layers: List[int] = field(default_factory=lambda: [8])
    # ── SAE 超参 ──────────────────────────────────────────────
    latent_mult: int = 32      # latent_dim = hidden_dim * latent_mult
    topk: int = 64           
    # ── 训练超参 ──────────────────────────────────────────────
    batch_size:    int   = 1
    sae_epochs:    int   = 1
    epochs:        int   = 2
    lr:            float = 3e-4   
    sparsity_coef: float = 0.0    
    grad_accum:    int   = 8
    save_every: int = 5000
    # ── auxiliary loss（新增）────────────────────────────────
    aux_coef:       float = 1/32   # 
    dead_threshold: int   = 40     # 
    # ── 路径 ──────────────────────────────────────────────────
    save_llava_dir: str = "outputs/llava/sae_ckpt"
    save_dir: str = "outputs/qwen/sae_ckpt"
    llava_cache_dir: str = "outputs/llava_next"
    cache_dir: str = "outputs/qwen"
    label_dir: str = "assets"
    # ── 推理 / 可视化 
    vis_layer: int = 8
    vis_feature_id: int = 120
    top_n_patches: int = 240
    top_n_images: int = 5
    # ── 设备 ──────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
# 全局单例，所有模块直接 from config import CFG
CFG = Config()