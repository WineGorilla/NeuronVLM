import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    """Sparse Autoencoder，使用 top-k + ReLU 实现稀疏激活。

    Args:
        hidden_dim: 输入/输出维度（对应 LLM 的 hidden_size）
        latent_dim: 潜空间维度，通常为 hidden_dim * latent_mult
        topk: 每个 token 保留激活最强的 k 个 latent
    """

    def __init__(self, hidden_dim: int, latent_dim: int, topk: int):
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.topk = topk

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：先 top-k 选取，再 ReLU 保证非负稀疏性。"""
        z = self.encoder(x)
        values, indices = torch.topk(z, self.topk, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, indices, values)
        return F.relu(mask)

    @torch.no_grad()
    def normalize_decoder(self):
        """将 decoder 列归一化为单位范数，防止幅度坍缩。每次 optimizer.step() 后调用。"""
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_hat: 重建向量，shape 同 x
            z:     稀疏编码，shape (*, latent_dim)
        """
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z