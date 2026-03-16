import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    """
    Sparse Autoencoder，top-k 稀疏激活。
    稀疏性由 top-k 本身保证，不依赖 ReLU。
    """

    def __init__(self, hidden_dim: int, latent_dim: int, topk: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.topk       = topk

        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, hidden_dim, bias=False)

        # encoder 权重用较大的 std 初始化，保证初始激活值不会太小
        nn.init.normal_(self.encoder.weight, std=0.02)
        nn.init.zeros_(self.encoder.bias)

        # decoder 权重初始化为单位范数
        nn.init.normal_(self.decoder.weight, std=0.02)
        self.normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """top-k 稀疏编码，稀疏性由 top-k 保证。"""
        z = self.encoder(x)

        k = min(self.topk, z.shape[-1])
        values, indices = torch.topk(z, k, dim=-1)

        mask = torch.zeros_like(z)
        mask.scatter_(-1, indices, values)

        return mask

    @torch.no_grad()
    def normalize_decoder(self):
        """将 decoder 列归一化为单位范数，防止幅度坍缩。"""
        self.decoder.weight.data = F.normalize(
            self.decoder.weight.data, dim=0
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_hat: 重建向量，shape 同 x
            z:     稀疏编码，shape (*, latent_dim)
        """
        z     = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z