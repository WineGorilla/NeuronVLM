import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    """
    Sparse Autoencoder，top-k + ReLU 稀疏激活。
    先 top-k 选取，再 ReLU 保证非负。
    """

    def __init__(self, hidden_dim: int, latent_dim: int, topk: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.topk       = topk

        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, hidden_dim, bias=False)

        self.normalize_decoder()

    def encode(self, x):
        z = F.relu(self.encoder(x))          # 先 relu
        values, indices = torch.topk(z, self.topk, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, indices, values)
        return mask                           # 不需要再 relu

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(
            self.decoder.weight.data, dim=0
        )

    def forward(self, x: torch.Tensor):
        z     = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z