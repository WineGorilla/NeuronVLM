"""
Sparse Autoencoder — 按照论文公式实现:
    z = TopK(ReLU(W1 (x - b1) + b2))
    x̂ = W2 z + b3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, topk: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.topk       = topk

        # b1: pre-encoder centering bias
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        # W1 + b2 (encoder)
        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=True)
        # W2 + b3 (decoder)
        self.decoder = nn.Linear(latent_dim, hidden_dim, bias=True)

        self._init_weights()
        self.normalize_decoder()

        # dead feature 统计
        self.register_buffer("num_batches_since_fired",
                             torch.zeros(latent_dim, dtype=torch.long))

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        """z = TopK(ReLU(W1 (x - b1) + b2))"""
        pre_act = self.encoder(x - self.b1)       # W1(x-b1) + b2
        z_relu  = F.relu(pre_act)
        topk_vals, topk_idx = torch.topk(z_relu, self.topk, dim=-1)
        z_sparse = torch.zeros_like(z_relu)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        return z_sparse

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(
            self.decoder.weight.data, dim=0
        )

    def forward(self, x: torch.Tensor):
        z     = self.encode(x)
        x_hat = self.decoder(z)                    # W2 z + b3
        return x_hat, z

    # ── auxiliary loss 相关 ────────────────────────────────────

    @torch.no_grad()
    def update_fired_stats(self, z):
        """更新每个 feature 的'多久没被激活'计数。"""
        fired = (z.reshape(-1, z.shape[-1]).abs().sum(0) > 0)
        self.num_batches_since_fired[fired] = 0
        self.num_batches_since_fired[~fired] += 1

    def auxiliary_loss(self, x, x_hat, dead_threshold=40):
        """
        对 dead features 用 residual 做二次编码，给它们梯度。
        dead_threshold: 连续多少个 batch 没激活就算 dead。
        """
        dead_mask = (self.num_batches_since_fired >= dead_threshold)
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return torch.tensor(0.0, device=x.device)

        residual = (x - x_hat).detach()
        pre_act  = self.encoder(residual - self.b1)
        z_aux    = F.relu(pre_act)
        z_aux    = z_aux * dead_mask.float()       # 只保留 dead features（非 inplace）

        k_aux = min(self.topk, n_dead)
        topk_v, topk_i = torch.topk(z_aux, k_aux, dim=-1)
        z_aux_sparse = torch.zeros_like(z_aux)
        z_aux_sparse.scatter_(-1, topk_i, topk_v)

        x_hat_aux = self.decoder(z_aux_sparse)
        return F.mse_loss(x_hat_aux, residual)