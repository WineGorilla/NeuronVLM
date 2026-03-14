"""
FeatureRouter：根据问题动态决定增强哪些 SAE feature。

设计原则：
    1. question_vec 直接取 Qwen 对应层的 text token hidden state 均值
       （与 image token 在同一 hidden space，天然语义对齐，不需要额外编码器）

    2. Router 用 question_vec 和 SAE decoder weight 做内积打分
       decoder weight[:, i] 就是 feature i 在 hidden space 的方向向量
       内积 = 问题语义与 feature 语义的相似度，直接、高效、可解释

    3. 输出 alpha（每个 feature 的增强系数）
       alpha = 1 + scale * sigmoid(score)
       初始 scale=0，训练过程中学习，保证训练初期不破坏模型输出

    4. 只对已激活的 feature（z > 0）生效
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRouter(nn.Module):
    """
    基于 decoder weight 内积的 Feature Router。

    Args:
        hidden_dim:  Qwen hidden size
        latent_dim:  SAE latent 维度
        topk_route:  每次最多增强的 feature 数量（稀疏路由）
        max_alpha:   最大增强倍数，防止激活爆炸
    """

    def __init__(
        self,
        hidden_dim:  int,
        latent_dim:  int,
        topk_route:  int   = 64,
        max_alpha:   float = 3.0,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.latent_dim  = latent_dim
        self.topk_route  = topk_route
        self.max_alpha   = max_alpha

        # 问题向量的线性变换，把 question_vec 投影到 feature 打分空间
        # 初始化为接近 0，保证训练初期 alpha ≈ 1
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.normal_(self.query_proj.weight, std=0.01)

        # 可学习的温度系数（控制 alpha 的分布锐度）
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        question_vec:   torch.Tensor,   # (hidden_dim,) 或 (batch, hidden_dim)
        z:              torch.Tensor,   # (num_tokens, latent_dim)
        decoder_weight: torch.Tensor,   # (hidden_dim, latent_dim) = decoder.weight.T
    ) -> torch.Tensor:
        """
        Returns:
            alpha: (num_tokens, latent_dim)
                   已激活的 top-k feature 被增强，其余保持 1.0
        """
        if question_vec.dim() == 1:
            question_vec = question_vec.unsqueeze(0)   # (1, hidden_dim)

        # 问题向量投影
        q = self.query_proj(question_vec.float())      # (1, hidden_dim)

        # 与 decoder weight 做内积，得到每个 feature 的相关性分数
        # decoder_weight: (hidden_dim, latent_dim)
        # score[i] = q · decoder_weight[:, i]  =  问题语义与 feature i 的相似度
        scores = torch.matmul(q, decoder_weight.float())   # (1, latent_dim)
        scores = scores.squeeze(0)                          # (latent_dim,)

        # 只对已激活的 feature 打分（dead feature 不参与）
        # 取所有 token 的激活均值作为 feature 是否活跃的判断
        active_mask = (z > 0).float().mean(dim=0)      # (latent_dim,)  0~1

        # 用 active_mask 过滤分数，非激活 feature 分数置为 -inf
        scores = scores + (1 - (active_mask > 0).float()) * (-1e9)

        # 稀疏路由：只增强 top-k 个最相关的 feature
        topk_scores, topk_indices = torch.topk(
            scores, min(self.topk_route, (active_mask > 0).sum().item()), dim=-1
        )

        # 构建 alpha：初始全为 1.0
        alpha = torch.ones(self.latent_dim, device=z.device, dtype=z.dtype)

        # 对 top-k feature 计算增强系数
        # scale = exp(log_scale)，训练初期接近 1，之后学习增大
        scale = torch.exp(self.log_scale).clamp(max=10.0)
        boost = 1.0 + (self.max_alpha - 1.0) * torch.sigmoid(
            topk_scores * scale
        )
        alpha[topk_indices] = boost.to(alpha.dtype)

        # 广播到 (num_tokens, latent_dim)
        alpha = alpha.unsqueeze(0).expand(z.shape[0], -1)

        # 最终只对真正激活的 token-feature 对生效
        token_active = (z > 0).float()
        alpha = alpha * token_active + 1.0 * (1.0 - token_active)

        return alpha