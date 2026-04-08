"""
Random SAE 消融模型。

和正常增强模型完全一致，唯一区别：
  SAE 的 encoder/decoder 权重被随机重置。

这样 ClusterPredictor 仍然正常预测 cluster，
但 SAE encode → decode 出来的是随机特征，
注入到 hidden states 的内容是无意义的噪声。

用于证明：增强效果来自有意义的 SAE 特征，而不是随机扰动。
"""
import torch
import torch.nn as nn
from src.Model import QwenWithClusterPredictorAndSAE


class QwenWithRandomSAE(QwenWithClusterPredictorAndSAE):
    """
    继承正常的增强模型，重写 SAE 权重为随机值。
    其他一切不变（ClusterPredictor、SemanticCrossAttention、PCS 全部正常）。
    """

    def randomize_sae(self, seed=42):
        """将 SAE 的 encoder 和 decoder 权重随机重置。"""
        torch.manual_seed(seed)
        print(f"  [Random SAE] Randomizing SAE weights (seed={seed})...")

        # 随机重置 encoder
        nn.init.xavier_uniform_(self.sae.encoder.weight)
        print(f"    encoder: {self.sae.encoder.weight.shape} -> randomized")

        # 随机重置 decoder
        nn.init.xavier_uniform_(self.sae.decoder.weight)
        print(f"    decoder: {self.sae.decoder.weight.shape} -> randomized")

        # 重新做 decoder 归一化（和训练时一致）
        self.sae.normalize_decoder()

        # 保持 SAE 冻结
        self.sae.eval()
        for p in self.sae.parameters():
            p.requires_grad = False

        print(f"  [Random SAE] Done. SAE is now random noise generator.")

    @classmethod
    def from_pretrained(cls, **kwargs):
        """加载正常模型，然后随机化 SAE。"""
        seed = kwargs.pop("random_seed", 42)

        # 用父类的 from_pretrained 加载完整模型
        # 需要临时把类替换回去
        model = QwenWithClusterPredictorAndSAE.from_pretrained(**kwargs)

        # 把 __class__ 换成当前类，这样就能调用 randomize_sae
        model.__class__ = cls

        # 随机化 SAE
        model.randomize_sae(seed=seed)

        return model