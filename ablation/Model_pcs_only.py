"""
PCS-Only 消融模型。

只保留 PCA Suppression（suppress hook），去掉 SAE 注入（inject hook 改为只读）。
用于证明 PCS 单独的贡献。

和完整模型的区别：
  - _mid_layer_hook: 只做 cluster 预测（READ_ONLY），不做 SAE 注入
  - _suppress_hook: 正常做 PCA suppression
  - 不需要单独训练，直接用 Stage 2 训完的权重
"""
from src.Model import QwenWithClusterPredictorAndSAE


class QwenWithPCSOnly(QwenWithClusterPredictorAndSAE):
    """
    继承完整模型，重写 inject hook 为只读模式。
    suppress hook 保持不变（正常做 PCA suppression）。
    """

    def _mid_layer_hook(self, module, input, output):
        """
        重写：始终以 READ_ONLY 模式运行。
        只做 cluster 预测，不做 SAE 注入。
        """
        if self._hook_fired:
            return output

        is_tuple = isinstance(output, tuple)
        raw_hs = output[0] if is_tuple else output
        was_2d = (raw_hs.dim() == 2)
        hs = raw_hs.unsqueeze(0) if was_2d else raw_hs

        if hs.shape[1] <= 1:
            return output

        self._hook_fired = True
        v_pos, n_img, text_pos = self._cached_positions

        # ── 只做 cluster 预测，不注入 ─────────────────────────────────────────
        import torch
        logits = self.cluster_predictor(hs, text_pos)
        self._stashed_logits = logits

        h_vision_float = hs[0, v_pos:v_pos + n_img, :].float()
        self._stashed_h_vision = h_vision_float

        probs_with_grad = torch.sigmoid(logits)[0]
        probs_detached = probs_with_grad.detach()

        top_k_clusters = min(self.top_k_clusters, len(probs_with_grad))
        _, top_cids = torch.topk(probs_detached, top_k_clusters)
        top_cids = top_cids.tolist()

        self._last_cluster_ids = [cid for cid in top_cids if probs_detached[cid] > 0.1]
        self._last_cluster_probs = probs_with_grad

        # ── 不做注入，直接返回原始 output ─────────────────────────────────────
        return output

    # _suppress_hook 继承自父类，正常做 PCA suppression，不需要重写

    @classmethod
    def from_pretrained(cls, **kwargs):
        """加载完整模型，然后切换为 PCS-only 模式。"""
        model = QwenWithClusterPredictorAndSAE.from_pretrained(**kwargs)
        model.__class__ = cls
        print("  [PCS-Only] SAE injection disabled, only PCA suppression active.")
        return model