"""
公共工具函数，供 scripts/visualize.py 和 scripts/interpret.py 共用。
"""
import numpy as np
import cv2
import base64


# ── Feature 分析 ──────────────────────────────────────────────────────────────

def get_top_features(z: np.ndarray, top_n: int = 32, mode: str = "mean"):
    """找激活最强的 top_n 个 feature。

    判断 feature 是否活跃用绝对值，排序用绝对值找最显著的 feature。

    Args:
        z:      shape (num_tokens, latent_dim)
        top_n:  返回的 feature 数量
        mode:   "mean" 取均值，"max" 取最大值

    Returns:
        top_feature_ids: shape (top_n,)，feature 索引（降序）
        feature_scores:  shape (top_n,)，对应分数
    """
    if mode == "mean":
        scores = np.abs(z).mean(axis=0)
    elif mode == "max":
        scores = np.abs(z).max(axis=0)
    else:
        raise ValueError(f"mode must be 'mean' or 'max', got '{mode}'")

    top_feature_ids = scores.argsort()[::-1][:top_n]
    feature_scores  = scores[top_feature_ids]
    return top_feature_ids, feature_scores


def get_peak_blocks(z: np.ndarray, feature_ids, H_tok: int, W_tok: int,
                    img_h: int, img_w: int) -> list:
    """找每个 feature 正激活最强的图块，返回像素坐标信息。"""
    blocks = []
    for fid in feature_ids:
        activations   = z[:, fid]
        # 只看正激活最强的 token
        max_token_idx = int(activations.argmax())
        tok_y = max_token_idx // W_tok
        tok_x = max_token_idx %  W_tok

        px_x    = int(tok_x / W_tok * img_w)
        px_y    = int(tok_y / H_tok * img_h)
        block_w = int(img_w / W_tok)
        block_h = int(img_h / H_tok)

        blocks.append({
            "fid":        int(fid),
            "tok_y":      tok_y,
            "tok_x":      tok_x,
            "px_x":       px_x,
            "px_y":       px_y,
            "block_w":    block_w,
            "block_h":    block_h,
            "activation": float(activations[max_token_idx]),
        })
    return blocks


# ── Patch masking ─────────────────────────────────────────────────────────────

def make_masked_image(image_path: str, z: np.ndarray, feature_id: int,
                      H_tok: int, W_tok: int, top_n: int = 32):
    """只保留正激活最强的 top_n 个 patch，其余置黑。

    语义标注用正激活：正激活代表"这里有这个概念"，负激活是抑制信号。

    Returns:
        masked:      np.ndarray (H, W, 3)，BGR
        image_score: float，top_n patch 的平均正激活值（用于跨图排序）
    """
    activations       = z[:, feature_id]
    # 只取正激活，按正值降序排序
    top_patch_indices = activations.argsort()[::-1][:top_n]
    image_score       = float(activations[top_patch_indices].mean())

    img            = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]
    masked         = np.zeros_like(img)
    block_w        = int(orig_w / W_tok)
    block_h        = int(orig_h / H_tok)

    for idx in top_patch_indices:
        tok_y = idx // W_tok
        tok_x = idx %  W_tok
        px_x  = int(tok_x / W_tok * orig_w)
        px_y  = int(tok_y / H_tok * orig_h)
        masked[px_y : px_y + block_h, px_x : px_x + block_w] = \
            img[px_y : px_y + block_h, px_x : px_x + block_w]

    return masked, image_score


# ── 图像工具 ──────────────────────────────────────────────────────────────────

def image_to_base64(img_bgr: np.ndarray) -> str:
    """将 BGR numpy 图像编码为 base64 PNG 字符串（供 Anthropic API 使用）。"""
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")