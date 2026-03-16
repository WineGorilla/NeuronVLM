"""
批量 SAE feature 语义标注流程：
  1. 缓存所有训练图像的 SAE latent（cache_layer{N}.pkl，稀疏存储）
  2. 找所有有激活的 feature
  3. 对每个 feature，找激活值 > 0 的 top-3 图像，生成 patch-masked 图
  4. 用本地 Qwen2.5-VL-7B 解读视觉语义，输出简短标签
  5. 断点续跑，结果保存到 feature_labels_layer{N}.json

用法：
    python scripts/interpret.py --layer 8
    python scripts/interpret.py --layer 8 --debug_feature 100
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import argparse
import tempfile

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import CFG
from src.SAE import SAE
from src.utils import make_masked_image


# ── 模型 ID ───────────────────────────────────────────────────────────────────

SAE_MODEL_ID       = CFG.model_id
ANNOTATOR_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
TOP_N_IMAGES       = 3   # 只取有正激活的 top-3 图片


# ── 加载 SAE 推理模型（3B）────────────────────────────────────────────────────

def load_sae_model(layer: int):
    processor = AutoProcessor.from_pretrained(SAE_MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        SAE_MODEL_ID, torch_dtype=torch.float16
    ).to(CFG.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.text_config.hidden_size
    latent_dim = hidden_dim * CFG.latent_mult

    ckpt_path = os.path.join(CFG.save_dir, f"sae_layer{layer}.pt")
    sae = SAE(hidden_dim, latent_dim, CFG.topk).float().to(CFG.device)
    sae.load_state_dict(torch.load(ckpt_path, map_location=CFG.device))
    sae.eval()

    return processor, model, sae


# ── 加载标注模型（7B）─────────────────────────────────────────────────────────

def load_annotator():
    print(f"Loading annotator: {ANNOTATOR_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(ANNOTATOR_MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ANNOTATOR_MODEL_ID, torch_dtype=torch.float16
    ).to(CFG.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("Annotator loaded.")
    return processor, model


# ── 单图前向：返回 SAE 编码 + token 网格尺寸 ─────────────────────────────────

def forward_single(image_path: str, layer: int, processor, model, sae):
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": "Describe the image"},
        ],
    }]]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts, images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(CFG.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    h             = outputs.hidden_states[layer + 1]
    image_grid    = inputs["image_grid_thw"]
    H_grid        = image_grid[0, 1].item()
    W_grid        = image_grid[0, 2].item()
    spatial_merge = model.config.vision_config.spatial_merge_size
    num_img_tokens = int(H_grid * W_grid / (spatial_merge ** 2))
    H_tok         = int(H_grid // spatial_merge)
    W_tok         = int(W_grid  // spatial_merge)

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    input_ids   = inputs["input_ids"][0]
    vision_pos  = (input_ids == vision_start_id).nonzero()[0].item() + 1

    img_tokens = h[:, vision_pos : vision_pos + num_img_tokens, :]
    tokens     = img_tokens.reshape(-1, img_tokens.shape[-1]).float()
    tokens     = F.layer_norm(tokens, [tokens.shape[-1]])

    with torch.no_grad():
        z = sae.encode(tokens).cpu().numpy()

    z_sparse = sp.csr_matrix(z)
    return z_sparse, H_tok, W_tok


# ── Step 1：缓存所有图的 latent ───────────────────────────────────────────────

def build_cache(layer: int, processor, model, sae) -> list:
    cache_path = os.path.join(CFG.cache_dir, f"cache_layer{layer}.pkl")

    if os.path.exists(cache_path):
        print(f"loading cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"loaded {len(cache)} items.")
        return cache

    samples = []
    with open(CFG.train_file) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"caching {len(samples)} images...")
    cache = []
    for i, sample in enumerate(samples):
        image_path = sample["image"]
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(samples)}]")
        try:
            z_sparse, H_tok, W_tok = forward_single(image_path, layer, processor, model, sae)
            cache.append({
                "image_path": image_path,
                "z":          z_sparse,
                "H_tok":      H_tok,
                "W_tok":      W_tok,
            })
        except Exception as e:
            print(f"  [skip] {image_path}: {e}")

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"saved: {cache_path}")
    return cache


def get_dense_z(item: dict) -> np.ndarray:
    z = item["z"]
    if sp.issparse(z):
        return z.toarray()
    return z


def get_top_results(cache: list, feature_id: int, top_n: int = TOP_N_IMAGES) -> list:
    """
    遍历所有图，只保留正激活 > 0 的图片，取 top_n 个。
    """
    results = []
    for item in cache:
        z = get_dense_z(item)
        masked_img, image_score = make_masked_image(
            item["image_path"], z, feature_id,
            item["H_tok"], item["W_tok"], top_n=CFG.top_n_patches,
        )
        # 只保留有正激活的图片
        if image_score > 0:
            results.append({
                "image_path":  item["image_path"],
                "masked_img":  masked_img,
                "image_score": image_score,
            })

    results.sort(key=lambda x: x["image_score"], reverse=True)
    return results[:top_n]


# ── Step 2：用 7B Qwen 标注 feature ──────────────────────────────────────────

def interpret_feature_with_qwen(
    top_results:  list,
    feature_id:   int,
    layer:        int,
    ann_processor,
    ann_model,
) -> str:
    tmp_paths = []
    for res in top_results:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, res["masked_img"])
        tmp_paths.append(tmp.name)

    prompt = (
        f"These are {len(tmp_paths)} images showing the regions most strongly activating "
        f"SAE feature {feature_id} (layer {layer}) of a vision-language model. "
        "In each image, only the patches with the strongest activation are visible; "
        "the rest are blacked out. "
        "Based on the visible regions across all images, "
        "what visual concept or semantic pattern does this feature likely represent? "
        "Please give a concise label of 1-5 words only, no explanation."
    )

    content = []
    for path in tmp_paths:
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": prompt})

    messages = [[{"role": "user", "content": content}]]
    texts = [
        ann_processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]

    try:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = ann_processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(CFG.device)

        with torch.no_grad():
            output_ids = ann_model.generate(
                **inputs,
                max_new_tokens = 32,
                do_sample      = False,
            )

        input_len = inputs["input_ids"].shape[1]
        new_ids   = output_ids[:, input_len:]
        label     = ann_processor.decode(new_ids[0], skip_special_tokens=True).strip()

    except Exception as e:
        print(f"  [warn] Qwen inference failed for feature {feature_id}: {e}")
        label = "unknown"

    finally:
        for path in tmp_paths:
            try:
                os.remove(path)
            except Exception:
                pass

    return label


# ── 调试用：单 feature 可视化 ─────────────────────────────────────────────────

def visualize_feature(feature_id: int, top_results: list, layer: int):
    n = len(top_results)
    if n == 0:
        print(f"No positive activation found for feature {feature_id}")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, res in enumerate(top_results):
        axes[i].imshow(cv2.cvtColor(res["masked_img"], cv2.COLOR_BGR2RGB))
        axes[i].axis("off")
        axes[i].set_title(
            f"{os.path.basename(res['image_path'])}\nscore={res['image_score']:.3f}",
            fontsize=9,
        )
    plt.suptitle(f"Feature {feature_id} (layer {layer}) — top {n} positive activations",
                 fontsize=12)
    plt.tight_layout()
    plt.show()

    target_h    = 400
    concat_imgs = []
    for res in top_results:
        img     = res["masked_img"]
        scale   = target_h / img.shape[0]
        resized = cv2.resize(img, (int(img.shape[1] * scale), target_h))
        bordered = cv2.copyMakeBorder(
            resized, 5, 5, 5, 5,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        concat_imgs.append(bordered)

    combined  = np.concatenate(concat_imgs, axis=1)
    save_path = os.path.join(CFG.cache_dir, f"feature_{feature_id}_layer{layer}_top{n}.png")
    cv2.imwrite(save_path, combined)
    print(f"saved: {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",         type=int, default=CFG.vis_layer)
    parser.add_argument("--debug_feature", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(CFG.label_dir, exist_ok=True)
    os.makedirs(CFG.cache_dir, exist_ok=True)

    print(f"Loading SAE model & layer {args.layer}...")
    sae_processor, sae_model, sae = load_sae_model(args.layer)

    cache = build_cache(args.layer, sae_processor, sae_model, sae)

    # 调试模式：只看图，不标注
    if args.debug_feature is not None:
        top_results = get_top_results(cache, args.debug_feature)
        visualize_feature(args.debug_feature, top_results, args.layer)
        return

    # 缓存构建完后释放 3B 模型，加载 7B 标注模型
    del sae_model
    torch.cuda.empty_cache()

    ann_processor, ann_model = load_annotator()

    # 遍历所有 active feature
    print("Computing active features...")
    all_z              = np.concatenate([get_dense_z(item) for item in cache], axis=0)
    active_feature_ids = np.where(np.abs(all_z).max(axis=0) > 0)[0]
    print(f"active features: {len(active_feature_ids)} / {all_z.shape[1]}")
    del all_z

    label_path = os.path.join(CFG.label_dir, f"feature_labels_layer{args.layer}.json")
    if os.path.exists(label_path):
        with open(label_path) as f:
            feature_label_dict = {int(k): v for k, v in json.load(f).items()}
        print(f"resuming from {len(feature_label_dict)} done features")
    else:
        feature_label_dict = {}

    for idx, target_feature_id in enumerate(active_feature_ids):
        if int(target_feature_id) in feature_label_dict:
            continue

        # 只取有正激活的 top-3 图片
        top_results = get_top_results(cache, target_feature_id, top_n=TOP_N_IMAGES)

        if not top_results:
            continue

        label = interpret_feature_with_qwen(
            top_results, target_feature_id, args.layer,
            ann_processor, ann_model,
        )
        feature_label_dict[int(target_feature_id)] = label
        print(f"  [{idx+1}/{len(active_feature_ids)}] feature {target_feature_id:6d} -> {label}")

        if len(feature_label_dict) % 10 == 0:
            with open(label_path, "w") as f:
                json.dump(feature_label_dict, f, indent=2, ensure_ascii=False)

    with open(label_path, "w") as f:
        json.dump(feature_label_dict, f, indent=2, ensure_ascii=False)
    print(f"saved {len(feature_label_dict)} features -> {label_path}")


if __name__ == "__main__":
    main()