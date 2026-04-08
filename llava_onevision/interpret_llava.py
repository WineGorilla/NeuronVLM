"""
批量 SAE feature 语义标注流程 — LLaVA-OneVision 版。

用法：
    python llava/build_feature_index_llava.py --layer 8
    python llava/interpret_llava.py --layer 8
    python llava/interpret_llava.py --layer 8 --debug_feature 100
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import argparse
import tempfile

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from config import CFG


ANNOTATOR_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
TOP_N_IMAGES       = 3
PREVIEW_DIR        = "newoutputs/llava_feature_previews"


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


def make_masked_from_index(entry: tuple) -> dict:
    score, image_path, H_tok, W_tok, top_patch_idx = entry

    img = cv2.imread(image_path)
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]
    masked = np.zeros_like(img)

    for idx in top_patch_idx:
        tok_y = idx // W_tok
        tok_x = idx %  W_tok
        # 用相邻 patch 起点作为当前 patch 终点，消除间隙
        x0 = int(tok_x       / W_tok * orig_w)
        x1 = int((tok_x + 1) / W_tok * orig_w)
        y0 = int(tok_y       / H_tok * orig_h)
        y1 = int((tok_y + 1) / H_tok * orig_h)
        masked[y0:y1, x0:x1] = img[y0:y1, x0:x1]

    return {
        "image_path":  image_path,
        "masked_img":  masked,
        "image_score": score,
    }


def save_concat_image(top_results: list, feature_id: int, layer: int, label: str = ""):
    os.makedirs(PREVIEW_DIR, exist_ok=True)

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

    combined = np.concatenate(concat_imgs, axis=1)

    label_str = label.replace(" ", "_").replace("/", "-")[:30] if label else "unlabeled"
    save_path = os.path.join(
        PREVIEW_DIR,
        f"layer{layer}_f{feature_id:05d}_{label_str}.png"
    )
    cv2.imwrite(save_path, combined)
    return save_path


def interpret_feature_with_qwen(
    top_results, feature_id, layer, ann_processor, ann_model,
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
                max_new_tokens=32,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        new_ids   = output_ids[:, input_len:]
        label     = ann_processor.decode(new_ids[0], skip_special_tokens=True).strip()

    except Exception as e:
        print(f"  [warn] Qwen failed for feature {feature_id}: {e}")
        label = "unknown"

    finally:
        for path in tmp_paths:
            try:
                os.remove(path)
            except Exception:
                pass

    return label


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
    plt.suptitle(f"Feature {feature_id} (layer {layer})", fontsize=12)
    plt.tight_layout()
    plt.show()

    save_path = save_concat_image(top_results, feature_id, layer)
    print(f"saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",         type=int, default=CFG.vis_layer)
    parser.add_argument("--debug_feature", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(CFG.label_dir,        exist_ok=True)
    os.makedirs(CFG.llava_cache_dir,  exist_ok=True)
    os.makedirs(PREVIEW_DIR,          exist_ok=True)

    index_path = os.path.join(CFG.llava_cache_dir, f"feature_index_layer{args.layer}.pkl")
    if not os.path.exists(index_path):
        print(f"Feature index not found: {index_path}")
        print(f"Please run: python llava/build_feature_index_llava.py --layer {args.layer}")
        return

    print(f"Loading feature index: {index_path}")
    with open(index_path, "rb") as f:
        feature_index = pickle.load(f)
    print(f"Loaded {len(feature_index)} features")

    if args.debug_feature is not None:
        fid = args.debug_feature
        if fid not in feature_index:
            print(f"Feature {fid} not found in index")
            return
        entries     = feature_index[fid][:TOP_N_IMAGES]
        top_results = [r for r in (make_masked_from_index(e) for e in entries) if r]
        visualize_feature(fid, top_results, args.layer)
        return

    ann_processor, ann_model = load_annotator()

    label_path = os.path.join(CFG.label_dir, f"feature_labels_layer{args.layer}.json")
    if os.path.exists(label_path):
        with open(label_path) as f:
            feature_label_dict = {int(k): v for k, v in json.load(f).items()}
        print(f"Resuming from {len(feature_label_dict)} done features")
    else:
        feature_label_dict = {}

    feature_ids = sorted(feature_index.keys())
    print(f"Total features to label: {len(feature_ids)}")
    print(f"Preview images saved to: {PREVIEW_DIR}")

    for idx, fid in enumerate(feature_ids):
        if fid in feature_label_dict:
            continue

        entries     = feature_index[fid][:TOP_N_IMAGES]
        top_results = [r for r in (make_masked_from_index(e) for e in entries) if r]

        if not top_results:
            continue

        label = interpret_feature_with_qwen(
            top_results, fid, args.layer,
            ann_processor, ann_model,
        )
        feature_label_dict[fid] = label

        save_concat_image(top_results, fid, args.layer, label)

        print(f"  [{idx+1}/{len(feature_ids)}] feature {fid:6d} -> {label}")

        if len(feature_label_dict) % 10 == 0:
            with open(label_path, "w") as f:
                json.dump(feature_label_dict, f, indent=2, ensure_ascii=False)

    with open(label_path, "w") as f:
        json.dump(feature_label_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(feature_label_dict)} features -> {label_path}")


if __name__ == "__main__":
    main()