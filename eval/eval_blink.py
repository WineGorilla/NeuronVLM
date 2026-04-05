"""
BLINK 评估脚本。

BLINK 包含 14 个经典 CV 任务，重新格式化为多选题。
数据集: https://huggingface.co/datasets/BLINK-Benchmark/BLINK
论文: https://arxiv.org/abs/2404.12390 (ECCV 2024)

子任务列表:
  Art_Style, Counting, Forensic_Detection, Functional_Correspondence,
  IQ_Test, Jigsaw, Multi-view_Reasoning, Object_Localization,
  Relative_Depth, Relative_Reflectance, Semantic_Correspondence,
  Spatial_Relation, Visual_Correspondence, Visual_Similarity

用法：
    # 全部子任务
    python eval/eval_blink.py --mode base
    python eval/eval_blink.py --mode enhanced
    python eval/eval_blink.py --mode spatial --spatial_qwen_ckpt outputs/focus_v2_ckpt/qwen_best.pt
    python eval/eval_blink.py --mode finetune_baseline --qwen_ckpt outputs/ablation_baseline/qwen_best.pt
    python eval/eval_blink.py --mode no_pcs --no_pcs_qwen_ckpt outputs/ablation_no_pcs/qwen_best.pt
    CUDA_VISIBLE_DEVICES=1 python eval/eval_blink.py --mode both
    python eval/eval_blink.py --mode all --qwen_ckpt outputs/ablation_baseline/qwen_best.pt

    # 只跑深度/空间相关子任务（推荐首次测试）
    python eval/eval_blink.py --mode both --depth_spatial_only
    python eval/eval_blink.py --mode both --subtasks Relative_Depth Spatial_Relation

    # 限制样本数
    python eval/eval_blink.py --mode both --max_samples 50
"""
import os
import sys
import re
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import CFG


# ── 所有 BLINK 子任务 ─────────────────────────────────────────────────────────

ALL_SUBTASKS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]

# 与你模型 distance 能力最相关的子任务
DEPTH_SPATIAL_SUBTASKS = [
    "Relative_Depth",
    "Spatial_Relation",
    "Multi-view_Reasoning",
    "Object_Localization",
]


# ── Prompt 构造 ───────────────────────────────────────────────────────────────

def build_blink_prompt(prompt: str, choices: list, has_image_choices: bool = False) -> str:
    """
    构造 BLINK 评估 prompt。
    BLINK 的 prompt 字段本身已经包含了问题描述，
    choices 可能是文本也可能对应图片选项。
    """
    if has_image_choices:
        choice_text = "\n".join(
            [f"{chr(ord('A') + i)}. (Image {i+1})" for i in range(len(choices))]
        )
    else:
        choice_text = "\n".join(
            [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
        )

    instruction = (
        "Answer the following question.\n"
        "Select the correct option and output ONLY the letter.\n"
        "Do NOT output explanation.\n"
    )
    return f"{instruction}\n{prompt}\n\nChoices:\n{choice_text}\n\nAnswer:"


# ── 答案匹配 ──────────────────────────────────────────────────────────────────

def extract_choice_letter(response: str) -> str | None:
    """从模型输出中提取选项字母。"""
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None
    m = re.search(r'\(([A-Z])\)', text)
    if m:
        return f"({m.group(1)})"
    m = re.search(r'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?([A-Z])\)?', text)
    if m:
        return f"({m.group(1)})"
    m = re.match(r'^([A-Z])(?:[\s.,):]|$)', text)
    if m:
        return f"({m.group(1)})"
    return None


def match_by_content(response: str, choices: list) -> str | None:
    resp = response.strip().split("\n")[0].strip().lower()
    for i, choice in enumerate(choices):
        if isinstance(choice, str) and resp == choice.strip().lower():
            return f"({chr(ord('A') + i)})"
    return None


def match_answer(response: str, choices: list, answer: str) -> tuple[bool, str | None]:
    """
    匹配模型响应与标准答案。
    BLINK 的 answer 字段格式为 "(A)", "(B)" 等。
    """
    extracted = extract_choice_letter(response)
    if extracted is not None:
        return extracted == answer, extracted
    extracted = match_by_content(response, choices)
    if extracted is not None:
        return extracted == answer, extracted
    return False, None


# ── 图片处理工具 ──────────────────────────────────────────────────────────────

def save_images_for_item(item, tmp_dir="/tmp/blink_eval"):
    """
    将 BLINK 数据项中的图片保存到临时目录。
    BLINK 每个样本可能有 1~4 张图片（image_1, image_2, ...）。
    返回: 图片路径列表。
    """
    os.makedirs(tmp_dir, exist_ok=True)
    image_paths = []
    for key in ["image_1", "image_2", "image_3", "image_4"]:
        img = item.get(key)
        if img is not None and isinstance(img, Image.Image):
            path = os.path.join(tmp_dir, f"{key}.png")
            img.save(path)
            image_paths.append(path)
    return image_paths


def concat_images_horizontal(image_paths: list, max_height: int = 768) -> str:
    """
    水平拼接多张图片为一张，方便输入单图模型。
    返回拼接后的图片路径。
    """
    if len(image_paths) == 1:
        return image_paths[0]

    images = [Image.open(p).convert("RGB") for p in image_paths]
    target_h = min(max_height, max(img.height for img in images))
    resized = []
    for img in images:
        ratio = target_h / img.height
        new_w = int(img.width * ratio)
        resized.append(img.resize((new_w, target_h), Image.LANCZOS))

    total_w = sum(img.width for img in resized)
    concat = Image.new("RGB", (total_w, target_h))
    x_offset = 0
    for img in resized:
        concat.paste(img, (x_offset, 0))
        x_offset += img.width

    out_path = "/tmp/blink_eval/concat.png"
    concat.save(out_path)
    return out_path


# ── 加载模型 ──────────────────────────────────────────────────────────────────

def load_base_model():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading vanilla Qwen2.5-VL for baseline...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16, device_map=CFG.device,
    )
    base_model.eval()
    return base_model, processor


def load_finetune_baseline(qwen_ckpt):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading finetune baseline (Qwen + finetuned layers, no hooks)...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16, device_map=CFG.device,
    )
    if qwen_ckpt and os.path.exists(qwen_ckpt):
        print(f"  Loading finetuned weights: {qwen_ckpt}")
        state = torch.load(qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"  Loaded {len(state)} weight tensors.")
    else:
        print("  WARNING: --qwen_ckpt not provided or not found, using vanilla model!")
    model.eval()
    return model, processor


def load_enhanced_model(args):
    from src.Model import QwenWithClusterPredictorAndSAE
    print("Loading enhanced model (ClusterPredictor + SAE)...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id, sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path, inject_layer=args.layer,
        latent_mult=CFG.latent_mult, topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.predictor_ckpt, device=CFG.device,
    )
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        state = torch.load(args.qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("  Enhanced model weights (Stage 2) loaded.")
    model.to(CFG.device)
    model.eval()
    return model


def load_spatial_model(args):
    from src.Model_v2 import QwenWithClusterPredictorAndSAE
    print("Loading spatial model (Model_v2: enhanced + SpatialPatchInteraction)...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id, sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path, inject_layer=args.layer,
        latent_mult=CFG.latent_mult, topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.spatial_predictor_ckpt, device=CFG.device,
    )
    if args.spatial_qwen_ckpt and os.path.exists(args.spatial_qwen_ckpt):
        state = torch.load(args.spatial_qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("  Spatial model weights (Stage 2) loaded.")
    model.to(CFG.device)
    model.eval()
    return model


def load_enhanced_model_no_pcs(args):
    from ablation.Model_no_pcs import QwenWithClusterPredictorAndSAE
    print("Loading enhanced model (NO PCS ablation)...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id, sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path, inject_layer=args.layer,
        latent_mult=CFG.latent_mult, topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.no_pcs_predictor_ckpt, device=CFG.device,
    )
    if args.no_pcs_qwen_ckpt and os.path.exists(args.no_pcs_qwen_ckpt):
        state = torch.load(args.no_pcs_qwen_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("  No-PCS model weights loaded.")
    model.to(CFG.device)
    model.eval()
    return model


# ── 评估循环 ──────────────────────────────────────────────────────────────────

def evaluate_base(base_model, processor, dataset, save_path=None, label="base"):
    from qwen_vl_utils import process_vision_info

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        subtask = item["subtask"]
        prompt = item["prompt"]
        answer = item["answer"]
        choices = item["choices"]

        has_image_choices = (
            isinstance(choices, list) and len(choices) > 0
            and isinstance(choices[0], Image.Image)
        )
        prompt_text = build_blink_prompt(prompt, choices, has_image_choices)

        image_paths = save_images_for_item(item)
        if not image_paths:
            print(f"  [warning] No images for idx={item.get('idx', '?')}")
            results.append({
                "idx": item.get("idx", len(results)),
                "subtask": subtask, "answer": answer,
                "response": "", "extracted": None, "correct": False,
            })
            continue

        concat_path = concat_images_horizontal(image_paths)

        response = ""
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{concat_path}"},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(base_model.device)

            with torch.no_grad():
                output_ids = base_model.generate(
                    **inputs, max_new_tokens=32, do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [error] idx={item.get('idx', '?')}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({
            "idx": item.get("idx", len(results)),
            "subtask": subtask, "answer": answer,
            "response": response, "extracted": extracted, "correct": correct,
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")
    return results


def evaluate_enhanced(model, dataset, save_path=None, label="enhanced"):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc=f"{label} eval"):
        subtask = item["subtask"]
        prompt = item["prompt"]
        answer = item["answer"]
        choices = item["choices"]

        has_image_choices = (
            isinstance(choices, list) and len(choices) > 0
            and isinstance(choices[0], Image.Image)
        )
        prompt_text = build_blink_prompt(prompt, choices, has_image_choices)

        image_paths = save_images_for_item(item)
        if not image_paths:
            results.append({
                "idx": item.get("idx", len(results)),
                "subtask": subtask, "answer": answer,
                "response": "", "extracted": None, "correct": False,
                "cluster_ids": [],
            })
            continue

        concat_path = concat_images_horizontal(image_paths)

        response = ""
        cluster_ids = []
        try:
            result = model.generate(
                image_path=concat_path, question=prompt_text, verbose=False,
            )
            response = result["final_answer"]
            cluster_ids = result.get("cluster_ids", [])
        except Exception as e:
            print(f"  [error] idx={item.get('idx', '?')}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({
            "idx": item.get("idx", len(results)),
            "subtask": subtask, "answer": answer,
            "response": response, "extracted": extracted, "correct": correct,
            "cluster_ids": cluster_ids,
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")
    return results


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_blink_dataset(subtasks: list, split: str = "val",
                       max_samples: int | None = None):
    """
    加载 BLINK 数据集，合并多个子任务。
    val split 有答案，test split 无答案（需提交 EvalAI）。
    """
    all_items = []
    for subtask in subtasks:
        print(f"  Loading {subtask}...")
        try:
            ds = load_dataset("BLINK-Benchmark/BLINK", subtask, split=split)
            columns = ds.column_names
            print(f"    columns: {columns}")

            for i, item in enumerate(ds):
                unified = {
                    "idx": f"{subtask}_{i}",
                    "subtask": subtask,
                    "prompt": item.get("prompt", item.get("question", "")),
                    "answer": item.get("answer", ""),
                    "choices": [],
                    "image_1": item.get("image_1", item.get("image", None)),
                    "image_2": item.get("image_2", None),
                    "image_3": item.get("image_3", None),
                    "image_4": item.get("image_4", None),
                }

                if "choices" in item and item["choices"] is not None:
                    unified["choices"] = item["choices"]
                elif "options" in item and item["options"] is not None:
                    unified["choices"] = item["options"]

                if isinstance(unified["choices"], str):
                    try:
                        unified["choices"] = json.loads(unified["choices"])
                    except json.JSONDecodeError:
                        unified["choices"] = [
                            c.strip() for c in unified["choices"].split(",")
                        ]

                all_items.append(unified)
            print(f"    {subtask}: {len(ds)} samples loaded")
        except Exception as e:
            print(f"    [error] Failed to load {subtask}: {e}")

    if max_samples and len(all_items) > max_samples:
        per_task = max(1, max_samples // len(subtasks))
        sampled = []
        by_task = defaultdict(list)
        for item in all_items:
            by_task[item["subtask"]].append(item)
        for task_items in by_task.values():
            sampled.extend(task_items[:per_task])
        all_items = sampled[:max_samples]

    print(f"  Total: {len(all_items)} samples")
    return all_items


# ── 统计 ──────────────────────────────────────────────────────────────────────

def compute_metrics(results, label=""):
    by_subtask = defaultdict(list)
    for r in results:
        by_subtask[r["subtask"]].append(r)

    def acc(items):
        if not items:
            return 0.0
        return sum(1 for i in items if i["correct"]) / len(items) * 100

    n_unparsed = sum(1 for r in results if r.get("extracted") is None)
    n_total = len(results)

    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) "
              f"responses could not be parsed")
        unparsed_by_task = defaultdict(int)
        for r in results:
            if r.get("extracted") is None:
                unparsed_by_task[r["subtask"]] += 1
        for t, c in sorted(unparsed_by_task.items()):
            print(f"    {t}: {c}/{len(by_subtask[t])} unparsed")
        for s in [r for r in results if r.get("extracted") is None][:3]:
            print(f"    example: subtask={s['subtask']} ans={s['answer']}")
            print(f"      resp=\"{s['response'][:100]}\"")

    per_subtask = {t: acc(by_subtask[t]) for t in sorted(by_subtask.keys())}
    overall_acc = (
        sum(per_subtask.values()) / len(per_subtask) if per_subtask else 0.0
    )

    depth_spatial_items = []
    for t in DEPTH_SPATIAL_SUBTASKS:
        if t in by_subtask:
            depth_spatial_items.extend(by_subtask[t])
    acc_depth_spatial = acc(depth_spatial_items)

    other_items = []
    for t in by_subtask:
        if t not in DEPTH_SPATIAL_SUBTASKS:
            other_items.extend(by_subtask[t])
    acc_other = acc(other_items)

    print(f"\n{'='*60}")
    print(f"  BLINK Results: {label}")
    print(f"{'='*60}")
    print(f"  BLINK Overall (macro avg) : {overall_acc:.2f}")
    print(f"  Depth/Spatial Accuracy    : {acc_depth_spatial:.2f}  "
          f"(n={len(depth_spatial_items)})")
    print(f"  Other Tasks Accuracy      : {acc_other:.2f}  "
          f"(n={len(other_items)})")
    print(f"{'─'*60}")
    print(f"  Per-Subtask:")
    for t in sorted(by_subtask.keys()):
        items = by_subtask[t]
        marker = " *" if t in DEPTH_SPATIAL_SUBTASKS else ""
        print(f"    {t:30s}: {acc(items):6.2f}  (n={len(items)}){marker}")
    print(f"  (* = depth/spatial related)")
    print(f"  Unparsed: {n_unparsed}/{n_total}")
    print(f"{'='*60}")

    return {
        "overall": overall_acc,
        "acc_depth_spatial": acc_depth_spatial,
        "acc_other": acc_other,
        "per_subtask": per_subtask,
        "n_unparsed": n_unparsed,
        "n_total": n_total,
    }


def print_comparison(metrics_dict):
    labels = list(metrics_dict.keys())
    if len(labels) < 2:
        return

    print(f"\n{'='*70}")
    print(f"  BLINK Comparison")
    print(f"{'='*70}")

    header = f"  {'Metric':<32s}"
    for label in labels:
        header += f" {label:>12s}"
    if len(labels) >= 2:
        header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*65}")

    for name, key in [("BLINK Overall", "overall"),
                      ("Depth/Spatial Avg", "acc_depth_spatial"),
                      ("Other Tasks Avg", "acc_other")]:
        line = f"  {name:<32s}"
        vals = [metrics_dict[l][key] for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    all_subtasks = sorted(set(
        t for m in metrics_dict.values() for t in m["per_subtask"]
    ))
    print(f"  {'─'*65}")
    for t in all_subtasks:
        marker = "*" if t in DEPTH_SPATIAL_SUBTASKS else " "
        line = f"  {marker} {t:<30s}"
        vals = [metrics_dict[l]["per_subtask"].get(t, 0) for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    print(f"  {'─'*65}")
    line = f"  {'Unparsed':<32s}"
    for l in labels:
        line += f" {metrics_dict[l]['n_unparsed']:>12d}"
    print(line)
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BLINK Benchmark Evaluation")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "spatial", "no_pcs",
                                 "finetune_baseline", "both", "all"])
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_0.75_64_5000/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None,
                        help="For enhanced: Stage 2 weights. "
                             "For finetune_baseline: ablation weights.")
    parser.add_argument("--baseline_ckpt", type=str, default=None,
                        help="Finetune baseline weights (used in --mode all)")
    parser.add_argument("--spatial_predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_spatial/predictor_best.pt")
    parser.add_argument("--spatial_qwen_ckpt", type=str, default=None)
    parser.add_argument("--no_pcs_predictor_ckpt", type=str,
                        default="outputs/ablation_no_pcs/predictor_best.pt")
    parser.add_argument("--no_pcs_qwen_ckpt", type=str, default=None)
    # ── BLINK 特有参数 ──
    parser.add_argument("--subtasks", type=str, nargs="+", default=None,
                        help="指定子任务，例: --subtasks Relative_Depth Spatial_Relation")
    parser.add_argument("--depth_spatial_only", action="store_true",
                        help="只评估 depth/spatial 相关的4个子任务")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="val 有答案; test 无答案 (需提交 EvalAI)")
    parser.add_argument("--save_dir", type=str, default="outputs/blink_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 确定要评估的子任务 ──
    if args.subtasks:
        subtasks = args.subtasks
    elif args.depth_spatial_only:
        subtasks = DEPTH_SPATIAL_SUBTASKS
    else:
        subtasks = ALL_SUBTASKS

    print(f"Subtasks to evaluate ({len(subtasks)}):")
    for t in subtasks:
        marker = " (depth/spatial)" if t in DEPTH_SPATIAL_SUBTASKS else ""
        print(f"  - {t}{marker}")

    # ── 加载数据集 ──
    print("\nLoading BLINK...")
    dataset = load_blink_dataset(subtasks, args.split, args.max_samples)

    subtask_counts = defaultdict(int)
    for item in dataset:
        subtask_counts[item["subtask"]] += 1
    for t, c in sorted(subtask_counts.items()):
        print(f"    {t}: {c}")

    # ── 评估 ──
    all_metrics = {}
    run_base     = args.mode in ["base", "both", "all"]
    run_ft_base  = args.mode in ["finetune_baseline", "all"]
    run_no_pcs   = args.mode in ["no_pcs", "all"]
    run_enhanced = args.mode in ["enhanced", "both", "all"]
    run_spatial  = args.mode in ["spatial", "all"]

    if run_base:
        base_model, processor = load_base_model()
        res = evaluate_base(
            base_model, processor, dataset,
            save_path=os.path.join(args.save_dir, "base_results.json"),
            label="base (vanilla)",
        )
        all_metrics["base"] = compute_metrics(res, label="Base Qwen2.5-VL")
        del base_model, processor; torch.cuda.empty_cache()

    if run_ft_base:
        ft_ckpt = args.baseline_ckpt or args.qwen_ckpt
        ft_model, ft_proc = load_finetune_baseline(ft_ckpt)
        res = evaluate_base(
            ft_model, ft_proc, dataset,
            save_path=os.path.join(args.save_dir, "ft_baseline_results.json"),
            label="finetune_baseline",
        )
        all_metrics["ft_baseline"] = compute_metrics(
            res, label="Finetune Baseline")
        del ft_model, ft_proc; torch.cuda.empty_cache()

    if run_no_pcs:
        no_pcs_model = load_enhanced_model_no_pcs(args)
        res = evaluate_enhanced(
            no_pcs_model, dataset,
            save_path=os.path.join(args.save_dir, "no_pcs_results.json"),
            label="enhanced (no PCS)",
        )
        all_metrics["no_pcs"] = compute_metrics(res, label="Enhanced (no PCS)")
        del no_pcs_model; torch.cuda.empty_cache()

    if run_enhanced:
        enhanced_model = load_enhanced_model(args)
        res = evaluate_enhanced(
            enhanced_model, dataset,
            save_path=os.path.join(args.save_dir, "enhanced_results.json"),
            label="enhanced (v1)",
        )
        all_metrics["enhanced"] = compute_metrics(res, label="Enhanced (v1)")
        del enhanced_model; torch.cuda.empty_cache()

    if run_spatial:
        spatial_model = load_spatial_model(args)
        res = evaluate_enhanced(
            spatial_model, dataset,
            save_path=os.path.join(args.save_dir, "spatial_results.json"),
            label="spatial (v2)",
        )
        all_metrics["spatial"] = compute_metrics(res, label="Spatial (v2)")
        del spatial_model; torch.cuda.empty_cache()

    if len(all_metrics) >= 2:
        print_comparison(all_metrics)

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nDone. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()