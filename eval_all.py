"""
统一评估脚本：一次性测试 CV-Bench, BLINK, RealWorldQA, MMStar。

支持模式: base, enhanced, spatial, finetune_baseline, no_pcs, both, all

用法：
    # 全部 benchmark + base vs enhanced
    python scripts/eval_all.py --mode enhanced --layer 8 --no_pcs --benchmarks cvbench blink --num_runs 3 --subsample_ratio 0.8
    python eval_all.py --mode no_pcs --layer 8 --benchmarks cvbench blink --num_runs 3 --subsample_ratio 0.8
    # 只跑指定 benchmark
    python eval/eval_all.py --mode both --benchmarks cvbench blink

    # 全部模型对比
    python eval/eval_all.py --mode all --qwen_ckpt outputs/ablation_baseline/qwen_best.pt

    # 限制样本数（快速测试）
    python eval/eval_all.py --mode both --max_samples 50

    # BLINK 只跑 depth/spatial 子任务
    python eval/eval_all.py --mode both --depth_spatial_only

    # 指定 spatial 权重
    python eval/eval_all.py --mode spatial --spatial_qwen_ckpt outputs/focus_v2_ckpt/qwen_best.pt

    # 跑3次，报告 mean±std（每次随机采样80%数据）
    python eval_all.py --mode both --num_runs 3 --subsample_ratio 0.8

    # 跑5次，采样90%
    python eval/eval_all.py --mode both --num_runs 5 --subsample_ratio 0.9
"""
import os
import sys
import re
import json
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import CFG


# ══════════════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════════════

ALL_BLINK_SUBTASKS = [
    "Art_Style", "Counting", "Forensic_Detection",
    "Functional_Correspondence", "IQ_Test", "Jigsaw",
    "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance",
    "Semantic_Correspondence", "Spatial_Relation",
    "Visual_Correspondence", "Visual_Similarity",
]

DEPTH_SPATIAL_SUBTASKS = [
    "Relative_Depth", "Spatial_Relation",
    "Multi-view_Reasoning", "Object_Localization",
]


# ══════════════════════════════════════════════════════════════════════════════
# 通用工具
# ══════════════════════════════════════════════════════════════════════════════

def extract_choice_letter(response: str, max_letter: str = "Z") -> str | None:
    """从模型输出中提取选项字母。"""
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None
    pat = f'[A-{max_letter}]'
    m = re.search(rf'\(({pat})\)', text)
    if m:
        return m.group(1)
    m = re.search(rf'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?({pat})\)?', text)
    if m:
        return m.group(1)
    m = re.match(rf'^({pat})(?:[\s.,):]|$)', text)
    if m:
        return m.group(1)
    return None


def match_by_content(response: str, choices: list) -> str | None:
    resp = response.strip().split("\n")[0].strip().lower()
    for i, choice in enumerate(choices):
        if isinstance(choice, str) and resp == choice.strip().lower():
            return chr(ord('A') + i)
    return None


def concat_images_horizontal(image_paths: list, max_height: int = 768) -> str:
    """水平拼接多张图片。"""
    if len(image_paths) == 1:
        return image_paths[0]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    target_h = min(max_height, max(img.height for img in images))
    resized = []
    for img in images:
        ratio = target_h / img.height
        resized.append(img.resize((int(img.width * ratio), target_h), Image.LANCZOS))
    total_w = sum(img.width for img in resized)
    concat = Image.new("RGB", (total_w, target_h))
    x = 0
    for img in resized:
        concat.paste(img, (x, 0))
        x += img.width
    out = "/tmp/eval_all_concat.png"
    concat.save(out)
    return out


def subsample(ds, ratio, seed, is_list=False):
    """从 dataset 中随机采样 ratio 比例的数据，用于 bootstrap 多次评估。"""
    random.seed(seed)
    n = len(ds)
    k = int(n * ratio)
    indices = random.sample(range(n), k)
    if is_list:
        return [ds[i] for i in indices]
    else:
        return ds.select(indices)


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_base_model():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print("Loading vanilla Qwen2.5-VL for baseline...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id, torch_dtype=torch.float16, device_map=CFG.device,
    )
    model.eval()
    return model, processor


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


# ══════════════════════════════════════════════════════════════════════════════
# 推理封装：base 模型 (Qwen2.5-VL 原始推理)
# ══════════════════════════════════════════════════════════════════════════════

def base_generate(model, processor, image_path: str, prompt: str,
                  max_new_tokens: int = 32) -> str:
    """用原始 Qwen2.5-VL 推理。"""
    from qwen_vl_utils import process_vision_info
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        return processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()
    except Exception as e:
        print(f"  [error] base_generate: {e}")
        return ""


def enhanced_generate(model, image_path: str, prompt: str) -> dict:
    """用增强模型推理，返回 dict(final_answer, cluster_ids)。"""
    try:
        return model.generate(image_path=image_path, question=prompt, verbose=False)
    except Exception as e:
        print(f"  [error] enhanced_generate: {e}")
        return {"final_answer": "", "cluster_ids": []}


# ══════════════════════════════════════════════════════════════════════════════
# CV-Bench
# ══════════════════════════════════════════════════════════════════════════════

def load_cvbench(max_samples=None, subset=None):
    print("Loading CV-Bench...")
    if subset:
        ds = load_dataset("nyu-visionx/CV-Bench", subset, split="test")
    else:
        ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")
    return ds


def eval_cvbench_single(model, processor, ds, is_enhanced=False, label=""):
    print(f"\n  Evaluating CV-Bench: {label}")
    results = []
    for item in tqdm(ds, desc=f"CV-Bench [{label}]"):
        choices = item["choices"]
        choice_text = "\n".join(
            [f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)]
        )
        prompt = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter.\n"
            "Do NOT output explanation.\n\n"
            f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
        )

        tmp_path = "/tmp/eval_all_cvbench.png"
        img = item["image"]
        if isinstance(img, Image.Image):
            img.save(tmp_path)
        else:
            tmp_path = img

        cluster_ids = []
        if is_enhanced:
            res = enhanced_generate(model, tmp_path, prompt)
            response = res["final_answer"]
            cluster_ids = res.get("cluster_ids", [])
        else:
            response = base_generate(model, processor, tmp_path, prompt)

        extracted = extract_choice_letter(response)
        if extracted is None:
            extracted = match_by_content(response, choices)
        answer = item["answer"]
        extracted_fmt = f"({extracted})" if extracted else None
        correct = extracted_fmt == answer

        r = {
            "idx": item["idx"], "task": item["task"],
            "source": item["source"], "type": item["type"],
            "answer": answer, "response": response,
            "extracted": extracted_fmt, "correct": correct,
        }
        if is_enhanced:
            r["cluster_ids"] = cluster_ids
        results.append(r)

    return results


def compute_cvbench_metrics(results, label=""):
    by_source = defaultdict(list)
    by_task = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)
        by_task[r["task"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    acc_ade = acc(by_source.get("ADE20K", []))
    acc_coco = acc(by_source.get("COCO", []))
    acc_omni = acc(by_source.get("Omni3D", []))
    acc_2d = (acc_ade + acc_coco) / 2
    acc_3d = acc_omni
    cv_bench = (acc_2d + acc_3d) / 2
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  CV-Bench [{label}]: Overall={cv_bench:.2f}  "
          f"2D={acc_2d:.2f}  3D={acc_3d:.2f}  "
          f"Unparsed={n_unparsed}/{len(results)}")

    return {
        "cv_bench": cv_bench, "acc_2d": acc_2d, "acc_3d": acc_3d,
        "acc_ade": acc_ade, "acc_coco": acc_coco, "acc_omni": acc_omni,
        "per_task": {t: acc(items) for t, items in by_task.items()},
        "n_unparsed": n_unparsed, "n_total": len(results),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MMStar
# ══════════════════════════════════════════════════════════════════════════════

def load_mmstar(max_samples=None):
    print("Loading MMStar...")
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")
    return ds


def eval_mmstar_single(model, processor, ds, is_enhanced=False, label=""):
    print(f"\n  Evaluating MMStar: {label}")
    results = []
    for item in tqdm(ds, desc=f"MMStar [{label}]"):
        prompt = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter (A, B, C, or D).\n"
            "Do NOT output explanation.\n\n"
            f"{item['question']}\n\nAnswer:"
        )

        tmp_path = "/tmp/eval_all_mmstar.png"
        img = item["image"]
        if isinstance(img, Image.Image):
            img.save(tmp_path)
        else:
            tmp_path = img

        cluster_ids = []
        if is_enhanced:
            res = enhanced_generate(model, tmp_path, prompt)
            response = res["final_answer"]
            cluster_ids = res.get("cluster_ids", [])
        else:
            response = base_generate(model, processor, tmp_path, prompt)

        extracted = extract_choice_letter(response, max_letter="D")
        answer = item["answer"].strip().upper()
        correct = extracted == answer

        r = {
            "index": item["index"], "category": item["category"],
            "l2_category": item["l2_category"],
            "answer": answer, "response": response,
            "extracted": extracted, "correct": correct,
        }
        if is_enhanced:
            r["cluster_ids"] = cluster_ids
        results.append(r)

    return results


def compute_mmstar_metrics(results, label=""):
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    overall = acc(results)
    cat_accs = {c: acc(items) for c, items in sorted(by_category.items())}
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  MMStar [{label}]: Overall={overall:.2f}  "
          f"Unparsed={n_unparsed}/{len(results)}")

    return {
        "overall": overall, "per_category": cat_accs,
        "n_unparsed": n_unparsed, "n_total": len(results),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RealWorldQA
# ══════════════════════════════════════════════════════════════════════════════

def detect_question_type(question: str, answer: str) -> str:
    ans = answer.strip()
    if ans in ("A", "B", "C", "D"):
        return "mcq"
    if ans.lower() in ("yes", "no"):
        return "yes_no"
    return "short"


def match_realworldqa(response: str, answer: str, q_type: str):
    resp = response.strip()
    ans = answer.strip()

    if q_type == "mcq":
        extracted = extract_choice_letter(resp, max_letter="D")
        if extracted:
            return extracted == ans, extracted
        return False, None

    if q_type == "yes_no":
        resp_lower = resp.lower()
        if resp_lower.startswith("yes"):
            return "Yes" == ans, "Yes"
        if resp_lower.startswith("no"):
            return "No" == ans, "No"
        return False, None

    # short answer
    resp_norm = resp.split("\n")[0].strip().lower().rstrip(".")
    ans_norm = ans.lower().rstrip(".")
    if resp_norm == ans_norm or resp_norm.startswith(ans_norm):
        return True, resp
    try:
        if float(resp_norm) == float(ans_norm):
            return True, resp
    except ValueError:
        pass
    return False, resp


def load_realworldqa(max_samples=None):
    print("Loading RealWorldQA...")
    ds = load_dataset("xai-org/RealworldQA", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")
    return ds


def eval_realworldqa_single(model, processor, ds, is_enhanced=False, label=""):
    print(f"\n  Evaluating RealWorldQA: {label}")
    results = []
    for i, item in enumerate(tqdm(ds, desc=f"RealWorldQA [{label}]")):
        question = item["question"]
        answer = item["answer"]
        q_type = detect_question_type(question, answer)

        tmp_path = "/tmp/eval_all_rwqa.png"
        img = item["image"]
        if isinstance(img, Image.Image):
            img.save(tmp_path)
        else:
            tmp_path = img

        cluster_ids = []
        if is_enhanced:
            res = enhanced_generate(model, tmp_path, question)
            response = res["final_answer"]
            cluster_ids = res.get("cluster_ids", [])
        else:
            response = base_generate(model, processor, tmp_path, question)

        correct, extracted = match_realworldqa(response, answer, q_type)

        r = {
            "idx": i, "q_type": q_type, "answer": answer,
            "response": response, "extracted": extracted, "correct": correct,
        }
        if is_enhanced:
            r["cluster_ids"] = cluster_ids
        results.append(r)

    return results


def compute_realworldqa_metrics(results, label=""):
    by_type = defaultdict(list)
    for r in results:
        by_type[r["q_type"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    overall = acc(results)
    per_type = {t: acc(items) for t, items in sorted(by_type.items())}
    n_unparsed = sum(1 for r in results
                     if r.get("extracted") is None and r["q_type"] == "mcq")

    print(f"\n  RealWorldQA [{label}]: Overall={overall:.2f}  "
          f"Unparsed MCQ={n_unparsed}")

    return {
        "overall": overall, "per_type": per_type,
        "n_unparsed": n_unparsed, "n_total": len(results),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BLINK
# ══════════════════════════════════════════════════════════════════════════════

def load_blink(subtasks, max_samples=None):
    print("Loading BLINK...")
    all_items = []
    for subtask in subtasks:
        try:
            ds = load_dataset("BLINK-Benchmark/BLINK", subtask, split="val")
            for i, item in enumerate(ds):
                choices = item.get("choices", item.get("options", []))
                if isinstance(choices, str):
                    try:
                        choices = json.loads(choices)
                    except json.JSONDecodeError:
                        choices = [c.strip() for c in choices.split(",")]
                all_items.append({
                    "idx": f"{subtask}_{i}", "subtask": subtask,
                    "prompt": item.get("prompt", item.get("question", "")),
                    "answer": item.get("answer", ""), "choices": choices,
                    "image_1": item.get("image_1", item.get("image", None)),
                    "image_2": item.get("image_2", None),
                    "image_3": item.get("image_3", None),
                    "image_4": item.get("image_4", None),
                })
            print(f"  {subtask}: {len(ds)} samples")
        except Exception as e:
            print(f"  [error] {subtask}: {e}")

    if max_samples and len(all_items) > max_samples:
        per_task = max(1, max_samples // len(subtasks))
        by_task = defaultdict(list)
        for item in all_items:
            by_task[item["subtask"]].append(item)
        sampled = []
        for items in by_task.values():
            sampled.extend(items[:per_task])
        all_items = sampled[:max_samples]

    print(f"  Total: {len(all_items)} samples")
    return all_items


def eval_blink_single(model, processor, dataset, is_enhanced=False, label=""):
    print(f"\n  Evaluating BLINK: {label}")
    tmp_dir = "/tmp/eval_all_blink"
    os.makedirs(tmp_dir, exist_ok=True)

    results = []
    for item in tqdm(dataset, desc=f"BLINK [{label}]"):
        choices = item["choices"]
        has_image_choices = (
            isinstance(choices, list) and len(choices) > 0
            and isinstance(choices[0], Image.Image)
        )
        if has_image_choices:
            choice_text = "\n".join(
                [f"{chr(ord('A')+i)}. (Image {i+1})" for i in range(len(choices))]
            )
        else:
            choice_text = "\n".join(
                [f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)]
            )
        prompt = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter.\n"
            "Do NOT output explanation.\n\n"
            f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
        )

        # 保存图片
        image_paths = []
        for key in ["image_1", "image_2", "image_3", "image_4"]:
            img = item.get(key)
            if img is not None and isinstance(img, Image.Image):
                p = os.path.join(tmp_dir, f"{key}.png")
                img.save(p)
                image_paths.append(p)

        if not image_paths:
            results.append({
                "idx": item["idx"], "subtask": item["subtask"],
                "answer": item["answer"], "response": "",
                "extracted": None, "correct": False,
            })
            continue

        concat_path = concat_images_horizontal(image_paths)

        cluster_ids = []
        if is_enhanced:
            res = enhanced_generate(model, concat_path, prompt)
            response = res["final_answer"]
            cluster_ids = res.get("cluster_ids", [])
        else:
            response = base_generate(model, processor, concat_path, prompt)

        extracted = extract_choice_letter(response)
        if extracted is None:
            extracted_content = match_by_content(response, choices)
            if extracted_content:
                extracted = extracted_content
        answer = item["answer"]
        extracted_fmt = f"({extracted})" if extracted else None
        correct = extracted_fmt == answer

        r = {
            "idx": item["idx"], "subtask": item["subtask"],
            "answer": answer, "response": response,
            "extracted": extracted_fmt, "correct": correct,
        }
        if is_enhanced:
            r["cluster_ids"] = cluster_ids
        results.append(r)

    return results


def compute_blink_metrics(results, label=""):
    by_subtask = defaultdict(list)
    for r in results:
        by_subtask[r["subtask"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    per_subtask = {t: acc(by_subtask[t]) for t in sorted(by_subtask.keys())}
    overall = sum(per_subtask.values()) / len(per_subtask) if per_subtask else 0
    ds_items = [r for t in DEPTH_SPATIAL_SUBTASKS for r in by_subtask.get(t, [])]
    acc_ds = acc(ds_items)
    other_items = [r for t in by_subtask if t not in DEPTH_SPATIAL_SUBTASKS
                   for r in by_subtask[t]]
    acc_other = acc(other_items)
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  BLINK [{label}]: Overall={overall:.2f}  "
          f"Depth/Spatial={acc_ds:.2f}  Other={acc_other:.2f}  "
          f"Unparsed={n_unparsed}/{len(results)}")

    return {
        "overall": overall, "acc_depth_spatial": acc_ds, "acc_other": acc_other,
        "per_subtask": per_subtask,
        "n_unparsed": n_unparsed, "n_total": len(results),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 综合对比打印
# ══════════════════════════════════════════════════════════════════════════════

def print_grand_summary(all_results: dict, model_labels: list):
    """
    all_results: {benchmark: {model_label: metrics_dict}}
    """
    print(f"\n{'═'*75}")
    print(f"  Grand Summary — All Benchmarks × All Models")
    print(f"{'═'*75}")

    header = f"  {'Benchmark / Metric':<30s}"
    for ml in model_labels:
        header += f" {ml:>12s}"
    if len(model_labels) >= 2:
        header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*70}")

    bench_keys = [
        ("CV-Bench", "cvbench", "cv_bench"),
        ("  2D", "cvbench", "acc_2d"),
        ("  3D", "cvbench", "acc_3d"),
        ("MMStar", "mmstar", "overall"),
        ("RealWorldQA", "realworldqa", "overall"),
        ("BLINK Overall", "blink", "overall"),
        ("  Depth/Spatial", "blink", "acc_depth_spatial"),
        ("  Other", "blink", "acc_other"),
    ]

    for name, bench, key in bench_keys:
        if bench not in all_results:
            continue
        line = f"  {name:<30s}"
        vals = []
        for ml in model_labels:
            v = all_results[bench].get(ml, {}).get(key, 0)
            vals.append(v)
            line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    print(f"{'═'*75}")


def aggregate_multi_runs(all_runs_results: list, model_labels: list):
    """
    all_runs_results: list of {benchmark: {model_label: metrics_dict}}
    返回: {benchmark: {model_label: {metric: {"mean": x, "std": x, "runs": [...]}}}}
    """
    benchmarks = set()
    for run in all_runs_results:
        benchmarks.update(run.keys())

    aggregated = {}
    for bench in benchmarks:
        aggregated[bench] = {}
        for ml in model_labels:
            # 收集每次 run 的 metrics
            all_metrics = [run[bench][ml] for run in all_runs_results
                           if bench in run and ml in run[bench]]
            if not all_metrics:
                continue
            # 提取所有数值型 key
            agg = {}
            for key in all_metrics[0]:
                vals = [m[key] for m in all_metrics if key in m]
                if vals and isinstance(vals[0], (int, float)):
                    agg[key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "runs": [float(v) for v in vals],
                    }
            aggregated[bench][ml] = agg
    return aggregated


def print_grand_summary_with_std(aggregated: dict, model_labels: list):
    print(f"\n{'═'*85}")
    print(f"  Grand Summary (mean ± std) — All Benchmarks × All Models")
    print(f"{'═'*85}")

    header = f"  {'Benchmark / Metric':<30s}"
    for ml in model_labels:
        header += f" {ml:>18s}"
    if len(model_labels) >= 2:
        header += f"  {'Delta':>10s}"
    print(header)
    print(f"  {'─'*80}")

    bench_keys = [
        ("CV-Bench",        "cvbench",     "cv_bench"),
        ("  2D",            "cvbench",     "acc_2d"),
        ("  3D",            "cvbench",     "acc_3d"),
        ("MMStar",          "mmstar",      "overall"),
        ("RealWorldQA",     "realworldqa", "overall"),
        ("BLINK Overall",   "blink",       "overall"),
        ("  Depth/Spatial", "blink",       "acc_depth_spatial"),
        ("  Other",         "blink",       "acc_other"),
    ]

    for name, bench, key in bench_keys:
        if bench not in aggregated:
            continue
        line = f"  {name:<30s}"
        means = []
        for ml in model_labels:
            stats = aggregated.get(bench, {}).get(ml, {}).get(key, None)
            if stats:
                line += f" {stats['mean']:6.2f}±{stats['std']:4.2f}   "
                means.append(stats["mean"])
            else:
                line += f" {'N/A':>18s}"
                means.append(None)
        if len(means) >= 2 and means[0] is not None and means[-1] is not None:
            d = means[-1] - means[0]
            line += f"  {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    print(f"{'═'*85}")


# ══════════════════════════════════════════════════════════════════════════════
# 单个模型跑所有 benchmark
# ══════════════════════════════════════════════════════════════════════════════

def run_all_benchmarks_base(model, processor, datasets, model_label, save_dir):
    """用 base 类模型跑所有已加载的 benchmark。"""
    metrics = {}

    if "cvbench" in datasets:
        res = eval_cvbench_single(model, processor, datasets["cvbench"],
                                  is_enhanced=False, label=model_label)
        metrics["cvbench"] = compute_cvbench_metrics(res, model_label)
        _save_results(res, save_dir, f"cvbench_{model_label}.json")

    if "mmstar" in datasets:
        res = eval_mmstar_single(model, processor, datasets["mmstar"],
                                 is_enhanced=False, label=model_label)
        metrics["mmstar"] = compute_mmstar_metrics(res, model_label)
        _save_results(res, save_dir, f"mmstar_{model_label}.json")

    if "realworldqa" in datasets:
        res = eval_realworldqa_single(model, processor, datasets["realworldqa"],
                                      is_enhanced=False, label=model_label)
        metrics["realworldqa"] = compute_realworldqa_metrics(res, model_label)
        _save_results(res, save_dir, f"realworldqa_{model_label}.json")

    if "blink" in datasets:
        res = eval_blink_single(model, processor, datasets["blink"],
                                is_enhanced=False, label=model_label)
        metrics["blink"] = compute_blink_metrics(res, model_label)
        _save_results(res, save_dir, f"blink_{model_label}.json")

    return metrics


def run_all_benchmarks_enhanced(model, datasets, model_label, save_dir):
    """用增强类模型跑所有已加载的 benchmark。"""
    metrics = {}

    if "cvbench" in datasets:
        res = eval_cvbench_single(model, None, datasets["cvbench"],
                                  is_enhanced=True, label=model_label)
        metrics["cvbench"] = compute_cvbench_metrics(res, model_label)
        _save_results(res, save_dir, f"cvbench_{model_label}.json")

    if "mmstar" in datasets:
        res = eval_mmstar_single(model, None, datasets["mmstar"],
                                 is_enhanced=True, label=model_label)
        metrics["mmstar"] = compute_mmstar_metrics(res, model_label)
        _save_results(res, save_dir, f"mmstar_{model_label}.json")

    if "realworldqa" in datasets:
        res = eval_realworldqa_single(model, None, datasets["realworldqa"],
                                      is_enhanced=True, label=model_label)
        metrics["realworldqa"] = compute_realworldqa_metrics(res, model_label)
        _save_results(res, save_dir, f"realworldqa_{model_label}.json")

    if "blink" in datasets:
        res = eval_blink_single(model, None, datasets["blink"],
                                is_enhanced=True, label=model_label)
        metrics["blink"] = compute_blink_metrics(res, model_label)
        _save_results(res, save_dir, f"blink_{model_label}.json")

    return metrics


def _save_results(results, save_dir, filename):
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation: 4 Benchmarks")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "spatial", "no_pcs",
                                 "finetune_baseline", "both", "all"])
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["cvbench", "mmstar", "realworldqa", "blink"],
                        choices=["cvbench", "mmstar", "realworldqa", "blink"])
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/qwen_layer8_old/focus_ckpt_0.75_64_5000/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None,
                        help="For enhanced: Stage 2 weights. "
                             "For finetune_baseline: ablation weights.")
    parser.add_argument("--baseline_ckpt", type=str, default=None,
                        help="Finetune baseline weights (used in --mode all)")
    parser.add_argument("--spatial_predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_spatial/predictor_best.pt")
    parser.add_argument("--spatial_qwen_ckpt", type=str, default=None)
    parser.add_argument("--no_pcs_predictor_ckpt", type=str,
                        default="outputs/evaluation/ablation_no_pcs/predictor_best.pt")
    parser.add_argument("--no_pcs_qwen_ckpt", type=str, default=None)
    # BLINK 特有
    parser.add_argument("--blink_subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    # 多次运行
    parser.add_argument("--num_runs", type=int, default=1,
                        help="重复跑几次，报告 mean±std（每次随机采样子集）")
    parser.add_argument("--subsample_ratio", type=float, default=0.8,
                        help="每次 run 采样数据的比例（默认 0.8 即 80%%）")
    # 通用
    parser.add_argument("--save_dir", type=str, default="outputs/eval_all_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 确定哪些模型要跑 ──
    run_base     = args.mode in ["base", "both", "all"]
    run_ft_base  = args.mode in ["finetune_baseline", "all"]
    run_no_pcs   = args.mode in ["no_pcs", "all"]
    run_enhanced = args.mode in ["enhanced", "both", "all"]
    run_spatial  = args.mode in ["spatial", "all"]

    # ── 加载完整数据集（只加载一次，所有模型和所有 run 共用） ──
    datasets_full = {}

    if "cvbench" in args.benchmarks:
        datasets_full["cvbench"] = load_cvbench(args.max_samples)

    if "mmstar" in args.benchmarks:
        datasets_full["mmstar"] = load_mmstar(args.max_samples)

    if "realworldqa" in args.benchmarks:
        datasets_full["realworldqa"] = load_realworldqa(args.max_samples)

    if "blink" in args.benchmarks:
        blink_subtasks = args.blink_subtasks
        if args.depth_spatial_only:
            blink_subtasks = DEPTH_SPATIAL_SUBTASKS
        if blink_subtasks is None:
            blink_subtasks = ALL_BLINK_SUBTASKS
        datasets_full["blink"] = load_blink(blink_subtasks, args.max_samples)

    # ── 预加载所有需要的模型（避免每次 run 重复加载） ──
    base_model, processor = None, None
    ft_model, ft_proc = None, None
    no_pcs_model = None
    enhanced_model = None
    spatial_model = None

    if run_base:
        base_model, processor = load_base_model()
    if run_ft_base:
        ft_ckpt = args.baseline_ckpt or args.qwen_ckpt
        ft_model, ft_proc = load_finetune_baseline(ft_ckpt)
    if run_no_pcs:
        no_pcs_model = load_enhanced_model_no_pcs(args)
    if run_enhanced:
        enhanced_model = load_enhanced_model(args)
    if run_spatial:
        spatial_model = load_spatial_model(args)

    # ── 多次运行循环 ──
    all_runs_results = []
    model_labels = []

    for run_idx in range(args.num_runs):
        if args.num_runs > 1:
            print(f"\n{'▶'*30} Run {run_idx+1}/{args.num_runs} "
                  f"(seed={42+run_idx}, ratio={args.subsample_ratio}) {'◀'*30}\n")

        run_save_dir = (os.path.join(args.save_dir, f"run_{run_idx}")
                        if args.num_runs > 1 else args.save_dir)
        os.makedirs(run_save_dir, exist_ok=True)

        # ── 对数据集做随机子采样 ──
        seed = 42 + run_idx
        if args.num_runs > 1:
            datasets = {}
            for key, ds in datasets_full.items():
                datasets[key] = subsample(
                    ds, ratio=args.subsample_ratio, seed=seed,
                    is_list=(key == "blink"),
                )
                print(f"  Run {run_idx+1}: {key} sampled "
                      f"{len(datasets[key])}/{len(ds)} (seed={seed})")
        else:
            datasets = datasets_full

        run_results = defaultdict(dict)
        model_labels_this_run = []

        # ── 1. Base ──
        if run_base:
            label = "base"
            model_labels_this_run.append(label)
            metrics = run_all_benchmarks_base(
                base_model, processor, datasets, label, run_save_dir)
            for bench, m in metrics.items():
                run_results[bench][label] = m

        # ── 2. Finetune Baseline ──
        if run_ft_base:
            label = "ft_baseline"
            model_labels_this_run.append(label)
            metrics = run_all_benchmarks_base(
                ft_model, ft_proc, datasets, label, run_save_dir)
            for bench, m in metrics.items():
                run_results[bench][label] = m

        # ── 3. No-PCS Ablation ──
        if run_no_pcs:
            label = "no_pcs"
            model_labels_this_run.append(label)
            metrics = run_all_benchmarks_enhanced(
                no_pcs_model, datasets, label, run_save_dir)
            for bench, m in metrics.items():
                run_results[bench][label] = m

        # ── 4. Enhanced (v1) ──
        if run_enhanced:
            label = "enhanced"
            model_labels_this_run.append(label)
            metrics = run_all_benchmarks_enhanced(
                enhanced_model, datasets, label, run_save_dir)
            for bench, m in metrics.items():
                run_results[bench][label] = m

        # ── 5. Spatial (v2) ──
        if run_spatial:
            label = "spatial"
            model_labels_this_run.append(label)
            metrics = run_all_benchmarks_enhanced(
                spatial_model, datasets, label, run_save_dir)
            for bench, m in metrics.items():
                run_results[bench][label] = m

        all_runs_results.append(dict(run_results))
        model_labels = model_labels_this_run  # 每次 run 的 label 相同

        # 单次打印
        if len(model_labels) >= 2:
            print_grand_summary(dict(run_results), model_labels)

    # ── 释放模型显存 ──
    del base_model, processor, ft_model, ft_proc
    del no_pcs_model, enhanced_model, spatial_model
    torch.cuda.empty_cache()

    # ── 多次运行汇总 ──
    if args.num_runs > 1:
        aggregated = aggregate_multi_runs(all_runs_results, model_labels)
        print_grand_summary_with_std(aggregated, model_labels)

        # 保存汇总
        summary = {
            "config": {
                "num_runs": args.num_runs,
                "subsample_ratio": args.subsample_ratio,
                "seeds": [42 + i for i in range(args.num_runs)],
            },
            "aggregated": aggregated,
            "per_run": all_runs_results,
        }
        summary_path = os.path.join(args.save_dir, "summary_mean_std.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nDone. Aggregated summary (mean±std) saved to {summary_path}")
    else:
        summary = all_runs_results[0]
        summary_path = os.path.join(args.save_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nDone. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()