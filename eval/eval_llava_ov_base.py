"""
LLaVA-OneVision 基础模型评估脚本。

支持 4 个 benchmark：CV-Bench, BLINK, RealWorldQA, MMStar。
只测原始 LLaVA-OV 模型（无 hook / 无增强），用于获取 baseline 分数。

用法：
    # 跑全部 benchmark
    CUDA_VISIBLE_DEVICES=2 python eval/eval_llava_ov_base.py

    # 只跑某个 benchmark
    python eval/eval_llava_ov_base.py --benchmarks cvbench
    python eval/eval_llava_ov_base.py --benchmarks blink --depth_spatial_only
    python eval/eval_llava_ov_base.py --benchmarks mmstar realworldqa

    # 限制样本数（调试用）
    python eval/eval_llava_ov_base.py --max_samples 50

    # 指定 GPU
    CUDA_VISIBLE_DEVICES=1 python eval/eval_llava_ov_base.py
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


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_llava_ov():
    """加载 LLaVA-OneVision 原始模型。"""
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    model_id = getattr(CFG, "llava_model_id", "llava-hf/llava-onevision-qwen2-7b-ov-hf")
    print(f"Loading LLaVA-OneVision: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(CFG.device)
    model.eval()
    print("  Model loaded.")
    return model, processor


def llava_ov_generate(model, processor, image, question, max_new_tokens=32):
    """LLaVA-OV 单次推理。"""
    tmp_path = "/tmp/llava_ov_eval_tmp.png"
    if isinstance(image, Image.Image):
        image.save(tmp_path)
    elif isinstance(image, str):
        tmp_path = image
    else:
        return ""

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True,
    )
    pil_image = Image.open(tmp_path).convert("RGB") if isinstance(tmp_path, str) else image
    inputs = processor(
        images=pil_image, text=prompt, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(
        output_ids[0, input_len:], skip_special_tokens=True,
    ).strip()
    return response


# ══════════════════════════════════════════════════════════════════════════════
# 答案匹配工具
# ══════════════════════════════════════════════════════════════════════════════

def extract_choice_letter(response: str, max_letter="Z") -> str | None:
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


# ══════════════════════════════════════════════════════════════════════════════
# CV-Bench
# ══════════════════════════════════════════════════════════════════════════════

def eval_cvbench(model, processor, max_samples=None):
    print(f"\n{'='*60}")
    print(f"  CV-Bench Evaluation")
    print(f"{'='*60}")

    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for item in tqdm(ds, desc="CV-Bench"):
        choices = item["choices"]
        choice_text = "\n".join(
            [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
        )
        prompt_text = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter.\n"
            "Do NOT output explanation.\n\n"
            f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
        )

        try:
            response = llava_ov_generate(model, processor, item["image"], prompt_text)
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")
            response = ""

        extracted = extract_choice_letter(response)
        answer = item["answer"]  # 格式: "(A)"
        extracted_fmt = f"({extracted})" if extracted else None
        correct = extracted_fmt == answer

        results.append({
            "idx": item["idx"], "task": item["task"],
            "source": item["source"], "type": item["type"],
            "answer": answer, "response": response,
            "extracted": extracted_fmt, "correct": correct,
        })

    # ── 统计 ──
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

    print(f"\n  CV-Bench Overall : {cv_bench:.2f}")
    print(f"  2D Accuracy      : {acc_2d:.2f}  (ADE={acc_ade:.2f}, COCO={acc_coco:.2f})")
    print(f"  3D Accuracy      : {acc_3d:.2f}  (Omni3D={acc_omni:.2f})")
    print(f"  Per-Task:")
    for t in sorted(by_task.keys()):
        print(f"    {t:20s}: {acc(by_task[t]):6.2f}  (n={len(by_task[t])})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {
        "benchmark": "CV-Bench",
        "cv_bench": cv_bench, "acc_2d": acc_2d, "acc_3d": acc_3d,
        "acc_ade": acc_ade, "acc_coco": acc_coco, "acc_omni": acc_omni,
        "per_task": {t: acc(items) for t, items in by_task.items()},
        "n_unparsed": n_unparsed, "n_total": len(results),
    }, results


# ══════════════════════════════════════════════════════════════════════════════
# MMStar
# ══════════════════════════════════════════════════════════════════════════════

def eval_mmstar(model, processor, max_samples=None):
    print(f"\n{'='*60}")
    print(f"  MMStar Evaluation")
    print(f"{'='*60}")

    ds = load_dataset("Lin-Chen/MMStar", split="val")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for item in tqdm(ds, desc="MMStar"):
        prompt_text = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter (A, B, C, or D).\n"
            "Do NOT output explanation.\n\n"
            f"{item['question']}\n\nAnswer:"
        )

        try:
            response = llava_ov_generate(model, processor, item["image"], prompt_text)
        except Exception as e:
            print(f"  [error] idx={item['index']}: {e}")
            response = ""

        extracted = extract_choice_letter(response, max_letter="D")
        answer = item["answer"].strip().upper()
        correct = extracted == answer

        results.append({
            "index": item["index"], "category": item["category"],
            "l2_category": item["l2_category"],
            "answer": answer, "response": response,
            "extracted": extracted, "correct": correct,
        })

    # ── 统计 ──
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    overall = acc(results)
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  Overall Accuracy : {overall:.2f}  (n={len(results)})")
    print(f"  Per Category:")
    cat_accs = {}
    for cat in sorted(by_category.keys()):
        a = acc(by_category[cat])
        cat_accs[cat] = a
        print(f"    {cat:30s}: {a:6.2f}  (n={len(by_category[cat])})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {
        "benchmark": "MMStar", "overall": overall,
        "per_category": cat_accs,
        "n_unparsed": n_unparsed, "n_total": len(results),
    }, results


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


def eval_realworldqa(model, processor, max_samples=None):
    print(f"\n{'='*60}")
    print(f"  RealWorldQA Evaluation")
    print(f"{'='*60}")

    ds = load_dataset("xai-org/RealworldQA", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for i, item in enumerate(tqdm(ds, desc="RealWorldQA")):
        question = item["question"]
        answer = item["answer"]
        q_type = detect_question_type(question, answer)

        try:
            response = llava_ov_generate(model, processor, item["image"], question)
        except Exception as e:
            print(f"  [error] idx={i}: {e}")
            response = ""

        correct, extracted = match_realworldqa(response, answer, q_type)
        results.append({
            "idx": i, "q_type": q_type, "answer": answer,
            "response": response, "extracted": extracted, "correct": correct,
        })

    # ── 统计 ──
    by_type = defaultdict(list)
    for r in results:
        by_type[r["q_type"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    overall = acc(results)
    n_unparsed = sum(1 for r in results if r.get("extracted") is None and r["q_type"] == "mcq")

    print(f"\n  Overall Accuracy : {overall:.2f}  (n={len(results)})")
    print(f"  Per Type:")
    per_type = {}
    for t in ["mcq", "yes_no", "short"]:
        if t in by_type:
            a = acc(by_type[t])
            per_type[t] = a
            print(f"    {t:10s}: {a:6.2f}  (n={len(by_type[t])})")
    print(f"  Unparsed MCQ: {n_unparsed}/{len(by_type.get('mcq', []))}")

    return {
        "benchmark": "RealWorldQA", "overall": overall,
        "per_type": per_type,
        "n_unparsed": n_unparsed, "n_total": len(results),
    }, results


# ══════════════════════════════════════════════════════════════════════════════
# BLINK
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


def concat_images_horizontal(image_paths, max_height=768):
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
    out = "/tmp/blink_eval_concat.png"
    concat.save(out)
    return out


def eval_blink(model, processor, subtasks=None, max_samples=None):
    print(f"\n{'='*60}")
    print(f"  BLINK Evaluation")
    print(f"{'='*60}")

    if subtasks is None:
        subtasks = ALL_BLINK_SUBTASKS

    # ── 加载数据 ──
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

    # ── 评估 ──
    results = []
    for item in tqdm(all_items, desc="BLINK"):
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
        prompt_text = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter.\n"
            "Do NOT output explanation.\n\n"
            f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
        )

        # 处理图片
        tmp_dir = "/tmp/blink_eval"
        os.makedirs(tmp_dir, exist_ok=True)
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

        try:
            response = llava_ov_generate(model, processor, concat_path, prompt_text)
        except Exception as e:
            print(f"  [error] {item['idx']}: {e}")
            response = ""

        extracted = extract_choice_letter(response)
        answer = item["answer"]
        extracted_fmt = f"({extracted})" if extracted else None
        correct = extracted_fmt == answer

        results.append({
            "idx": item["idx"], "subtask": item["subtask"],
            "answer": answer, "response": response,
            "extracted": extracted_fmt, "correct": correct,
        })

    # ── 统计 ──
    by_subtask = defaultdict(list)
    for r in results:
        by_subtask[r["subtask"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    per_subtask = {t: acc(by_subtask[t]) for t in sorted(by_subtask.keys())}
    overall = sum(per_subtask.values()) / len(per_subtask) if per_subtask else 0

    ds_items = [r for t in DEPTH_SPATIAL_SUBTASKS for r in by_subtask.get(t, [])]
    acc_ds = acc(ds_items)
    other_items = [r for t in by_subtask if t not in DEPTH_SPATIAL_SUBTASKS for r in by_subtask[t]]
    acc_other = acc(other_items)
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  BLINK Overall (macro) : {overall:.2f}")
    print(f"  Depth/Spatial         : {acc_ds:.2f}  (n={len(ds_items)})")
    print(f"  Other Tasks           : {acc_other:.2f}  (n={len(other_items)})")
    print(f"  Per-Subtask:")
    for t in sorted(by_subtask.keys()):
        marker = " *" if t in DEPTH_SPATIAL_SUBTASKS else ""
        print(f"    {t:30s}: {acc(by_subtask[t]):6.2f}  (n={len(by_subtask[t])}){marker}")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {
        "benchmark": "BLINK", "overall": overall,
        "acc_depth_spatial": acc_ds, "acc_other": acc_other,
        "per_subtask": per_subtask,
        "n_unparsed": n_unparsed, "n_total": len(results),
    }, results


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLaVA-OneVision Base Model Evaluation")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["cvbench", "mmstar", "realworldqa", "blink"],
                        choices=["cvbench", "mmstar", "realworldqa", "blink"],
                        help="要评估的 benchmark")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/llava_ov_base_results")
    # BLINK 专用
    parser.add_argument("--blink_subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载模型（只加载一次）──
    model, processor = load_llava_ov()

    all_summary = {}

    # ── CV-Bench ──
    if "cvbench" in args.benchmarks:
        metrics, results = eval_cvbench(model, processor, args.max_samples)
        all_summary["cvbench"] = metrics
        with open(os.path.join(args.save_dir, "cvbench_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── MMStar ──
    if "mmstar" in args.benchmarks:
        metrics, results = eval_mmstar(model, processor, args.max_samples)
        all_summary["mmstar"] = metrics
        with open(os.path.join(args.save_dir, "mmstar_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── RealWorldQA ──
    if "realworldqa" in args.benchmarks:
        metrics, results = eval_realworldqa(model, processor, args.max_samples)
        all_summary["realworldqa"] = metrics
        with open(os.path.join(args.save_dir, "realworldqa_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── BLINK ──
    if "blink" in args.benchmarks:
        subtasks = args.blink_subtasks
        if args.depth_spatial_only:
            subtasks = DEPTH_SPATIAL_SUBTASKS
        metrics, results = eval_blink(model, processor, subtasks, args.max_samples)
        all_summary["blink"] = metrics
        with open(os.path.join(args.save_dir, "blink_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  LLaVA-OneVision Base — Summary")
    print(f"{'='*60}")
    for name, m in all_summary.items():
        key = "cv_bench" if name == "cvbench" else "overall"
        score = m.get(key, 0)
        print(f"  {name:15s}: {score:.2f}")
    print(f"{'='*60}")

    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nDone. Results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()