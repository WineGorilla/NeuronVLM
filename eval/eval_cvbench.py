"""
CV-Bench 评估脚本。

用法：
    python eval/eval_cvbench.py --mode base
    python eval/eval_cvbench.py --mode enhanced
    python eval/eval_cvbench.py --mode both
    python eval/eval_cvbench.py --mode both --max_samples 50
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


# ── Prompt 构造 ───────────────────────────────────────────────────────────────

def build_prompt(prompt: str, choices: list) -> str:
    """
    构造带 instruction + 选项列表的 prompt，引导模型只输出字母。
    """
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
    """
    从模型输出中提取选项字母，返回 '(A)' 格式或 None。
    """
    # 去 CoT：只看第一行
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None

    # "(A)" / "(B)"
    m = re.search(r'\(([A-D])\)', text)
    if m:
        return f"({m.group(1)})"

    # "Answer: A" / "answer is B" / "option C"
    m = re.search(r'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?([A-D])\)?', text)
    if m:
        return f"({m.group(1)})"

    # 开头单字母: "A" / "A." / "A,"
    m = re.match(r'^([A-D])(?:[\s.,):]|$)', text)
    if m:
        return f"({m.group(1)})"

    return None


def match_by_content(response: str, choices: list) -> str | None:
    """
    当提取不到字母时，用选项内容做精确匹配。
    只有 response 完全等于某个选项内容时才匹配（避免 substring 虚高）。
    """
    resp = response.strip().split("\n")[0].strip().lower()
    for i, choice in enumerate(choices):
        if resp == str(choice).strip().lower():
            return f"({chr(ord('A') + i)})"
    return None


def match_answer(response: str, choices: list, answer: str) -> tuple[bool, str | None]:
    """
    判断模型输出是否正确。返回 (correct, extracted_letter)。
    策略：字母提取 → 精确内容匹配 → 判错
    """
    extracted = extract_choice_letter(response)
    if extracted is not None:
        return extracted == answer, extracted

    extracted = match_by_content(response, choices)
    if extracted is not None:
        return extracted == answer, extracted

    return False, None


# ── 加载模型 ──────────────────────────────────────────────────────────────────

def load_base_model():
    """加载原始 Qwen2.5-VL，不附加任何自定义模块。"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print("Loading vanilla Qwen2.5-VL for baseline...")
    processor = AutoProcessor.from_pretrained(CFG.model_id)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id,
        torch_dtype=torch.float16,
        device_map=CFG.device,
    )
    base_model.eval()
    return base_model, processor


def load_enhanced_model(args):
    """加载带 ClusterPredictor + SAE 的增强模型。"""
    from src.Model import QwenWithClusterPredictorAndSAE

    print("Loading enhanced model (ClusterPredictor + SAE)...")
    cluster_path = os.path.join(CFG.label_dir, f"feature_clusters_layer{args.layer}.json")
    model = QwenWithClusterPredictorAndSAE.from_pretrained(
        model_id=CFG.model_id,
        sae_ckpt_dir=CFG.save_dir,
        cluster_path=cluster_path,
        inject_layer=args.layer,
        latent_mult=CFG.latent_mult,
        topk=CFG.topk,
        top_n_patches=CFG.top_n_patches,
        predictor_ckpt=args.predictor_ckpt,
        device=CFG.device,
    )
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
            state = torch.load(args.qwen_ckpt, map_location="cpu")
            # 1. 必须让最外层 model 加载，这样 semantic_cross_attn 和 lambda 才能被正确赋值
            model.load_state_dict(state, strict=False) 
            print("  Enhanced model weights (Stage 2) loaded.")
    
    # 2. 放到 GPU 上
    model.to(CFG.device)
    
    # 3. 极其重要：开启评估模式，关闭 Dropout！
    model.eval() 
    
    return model


# ── 评估循环 ──────────────────────────────────────────────────────────────────

def evaluate_base(base_model, processor, dataset, save_path=None):
    """用原始 Qwen2.5-VL 评估（纯 baseline）。"""
    from qwen_vl_utils import process_vision_info

    print(f"\n{'='*60}")
    print(f"Evaluating: base (vanilla Qwen2.5-VL)")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc="base eval"):
        image = item["image"]
        prompt = item["prompt"]
        answer = item["answer"]
        choices = item["choices"]

        # 构造带 instruction 的 prompt
        prompt_text = build_prompt(prompt, choices)

        tmp_path = "/tmp/cvbench_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        response = ""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{tmp_path}"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(base_model.device)

            with torch.no_grad():
                output_ids = base_model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({
            "idx": item["idx"],
            "task": item["task"],
            "source": item["source"],
            "type": item["type"],
            "answer": answer,
            "response": response,
            "extracted": extracted,
            "correct": correct,
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")

    return results


def evaluate_enhanced(model, dataset, save_path=None):
    """用增强模型（ClusterPredictor + SAE）评估。"""
    print(f"\n{'='*60}")
    print(f"Evaluating: enhanced (ClusterPredictor + SAE)")
    print(f"{'='*60}")

    results = []
    for item in tqdm(dataset, desc="enhanced eval"):
        image = item["image"]
        prompt = item["prompt"]
        answer = item["answer"]
        choices = item["choices"]

        # enhanced 也用同样的 prompt 构造，保证公平对比
        prompt_text = build_prompt(prompt, choices)

        tmp_path = "/tmp/cvbench_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        response = ""
        cluster_ids = []
        try:
            result = model.generate(
                image_path=tmp_path,
                question=prompt_text,
                verbose=False,
            )
            response = result["final_answer"]
            cluster_ids = result.get("cluster_ids", [])
        except Exception as e:
            print(f"  [error] idx={item['idx']}: {e}")

        correct, extracted = match_answer(response, choices, answer)
        results.append({
            "idx": item["idx"],
            "task": item["task"],
            "source": item["source"],
            "type": item["type"],
            "answer": answer,
            "response": response,
            "extracted": extracted,
            "correct": correct,
            "cluster_ids": cluster_ids,
        })

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {save_path}")

    return results


# ── 统计 ──────────────────────────────────────────────────────────────────────

def compute_metrics(results, label=""):
    by_source = defaultdict(list)
    by_task   = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r)
        by_task[r["task"]].append(r)

    def acc(items):
        if not items:
            return 0.0
        return sum(1 for i in items if i["correct"]) / len(items) * 100

    # 统计无法解析的 response
    n_unparsed = sum(1 for r in results if r.get("extracted") is None)
    n_total = len(results)
    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) responses could not be parsed")
        unparsed_by_task = defaultdict(int)
        for r in results:
            if r.get("extracted") is None:
                unparsed_by_task[r["task"]] += 1
        for t, c in sorted(unparsed_by_task.items()):
            total_t = len(by_task[t])
            print(f"    {t}: {c}/{total_t} unparsed")
        samples = [r for r in results if r.get("extracted") is None][:3]
        for s in samples:
            print(f"    example: task={s['task']} ans={s['answer']}")
            print(f"      resp=\"{s['response'][:100]}\"")

    acc_ade  = acc(by_source.get("ADE20K", []))
    acc_coco = acc(by_source.get("COCO", []))
    acc_omni = acc(by_source.get("Omni3D", []))
    acc_2d   = (acc_ade + acc_coco) / 2
    acc_3d   = acc_omni
    cv_bench = (acc_2d + acc_3d) / 2

    print(f"\n{'='*60}")
    print(f"  CV-Bench Results: {label}")
    print(f"{'='*60}")
    print(f"  CV-Bench Overall : {cv_bench:.2f}")
    print(f"  2D Accuracy      : {acc_2d:.2f}")
    print(f"    ADE20K         : {acc_ade:.2f}  (n={len(by_source.get('ADE20K', []))})")
    print(f"    COCO           : {acc_coco:.2f}  (n={len(by_source.get('COCO', []))})")
    print(f"  3D Accuracy      : {acc_3d:.2f}")
    print(f"    Omni3D         : {acc_omni:.2f}  (n={len(by_source.get('Omni3D', []))})")
    print(f"{'─'*60}")
    print(f"  Per-Task:")
    for task_name in sorted(by_task.keys()):
        items = by_task[task_name]
        print(f"    {task_name:20s}: {acc(items):6.2f}  (n={len(items)})")
    print(f"  Unparsed: {n_unparsed}/{n_total}")
    print(f"{'='*60}")

    return {
        "cv_bench": cv_bench, "acc_2d": acc_2d, "acc_3d": acc_3d,
        "acc_ade": acc_ade, "acc_coco": acc_coco, "acc_omni": acc_omni,
        "per_task": {t: acc(items) for t, items in by_task.items()},
        "n_unparsed": n_unparsed, "n_total": n_total,
    }


def print_comparison(base_m, enh_m):
    print(f"\n{'='*70}")
    print(f"  CV-Bench Comparison: Base vs Enhanced")
    print(f"{'='*70}")
    print(f"  {'Metric':<25s} {'Base':>8s} {'Enhanced':>10s} {'Delta':>8s}")
    print(f"  {'─'*55}")
    for name, key in [("CV-Bench Overall","cv_bench"), ("2D Accuracy","acc_2d"),
                       ("  ADE20K","acc_ade"), ("  COCO","acc_coco"),
                       ("3D Accuracy","acc_3d"), ("  Omni3D","acc_omni")]:
        b, e = base_m[key], enh_m[key]
        d = e - b
        print(f"  {name:<25s} {b:8.2f} {e:10.2f} {'+'if d>0 else ''}{d:8.2f}")
    all_tasks = sorted(set(list(base_m["per_task"]) + list(enh_m["per_task"])))
    print(f"  {'─'*55}")
    for t in all_tasks:
        b = base_m["per_task"].get(t, 0)
        e = enh_m["per_task"].get(t, 0)
        d = e - b
        print(f"    {t:<23s} {b:8.2f} {e:10.2f} {'+'if d>0 else ''}{d:8.2f}")
    print(f"  {'─'*55}")
    print(f"  {'Unparsed':<25s} {base_m['n_unparsed']:>8d} {enh_m['n_unparsed']:>10d}")
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["base","enhanced","both"])
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str, default="outputs/focus_ckpt/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None, choices=["2D","3D"])
    parser.add_argument("--save_dir", type=str, default="outputs/cvbench_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载数据集 ──
    print("Loading CV-Bench...")
    if args.subset:
        ds = load_dataset("nyu-visionx/CV-Bench", args.subset, split="test")
    else:
        ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    task_counts = defaultdict(int)
    for item in ds:
        task_counts[item["task"]] += 1
    for t, c in sorted(task_counts.items()):
        print(f"    {t}: {c}")

    # ── 评估 ──
    base_m = enh_m = None

    if args.mode in ["base", "both"]:
        base_model, processor = load_base_model()
        res = evaluate_base(
            base_model, processor, ds,
            save_path=os.path.join(args.save_dir, "base_results.json"),
        )
        base_m = compute_metrics(res, label="Base Qwen2.5-VL")
        # 释放显存，给 enhanced 模型腾空间
        if args.mode == "both":
            del base_model, processor
            torch.cuda.empty_cache()

    if args.mode in ["enhanced", "both"]:
        enhanced_model = load_enhanced_model(args)
        res = evaluate_enhanced(
            enhanced_model, ds,
            save_path=os.path.join(args.save_dir, "enhanced_results.json"),
        )
        enh_m = compute_metrics(res, label="Enhanced")

    if base_m and enh_m:
        print_comparison(base_m, enh_m)

    summary = {}
    if base_m: summary["base"] = base_m
    if enh_m:  summary["enhanced"] = enh_m
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone.")


if __name__ == "__main__":
    main()