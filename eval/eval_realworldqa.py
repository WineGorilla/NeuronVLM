"""
RealWorldQA 评估脚本。

RealWorldQA 由 xAI 发布，765 道真实场景理解题目。
数据集: https://huggingface.co/datasets/xai-org/RealworldQA
相关: Grok-1.5 Vision Preview (https://x.ai/blog/grok-1.5v)

特点:
  - 图片来自车载摄像头和其他真实场景
  - 混合格式: 多选题 (A/B/C/D) + Yes/No + 简答 (数字/单词)
  - question 字段已包含选项和作答指令
  - answer 字段为标准答案 (如 "A", "B", "Yes", "3" 等)
  - 很多题目涉及空间关系、距离、方向 → 适合展示你的 distance 提升

用法：
    CUDA_VISIBLE_DEVICES=1 python eval/eval_realworldqa.py --mode base
    CUDA_VISIBLE_DEVICES=1 python eval/eval_realworldqa.py --mode enhanced
    python eval/eval_realworldqa.py --mode both
    python eval/eval_realworldqa.py --mode spatial --spatial_qwen_ckpt outputs/focus_v2_ckpt/qwen_best.pt
    python eval/eval_realworldqa.py --mode all --qwen_ckpt outputs/ablation_baseline/qwen_best.pt
    python eval/eval_realworldqa.py --mode both --max_samples 100
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


# ── 题目类型自动检测 ─────────────────────────────────────────────────────────

def detect_question_type(question: str, answer: str) -> str:
    """
    检测题目类型:
      'mcq'       — 多选题 (answer 是 A/B/C/D)
      'yes_no'    — 是非题 (answer 是 Yes/No)
      'short'     — 简答题 (answer 是数字或单词)
    """
    ans = answer.strip()
    if ans in ("A", "B", "C", "D"):
        return "mcq"
    if ans.lower() in ("yes", "no"):
        return "yes_no"
    return "short"


# ── 答案匹配 ──────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """标准化文本用于比较。"""
    return text.strip().lower().rstrip(".")


def extract_mcq_letter(response: str) -> str | None:
    """从模型输出中提取选项字母。"""
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None
    m = re.search(r'\(([A-D])\)', text)
    if m:
        return m.group(1)
    m = re.search(r'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?([A-D])\)?', text)
    if m:
        return m.group(1)
    m = re.match(r'^([A-D])(?:[\s.,):]|$)', text)
    if m:
        return m.group(1)
    return None


def match_answer(response: str, answer: str, q_type: str) -> tuple[bool, str | None]:
    """
    匹配模型输出与标准答案。
    RealWorldQA 的 answer 可能是:
      - "A"/"B"/"C"/"D" (MCQ)
      - "Yes"/"No"
      - 数字 ("0", "3") 或单词 ("Green", "Bus")
    """
    resp = response.strip()
    ans = answer.strip()

    if q_type == "mcq":
        extracted = extract_mcq_letter(resp)
        if extracted is not None:
            return extracted == ans, extracted
        return False, None

    if q_type == "yes_no":
        resp_lower = normalize(resp)
        if resp_lower.startswith("yes"):
            extracted = "Yes"
        elif resp_lower.startswith("no"):
            extracted = "No"
        else:
            return False, None
        return extracted == ans, extracted

    # short answer: 直接比较 (宽松匹配)
    resp_norm = normalize(resp.split("\n")[0])
    ans_norm = normalize(ans)

    # 完全匹配
    if resp_norm == ans_norm:
        return True, resp

    # 答案出现在回复开头
    if resp_norm.startswith(ans_norm):
        return True, resp

    # 数字匹配
    try:
        if float(resp_norm) == float(ans_norm):
            return True, resp
    except ValueError:
        pass

    return False, resp


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
    print("Loading finetune baseline...")
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
        print("  WARNING: --qwen_ckpt not found, using vanilla model!")
    model.eval()
    return model, processor


def load_enhanced_model(args):
    from src.Model import QwenWithClusterPredictorAndSAE
    print("Loading enhanced model (ClusterPredictor + SAE)...")
    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
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
    print("Loading spatial model (Model_v2)...")
    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
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
        print("  Spatial model weights loaded.")
    model.to(CFG.device)
    model.eval()
    return model


def load_enhanced_model_no_pcs(args):
    from ablation.Model_no_pcs import QwenWithClusterPredictorAndSAE
    print("Loading enhanced model (NO PCS ablation)...")
    cluster_path = os.path.join(
        CFG.label_dir, f"feature_clusters_layer{args.layer}.json"
    )
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
    for i, item in enumerate(tqdm(dataset, desc=f"{label} eval")):
        image = item["image"]
        question = item["question"]
        answer = item["answer"]
        q_type = detect_question_type(question, answer)

        # RealWorldQA 的 question 已自带选项和指令，直接使用
        tmp_path = "/tmp/realworldqa_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        response = ""
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{tmp_path}"},
                    {"type": "text", "text": question},
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
            print(f"  [error] idx={i}: {e}")

        correct, extracted = match_answer(response, answer, q_type)
        results.append({
            "idx": i,
            "question": question[:120],
            "answer": answer,
            "q_type": q_type,
            "response": response,
            "extracted": extracted,
            "correct": correct,
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
    for i, item in enumerate(tqdm(dataset, desc=f"{label} eval")):
        image = item["image"]
        question = item["question"]
        answer = item["answer"]
        q_type = detect_question_type(question, answer)

        tmp_path = "/tmp/realworldqa_tmp.png"
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            tmp_path = image

        response = ""
        cluster_ids = []
        try:
            result = model.generate(
                image_path=tmp_path, question=question, verbose=False,
            )
            response = result["final_answer"]
            cluster_ids = result.get("cluster_ids", [])
        except Exception as e:
            print(f"  [error] idx={i}: {e}")

        correct, extracted = match_answer(response, answer, q_type)
        results.append({
            "idx": i,
            "question": question[:120],
            "answer": answer,
            "q_type": q_type,
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
    n_total = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    overall_acc = n_correct / n_total * 100 if n_total else 0.0

    # 按题目类型统计
    by_type = defaultdict(list)
    for r in results:
        by_type[r["q_type"]].append(r)

    def acc(items):
        if not items:
            return 0.0
        return sum(1 for i in items if i["correct"]) / len(items) * 100

    per_type = {t: acc(items) for t, items in sorted(by_type.items())}

    n_unparsed = sum(1 for r in results
                     if r.get("extracted") is None and r["q_type"] == "mcq")

    if n_unparsed > 0:
        pct = n_unparsed / n_total * 100
        print(f"\n  [warning] {n_unparsed}/{n_total} ({pct:.1f}%) "
              f"MCQ responses could not be parsed")
        for s in [r for r in results
                  if r.get("extracted") is None and r["q_type"] == "mcq"][:3]:
            print(f"    example: ans={s['answer']}")
            print(f"      resp=\"{s['response'][:100]}\"")

    print(f"\n{'='*60}")
    print(f"  RealWorldQA Results: {label}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy : {overall_acc:.2f}  ({n_correct}/{n_total})")
    print(f"{'─'*60}")
    print(f"  Per Question Type:")
    for t in ["mcq", "yes_no", "short"]:
        if t in per_type:
            n = len(by_type[t])
            print(f"    {t:10s}: {per_type[t]:6.2f}  (n={n})")
    print(f"  Unparsed MCQ: {n_unparsed}/{len(by_type.get('mcq', []))}")
    print(f"{'='*60}")

    return {
        "overall": overall_acc,
        "per_type": per_type,
        "n_correct": n_correct,
        "n_total": n_total,
        "n_unparsed": n_unparsed,
    }


def print_comparison(metrics_dict):
    labels = list(metrics_dict.keys())
    if len(labels) < 2:
        return

    print(f"\n{'='*70}")
    print(f"  RealWorldQA Comparison")
    print(f"{'='*70}")

    header = f"  {'Metric':<20s}"
    for label in labels:
        header += f" {label:>12s}"
    if len(labels) >= 2:
        header += f" {'Delta':>8s}"
    print(header)
    print(f"  {'─'*60}")

    # Overall
    line = f"  {'Overall':<20s}"
    vals = [metrics_dict[l]["overall"] for l in labels]
    for v in vals:
        line += f" {v:12.2f}"
    if len(vals) >= 2:
        d = vals[-1] - vals[0]
        line += f" {'+'if d>0 else ''}{d:7.2f}"
    print(line)

    # Per type
    print(f"  {'─'*60}")
    for t in ["mcq", "yes_no", "short"]:
        line = f"    {t:<18s}"
        vals = [metrics_dict[l]["per_type"].get(t, 0) for l in labels]
        for v in vals:
            line += f" {v:12.2f}"
        if len(vals) >= 2:
            d = vals[-1] - vals[0]
            line += f" {'+'if d>0 else ''}{d:7.2f}"
        print(line)

    print(f"  {'─'*60}")
    line = f"  {'Unparsed MCQ':<20s}"
    for l in labels:
        line += f" {metrics_dict[l]['n_unparsed']:>12d}"
    print(line)
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RealWorldQA Evaluation")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["base", "enhanced", "spatial", "no_pcs",
                                 "finetune_baseline", "both", "all"])
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_0.75_64_5000/predictor_best.pt")
    parser.add_argument("--qwen_ckpt", type=str, default=None)
    parser.add_argument("--baseline_ckpt", type=str, default=None)
    parser.add_argument("--spatial_predictor_ckpt", type=str,
                        default="outputs/focus_ckpt_spatial/predictor_best.pt")
    parser.add_argument("--spatial_qwen_ckpt", type=str, default=None)
    parser.add_argument("--no_pcs_predictor_ckpt", type=str,
                        default="outputs/ablation_no_pcs/predictor_best.pt")
    parser.add_argument("--no_pcs_qwen_ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str,
                        default="outputs/realworldqa_results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 加载数据集 ──
    print("Loading RealWorldQA from HuggingFace...")
    ds = load_dataset("xai-org/RealworldQA", split="test")
    print(f"  columns: {ds.column_names}")
    print(f"  Samples: {len(ds)}")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"  Using first {len(ds)} samples")

    # 统计题目类型
    type_counts = defaultdict(int)
    for item in ds:
        q_type = detect_question_type(item["question"], item["answer"])
        type_counts[q_type] += 1
    for t, c in sorted(type_counts.items()):
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
            base_model, processor, ds,
            save_path=os.path.join(args.save_dir, "base_results.json"),
            label="base (vanilla)",
        )
        all_metrics["base"] = compute_metrics(res, label="Base Qwen2.5-VL")
        del base_model, processor; torch.cuda.empty_cache()

    if run_ft_base:
        ft_ckpt = args.baseline_ckpt or args.qwen_ckpt
        ft_model, ft_proc = load_finetune_baseline(ft_ckpt)
        res = evaluate_base(
            ft_model, ft_proc, ds,
            save_path=os.path.join(args.save_dir, "ft_baseline_results.json"),
            label="finetune_baseline",
        )
        all_metrics["ft_baseline"] = compute_metrics(
            res, label="Finetune Baseline")
        del ft_model, ft_proc; torch.cuda.empty_cache()

    if run_no_pcs:
        no_pcs_model = load_enhanced_model_no_pcs(args)
        res = evaluate_enhanced(
            no_pcs_model, ds,
            save_path=os.path.join(args.save_dir, "no_pcs_results.json"),
            label="enhanced (no PCS)",
        )
        all_metrics["no_pcs"] = compute_metrics(res, label="Enhanced (no PCS)")
        del no_pcs_model; torch.cuda.empty_cache()

    if run_enhanced:
        enhanced_model = load_enhanced_model(args)
        res = evaluate_enhanced(
            enhanced_model, ds,
            save_path=os.path.join(args.save_dir, "enhanced_results.json"),
            label="enhanced (v1)",
        )
        all_metrics["enhanced"] = compute_metrics(res, label="Enhanced (v1)")
        del enhanced_model; torch.cuda.empty_cache()

    if run_spatial:
        spatial_model = load_spatial_model(args)
        res = evaluate_enhanced(
            spatial_model, ds,
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