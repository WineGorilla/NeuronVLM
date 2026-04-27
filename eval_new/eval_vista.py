"""
VISTA Baseline 评估脚本 — 在 Qwen2.5-VL 上复现 VISTA (ICML 2025)。

VISTA 核心思想 (Li et al., ICML 2025):
  两个互补模块：
  1. VSV (Visual Steering Vector): 提取"有图 vs 无图"的 hidden states 差异方向，
     推理时沿该方向 steering 以保持 visual grounding
  2. SLA (Self-Logits Augmentation): 用早期层的 logits 辅助最终层 decode

本脚本实现 VSV 部分（VISTA 的主要贡献），适配到 Qwen2.5-VL。

与 NeuronEye 的对比：
  - VISTA VSV: 固定的 steering 方向，不看 query，全局作用于所有层
  - NeuronEye: query-conditioned routing，局部作用于特定 patches

用法：
    # Step 1: 提取 VSV（用少量校准样本）
    python eval/eval_vista.py --mode extract --extract_samples 50

    # Step 2: 用 VSV 做 steering 评估
    python eval/eval_vista.py --mode eval --vsv_lambda 0.17 --benchmarks cvbench mmstar

    # 一步到位
    python eval_new/eval_vista.py --mode both --benchmarks cvbench mmstar

依赖：
    pip install transformers torch accelerate qwen-vl-utils datasets
"""
import os
import sys
import re
import json
import argparse
from collections import defaultdict
from functools import wraps
from copy import deepcopy

import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from config import CFG
except ImportError:
    class CFG:
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        device = "cuda"


# ══════════════════════════════════════════════════════════════════════════════
# VISTA: Visual Steering Vector
# ══════════════════════════════════════════════════════════════════════════════

class VISTASteering:
    """
    VISTA VSV (Visual Steering Vector) 实现。

    核心算法：
      1. Extract Phase: 对同一张图+同一个 prompt，分别跑两次 forward：
         - Positive: 正常输入（有 image tokens）
         - Negative: 把 image tokens 的 hidden states 置零（模拟无图）
         对每一层，计算 V_s = mean(H_pos - H_neg)，得到每层的 steering vector

      2. Steering Phase: 推理时，在每层的 hook 中：
         H' = H + lambda * V_s[layer]
         对所有 token 都加（不区分 vision/text）

    与 SAVE 的区别：
      - SAVE: 在 SAE feature space 做 steering，只作用于 vision tokens
      - VISTA: 在 dense hidden space 做 steering，作用于所有 tokens，每层不同方向
    """

    def __init__(self, model, processor, vsv_lambda=0.17):
        self.model = model
        self.processor = processor
        self.vsv_lambda = vsv_lambda

        self.n_layers = len(model.model.language_model.layers)
        # 每层一个 steering vector: (n_layers, hidden_dim)
        self.steering_vectors = None

        self._hooks = []
        self._steered_layers = set()

    def extract_vsv(self, max_samples=50):
        """
        提取 Visual Steering Vectors。

        对每个样本：
          1. 正常 forward → 收集每层 hidden states (positive)
          2. 把 vision tokens 置零后 forward → 收集每层 hidden states (negative)
          3. 差异 = positive - negative
        对所有样本求平均，得到每层的 VSV。
        """
        from qwen_vl_utils import process_vision_info

        print("\n" + "=" * 60)
        print("  VISTA: Extracting Visual Steering Vectors")
        print("=" * 60)

        # 用 CV-Bench 的数据做校准
        ds = load_dataset("nyu-visionx/CV-Bench", split="test")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        print(f"  Calibration samples: {len(ds)}")

        layers = self.model.model.language_model.layers
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        spatial_merge = self.model.config.vision_config.spatial_merge_size
        hidden_dim = self.model.config.text_config.hidden_size

        # 累积差异
        vsv_sum = torch.zeros(self.n_layers, hidden_dim, device="cpu")
        n_valid = 0

        for item in tqdm(ds, desc="Extracting VSV"):
            image = item["image"]
            choices = item["choices"]
            choice_text = "\n".join(
                [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
            )
            prompt = (
                "Answer the following question.\n"
                "Select the correct option and output ONLY the letter.\n"
                "Do NOT output explanation.\n\n"
                f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
            )

            tmp_path = "/tmp/vista_cal_tmp.png"
            if isinstance(image, Image.Image):
                image.save(tmp_path)
            else:
                tmp_path = image

            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{tmp_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(self.model.device)

                # 定位 vision tokens
                input_ids = inputs["input_ids"][0]
                image_grid = inputs["image_grid_thw"]
                num_img_tokens = int(
                    image_grid[0, 1] * image_grid[0, 2] / (spatial_merge ** 2)
                )
                vs_positions = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
                if len(vs_positions) == 0:
                    continue
                vision_pos = vs_positions[0].item() + 1

                # === Positive forward: 正常输入 ===
                with torch.no_grad():
                    pos_outputs = self.model(
                        **{k: v for k, v in inputs.items() if torch.is_tensor(v)},
                        output_hidden_states=True,
                        return_dict=True,
                    )

                # === Negative forward: vision tokens 置零 ===
                # 克隆 inputs，把 pixel_values 置零
                neg_inputs = {k: v.clone() if torch.is_tensor(v) else v
                              for k, v in inputs.items()}
                if "pixel_values" in neg_inputs:
                    neg_inputs["pixel_values"] = torch.zeros_like(neg_inputs["pixel_values"])

                with torch.no_grad():
                    neg_outputs = self.model(
                        **{k: v for k, v in neg_inputs.items() if torch.is_tensor(v)},
                        output_hidden_states=True,
                        return_dict=True,
                    )

                # === 计算每层差异 ===
                for l in range(self.n_layers):
                    # hidden_states[0] 是 embedding，[l+1] 是 layer l 的输出
                    pos_h = pos_outputs.hidden_states[l + 1]  # (1, seq, dim)
                    neg_h = neg_outputs.hidden_states[l + 1]

                    # 只看 vision token 区域的差异
                    pos_vision = pos_h[0, vision_pos:vision_pos + num_img_tokens, :].float()
                    neg_vision = neg_h[0, vision_pos:vision_pos + num_img_tokens, :].float()

                    diff = (pos_vision - neg_vision).mean(dim=0)  # (dim,)
                    vsv_sum[l] += diff.cpu()

                n_valid += 1

            except Exception as e:
                print(f"  [error]: {e}")
                continue

        if n_valid == 0:
            print("  ERROR: No valid samples for VSV extraction!")
            return False

        # 平均 + 归一化
        self.steering_vectors = vsv_sum / n_valid
        # 逐层归一化
        for l in range(self.n_layers):
            norm = self.steering_vectors[l].norm()
            if norm > 0:
                self.steering_vectors[l] = self.steering_vectors[l] / norm

        print(f"  VSV extracted from {n_valid} samples")
        print(f"  VSV shape: {self.steering_vectors.shape}")
        norms = [self.steering_vectors[l].norm().item() for l in range(self.n_layers)]
        print(f"  VSV norms (first 5 layers): {norms[:5]}")

        return True

    def save_vsv(self, path):
        torch.save(self.steering_vectors, path)
        print(f"  VSV saved: {path}")

    def load_vsv(self, path):
        self.steering_vectors = torch.load(path, map_location="cpu")
        print(f"  VSV loaded: {path}, shape={self.steering_vectors.shape}")

    def install_hooks(self):
        """在每一层安装 steering hook。"""
        self._hooks = []
        self._steered_layers = set()
        layers = self.model.model.language_model.layers

        for l in range(self.n_layers):
            hook = layers[l].register_forward_hook(self._make_hook(l))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, args, output):
            # 只在 prefill 阶段 steering（seq_len > 1）
            # 并且每层只 steer 一次
            if layer_idx in self._steered_layers:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if hidden_states.shape[1] <= 1:
                return output

            self._steered_layers.add(layer_idx)

            if self.steering_vectors is None:
                return output

            device = hidden_states.device
            vsv = self.steering_vectors[layer_idx].to(device).to(hidden_states.dtype)

            # Global steering: 所有 token += lambda * VSV
            hidden_states[:, :, :] += self.vsv_lambda * vsv

            return output
        return hook_fn

    def patch_generate(self):
        original_generate = self.model.generate

        @wraps(original_generate)
        def patched_generate(*args, **kwargs):
            self._steered_layers = set()
            self.install_hooks()
            try:
                result = original_generate(*args, **kwargs)
            finally:
                self.remove_hooks()
            return result

        self.model.generate = patched_generate
        self._original_generate = original_generate

    def unpatch_generate(self):
        if hasattr(self, "_original_generate"):
            self.model.generate = self._original_generate


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_id, device="cuda"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading Qwen2.5-VL: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id,
        max_pixels=1280 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    n_layers = len(model.model.language_model.layers)
    print(f"  Model loaded. Layers: {n_layers}")
    return model, processor


def qwen25vl_generate(model, processor, image, question, max_new_tokens=32):
    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            return ""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(
            output_ids[0, input_len:],
            skip_special_tokens=True,
        ).strip()
        return response

    except Exception as e:
        print(f"  [error] generate failed: {e}")
        torch.cuda.empty_cache()
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 答案匹配工具
# ══════════════════════════════════════════════════════════════════════════════

def extract_choice_letter(response, max_letter="Z"):
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
# Benchmark 评估函数
# ══════════════════════════════════════════════════════════════════════════════

def eval_cvbench(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  CV-Bench Evaluation (VISTA)\n{'='*60}")
    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for item in tqdm(ds, desc="CV-Bench"):
        choices = item["choices"]
        choice_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
        prompt = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter.\n"
            "Do NOT output explanation.\n\n"
            f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
        )
        response = qwen25vl_generate(model, processor, item["image"], prompt)
        extracted = extract_choice_letter(response)
        answer = item["answer"]
        extracted_fmt = f"({extracted})" if extracted else None
        correct = extracted_fmt == answer
        results.append({
            "idx": item["idx"], "task": item["task"],
            "source": item["source"], "type": item["type"],
            "answer": answer, "response": response,
            "extracted": extracted_fmt, "correct": correct,
        })

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

    return {"benchmark": "CV-Bench", "cv_bench": cv_bench, "acc_2d": acc_2d,
            "acc_3d": acc_3d, "per_task": {t: acc(items) for t, items in by_task.items()},
            "n_unparsed": n_unparsed, "n_total": len(results)}, results


def eval_mmstar(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  MMStar Evaluation (VISTA)\n{'='*60}")
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for item in tqdm(ds, desc="MMStar"):
        prompt = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter (A, B, C, or D).\n"
            "Do NOT output explanation.\n\n"
            f"{item['question']}\n\nAnswer:"
        )
        response = qwen25vl_generate(model, processor, item["image"], prompt)
        extracted = extract_choice_letter(response, max_letter="D")
        answer = item["answer"].strip().upper()
        correct = extracted == answer
        results.append({
            "index": item["index"], "category": item["category"],
            "l2_category": item["l2_category"],
            "answer": answer, "response": response,
            "extracted": extracted, "correct": correct,
        })

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0
    overall = acc(results)
    n_unparsed = sum(1 for r in results if r["extracted"] is None)
    print(f"\n  Overall Accuracy : {overall:.2f}  (n={len(results)})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {"benchmark": "MMStar", "overall": overall,
            "n_unparsed": n_unparsed, "n_total": len(results)}, results


def eval_realworldqa(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  RealWorldQA Evaluation (VISTA)\n{'='*60}")
    ds = load_dataset("xai-org/RealworldQA", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for i, item in enumerate(tqdm(ds, desc="RealWorldQA")):
        question = item["question"]
        answer = item["answer"]
        response = qwen25vl_generate(model, processor, item["image"], question)
        ans = answer.strip()
        if ans in ("A", "B", "C", "D"):
            extracted = extract_choice_letter(response, max_letter="D")
            correct = extracted == ans
        else:
            resp_norm = response.strip().split("\n")[0].strip().lower().rstrip(".")
            ans_norm = ans.lower().rstrip(".")
            correct = resp_norm == ans_norm or resp_norm.startswith(ans_norm)
            extracted = response.strip()[:50]
        results.append({"idx": i, "answer": answer, "response": response,
                        "extracted": extracted, "correct": correct})

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0
    overall = acc(results)
    print(f"\n  Overall Accuracy : {overall:.2f}  (n={len(results)})")

    return {"benchmark": "RealWorldQA", "overall": overall,
            "n_total": len(results)}, results


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
        concat.paste(img, (x, 0)); x += img.width
    out = "/tmp/blink_eval_concat.png"
    concat.save(out)
    return out

def eval_blink(model, processor, subtasks=None, max_samples=None):
    print(f"\n{'='*60}\n  BLINK Evaluation (VISTA)\n{'='*60}")
    if subtasks is None:
        subtasks = ALL_BLINK_SUBTASKS
    all_items = []
    for subtask in subtasks:
        try:
            ds = load_dataset("BLINK-Benchmark/BLINK", subtask, split="val")
            for i, item in enumerate(ds):
                choices = item.get("choices", item.get("options", []))
                if isinstance(choices, str):
                    try: choices = json.loads(choices)
                    except: choices = [c.strip() for c in choices.split(",")]
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
        for item in all_items: by_task[item["subtask"]].append(item)
        sampled = []
        for items in by_task.values(): sampled.extend(items[:per_task])
        all_items = sampled[:max_samples]
    print(f"  Total: {len(all_items)} samples")

    results = []
    for item in tqdm(all_items, desc="BLINK"):
        choices = item["choices"]
        has_image_choices = (isinstance(choices, list) and len(choices) > 0
                            and isinstance(choices[0], Image.Image))
        if has_image_choices:
            choice_text = "\n".join([f"{chr(ord('A')+i)}. (Image {i+1})" for i in range(len(choices))])
        else:
            choice_text = "\n".join([f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
        prompt_text = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter.\n"
            "Do NOT output explanation.\n\n"
            f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
        )
        tmp_dir = "/tmp/blink_eval"
        os.makedirs(tmp_dir, exist_ok=True)
        image_paths = []
        for key in ["image_1", "image_2", "image_3", "image_4"]:
            img = item.get(key)
            if img is not None and isinstance(img, Image.Image):
                p = os.path.join(tmp_dir, f"{key}.png")
                img.save(p); image_paths.append(p)
        if not image_paths:
            results.append({"idx": item["idx"], "subtask": item["subtask"],
                            "answer": item["answer"], "response": "",
                            "extracted": None, "correct": False})
            continue
        concat_path = concat_images_horizontal(image_paths)
        response = qwen25vl_generate(model, processor, concat_path, prompt_text)
        extracted = extract_choice_letter(response)
        answer = item["answer"]
        extracted_fmt = f"({extracted})" if extracted else None
        correct = extracted_fmt == answer
        results.append({"idx": item["idx"], "subtask": item["subtask"],
                        "answer": answer, "response": response,
                        "extracted": extracted_fmt, "correct": correct})

    by_subtask = defaultdict(list)
    for r in results: by_subtask[r["subtask"]].append(r)
    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0
    per_subtask = {t: acc(by_subtask[t]) for t in sorted(by_subtask.keys())}
    overall = sum(per_subtask.values()) / len(per_subtask) if per_subtask else 0
    ds_items = [r for t in DEPTH_SPATIAL_SUBTASKS for r in by_subtask.get(t, [])]
    acc_ds = acc(ds_items)
    n_unparsed = sum(1 for r in results if r["extracted"] is None)

    print(f"\n  BLINK Overall (macro) : {overall:.2f}")
    print(f"  Depth/Spatial         : {acc_ds:.2f}  (n={len(ds_items)})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {"benchmark": "BLINK", "overall": overall,
            "acc_depth_spatial": acc_ds, "per_subtask": per_subtask,
            "n_unparsed": n_unparsed, "n_total": len(results)}, results


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VISTA (ICML 2025) on Qwen2.5-VL")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["extract", "eval", "both"])
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["cvbench", "mmstar", "realworldqa", "blink"],
                        choices=["cvbench", "mmstar", "realworldqa", "blink"])
    parser.add_argument("--model_id", type=str, default=CFG.model_id)
    parser.add_argument("--vsv_lambda", type=float, default=0.17,
                        help="VSV steering strength (VISTA default for LLaVA: 0.17)")
    parser.add_argument("--extract_samples", type=int, default=50,
                        help="Number of samples for VSV extraction")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/vista_qwen25vl_results")
    parser.add_argument("--vsv_path", type=str, default=None,
                        help="Path to pre-extracted VSV (skip extract step)")
    parser.add_argument("--blink_subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    model, processor = load_model(args.model_id)

    vista = VISTASteering(model, processor, vsv_lambda=args.vsv_lambda)

    vsv_path = args.vsv_path or os.path.join(args.save_dir, "vsv_vectors.pt")

    # Step 1: Extract VSV
    if args.mode in ["extract", "both"]:
        success = vista.extract_vsv(max_samples=args.extract_samples)
        if not success:
            print("ERROR: VSV extraction failed!")
            return
        vista.save_vsv(vsv_path)

    # Step 2: Evaluate with VSV steering
    if args.mode in ["eval", "both"]:
        if vista.steering_vectors is None:
            vista.load_vsv(vsv_path)

        vista.vsv_lambda = args.vsv_lambda
        vista.patch_generate()
        print(f"\n  VISTA VSV active: lambda={args.vsv_lambda}")

        config = {
            "method": "VISTA (Visual Steering Vector)",
            "model_id": args.model_id,
            "vsv_lambda": args.vsv_lambda,
            "extract_samples": args.extract_samples,
            "n_layers": vista.n_layers,
        }
        with open(os.path.join(args.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        all_summary = {}

        if "cvbench" in args.benchmarks:
            metrics, results = eval_cvbench(model, processor, args.max_samples)
            all_summary["cvbench"] = metrics
            with open(os.path.join(args.save_dir, "cvbench_results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if "mmstar" in args.benchmarks:
            metrics, results = eval_mmstar(model, processor, args.max_samples)
            all_summary["mmstar"] = metrics
            with open(os.path.join(args.save_dir, "mmstar_results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if "realworldqa" in args.benchmarks:
            metrics, results = eval_realworldqa(model, processor, args.max_samples)
            all_summary["realworldqa"] = metrics
            with open(os.path.join(args.save_dir, "realworldqa_results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        if "blink" in args.benchmarks:
            subtasks = args.blink_subtasks
            if args.depth_spatial_only:
                subtasks = DEPTH_SPATIAL_SUBTASKS
            metrics, results = eval_blink(model, processor, subtasks, args.max_samples)
            all_summary["blink"] = metrics
            with open(os.path.join(args.save_dir, "blink_results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"  VISTA (lambda={args.vsv_lambda}) on Qwen2.5-VL — Summary")
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