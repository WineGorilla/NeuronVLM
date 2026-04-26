"""
SAVE Baseline 评估脚本 — 在 Qwen2.5-VL 上复现 SAVE 的 Global SAE Steering。

SAVE 核心思想 (Park et al., WACV 2026):
  1. 用 SAE 分解 vision token 的 hidden states
  2. 通过 separation score 找到"visual understanding features"（与正确回答关联的 SAE 特征）
  3. 在推理时，沿这些特征方向做 global steering（对所有 vision token 加偏移）

与 NeuronEye 的关键区别：
  - SAVE: 固定的 steering 方向，不看 query 内容，全局作用于所有 vision token
  - NeuronEye: query-conditioned routing 到 neuron clusters，局部作用于相关 patches

本脚本直接复用 NeuronEye 的 SAE，实现 SAVE 的 global steering 逻辑，
确保对比公平（同一个 SAE，同一个基座模型）。

用法：
    # Step 1: 先收集 SAE activations 并识别 visual understanding features
    python eval/eval_save.py --mode identify --max_samples 500

    # Step 2: 用识别出的 features 做 steering 评估
    python eval/eval_save.py --mode eval --benchmarks cvbench mmstar
    python eval/eval_save.py --mode eval --steer_alpha 5.0 --benchmarks cvbench

    # 一步到位（先 identify 再 eval）
    python eval_new/eval_save.py --mode both --benchmarks cvbench mmstar
"""
import os
import sys
import re
import json
import argparse
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.SAE import SAE

try:
    from config import CFG
except ImportError:
    # Fallback if config not available
    class CFG:
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        save_dir = "output/qwen_layer8_old/sae_ckpt"
        vis_layer = 8
        latent_mult = 32
        topk = 32
        device = "cuda"


# ══════════════════════════════════════════════════════════════════════════════
# SAVE: Global SAE Steering
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656


class SAVESteering:
    """
    SAVE-style global steering using SAE features.

    算法：
      1. Identify Phase: 在校准集上收集 SAE activations，
         比较正确/错误回答时的 activation 差异，找到 visual understanding features
      2. Steering Phase: 推理时，在指定层的 hook 中，
         对所有 vision token 沿 visual understanding features 方向做加法偏移

    与 NeuronEye 的对比点：
      - 不做 query-conditioned routing（SAVE 不看问题内容）
      - 不做 localized injection（SAVE 对所有 vision token 统一操作）
      - 不做 semantic completion（SAVE 不补全缺失信息）
      - 不做 PCS（SAVE 无 anisotropy 抑制）
    """

    def __init__(self, model, sae, processor=None, steer_layer=8, steer_alpha=5.0,
                 n_visual_features=50, n_hallucinated_features=50):
        """
        Args:
            model: Qwen2_5_VLForConditionalGeneration
            sae: 训练好的 SAE
            processor: AutoProcessor (用于获取 vision_start_id)
            steer_layer: 在哪一层做 steering
            steer_alpha: steering 强度
            n_visual_features: 保留的 visual understanding features 数量
            n_hallucinated_features: 抑制的 hallucinated features 数量
        """
        self.model = model
        self.sae = sae
        self.processor = processor
        self.steer_layer = steer_layer
        self.steer_alpha = steer_alpha
        self.n_visual_features = n_visual_features
        self.n_hallucinated_features = n_hallucinated_features

        # steering 方向（identify 后设置）
        self.visual_feature_ids = None
        self.hallucinated_feature_ids = None
        self.steering_vector = None

        self._hooks = []
        self._vision_pos = None
        self._num_img_tokens = None
        self._steered = False

    def identify_features(self, correct_activations, incorrect_activations):
        """
        SAVE 的 feature identification：
        计算每个 SAE feature 的 separation score，
        找到 visual understanding features 和 hallucinated features。

        Args:
            correct_activations: list of (n_patches, latent_dim) tensors
                正确回答时的 SAE activations
            incorrect_activations: list of (n_patches, latent_dim) tensors
                错误回答时的 SAE activations

        SAVE 的 separation score:
            对每个 feature j:
            P(active | correct) - P(active | incorrect)
            正值大 = visual understanding feature
            负值大 = hallucinated feature
        """
        latent_dim = correct_activations[0].shape[-1] if correct_activations else \
                     incorrect_activations[0].shape[-1]

        # 计算每个 feature 在 correct/incorrect 中的平均激活概率
        correct_probs = torch.zeros(latent_dim)
        incorrect_probs = torch.zeros(latent_dim)

        for act in correct_activations:
            # act: (n_patches, latent_dim)
            active = (act > 0).float().mean(dim=0)  # (latent_dim,)
            correct_probs += active.cpu()
        if correct_activations:
            correct_probs /= len(correct_activations)

        for act in incorrect_activations:
            active = (act > 0).float().mean(dim=0)
            incorrect_probs += active.cpu()
        if incorrect_activations:
            incorrect_probs /= len(incorrect_activations)

        # Separation score
        separation = correct_probs - incorrect_probs

        # Visual understanding features: 正确时更活跃的 top-k
        _, visual_ids = separation.topk(self.n_visual_features)
        self.visual_feature_ids = visual_ids.tolist()

        # Hallucinated features: 错误时更活跃的 top-k
        _, hal_ids = (-separation).topk(self.n_hallucinated_features)
        self.hallucinated_feature_ids = hal_ids.tolist()

        print(f"  Identified {len(self.visual_feature_ids)} visual understanding features")
        print(f"  Identified {len(self.hallucinated_feature_ids)} hallucinated features")
        print(f"  Top visual features separation: "
              f"{separation[visual_ids[:5]].tolist()}")
        print(f"  Top hallucinated features separation: "
              f"{separation[hal_ids[:5]].tolist()}")

        # 预计算 steering vector:
        # 沿 visual understanding features 的 SAE decoder 列方向加法
        # 同时沿 hallucinated features 方向减法
        device = next(self.sae.parameters()).device
        steering = torch.zeros(self.sae.decoder.weight.shape[0], device=device)

        # decoder.weight: (dim, latent_dim) — 每列是一个 feature 的方向
        for fid in self.visual_feature_ids:
            steering += self.sae.decoder.weight[:, fid]
        for fid in self.hallucinated_feature_ids:
            steering -= self.sae.decoder.weight[:, fid]

        # 归一化
        steering = steering / (steering.norm() + 1e-8)
        self.steering_vector = steering

        return {
            "visual_feature_ids": self.visual_feature_ids,
            "hallucinated_feature_ids": self.hallucinated_feature_ids,
            "separation_scores": separation.tolist(),
        }

    def set_vision_positions(self, input_ids, image_grid_thw):
        """根据 vision_start_id + grid 定位 vision tokens。"""
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        spatial_merge = self.model.config.vision_config.spatial_merge_size

        ids = input_ids[0]
        vs_positions = (ids == vision_start_id).nonzero(as_tuple=True)[0]
        if len(vs_positions) > 0:
            self._vision_pos = vs_positions[0].item() + 1
            self._num_img_tokens = int(
                image_grid_thw[0, 1] * image_grid_thw[0, 2] / (spatial_merge ** 2)
            )
        else:
            self._vision_pos = None
            self._num_img_tokens = None
        self._steered = False

    def install_hooks(self):
        self._hooks = []
        layers = self.model.model.language_model.layers
        hook = layers[self.steer_layer].register_forward_hook(self._steer_hook)
        self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _steer_hook(self, module, args, output):
        """
        SAVE 的 global steering hook：
        对所有 vision token 沿 steering_vector 方向加偏移。
        """
        if self._steered or self.steering_vector is None:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        if hidden_states.shape[1] <= 1:
            return output

        self._steered = True

        if self._vision_pos is None or self._num_img_tokens is None:
            return output

        v_pos = self._vision_pos
        n_img = self._num_img_tokens

        if v_pos + n_img > hidden_states.shape[1]:
            return output

        device = hidden_states.device
        steering = self.steering_vector.to(device).to(hidden_states.dtype)

        # Global steering: 所有 vision token += alpha * steering_vector
        hidden_states[0, v_pos:v_pos + n_img, :] += self.steer_alpha * steering

        return output

    def patch_generate(self):
        original_generate = self.model.generate

        @wraps(original_generate)
        def patched_generate(*args, **kwargs):
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            image_grid_thw = kwargs.get("image_grid_thw", None)
            if input_ids is not None and image_grid_thw is not None:
                self.set_vision_positions(input_ids, image_grid_thw)

            self.install_hooks()
            try:
                result = original_generate(*args, **kwargs)
            finally:
                self.remove_hooks()
                self._steered = False
            return result

        self.model.generate = patched_generate
        self._original_generate = original_generate

    def save_features(self, path, info):
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Features saved: {path}")

    def load_features(self, path):
        with open(path) as f:
            info = json.load(f)
        self.visual_feature_ids = info["visual_feature_ids"]
        self.hallucinated_feature_ids = info["hallucinated_feature_ids"]

        device = next(self.sae.parameters()).device
        steering = torch.zeros(self.sae.decoder.weight.shape[0], device=device)
        for fid in self.visual_feature_ids:
            steering += self.sae.decoder.weight[:, fid]
        for fid in self.hallucinated_feature_ids:
            steering -= self.sae.decoder.weight[:, fid]
        steering = steering / (steering.norm() + 1e-8)
        self.steering_vector = steering

        print(f"  Loaded {len(self.visual_feature_ids)} visual features, "
              f"{len(self.hallucinated_feature_ids)} hallucinated features")


# ══════════════════════════════════════════════════════════════════════════════
# Feature Identification (Step 1)
# ══════════════════════════════════════════════════════════════════════════════

def identify_save_features(model, processor, sae, steer_layer,
                           max_samples=500, save_path=None):
    """
    在 POPE-style 的 yes/no 数据上收集 SAE activations，
    根据模型回答正确/错误来标注，计算 separation score。

    简化版：直接用 CV-Bench 的数据，模型答对的 = correct，答错的 = incorrect。
    """
    from qwen_vl_utils import process_vision_info

    print("\n" + "=" * 60)
    print("  SAVE: Identifying Visual Understanding Features")
    print("=" * 60)

    ds = load_dataset("nyu-visionx/CV-Bench", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Calibration samples: {len(ds)}")

    layers = model.model.language_model.layers
    correct_acts = []
    incorrect_acts = []

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    spatial_merge = model.config.vision_config.spatial_merge_size

    for item in tqdm(ds, desc="Collecting activations"):
        image = item["image"]
        choices = item["choices"]
        answer = item["answer"]

        choice_text = "\n".join(
            [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
        )
        prompt = (
            "Answer the following question.\n"
            "Select the correct option and output ONLY the letter.\n"
            "Do NOT output explanation.\n\n"
            f"{item['prompt']}\n\nChoices:\n{choice_text}\n\nAnswer:"
        )

        tmp_path = "/tmp/save_cal_tmp.png"
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
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            # 用 vision_start_id + grid 定位 vision tokens（和 SAE 训练一致）
            input_ids = inputs["input_ids"][0]
            image_grid = inputs["image_grid_thw"]
            num_img_tokens = int(
                image_grid[0, 1] * image_grid[0, 2] / (spatial_merge ** 2)
            )
            vs_positions = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
            if len(vs_positions) == 0:
                continue
            vision_pos = vs_positions[0].item() + 1

            # 收集中间层 hidden states — 只在 prefill
            collected = {}
            collected_done = [False]

            def make_hook(layer_idx):
                def hook_fn(module, args, output):
                    if collected_done[0]:
                        return
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    if hs.shape[1] > 1:
                        collected[layer_idx] = hs.detach()
                        collected_done[0] = True
                return hook_fn

            handle = layers[steer_layer].register_forward_hook(make_hook(steer_layer))

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=32, do_sample=False,
                )

            handle.remove()

            # 解码回答
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                out_ids[0, input_len:], skip_special_tokens=True
            ).strip()

            # 提取选项字母
            text_upper = response.strip().split("\n")[0].strip().upper()
            extracted = None
            m = re.search(r'\(([A-Z])\)', text_upper)
            if m:
                extracted = f"({m.group(1)})"
            else:
                m = re.match(r'^([A-Z])(?:[\s.,):]|$)', text_upper)
                if m:
                    extracted = f"({m.group(1)})"

            is_correct = (extracted == answer)

            # 提取 vision token 的 hidden states 并 SAE encode
            if steer_layer in collected:
                hs = collected[steer_layer]
                if vision_pos + num_img_tokens <= hs.shape[1]:
                    vision_hs = hs[0, vision_pos:vision_pos + num_img_tokens, :].float()
                    sae_device = next(sae.parameters()).device
                    with torch.no_grad():
                        z = sae.encode(vision_hs.to(sae_device))
                    if is_correct:
                        correct_acts.append(z.cpu())
                    else:
                        incorrect_acts.append(z.cpu())

        except Exception as e:
            print(f"  [error] idx={item.get('idx', '?')}: {e}")
            continue

    print(f"  Correct samples: {len(correct_acts)}")
    print(f"  Incorrect samples: {len(incorrect_acts)}")

    if not correct_acts or not incorrect_acts:
        print("  WARNING: Not enough correct/incorrect samples for identification!")
        return None

    # 计算 separation scores
    save_steering = SAVESteering(model, sae, processor=processor, steer_layer=steer_layer)
    info = save_steering.identify_features(correct_acts, incorrect_acts)

    if save_path:
        save_steering.save_features(save_path, info)

    return save_steering


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_sae(model_id, sae_ckpt_dir, layer, latent_mult=32, topk=32,
                       device="cuda"):
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

    dim = model.config.text_config.hidden_size
    sae = SAE(dim, dim * latent_mult, topk).float().to(device)
    sae_path = os.path.join(sae_ckpt_dir, f"sae_layer{layer}.pt")
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()
    print(f"  SAE loaded: {sae_path}")
    print(f"  SAE dims: {dim} → {dim * latent_mult}, top-k={topk}")

    return model, processor, sae


def qwen25vl_generate(model, processor, image, question, max_new_tokens=32):
    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            return ""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

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
# CV-Bench 评估
# ══════════════════════════════════════════════════════════════════════════════

def eval_cvbench(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  CV-Bench Evaluation (SAVE Steering)\n{'='*60}")
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
    print(f"\n{'='*60}\n  MMStar Evaluation (SAVE Steering)\n{'='*60}")
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

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)
    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    overall = acc(results)
    n_unparsed = sum(1 for r in results if r["extracted"] is None)
    print(f"\n  Overall Accuracy : {overall:.2f}  (n={len(results)})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {"benchmark": "MMStar", "overall": overall,
            "n_unparsed": n_unparsed, "n_total": len(results)}, results


def eval_realworldqa(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  RealWorldQA Evaluation (SAVE Steering)\n{'='*60}")
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


# ══════════════════════════════════════════════════════════════════════════════
# BLINK (simplified)
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
        concat.paste(img, (x, 0)); x += img.width
    out = "/tmp/blink_eval_concat.png"
    concat.save(out)
    return out


def eval_blink(model, processor, subtasks=None, max_samples=None):
    print(f"\n{'='*60}\n  BLINK Evaluation (SAVE Steering)\n{'='*60}")
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
    parser = argparse.ArgumentParser(description="SAVE Baseline on Qwen2.5-VL")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["identify", "eval", "both"])
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["cvbench", "mmstar", "realworldqa", "blink"],
                        choices=["cvbench", "mmstar", "realworldqa", "blink"])
    parser.add_argument("--model_id", type=str, default=CFG.model_id)
    parser.add_argument("--sae_ckpt_dir", type=str, default=CFG.save_dir)
    parser.add_argument("--layer", type=int, default=CFG.vis_layer)
    parser.add_argument("--latent_mult", type=int, default=CFG.latent_mult)
    parser.add_argument("--topk", type=int, default=CFG.topk)
    parser.add_argument("--steer_alpha", type=float, default=5.0,
                        help="Steering strength (SAVE default: 5.0)")
    parser.add_argument("--n_visual_features", type=int, default=50)
    parser.add_argument("--n_hallucinated_features", type=int, default=50)
    parser.add_argument("--identify_samples", type=int, default=500,
                        help="Number of samples for feature identification")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/save_qwen25vl_results")
    parser.add_argument("--features_path", type=str, default=None,
                        help="Path to pre-identified features JSON (skip identify step)")
    parser.add_argument("--blink_subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型和 SAE
    model, processor, sae = load_model_and_sae(
        args.model_id, args.sae_ckpt_dir, args.layer,
        latent_mult=args.latent_mult, topk=args.topk,
    )

    features_path = args.features_path or os.path.join(
        args.save_dir, f"save_features_layer{args.layer}.json"
    )

    # Step 1: Identify features
    save_steering = None
    if args.mode in ["identify", "both"]:
        save_steering = identify_save_features(
            model, processor, sae,
            steer_layer=args.layer,
            max_samples=args.identify_samples,
            save_path=features_path,
        )
        if save_steering is None:
            print("ERROR: Feature identification failed!")
            return

    # Step 2: Evaluate with steering
    if args.mode in ["eval", "both"]:
        if save_steering is None:
            # Load pre-identified features
            save_steering = SAVESteering(
                model, sae, processor=processor, steer_layer=args.layer,
                steer_alpha=args.steer_alpha,
            )
            save_steering.load_features(features_path)
        else:
            save_steering.steer_alpha = args.steer_alpha

        # Install steering hook
        save_steering.patch_generate()
        print(f"\n  SAVE Steering active: alpha={args.steer_alpha}, "
              f"layer={args.layer}")

        config = {
            "method": "SAVE (Global SAE Steering)",
            "model_id": args.model_id,
            "steer_alpha": args.steer_alpha,
            "steer_layer": args.layer,
            "n_visual_features": len(save_steering.visual_feature_ids),
            "n_hallucinated_features": len(save_steering.hallucinated_feature_ids),
            "sae_latent_mult": args.latent_mult,
            "sae_topk": args.topk,
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
        print(f"  SAVE (alpha={args.steer_alpha}) on Qwen2.5-VL — Summary")
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