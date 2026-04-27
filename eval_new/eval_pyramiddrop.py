"""
PyramidDrop 评估脚本 — 在 Qwen2.5-VL 上复现 PyramidDrop (CVPR 2025)。

PyramidDrop 核心思想 (Xing et al., CVPR 2025):
  将 LLM 分成 S 个 stage（默认 4 个），每个 stage 结束时：
  1. 计算最后一个 text token（instruction token）对每个 visual token 的 attention
  2. 按 attention score 排序，保留 top λ 比例的 visual token
  3. 被剪 token 通过 scatter-copy 替换

  默认配置：S=4, λ=0.5
  28 层分 4 stage → stage 边界在 layer 7, 14, 21
  Token 数量变化：100% → 50% → 25% → 12.5%

与 NeuronEye 的区别：
  - PyramidDrop: 渐进式减少 token（加速），不增强表示
  - NeuronEye: 不减少 token，通过 SAE + query routing 增强表示

用法：
    python eval/eval_pyramiddrop.py --benchmarks cvbench mmstar
    python eval_new/eval_pyramiddrop.py --drop_ratio 0.5
    python eval_new/eval_pyramiddrop.py --drop_ratio 0.6 --n_stages 4

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
# PyramidDrop
# ══════════════════════════════════════════════════════════════════════════════

class PyramidDropWrapper:
    """
    PyramidDrop: 金字塔式渐进 visual token 剪枝。

    算法：
      1. 把 N 层 LLM 分成 S 个 stage
      2. 每个 stage 结束的那一层，计算 text→visual 的 cosine similarity
         作为每个 visual token 的重要性
      3. 保留最重要的 λ 比例，被剪 token scatter-copy 到最近的保留 token
      4. 下一个 stage 继续在剩余 token 上操作

    原版用 last instruction token 对 visual token 的 attention 做排序。
    本适配版用 cosine similarity 近似（不需要 eager attention）。
    """

    def __init__(self, model, processor, n_stages=4, drop_ratio=0.5):
        self.model = model
        self.processor = processor
        self.n_stages = n_stages
        self.drop_ratio = drop_ratio  # 每个 stage 保留的比例

        n_layers = len(model.model.language_model.layers)
        # 计算 stage 边界层
        stage_size = n_layers // n_stages
        self.stage_boundaries = [
            (s + 1) * stage_size - 1 for s in range(n_stages - 1)
        ]
        # 最后一个 stage 不剪（已经到最后了）

        print(f"  PyramidDrop: {n_stages} stages, λ={drop_ratio}")
        print(f"  Stage boundaries (drop at layers): {self.stage_boundaries}")
        ratios = [drop_ratio ** (i + 1) for i in range(len(self.stage_boundaries))]
        print(f"  Cumulative retention: {[f'{r*100:.1f}%' for r in ratios]}")

        self._hooks = []
        self._vision_pos = None
        self._num_img_tokens = None
        self._alive_mask = None  # (seq_len,) bool — 哪些 visual token 还存活
        self._dropped_at = set()

    def _set_positions(self, input_ids, image_grid_thw):
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        spatial_merge = self.model.config.vision_config.spatial_merge_size

        ids = input_ids[0]
        vs_pos = (ids == vision_start_id).nonzero(as_tuple=True)[0]
        if len(vs_pos) > 0:
            self._vision_pos = vs_pos[0].item() + 1
            self._num_img_tokens = int(
                image_grid_thw[0, 1] * image_grid_thw[0, 2] / (spatial_merge ** 2)
            )
        else:
            self._vision_pos = None
            self._num_img_tokens = None

        # 初始化 alive mask
        if self._vision_pos is not None and self._num_img_tokens is not None:
            self._alive_mask = torch.ones(
                self._num_img_tokens, dtype=torch.bool
            )
        self._dropped_at = set()

    def install_hooks(self):
        self._hooks = []
        layers = self.model.model.language_model.layers
        for boundary_layer in self.stage_boundaries:
            hook = layers[boundary_layer].register_forward_hook(
                self._make_drop_hook(boundary_layer)
            )
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_drop_hook(self, layer_idx):
        def hook_fn(module, args, output):
            if layer_idx in self._dropped_at:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if hidden_states.shape[1] <= 1:
                return output

            self._dropped_at.add(layer_idx)
            self._do_pyramid_drop(hidden_states)
            return output
        return hook_fn

    def _do_pyramid_drop(self, hidden_states):
        if self._vision_pos is None or self._alive_mask is None:
            return

        v_pos = self._vision_pos
        n_img = self._num_img_tokens
        device = hidden_states.device

        # 确保 alive_mask 在正确设备
        if self._alive_mask.device != device:
            self._alive_mask = self._alive_mask.to(device)

        if v_pos + n_img > hidden_states.shape[1]:
            return

        # 当前存活的 visual token 位置（相对于 vision 区域的局部索引）
        alive_local = self._alive_mask.nonzero(as_tuple=True)[0]
        n_alive = len(alive_local)

        if n_alive <= 1:
            return

        n_keep = max(1, int(n_alive * self.drop_ratio))
        if n_keep >= n_alive:
            return

        # 全局位置
        alive_global = v_pos + alive_local

        # === PyramidDrop 的排序方式：text-visual similarity ===
        # 用最后一个 text token（instruction token）和 visual tokens 的 cosine similarity
        # 找最后一个非 vision 的 token 位置
        last_text_pos = min(v_pos + n_img, hidden_states.shape[1] - 1)
        # 往后找到最后一个 text token
        for p in range(hidden_states.shape[1] - 1, v_pos + n_img - 1, -1):
            last_text_pos = p
            break

        text_h = hidden_states[0, last_text_pos, :].float().unsqueeze(0)  # (1, dim)
        vision_h = hidden_states[0, alive_global, :].float()  # (n_alive, dim)

        # Cosine similarity
        sim = F.cosine_similarity(
            text_h.expand(n_alive, -1), vision_h, dim=-1
        )  # (n_alive,)

        # 保留 similarity 最高的 n_keep 个
        _, keep_local_idx = sim.topk(n_keep)
        _, drop_local_idx = sim.topk(n_alive - n_keep, largest=False)

        keep_global = alive_global[keep_local_idx]
        drop_global = alive_global[drop_local_idx]

        # scatter-copy: 被剪 token → 最近的保留 token
        if len(drop_global) > 0 and len(keep_global) > 0:
            keep_h = hidden_states[0, keep_global, :].float()
            drop_h = hidden_states[0, drop_global, :].float()

            cos_sim = torch.matmul(
                F.normalize(drop_h, dim=-1),
                F.normalize(keep_h, dim=-1).T
            )
            nearest = cos_sim.argmax(dim=1)

            for ii, pos in enumerate(drop_global):
                hidden_states[0, pos, :] = hidden_states[0, keep_global[nearest[ii]], :]

        # 更新 alive mask
        drop_local_positions = alive_local[drop_local_idx]
        self._alive_mask[drop_local_positions] = False

    def patch_generate(self):
        original_generate = self.model.generate

        @wraps(original_generate)
        def patched_generate(*args, **kwargs):
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            image_grid_thw = kwargs.get("image_grid_thw", None)

            if input_ids is not None and image_grid_thw is not None:
                self._set_positions(input_ids, image_grid_thw)

            self.install_hooks()
            try:
                result = original_generate(*args, **kwargs)
            finally:
                self.remove_hooks()
                self._dropped_at = set()
            return result

        self.model.generate = patched_generate
        self._original_generate = original_generate


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_id):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading Qwen2.5-VL: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id, max_pixels=1280 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    print(f"  Model loaded. Layers: {len(model.model.language_model.layers)}")
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
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
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

    except Exception as e:
        print(f"  [error] generate failed: {e}")
        torch.cuda.empty_cache()
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 答案匹配
# ══════════════════════════════════════════════════════════════════════════════

def extract_choice_letter(response, max_letter="Z"):
    text = response.strip().split("\n")[0].strip().upper()
    if not text:
        return None
    pat = f'[A-{max_letter}]'
    m = re.search(rf'\(({pat})\)', text)
    if m: return m.group(1)
    m = re.search(rf'(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?({pat})\)?', text)
    if m: return m.group(1)
    m = re.match(rf'^({pat})(?:[\s.,):]|$)', text)
    if m: return m.group(1)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 评估
# ══════════════════════════════════════════════════════════════════════════════

def eval_cvbench(model, processor, max_samples=None):
    print(f"\n{'='*60}\n  CV-Bench Evaluation (PyramidDrop)\n{'='*60}")
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
    print(f"\n{'='*60}\n  MMStar Evaluation (PyramidDrop)\n{'='*60}")
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
    print(f"\n{'='*60}\n  RealWorldQA Evaluation (PyramidDrop)\n{'='*60}")
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
    if len(image_paths) == 1: return image_paths[0]
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
    print(f"\n{'='*60}\n  BLINK Evaluation (PyramidDrop)\n{'='*60}")
    if subtasks is None: subtasks = ALL_BLINK_SUBTASKS
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
    return {"benchmark": "BLINK", "overall": overall, "acc_depth_spatial": acc_ds,
            "per_subtask": per_subtask, "n_unparsed": n_unparsed,
            "n_total": len(results)}, results


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PyramidDrop (CVPR 2025) on Qwen2.5-VL")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["cvbench", "mmstar", "realworldqa", "blink"],
                        choices=["cvbench", "mmstar", "realworldqa", "blink"])
    parser.add_argument("--model_id", type=str, default=CFG.model_id)
    parser.add_argument("--n_stages", type=int, default=4)
    parser.add_argument("--drop_ratio", type=float, default=0.5,
                        help="每个 stage 保留的比例 (0.5 = 保留 50%)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/pyramiddrop_qwen25vl_results")
    parser.add_argument("--blink_subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    model, processor = load_model(args.model_id)

    pdrop = PyramidDropWrapper(model, processor,
                                n_stages=args.n_stages,
                                drop_ratio=args.drop_ratio)
    pdrop.patch_generate()

    total_retain = args.drop_ratio ** (args.n_stages - 1)
    config = {
        "method": "PyramidDrop",
        "model_id": args.model_id,
        "n_stages": args.n_stages,
        "drop_ratio": args.drop_ratio,
        "effective_retention": total_retain,
        "stage_boundaries": pdrop.stage_boundaries,
        "n_layers": len(model.model.language_model.layers),
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
    print(f"  PyramidDrop (S={args.n_stages}, λ={args.drop_ratio}) — Summary")
    print(f"  Effective retention: {total_retain*100:.1f}%")
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