"""
DeepSeek-VL2 基础模型评估脚本。

支持 4 个 benchmark：CV-Bench, BLINK, RealWorldQA, MMStar。

依赖（二选一）：
    # 如果 transformers < 4.48
    pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git

    # 如果 transformers >= 4.48（推荐）
    pip install git+https://github.com/sheryc/DeepSeek-VL2-Latest.git --no-deps
    pip install attrdict timm

用法：
    CUDA_VISIBLE_DEVICES=0 python eval/eval_deepseek_vl2_base.py
    python eval/eval_deepseek_vl2_base.py --model_id deepseek-ai/deepseek-vl2-small
    python eval/eval_deepseek_vl2_base.py --benchmarks cvbench mmstar
    python eval/eval_deepseek_vl2_base.py --max_samples 50
"""
import os
import sys
import re
import json
import argparse
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

MODEL_ID = "deepseek-ai/deepseek-vl2-small"


# ══════════════════════════════════════════════════════════════════════════════
# 兼容性 patch — 在 import deepseek_vl2 之前执行
# ══════════════════════════════════════════════════════════════════════════════

def _apply_compat_patches():
    """修复 deepseek_vl2 与新版 transformers 的兼容性问题。"""
    import importlib

    # patch 1: is_torch_fx_available 缺失
    try:
        from transformers.utils.import_utils import is_torch_fx_available
    except ImportError:
        import transformers.utils.import_utils as _iu
        _iu.is_torch_fx_available = lambda: False

    # patch 2: xformers 缺失
    try:
        import xformers
    except ImportError:
        import types
        xformers_mod = types.ModuleType("xformers")
        xformers_ops = types.ModuleType("xformers.ops")
        xformers_ops.memory_efficient_attention = None
        xformers_mod.ops = xformers_ops
        sys.modules["xformers"] = xformers_mod
        sys.modules["xformers.ops"] = xformers_ops

    # patch 3: torch.linspace 在 meta device 上执行时 .item() 会崩
    #          强制让 linspace 始终在 CPU 上创建
    _orig_linspace = torch.linspace

    def _safe_linspace(*args, **kwargs):
        kwargs.pop("device", None)
        return _orig_linspace(*args, **kwargs, device="cpu")

    torch.linspace = _safe_linspace

    # patch 4: 新版 transformers 会给 config 注入一个 rope_scaling 字典
    #          (包含 rope_type:"default" 等字段)，即使原始 config.json
    #          里根本没有 rope_scaling。
    #
    #          deepseek-vl2-small 的原始 config 没有 rope_scaling，
    #          使用标准 RoPE，不需要任何 scaling。
    #
    #          解决方案：
    #          - 读取原始 config.json 判断是否真正有 rope_scaling
    #          - 如果原始没有，强制设为 None（清除 transformers 注入的假值）
    #          - 如果原始有，用原始值覆盖（防止字段名被篡改）
    import deepseek_vl2.models.modeling_deepseek as _md

    # 从 HF Hub 缓存里直接读原始 config.json
    _original_rope_scaling = "NOT_FOUND"  # 用哨兵值区分"没有"和 None
    try:
        from huggingface_hub import hf_hub_download
        _config_path = hf_hub_download(MODEL_ID, "config.json")
        with open(_config_path, "r") as _f:
            _raw_config = json.load(_f)
        _lang_cfg = _raw_config.get("language_config", _raw_config)
        if "rope_scaling" in _lang_cfg:
            _original_rope_scaling = _lang_cfg["rope_scaling"]
        else:
            _original_rope_scaling = None  # 原始 config 没有 rope_scaling
        print(f"  [patch] Original rope_scaling: {_original_rope_scaling}")
    except Exception as _e:
        print(f"  [patch] Could not read original config.json: {_e}")
        _original_rope_scaling = "NOT_FOUND"

    _orig_attn_init = _md.DeepseekV2Attention.__init__

    def _patched_attn_init(self, config, layer_idx=None):
        if _original_rope_scaling is None:
            # 原始 config 没有 rope_scaling，清除 transformers 注入的假值
            config.rope_scaling = None
        elif _original_rope_scaling != "NOT_FOUND" and _original_rope_scaling is not None:
            # 原始 config 有 rope_scaling，用原始值覆盖
            config.rope_scaling = dict(_original_rope_scaling)
        else:
            # 无法读取原始 config，尝试修复
            if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                rs = config.rope_scaling
                if "type" not in rs and "rope_type" in rs:
                    rs["type"] = rs["rope_type"]
                if rs.get("type") in ("default",):
                    rs["type"] = "yarn"
                if "factor" not in rs:
                    rs["factor"] = rs.get("scaling_factor", 1.0)
        _orig_attn_init(self, config, layer_idx)

    _md.DeepseekV2Attention.__init__ = _patched_attn_init

    # patch 5: 新版 transformers (>=4.48) 在 post_init() 中会执行
    #            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(...)
    #          deepseek_vl2 没有这个属性，需要预先设为普通类属性
    from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM as _VLModel
    from deepseek_vl2.models.modeling_deepseek import DeepseekV2ForCausalLM as _LangModel

    for _cls in (_VLModel, _LangModel):
        if not hasattr(_cls, 'all_tied_weights_keys'):
            _cls.all_tied_weights_keys = {}
        if not hasattr(_cls, '_tied_weights_keys'):
            _cls._tied_weights_keys = []


_apply_compat_patches()

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    print(f"Loading DeepSeek-VL2: {MODEL_ID}")
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    vl_gpt.eval()

    print("  Model loaded.")
    return vl_gpt, vl_chat_processor, tokenizer


def generate(vl_gpt, vl_chat_processor, tokenizer, image, question,
             max_new_tokens=32):
    """DeepSeek-VL2 单次推理，严格按官方 inference.py 写法。"""
    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            return ""

        tmp_path = "/tmp/deepseek_vl2_eval_tmp.png"
        image.save(tmp_path)

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": [tmp_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(vl_gpt.device)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        response = tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True,
        ).strip()
        return response

    except Exception as e:
        print(f"  [error] generate failed: {e}")
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
# CV-Bench
# ══════════════════════════════════════════════════════════════════════════════

def eval_cvbench(vl_gpt, vl_chat_processor, tokenizer, max_samples=None):
    print(f"\n{'='*60}\n  CV-Bench Evaluation\n{'='*60}")

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

        response = generate(vl_gpt, vl_chat_processor, tokenizer, item["image"], prompt)
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
    for t in sorted(by_task.keys()):
        print(f"    {t:20s}: {acc(by_task[t]):6.2f}  (n={len(by_task[t])})")
    print(f"  Unparsed: {n_unparsed}/{len(results)}")

    return {
        "benchmark": "CV-Bench", "cv_bench": cv_bench,
        "acc_2d": acc_2d, "acc_3d": acc_3d,
        "per_task": {t: acc(items) for t, items in by_task.items()},
        "n_unparsed": n_unparsed, "n_total": len(results),
    }, results


# ══════════════════════════════════════════════════════════════════════════════
# MMStar
# ══════════════════════════════════════════════════════════════════════════════

def eval_mmstar(vl_gpt, vl_chat_processor, tokenizer, max_samples=None):
    print(f"\n{'='*60}\n  MMStar Evaluation\n{'='*60}")

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

        response = generate(vl_gpt, vl_chat_processor, tokenizer, item["image"], prompt)
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

def detect_question_type(question, answer):
    ans = answer.strip()
    if ans in ("A", "B", "C", "D"):
        return "mcq"
    if ans.lower() in ("yes", "no"):
        return "yes_no"
    return "short"


def match_realworldqa(response, answer, q_type):
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


def eval_realworldqa(vl_gpt, vl_chat_processor, tokenizer, max_samples=None):
    print(f"\n{'='*60}\n  RealWorldQA Evaluation\n{'='*60}")

    ds = load_dataset("xai-org/RealworldQA", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"  Samples: {len(ds)}")

    results = []
    for i, item in enumerate(tqdm(ds, desc="RealWorldQA")):
        question = item["question"]
        answer = item["answer"]
        q_type = detect_question_type(question, answer)

        response = generate(vl_gpt, vl_chat_processor, tokenizer, item["image"], question)
        correct, extracted = match_realworldqa(response, answer, q_type)

        results.append({
            "idx": i, "q_type": q_type, "answer": answer,
            "response": response, "extracted": extracted, "correct": correct,
        })

    by_type = defaultdict(list)
    for r in results:
        by_type[r["q_type"]].append(r)

    def acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) * 100 if items else 0

    overall = acc(results)
    n_unparsed = sum(1 for r in results if r.get("extracted") is None and r["q_type"] == "mcq")

    print(f"\n  Overall Accuracy : {overall:.2f}  (n={len(results)})")
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


def eval_blink(vl_gpt, vl_chat_processor, tokenizer, subtasks=None, max_samples=None):
    print(f"\n{'='*60}\n  BLINK Evaluation\n{'='*60}")

    if subtasks is None:
        subtasks = ALL_BLINK_SUBTASKS

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

    results = []
    for item in tqdm(all_items, desc="BLINK"):
        choices = item["choices"]
        has_image_choices = (
            isinstance(choices, list) and len(choices) > 0
            and isinstance(choices[0], Image.Image)
        )
        if has_image_choices:
            choice_text = "\n".join(
                [f"{chr(ord('A')+i)}. (Image {i+1})" for i in range(len(choices))])
        else:
            choice_text = "\n".join(
                [f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices)])
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
        response = generate(vl_gpt, vl_chat_processor, tokenizer, concat_path, prompt_text)

        extracted = extract_choice_letter(response)
        answer = item["answer"]
        extracted_fmt = f"({extracted})" if extracted else None
        correct = extracted_fmt == answer

        results.append({
            "idx": item["idx"], "subtask": item["subtask"],
            "answer": answer, "response": response,
            "extracted": extracted_fmt, "correct": correct,
        })

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
    global MODEL_ID

    parser = argparse.ArgumentParser(description="DeepSeek-VL2 Base Evaluation")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["cvbench", "mmstar", "realworldqa", "blink"],
                        choices=["cvbench", "mmstar", "realworldqa", "blink"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/deepseek_vl2_base_results")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--blink_subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    args = parser.parse_args()
    MODEL_ID = args.model_id

    os.makedirs(args.save_dir, exist_ok=True)

    vl_gpt, vl_chat_processor, tokenizer = load_model()

    all_summary = {}

    if "cvbench" in args.benchmarks:
        metrics, results = eval_cvbench(vl_gpt, vl_chat_processor, tokenizer, args.max_samples)
        all_summary["cvbench"] = metrics
        with open(os.path.join(args.save_dir, "cvbench_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if "mmstar" in args.benchmarks:
        metrics, results = eval_mmstar(vl_gpt, vl_chat_processor, tokenizer, args.max_samples)
        all_summary["mmstar"] = metrics
        with open(os.path.join(args.save_dir, "mmstar_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if "realworldqa" in args.benchmarks:
        metrics, results = eval_realworldqa(vl_gpt, vl_chat_processor, tokenizer, args.max_samples)
        all_summary["realworldqa"] = metrics
        with open(os.path.join(args.save_dir, "realworldqa_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if "blink" in args.benchmarks:
        subtasks = args.blink_subtasks
        if args.depth_spatial_only:
            subtasks = DEPTH_SPATIAL_SUBTASKS
        metrics, results = eval_blink(vl_gpt, vl_chat_processor, tokenizer, subtasks, args.max_samples)
        all_summary["blink"] = metrics
        with open(os.path.join(args.save_dir, "blink_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  DeepSeek-VL2 Base — Summary")
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