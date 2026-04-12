"""
DeepSeek-VL2 基础模型评估脚本。

支持 4 个 benchmark：CV-Bench, BLINK, RealWorldQA, MMStar。
只测原始模型（无 hook / 无增强），用于获取 baseline 分数。

模型变体（activated params）：
    deepseek-ai/deepseek-vl2-tiny   — 1.0B activated, ~3.4B total, 单卡 <40GB
    deepseek-ai/deepseek-vl2-small  — 2.8B activated, ~16B total,  单卡 40-80GB
    deepseek-ai/deepseek-vl2        — 4.5B activated, ~27.5B total, 需 80GB+

依赖安装（二选一）：
    # 官方版（需要 transformers<4.48）
    pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git --no-deps
    pip install attrdict timm 'transformers<4.48.0'

    # 社区兼容新版 transformers 的 fork
    pip install git+https://github.com/sheryc/DeepSeek-VL2-Latest.git --no-deps

用法：
    CUDA_VISIBLE_DEVICES=0 python eval/eval_deepseek_vl2_complex.py
    python eval/eval_deepseek_vl2_base.py --model_id deepseek-ai/deepseek-vl2-tiny
    python eval/eval_deepseek_vl2_base.py --model_id deepseek-ai/deepseek-vl2-small --chunk_size 512
    python eval/eval_deepseek_vl2_base.py --benchmarks cvbench mmstar
    python eval/eval_deepseek_vl2_base.py --benchmarks blink --depth_spatial_only
    python eval/eval_deepseek_vl2_base.py --max_samples 50
"""
import os
import sys
import re
import json
import argparse
import subprocess
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── 自动安装 deepseek_vl2 依赖 ──
try:
    import deepseek_vl2
except ImportError:
    print("deepseek_vl2 not found, installing from GitHub (community fork for latest transformers)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/sheryc/DeepSeek-VL2-Latest.git",
        "--no-deps", "--quiet",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "attrdict", "timm", "--quiet",
    ])
    import deepseek_vl2

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model(chunk_size=-1):
    from transformers import AutoModelForCausalLM
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

    print(f"Loading DeepSeek-VL2: {MODEL_ID}")

    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(MODEL_ID)
    tokenizer = vl_chat_processor.tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    model = model.to(torch.bfloat16).cuda().eval()

    print(f"  Model loaded. chunk_size={chunk_size}")
    return model, vl_chat_processor, tokenizer, chunk_size


def deepseek_vl2_generate(model, vl_chat_processor, tokenizer, image, question,
                           max_new_tokens=32, chunk_size=-1):
    """DeepSeek-VL2 单次推理。"""
    try:
        from deepseek_vl2.utils.io import load_pil_images

        # 处理图片输入
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            return ""

        # 保存临时图片
        tmp_path = "/tmp/deepseek_vl2_eval_tmp.png"
        image.save(tmp_path)

        # 构造对话 — VL2 格式
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": [tmp_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # 加载图片
        pil_images = load_pil_images(conversation)

        # 处理输入
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(model.device)

        # 获取 image embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        # 使用 incremental prefilling（对显存友好）
        if chunk_size > 0:
            # incremental prefilling for large models on limited GPU
            outputs = _incremental_generate(
                model, tokenizer, inputs_embeds,
                prepare_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
            )
        else:
            outputs = model.language_model.generate(
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
        import traceback
        traceback.print_exc()
        return ""


def _incremental_generate(model, tokenizer, inputs_embeds, attention_mask,
                           max_new_tokens=32, chunk_size=512):
    """
    Incremental prefilling: 将 inputs_embeds 分块送入，节省显存。
    适用于 40GB GPU 跑 deepseek-vl2-small。
    """
    seq_len = inputs_embeds.shape[1]
    past_key_values = None

    # prefill in chunks
    for i in range(0, seq_len, chunk_size):
        chunk_embeds = inputs_embeds[:, i:i+chunk_size]
        chunk_mask = attention_mask[:, :i+chunk_embeds.shape[1]]

        with torch.no_grad():
            out = model.language_model(
                inputs_embeds=chunk_embeds,
                attention_mask=chunk_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = out.past_key_values

    # generate
    outputs = model.language_model.generate(
        input_ids=torch.tensor([[tokenizer.eos_token_id]], device=model.device),
        attention_mask=torch.ones(1, seq_len + 1, device=model.device),
        past_key_values=past_key_values,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return outputs


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

def eval_cvbench(model, vl_chat_processor, tokenizer, max_samples=None, chunk_size=-1):
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

        response = deepseek_vl2_generate(
            model, vl_chat_processor, tokenizer, item["image"], prompt,
            chunk_size=chunk_size)
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

def eval_mmstar(model, vl_chat_processor, tokenizer, max_samples=None, chunk_size=-1):
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

        response = deepseek_vl2_generate(
            model, vl_chat_processor, tokenizer, item["image"], prompt,
            chunk_size=chunk_size)
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


def eval_realworldqa(model, vl_chat_processor, tokenizer, max_samples=None, chunk_size=-1):
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

        response = deepseek_vl2_generate(
            model, vl_chat_processor, tokenizer, item["image"], question,
            chunk_size=chunk_size)
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


def eval_blink(model, vl_chat_processor, tokenizer, subtasks=None, max_samples=None,
               chunk_size=-1):
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
        response = deepseek_vl2_generate(
            model, vl_chat_processor, tokenizer, concat_path, prompt_text,
            chunk_size=chunk_size)

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
    parser.add_argument("--model_id", type=str, default=MODEL_ID,
                        help="可选: deepseek-ai/deepseek-vl2-tiny (默认), "
                             "deepseek-ai/deepseek-vl2-small, "
                             "deepseek-ai/deepseek-vl2")
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="Incremental prefilling chunk size. "
                             "设为 512 可在 40GB GPU 上跑 vl2-small。"
                             "默认 -1 不启用。")
    parser.add_argument("--blink_subtasks", type=str, nargs="+", default=None)
    parser.add_argument("--depth_spatial_only", action="store_true")
    args = parser.parse_args()
    MODEL_ID = args.model_id

    os.makedirs(args.save_dir, exist_ok=True)

    model, vl_chat_processor, tokenizer, chunk_size = load_model(args.chunk_size)

    all_summary = {}

    if "cvbench" in args.benchmarks:
        metrics, results = eval_cvbench(
            model, vl_chat_processor, tokenizer, args.max_samples, chunk_size)
        all_summary["cvbench"] = metrics
        with open(os.path.join(args.save_dir, "cvbench_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if "mmstar" in args.benchmarks:
        metrics, results = eval_mmstar(
            model, vl_chat_processor, tokenizer, args.max_samples, chunk_size)
        all_summary["mmstar"] = metrics
        with open(os.path.join(args.save_dir, "mmstar_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if "realworldqa" in args.benchmarks:
        metrics, results = eval_realworldqa(
            model, vl_chat_processor, tokenizer, args.max_samples, chunk_size)
        all_summary["realworldqa"] = metrics
        with open(os.path.join(args.save_dir, "realworldqa_results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if "blink" in args.benchmarks:
        subtasks = args.blink_subtasks
        if args.depth_spatial_only:
            subtasks = DEPTH_SPATIAL_SUBTASKS
        metrics, results = eval_blink(
            model, vl_chat_processor, tokenizer, subtasks, args.max_samples,
            chunk_size)
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