"""
数据构造脚本 — 支持多种数据源 → 统一 JSONL 格式

输出格式（每行一个 JSON）：
{
    "image": "data/images/train2014/COCO_train2014_000000429741.jpg",
    "question": "What interesting interaction occurs between the animals?",
    "answer": "In the scene, there is an interaction between..."
}

用法：

  1. VQA v2（annotations.json + questions.json → 短答案）：
     python scripts/build_vqa_data.py --vqa2 \
         --anno data/annotations/v2_mscoco_train2014_annotations.json \
         --questions data/annotations/v2_OpenEnded_mscoco_train2014_questions.json \
         --imgs data/images/train2014 \
         --output data/train_vqa2.jsonl

  2. LLaVA 格式（conversations JSON → 长答案）：
     python src/build_data.py --llava \
         --anno data/llava_instruct_150k.json \
         --imgs data/images/train2014 \
         --output data/train_llava.jsonl

  3. 单张图片（测试用）：
     python src/build_data.py --single path/to/image.jpg

  4. 合并多个 JSONL：
     python src/build_data.py --merge data/train_vqa2.jsonl data/train_llava.jsonl \
         --output data/train_merged.jsonl
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
import random
from collections import Counter


def _find_image(image_id: int, img_dir: str) -> str:
    """尝试多种 COCO 图片命名格式，返回存在的路径或 None"""
    candidates = [
        os.path.join(img_dir, f"COCO_train2014_{image_id:012d}.jpg"),
        os.path.join(img_dir, f"COCO_val2014_{image_id:012d}.jpg"),
        os.path.join(img_dir, f"{image_id:012d}.jpg"),
        os.path.join(img_dir, f"{image_id}.jpg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def build_from_vqa2(anno_path: str, questions_path: str, img_dir: str,
                    output: str = "data/train_vqa2.jsonl", max_samples: int = None):
    """
    VQA v2 格式：
    - annotations.json: {"annotations": [{"question_id", "image_id", "answers": [...], "multiple_choice_answer"}]}
    - questions.json:   {"questions": [{"question_id", "image_id", "question"}]}
    
    answer 取 multiple_choice_answer（最高频答案），或 answers 列表的众数。
    """
    print(f"Loading VQA v2 annotations: {anno_path}")
    with open(anno_path) as f:
        anno_data = json.load(f)
    annotations = anno_data.get("annotations", anno_data)
    if isinstance(annotations, dict):
        annotations = list(annotations.values())

    print(f"Loading VQA v2 questions: {questions_path}")
    with open(questions_path) as f:
        q_data = json.load(f)
    questions = q_data.get("questions", q_data)
    if isinstance(questions, dict):
        questions = list(questions.values())

    # question_id → question text
    qid_to_question = {}
    for q in questions:
        qid_to_question[q["question_id"]] = q["question"]

    if max_samples:
        annotations = annotations[:max_samples]

    count = 0
    skipped = 0

    with open(output, "w") as out:
        for ann in annotations:
            image_id = ann["image_id"]
            qid = ann["question_id"]

            img_path = _find_image(image_id, img_dir)
            if img_path is None:
                skipped += 1
                continue

            question = qid_to_question.get(qid, "")
            if not question:
                skipped += 1
                continue

            # 取答案：优先 multiple_choice_answer，否则取众数
            answer = ann.get("multiple_choice_answer", "")
            if not answer and "answers" in ann:
                answer_texts = [a["answer"] for a in ann["answers"]]
                if answer_texts:
                    counter = Counter(answer_texts)
                    answer = counter.most_common(1)[0][0]

            if not answer:
                skipped += 1
                continue

            sample = {
                "image": img_path,
                "question": question.strip(),
                "answer": answer.strip(),
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} samples → {output}")
    if skipped:
        print(f"Skipped {skipped} (missing image or empty Q/A)")


def build_from_llava(anno_path: str, img_dir: str,
                     output: str = "data/train_llava.jsonl", max_samples: int = None):
    """
    LLaVA 格式：
    [{"image": "000000429741.jpg", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]
    """
    print(f"Loading LLaVA annotations: {anno_path}")
    with open(anno_path) as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    count = 0
    skipped = 0

    with open(output, "w") as out:
        for item in data:
            img_file = item.get("image", "")
            img_path = os.path.join(img_dir, img_file)

            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, "COCO_train2014_" + img_file)
            if not os.path.exists(img_path):
                # 尝试从文件名提取 image_id
                basename = os.path.splitext(img_file)[0]
                digits = ''.join(c for c in basename if c.isdigit())
                if digits:
                    found = _find_image(int(digits), img_dir)
                    if found:
                        img_path = found
            if not os.path.exists(img_path):
                skipped += 1
                continue

            question = ""
            answer = ""
            for turn in item.get("conversations", []):
                if turn.get("from") == "human":
                    question = turn.get("value", "")
                    question = question.replace("<image>\n", "").replace("<image>", "").strip()
                if turn.get("from") == "gpt":
                    answer = turn.get("value", "").strip()

            if not question:
                question = "Describe the image."

            sample = {
                "image": img_path,
                "question": question,
                "answer": answer,
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} samples → {output}")
    if skipped:
        print(f"Skipped {skipped} missing images")


def build_single(image_path: str, output: str = "data/train.jsonl",
                 question: str = "Describe the image.", answer: str = ""):
    """单张图片，测试用"""
    sample = {
        "image": image_path,
        "question": question,
        "answer": answer,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote 1 sample → {output}")


def merge_jsonl(inputs: list, output: str, shuffle: bool = True):
    """合并多个 JSONL 文件，可选 shuffle"""
    lines = []
    for path in inputs:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        print(f"  Read {path}: {sum(1 for _ in open(path))} lines")

    if shuffle:
        random.shuffle(lines)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Merged {len(lines)} samples → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training JSONL from various VQA data sources")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vqa2",   action="store_true", help="VQA v2 格式 (annotations + questions)")
    group.add_argument("--llava",  action="store_true", help="LLaVA 格式 (conversations JSON)")
    group.add_argument("--single", type=str,            help="单张图片路径（测试用）")
    group.add_argument("--merge",  nargs="+",           help="合并多个 JSONL 文件")

    parser.add_argument("--anno",      type=str, help="标注文件路径 (annotations.json)")
    parser.add_argument("--questions", type=str, help="VQA v2 questions 文件路径")
    parser.add_argument("--imgs",      default="data/images/train2014", help="图片文件夹")
    parser.add_argument("--output",    default="data/train.jsonl")
    parser.add_argument("--max",       type=int, default=None, help="最多读取条数")
    parser.add_argument("--no-shuffle", action="store_true", help="合并时不 shuffle")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.vqa2:
        if not args.anno or not args.questions:
            parser.error("--vqa2 requires both --anno and --questions")
        build_from_vqa2(args.anno, args.questions, args.imgs, args.output, args.max)

    elif args.llava:
        if not args.anno:
            parser.error("--llava requires --anno")
        build_from_llava(args.anno, args.imgs, args.output, args.max)

    elif args.single:
        build_single(args.single, args.output)

    elif args.merge:
        merge_jsonl(args.merge, args.output, shuffle=not args.no_shuffle)