import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json
import argparse


def build_from_llava(anno_path: str, img_dir: str,
                     output: str = "data/train.jsonl", max_samples: int = None):
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
                skipped += 1
                continue

            question = ""
            answer   = ""
            for turn in item.get("conversations", []):
                if turn.get("from") == "human":
                    question = turn.get("value", "").replace("<image>\n", "").replace("<image>", "").strip()
                if turn.get("from") == "gpt":
                    answer = turn.get("value", "").strip()

            if not question:
                question = "Describe the image."

            sample = {
                "image":    img_path,
                "question": question,
                "answer":   answer,
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"wrote {count} samples -> {output}")
    if skipped:
        print(f"skipped {skipped} missing images")


def build_single(image_path: str, output: str = "data/train.jsonl",
                 question: str = "Describe the image.", answer: str = ""):
    sample = {
        "image":    image_path,
        "question": question,
        "answer":   answer,
    }
    with open(output, "w") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"wrote 1 sample -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--anno",   help="LLaVA json 标注文件路径")
    group.add_argument("--single", help="单张图片路径（测试用）")
    parser.add_argument("--imgs",   default="data/images/train2014", help="图片文件夹")
    parser.add_argument("--output", default="data/train.jsonl")
    parser.add_argument("--max",    type=int, default=None, help="最多读取条数")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.anno:
        build_from_llava(args.anno, args.imgs, args.output, args.max)
    else:
        build_single(args.single, args.output)