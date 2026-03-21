"""
构建测试集：取 LLaVA 数据的最后 5000 条。

用法：
    python scripts/build_test.py \
        --anno data/annotations/llava_instruct_150k.json \
        --imgs data/images/train2014 \
        --output data/test.jsonl
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno",      required=True, help="LLaVA json 标注文件路径")
    parser.add_argument("--imgs",      default="data/images/train2014")
    parser.add_argument("--output",    default="data/test.jsonl")
    parser.add_argument("--test_size", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.anno) as f:
        data = json.load(f)

    # 取最后 test_size 条
    data = data[-args.test_size:]
    print(f"Test samples: {len(data)}")

    count   = 0
    skipped = 0

    with open(args.output, "w") as out:
        for item in data:
            img_file = item.get("image", "")
            img_path = os.path.join(args.imgs, img_file)
            if not os.path.exists(img_path):
                img_path = os.path.join(args.imgs, "COCO_train2014_" + img_file)
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

            out.write(json.dumps({
                "image":    img_path,
                "question": question,
                "answer":   answer,
            }, ensure_ascii=False) + "\n")
            count += 1

    print(f"wrote {count} samples -> {args.output}")
    if skipped:
        print(f"skipped {skipped} missing images")


if __name__ == "__main__":
    main()