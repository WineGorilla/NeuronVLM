import json

input_file = "data/train_vqa2.jsonl"
output_file = "train_vqa2_unique.jsonl"

seen_images = set()
kept = 0

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        record = json.loads(line)
        img = record["image"]
        if img not in seen_images:
            seen_images.add(img)
            fout.write(line)
            kept += 1

print(f"总记录数中 unique 图片: {kept}")
print(f"已保存到: {output_file}")