## 表情包数据集分割

import json
import os
import random

# 路径设置
input_path = "labeled_data_logits.json"
image_root = "emo"
train_output = "train_data.json"
val_output = "val_data.json"
train_ratio = 0.8

# 加载数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 构建标准数据格式
processed_data = []

for item in data:
    image_path = os.path.join(image_root, item["filename"])
    new_item = {
        "image_path": image_path,
        "text": item["content"],
        "logits": item["logits"]  # 直接使用已有的向量作为监督信号
    }
    processed_data.append(new_item)

# 打乱并划分
random.shuffle(processed_data)
split = int(len(processed_data) * train_ratio)
train_data = processed_data[:split]
val_data = processed_data[split:]

# 保存
with open(train_output, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(val_output, "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ 数据处理完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条。")
