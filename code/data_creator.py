## 通过表情包的文字构建情感向量，从而拓展数据集


import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# 模型路径
model_path = "./models/Erlangshen-Roberta-110M-Sentiment"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 读取数据
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)



# 打标并记录 logits
labeled_data = []

for item in tqdm(data, desc="提取 logits 中"):
    text = item["content"]


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze().tolist()  # 获取 logits 向量

    labeled_item = {
        "filename": item["filename"],
        "content": item["content"],
        "logits": logits  # 保存原始 logits 向量
    }
    labeled_data.append(labeled_item)

# 保存为 JSON
with open("labeled_data_logits.json", "w", encoding="utf-8") as f:
    json.dump(labeled_data, f, ensure_ascii=False, indent=4)

print("✅ Logits 提取完成，保存在 labeled_data_logits.json")
