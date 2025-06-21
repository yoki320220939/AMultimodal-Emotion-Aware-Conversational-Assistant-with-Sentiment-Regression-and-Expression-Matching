from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 模型路径
model_path = "./models/Erlangshen-Roberta-110M-Sentiment"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 设置模型为评估模式
model.eval()

# 示例文本
texts = [
    "非常开心！",
    "这种矛盾的心理状态在动物身上表现得非常可爱和有趣。同时，猫咪躲在凳子下只露出眼睛的样子，增加了一种戏剧性和幽默感。",
    "还行吧。"
]

# 情感标签
labels = ["负面",  "正面"]

# 进行情感分类
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        print(logits)
        predicted_class = torch.argmax(logits, dim=1).item()
        print(f"文本：{text}")
        print(f"预测情感：{labels[predicted_class]}\n")
