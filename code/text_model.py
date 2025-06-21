## 通过文字和情感特征进行文字数据库匹配



import json
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载BERT模型和情感模型
text_model_path = "./code/models/bert-base-chinese"
text_tokenizer = BertTokenizer.from_pretrained(text_model_path)
text_model = BertModel.from_pretrained(text_model_path)
text_model.eval()

emotion_model_path = "./code/models/Erlangshen-Roberta-110M-Sentiment"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
emotion_model.eval()

# 提取文本特征（BERT）
def get_text_features(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return sentence_embedding

# 提取情感特征
def get_emotion_features(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze().tolist()
    return logits

# 加载数据集
with open("./code/quotes_with_features.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 提取数据集中的特征
dataset_features = [item["features"] for item in dataset]
dataset_texts = [item["quote"] for item in dataset]

# 输入待匹配的句子
input_sentence = "难过"  # 替换为你要匹配的句子

# 提取待匹配句子的特征
input_text_features = get_text_features(input_sentence, text_tokenizer, text_model)
input_emotion_features = get_emotion_features(input_sentence, emotion_tokenizer, emotion_model)

# 计算文本特征相似度
text_similarities = []
for features in dataset_features:
    text_features = features[:len(input_text_features)]  # 文本特征部分
    similarity = cosine_similarity([input_text_features], [text_features])[0][0]
    text_similarities.append(similarity)

# 计算情感特征相似度
emotion_similarities = []
for features in dataset_features:
    emotion_features = features[len(input_text_features):]  # 情感特征部分
    similarity = cosine_similarity([input_emotion_features], [emotion_features])[0][0]
    emotion_similarities.append(similarity)

# 设置加权比例，假设文本相似度和情感相似度的权重分别是 0.7 和 0.3
text_weight = 0.7
emotion_weight = 0.3

# 计算加权平均相似度
combined_similarities = [
    text_weight * text_similarity + emotion_weight * emotion_similarity
    for text_similarity, emotion_similarity in zip(text_similarities, emotion_similarities)
]

# 找到最相似的句子
most_similar_index = np.argmax(combined_similarities)
most_similar_sentence = dataset_texts[most_similar_index]

print(f"最相似的句子: {most_similar_sentence}")
print(f"相似度: {combined_similarities[most_similar_index]}")
