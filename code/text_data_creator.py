## 文字数据库构建


import torch
import json
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSequenceClassification

# === 加载BERT模型 ===
text_model_path = "./code/models/bert-base-chinese"  # 你可以使用自己的 BERT 模型
text_tokenizer = BertTokenizer.from_pretrained(text_model_path)
text_model = BertModel.from_pretrained(text_model_path)
text_model.eval()

# === 加载情感模型 ===
emotion_model_path = "./code/models/Erlangshen-Roberta-110M-Sentiment"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
emotion_model.eval()

# 提取文本特征（BERT）
def get_text_features(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()  # 获取 [CLS] token 对应的向量
    return sentence_embedding

# 提取情感特征
def get_emotion_features(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze().tolist()  # 获取情感分类向量（例如，二分类：消极/积极）
    return logits

# 创建数据集，包含名人名言和诗句
# 数据集（名言+诗句+哄人句子）
dataset = [
    # 励志名言
    {"quote": "看似不起波澜的日复一日，一定会在某天让你看到坚持的意义。", "emotion": "积极"},
    {"quote": "你走的每一步都算数，即使慢，也从未后退。", "emotion": "积极"},
    {"quote": "书山有路勤为径，学海无涯苦作舟。", "emotion": "积极"},

    # 哄男朋友
    {"quote": "不要不开心啦，我带你去吃你最喜欢的烧烤！", "emotion": "积极"},
    {"quote": "你皱眉的样子虽然帅，但笑一笑更好看~", "emotion": "积极"},

    # 哄女朋友

    {"quote": "世上本没有路，走的人多了，也便成了路。", "emotion": "中性"},
    {"quote": "抽刀断水水更流，举杯消愁愁更愁。", "emotion": "消极"},
    {"quote": "长风破浪会有时，直挂云帆济沧海。", "emotion": "积极"},
    {"quote": "人生如逆旅，我亦是行人，终将走向虚无。", "emotion": "消极"},
    {"quote": "希望只是痛苦的延长线，终点从未改变。", "emotion": "消极"},
    {"quote": "所有的热情都会冷却，就像所有的光终被黑暗吞噬。", "emotion": "消极"},
    {"quote": "努力不过是命运偶尔施舍的错觉。", "emotion": "消极"},
    {"quote": "生活不是等待暴风雨过去，而是学会在雨中跳舞。", "emotion": "积极"},
    {"quote": "每一个不曾起舞的日子，都是对生命的辜负。", "emotion": "积极"},
    {"quote": "黑暗中最亮的星，往往诞生于最深的夜。", "emotion": "积极"},
    {"quote": "你若决定灿烂，山无遮，海无拦。", "emotion": "积极"},
    {"quote": "世界会向那些有目标和远见的人让路。", "emotion": "积极"},  
    {"quote": "不是因为看到希望才坚持，而是因为坚持才会看到希望。", "emotion": "积极"},  
    {"quote": "生活不会辜负每一个认真努力的人。", "emotion": "积极"},  
    {"quote": "即使慢，驰而不息，纵令落后，纵令失败，但一定可以达到他所向往的目标。", "emotion": "积极"},  
    {"quote": "成功的秘诀就是每天比别人多努力一点。", "emotion": "积极"},  
    {"quote": "没有不可治愈的伤痛，没有不能结束的沉沦。", "emotion": "积极"},  
    {"quote": "你现在的努力，是为了以后有更多的选择。", "emotion": "积极"},  
    {"quote": "生活总是让我们遍体鳞伤，但到后来，那些受伤的地方一定会变成我们最强壮的地方。", "emotion": "积极"},  
    {"quote": "人生没有白走的路，每一步都算数。", "emotion": "积极"},  
    {"quote": "只要路是对的，就不怕路远。", "emotion": "积极"},
    {"quote": "人生不过是一场必输的游戏，只是时间长短不同。", "emotion": "消极"},  
    {"quote": "所有的努力，最终都会被遗忘。", "emotion": "消极"},  
    {"quote": "希望是最大的谎言，它让人在痛苦中挣扎得更久。", "emotion": "消极"},  
    {"quote": "命运从不公平，有人生来就在终点，有人拼尽全力仍在起点。", "emotion": "消极"},  
    {"quote": "时间会带走一切，包括你曾以为的永恒。", "emotion": "消极"},  
    {"quote": "人生最大的悲剧，不是失败，而是从未真正活过。", "emotion": "消极"},  
    {"quote": "所有的相遇，最终都只是擦肩而过。", "emotion": "消极"},  
    {"quote": "梦想不过是现实的止痛药，药效过后，疼痛依旧。", "emotion": "消极"},  
    {"quote": "我们终将孤独地死去，就像我们孤独地活着。", "emotion": "消极"},  
    {"quote": "人生没有意义，只是偶然的存在。", "emotion": "消极"}  
]

# 提取数据集中的特征并保存到新的结构中
new_dataset = []
for item in dataset:
    text_features = get_text_features(item["quote"], text_tokenizer, text_model)
    emotion_features = get_emotion_features(item["emotion"]+item["quote"], emotion_tokenizer, emotion_model)
    
    # 将文本特征和情感特征拼接
    combined_features = text_features + emotion_features
    
    new_item = {
        "quote": item["quote"],
        "emotion": item["emotion"],
        "features": combined_features  # 添加合并后的特征向量
    }
    new_dataset.append(new_item)

# 将新的数据集保存为 JSON 文件
with open("./code/quotes_with_features.json", "w", encoding="utf-8") as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)

print("数据集和特征已保存到 quotes_with_features.json")
