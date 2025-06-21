import streamlit as st
from PIL import Image
import os
from datetime import datetime
import json
import torch
import easyocr
import cv2
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

# 文本处理相关
from transformers import (
    BertTokenizer, 
    BertModel,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    MarianTokenizer, 
    MarianMTModel,
    BlipProcessor, 
    BlipForConditionalGeneration,
    CLIPProcessor, 
    CLIPModel
)

# 机器学习相关
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

##### 有一个我们懒得修的bug,无法通过图片匹配输出重复的图片
class MultimodalRegressor(nn.Module):
    def __init__(self, text_encoder, image_encoder, hidden_size=512, output_dim=2):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
        self.fusion = nn.Sequential(
            nn.Linear(512 + 768, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.residual = nn.Linear(512 + 768, hidden_size)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_dim)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values = pixel_values.to(device)
        
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        image_feat = self.image_encoder.get_image_features(pixel_values=pixel_values)

        fused = torch.cat((text_feat, image_feat), dim=1)
        x = self.fusion(fused) + self.residual(fused)
        output = self.regressor(x)
        return output
    
# class MultimodalRegressor(nn.Module):
#     def __init__(self, text_encoder, image_encoder, hidden_size=512, output_dim=2):
#         super().__init__()
#         self.text_encoder = text_encoder
#         self.image_encoder = image_encoder

#         self.fusion = nn.Sequential(
#             nn.Linear(512 + 768, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#         )

#         self.residual = nn.Linear(512 + 768, hidden_size)  # 残差分支

#         self.regressor = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size // 2, output_dim)
#         )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        image_feat = self.image_encoder.get_image_features(pixel_values=pixel_values)

        fused = torch.cat((text_feat, image_feat), dim=1)
        x = self.fusion(fused) + self.residual(fused)  # 残差连接
        output = self.regressor(x)
        return output

# 1. 表情包匹配器实现
class MemeMatcher:
    def __init__(self, data_path):
        """初始化表情包匹配器"""
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 初始化情感分析模型
        self.tokenizer = AutoTokenizer.from_pretrained("./code/models/Erlangshen-Roberta-110M-Sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("./code/models/Erlangshen-Roberta-110M-Sentiment")
        self.model.eval()
    
    def bleu_match(self, query, top_n=5):
        """基于BLEU分数的内容匹配"""
        scores = []
        for item in self.data:
            score = self._calculate_bleu(query, item['content'])
            scores.append((item, score))
        
        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scores[:top_n]]
    
    def emotion_match(self, query, top_n=5):
        """基于情感向量的匹配"""
        query_logits = self._get_emotion_features(query)
        
        similarities = []
        for item in self.data:
            sim = cosine_similarity([query_logits], [item['logits']])[0][0]
            similarities.append((item, sim))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in similarities[:top_n]]
    
    def _get_emotion_features(self, text):
        """获取文本情感特征"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.logits.squeeze().tolist()
    
    def _calculate_bleu(self, candidate, reference, max_n=4):
        """计算BLEU分数"""
        def ngram_precision(cand, ref, n):
            cand_ngrams = [tuple(cand[i:i+n]) for i in range(len(cand)-n+1)]
            ref_ngrams = [tuple(ref[i:i+n]) for i in range(len(ref)-n+1)]
            
            cand_count = Counter(cand_ngrams)
            ref_count = Counter(ref_ngrams)
            
            overlap = sum(min(cand_count[ngram], ref_count[ngram]) for ngram in cand_count)
            return overlap / len(cand_ngrams) if cand_ngrams else 0
        
        precisions = [ngram_precision(candidate, reference, n) for n in range(1, max_n+1)]
        p_avg = np.mean(precisions)
        
        # 简洁惩罚
        brevity_penalty = 1 if len(candidate) > len(reference) else np.exp(1 - len(reference)/len(candidate))
        
        return brevity_penalty * np.exp(np.sum(np.log(np.maximum(p_avg, 1e-10))))

# 初始化匹配器（缓存以提高性能）
@st.cache_resource
def load_matcher():
    return MemeMatcher("./code/labeled_data_logits.json")

# 增强的消息保存函数
def save_message(sender, text=None, image=None, image_path=None):
    """保存消息到本地（支持直接图片和图片路径）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理直接上传的图片
    if image:
        image_path = f"chat_messages/{timestamp}_{sender}.png"
        image.save(image_path)
    
    # 自动填充默认文本
    if (image or image_path) and (text is None or not text.strip()):
        text = "分享了一张图片"
    
    with open("chat_messages/chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"{timestamp}|{sender}|{text or ''}|{image_path or ''}\n")

# 设置页面标题和图标
st.set_page_config(
    page_title="GuGu的双人聊天室",
    page_icon="💬",
    layout="centered"
)

# 创建保存消息的目录
if not os.path.exists("chat_messages"):
    os.makedirs("chat_messages")

# 加载所有模型（放在全局，避免重复加载）
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 文本匹配模型
    text_model_path = "./code/models/bert-base-chinese"
    text_tokenizer = BertTokenizer.from_pretrained(text_model_path)
    text_model = BertModel.from_pretrained(text_model_path).to(device)
    text_model.eval()

    # 情感分析模型
    emotion_model_path = "./code/models/Erlangshen-Roberta-110M-Sentiment"
    emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path).to(device)
    emotion_model.eval()
    
    # 翻译模型
    translation_model_path = "./code/models/opus-mt-en-zh_model"
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_path)
    translation_model = MarianMTModel.from_pretrained(translation_model_path).to(device)
    
    # 图片描述模型
    image_caption_path = "./code/models/blip-image-captioning-base"
    image_processor = BlipProcessor.from_pretrained(image_caption_path)
    image_model = BlipForConditionalGeneration.from_pretrained(image_caption_path).to(device)
    
    # 多模态模型组件
    clip_model_path = "./code/models/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_model.eval()
    
    # 多模态情感分析模型
    multimodal_model_path = "./code/models/multimodal-regressor"
    multimodal_model = MultimodalRegressor(
        text_encoder=text_model,  # 重用已有的BERT模型
        image_encoder=clip_model,
        output_dim=2
    ).to(device)
    # 加载预训练权重
    if os.path.exists(os.path.join(multimodal_model_path, "best_model.pt")):
        multimodal_model.load_state_dict(
            torch.load(
                os.path.join(multimodal_model_path, "best_model.pt"),
                map_location="cuda" if torch.cuda.is_available() else "cpu"
            )
        )
    multimodal_model.eval()

    # 加载数据集
    with open("./code/quotes_with_features.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    return {
        "text_tokenizer": text_tokenizer,
        "text_model": text_model,
        "emotion_tokenizer": emotion_tokenizer,
        "emotion_model": emotion_model,
        "translation_tokenizer": translation_tokenizer,
        "translation_model": translation_model,
        "image_processor": image_processor,
        "image_model": image_model,
        "clip_processor": clip_processor,
        "clip_model": clip_model,
        "multimodal_model": multimodal_model,
        "dataset": dataset,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

models = load_models()

def analyze_multimodal_sentiment(text=None, image=None):
    """
    多模态情感分析
    参数:
        text: 文本内容 (可选)
        image: PIL.Image对象 (可选)
    返回:
        (negative, positive) 情感logits
    """
    device = models["device"]
    model = models["multimodal_model"].to(device)
    
    # 处理文本输入
    if text:
        text_inputs = models["text_tokenizer"](
            text, return_tensors="pt", 
            truncation=True, padding=True, 
            max_length=128
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
    else:
        # 如果没有文本，使用空文本
        input_ids = torch.tensor([[0]]).to(device)  # [CLS] token
        attention_mask = torch.tensor([[1]]).to(device)
    
    # 处理图像输入
    if image:
        try:
            image = image.convert("RGB")
            image_inputs = models["clip_processor"](
                images=image, return_tensors="pt"
            )
            pixel_values = image_inputs["pixel_values"].to(device)
        except Exception as e:
            st.error(f"图像处理失败: {str(e)}")
            pixel_values = torch.zeros(1, 3, 224, 224).to(device)  
    else:
        pixel_values = torch.zeros(1, 3, 224, 224).to(device) 
    # 运行模型
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
    
    negative, positive = outputs.squeeze().tolist()
    return negative, positive

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ch_sim', 'en'])  # 中英文识别

# 修改 ocr_extract_text() 函数
def ocr_extract_text(image):
    reader = load_ocr_reader()  # 使用缓存的reader
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = reader.readtext(img_cv)
    return "\n".join([text for (_, text, _) in results])

# 情感分析功能
def analyze_sentiment(texts):
    """
    分析多条文本的情感
    返回: (negative_logits, positive_logits)
    """
    negative_logits = []
    positive_logits = []
    
    for text in texts:
        inputs = models["emotion_tokenizer"](text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = models["emotion_model"](**inputs)
            logits = outputs.logits.squeeze().tolist()
            negative_logits.append(logits[0])  # 负面情感logits
            positive_logits.append(logits[1])  # 正面情感logits
    
    return negative_logits, positive_logits

def plot_sentiment_curve(text_ids, negative_logits, positive_logits):
    """
    绘制情感曲线图
    :param text_ids: 文本ID列表
    :param negative_logits: 负面情感logits列表
    :param positive_logits: 正面情感logits列表
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 绘制曲线
    ax.plot(text_ids, negative_logits, 'b-', label='Negative emotions', linewidth=2)
    ax.plot(text_ids, positive_logits, 'r-', label='Positive emotion', linewidth=2)
    
    # 填充曲线之间的区域
    ax.fill_between(text_ids, negative_logits, positive_logits, 
                    where=np.array(positive_logits)>=np.array(negative_logits), 
                    facecolor='red', alpha=0.1)
    ax.fill_between(text_ids, negative_logits, positive_logits, 
                    where=np.array(positive_logits)<np.array(negative_logits), 
                    facecolor='blue', alpha=0.1)
    
    # 设置图表属性
    ax.set_title('GuGu sentiment analysis curve', pad=20)
    ax.set_ylabel('Emotional intensity')
    ax.legend(loc='upper right')
    
    # 隐藏横坐标标签
    ax.set_xticks([])
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

# BLEU匹配功能
epsilon = 1e-10

def ngram_precision(candidate, reference, n):
    candidate_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)]
    reference_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]
    
    candidate_ngrams_count = Counter(candidate_ngrams)
    reference_ngrams_count = Counter(reference_ngrams)

    overlap = sum(min(candidate_ngrams_count[ngram], reference_ngrams_count[ngram]) for ngram in candidate_ngrams_count)
    precision = overlap / len(candidate_ngrams) if candidate_ngrams else 0
    return precision

def calculate_bleu(candidate, reference, max_n=4):
    precisions = [ngram_precision(candidate, reference, n) for n in range(1, max_n+1)]
    p_avg = np.mean(precisions)
    
    candidate_length = len(candidate)
    reference_length = len(reference)
    
    if candidate_length > reference_length:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - reference_length / candidate_length) if candidate_length > 0 else 0
    
    bleu_score = brevity_penalty * np.exp(np.sum(np.log(np.maximum(p_avg, epsilon))))
    return bleu_score

def find_similar_by_bleu(input_sentence, top_n=3):
    dataset_texts = [item["quote"] for item in models["dataset"]]
    
    bleu_scores = []
    for quote in dataset_texts:
        bleu_score = calculate_bleu(input_sentence, quote, max_n=4)
        bleu_scores.append(bleu_score)
    
    top_indices = np.argsort(bleu_scores)[-top_n:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            "text": dataset_texts[idx],
            "score": bleu_scores[idx],  # BLEU分数
            "similarity": bleu_scores[idx]  # 也作为相似度存储
        })
    
    return results

# 翻译功能
def translate_text(text):
    device = models["device"]
    inputs = models["translation_tokenizer"]([text], return_tensors="pt", padding=True).to(device)
    translated = models["translation_model"].generate(**inputs)
    translated_text = models["translation_tokenizer"].decode(translated[0], skip_special_tokens=True)
    return translated_text

# 图片描述功能
def generate_image_description(image):
    device = models["device"] 
    try:
        # 将图片转换为RGB格式
        image = image.convert("RGB")
        # 生成描述
        inputs = models["image_processor"](images=image, return_tensors="pt").to(device)
        models["image_model"].to(device)
        with torch.no_grad():
            out = models["image_model"].generate(**inputs).to(device)
        
        description = models["image_processor"].decode(out[0], skip_special_tokens=True)
        
        # 将英文描述翻译成中文
        chinese_description = translate_text(description)
        
        return {
            "english": description,
            "chinese": chinese_description
        }
    except Exception as e:
        st.error(f"生成图片描述时出错: {str(e)}")
        return None

# 提取文本特征（BERT）
def get_text_features(sentence):
    device = models["device"]  # 获取模型所在的设备
    inputs = models["text_tokenizer"](
        sentence, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    ).to(device)  # 将输入张量移动到正确设备
    
    with torch.no_grad():
        outputs = models["text_model"](**inputs)
        sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return sentence_embedding

# 提取情感特征
def get_emotion_features(sentence):
    device = models["device"]
    inputs = models["emotion_tokenizer"](
        sentence, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = models["emotion_model"](**inputs)
        logits = outputs.logits.squeeze().tolist()
    return logits

# 查找最相似的多个句子
def find_similar_sentences(input_sentence, top_n=3):
    dataset_features = [item["features"] for item in models["dataset"]]
    dataset_texts = [item["quote"] for item in models["dataset"]]

    input_text_features = get_text_features(input_sentence)
    input_emotion_features = get_emotion_features(input_sentence)

    text_similarities = []
    for features in dataset_features:
        text_features = features[:len(input_text_features)]
        similarity = cosine_similarity([input_text_features], [text_features])[0][0]
        text_similarities.append(similarity)

    emotion_similarities = []
    for features in dataset_features:
        emotion_features = features[len(input_text_features):]
        similarity = cosine_similarity([input_emotion_features], [emotion_features])[0][0]
        emotion_similarities.append(similarity)

    text_weight = 0.3
    emotion_weight = 0.7

    combined_similarities = [
        text_weight * text_similarity + emotion_weight * emotion_similarity
        for text_similarity, emotion_similarity in zip(text_similarities, emotion_similarities)
    ]

    top_indices = np.argsort(combined_similarities)[-top_n:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            "text": dataset_texts[idx],
            "similarity": combined_similarities[idx],  # 相似度
            "score": combined_similarities[idx]  # 也作为分数存储
        })
    
    return results

def save_message(sender, text=None, image=None):
    """保存消息到本地（兼容图片对象和图片路径）
    参数:
        sender: 发送者名称
        text: 消息文本（可选）
        image: 可以是以下两种形式之一:
               - PIL.Image 对象（直接上传的图片）
               - str 图片路径（表情包路径）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    image_path = None
    if isinstance(image, str):  # 处理图片路径
        if os.path.exists(image):
            image_path = image
        else:
            st.error(f"图片文件不存在: {image}")
            return
    elif image is not None:  # 处理上传的图片对象
        try:
            image_path = f"chat_messages/{timestamp}_{sender}.png"
            image.save(image_path)
        except Exception as e:
            st.error(f"保存图片失败: {str(e)}")
            return
    
    # 如果没有文本但有图片，填充默认文本
    if image_path and (text is None or not text.strip()):
        text = "分享了一张图片"
    
    # 写入聊天记录
    with open("chat_messages/chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"{timestamp}|{sender}|{text or ''}|{image_path or ''}\n")

def load_messages():
    """加载所有聊天消息"""
    messages = []
    if os.path.exists("chat_messages/chat_history.txt"):
        with open("chat_messages/chat_history.txt", "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.strip().split("|")
                if len(parts) == 4:
                    timestamp, sender, text, image_path = parts
                    messages.append({
                        "time": datetime.strptime(timestamp, "%Y%m%d_%H%M%S"),
                        "sender": sender,
                        "text": text,
                        "image": image_path if image_path != "" else None
                    })
    return messages

def display_messages():
    """显示所有聊天消息（中文界面）"""
    messages = load_messages()
    messages.sort(key=lambda x: x["time"])
    
    # 初始化选中内容相关状态
    if "selected_items" not in st.session_state:
        st.session_state.selected_items = []  # 存储选中内容的字典列表
    if "item_ids" not in st.session_state:
        st.session_state.item_ids = []
    
    for idx, msg in enumerate(messages):
        # 判断是否为对方发送的消息
        is_other_user = msg["sender"] != st.session_state.current_user
        
        # 消息样式设置
        message_align = "left" if is_other_user else "right"
        message_color = "lightgray" if is_other_user else "lightblue"
        
        # 显示消息
        st.markdown(f"""
        <div style="text-align: {message_align}; margin: 5px;">
            <div style="display: inline-block; background-color: {message_color}; 
                        padding: 8px 12px; border-radius: 12px; max-width: 70%;">
                <div style="font-size: 0.8em; color: gray;">{msg['sender']}</div>
                {msg['text'] if msg['text'] else ''}
            </div>
            <div style="font-size: 0.7em; color: gray; margin-top: 2px;">
                {msg['time'].strftime("%H:%M")}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示图片（如果有）
        if msg["image"] and os.path.exists(msg["image"]):
            try:
                image = Image.open(msg["image"])
                st.image(image, width=200)
                
                # 图片下方添加操作按钮
                col1, col2, col3 = st.columns([1, 1, 1])  # 改为3列
                with col1:
                    # 只为对方发送的图片添加选择框
                    if is_other_user:
                        image_id = f"image_{idx}_{msg['time'].timestamp()}"
                        selected = st.checkbox(
                            "分析图片情感",
                            key=f"select_{image_id}",
                            help="勾选以分析此图片情感"
                        )
                        
                        if selected:
                            if image_id not in st.session_state.item_ids:
                                st.session_state.selected_items.append({
                                    "type": "image",
                                    "content": image,
                                    "text": ""  # 图片没有文本内容
                                })
                                st.session_state.item_ids.append(image_id)
                        else:
                            if image_id in st.session_state.item_ids:
                                index = st.session_state.item_ids.index(image_id)
                                st.session_state.selected_items.pop(index)
                                st.session_state.item_ids.pop(index)
                
                with col2:
                    # 添加OCR按钮（所有图片都可OCR）
                    if st.button("识别图中文字", key=f"ocr_{idx}_{msg['time'].timestamp()}"):
                        ocr_text = ocr_extract_text(image)
                        st.session_state.current_ocr_text = ocr_text
                        st.session_state.show_ocr_dialog = True
                        st.rerun()
                
                with col3:  # 新增图片描述按钮
                    if st.button("生成图片描述", key=f"desc_{idx}_{msg['time'].timestamp()}"):
                        description = generate_image_description(image)
                        if description:
                            st.session_state.current_description = description
                            st.session_state.show_description_dialog = True
                            st.rerun()
                        
            except Exception as e:
                st.error(f"加载图片失败: {str(e)}")
        
        # 只为对方发送的文本消息添加选择框
        if is_other_user and msg["text"] and msg["text"].strip():
            text_id = f"text_{idx}_{msg['time'].timestamp()}"
            selected = st.checkbox(
                "选择此消息", 
                key=f"select_{text_id}",
                help="勾选以分析此消息情感"
            )
            
            if selected:
                if text_id not in st.session_state.item_ids:
                    st.session_state.selected_items.append({
                        "type": "text",
                        "content": msg["text"],
                        "image": None  # 文本没有图片
                    })
                    st.session_state.item_ids.append(text_id)
            else:
                if text_id in st.session_state.item_ids:
                    index = st.session_state.item_ids.index(text_id)
                    st.session_state.selected_items.pop(index)
                    st.session_state.item_ids.pop(index)
    
    # 添加分析按钮（中文界面）
    if st.session_state.selected_items:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("分析选中内容", help="分析已选内容和图片的情感趋势"):
                st.session_state.show_sentiment_analysis = True
                st.rerun()
        with col2:
            if st.button("清除选择", help="清除所有已选内容"):
                st.session_state.selected_items = []
                st.session_state.item_ids = []
                st.rerun()



def main():
    # 初始化会话状态（新增配图相关状态）
    if "current_user" not in st.session_state:
        st.session_state.current_user = "用户A"
    if "show_gugu_dialog" not in st.session_state:
        st.session_state.show_gugu_dialog = False
    if "gugu_results" not in st.session_state:
        st.session_state.gugu_results = []
    if "show_translation_dialog" not in st.session_state:
        st.session_state.show_translation_dialog = False
    if "show_description_dialog" not in st.session_state:
        st.session_state.show_description_dialog = False
    if "current_description" not in st.session_state:
        st.session_state.current_description = None
    if "show_bleu_dialog" not in st.session_state:
        st.session_state.show_bleu_dialog = False
    if "show_sentiment_analysis" not in st.session_state:
        st.session_state.show_sentiment_analysis = False
    if "selected_texts" not in st.session_state:
        st.session_state.selected_texts = []
    if "text_ids" not in st.session_state:
        st.session_state.text_ids = []
    if "last_message_type" not in st.session_state:
        st.session_state.lastmessage_type = None
    if "show_ocr_dialog" not in st.session_state:
        st.session_state.show_ocr_dialog = False
    if "current_ocr_text" not in st.session_state:
        st.session_state.current_ocr_text = ""
    # 新增配图相关状态
    if "show_meme_dialog" not in st.session_state:
        st.session_state.show_meme_dialog = False
    if "preview_meme" not in st.session_state:
        st.session_state.preview_meme = None
    if "selected_meme_idx" not in st.session_state:
        st.session_state.selected_meme_idx = 0
    if "meme_match_mode" not in st.session_state:
        st.session_state.meme_match_mode = "内容匹配(BLEU)"

    # 加载匹配器
    matcher = load_matcher()

    # 页面标题
    st.title("💬 双人聊天室")
    
    # 用户选择
    st.sidebar.title("用户设置")
    user = st.sidebar.radio("当前用户", ["用户A", "用户B"])
    st.session_state.current_user = user
    
    # 聊天记录显示区域
    st.subheader("聊天记录")
    display_messages()
    
    # 消息输入区域 - 使用独立的form
    with st.form(key="message_form"):
        st.subheader("发送消息")
        
        text_input = st.text_area("输入消息", height=100, key="text_input")
        image_upload = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"], key="image_upload")
        
        # 按钮布局（从4列扩展为5列）
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        with col1:
            submitted = st.form_submit_button("发送")
        with col2:
            gugu_speak = st.form_submit_button("GuGu说话")
        with col3:
            gugu_translate = st.form_submit_button("GuGu翻译")
        with col4:
            bleu_match = st.form_submit_button("BLEU匹配")
        with col5:
            meme_btn = st.form_submit_button("GuGu配图")  # 新增按钮
            
            
        if submitted:
            img_obj = None
            if image_upload:
                img_obj = Image.open(image_upload)
            
            # 自动处理文本和图片的组合
            save_message(
                st.session_state.current_user,
                text=text_input if text_input and text_input.strip() else None,
                image=img_obj
            )
            st.rerun()
        
        if gugu_speak and text_input:
            results = find_similar_sentences(text_input, top_n=3)
            st.session_state.gugu_results = results
            st.session_state.show_gugu_dialog = True
            st.rerun()
        
        if gugu_translate and text_input:
            st.session_state.show_translation_dialog = True
            st.rerun()
            
        if bleu_match and text_input:
            results = find_similar_by_bleu(text_input, top_n=3)
            st.session_state.gugu_results = results
            st.session_state.show_bleu_dialog = True
            st.rerun()
        
        if meme_btn and text_input:  # 新增配图按钮处理
            st.session_state.show_meme_dialog = True
            st.rerun()
    
    # BLEU匹配对话框
    if st.session_state.show_bleu_dialog:
        with st.container():
            st.subheader("BLEU匹配结果")
            
            selected_index = st.radio(
                "匹配结果:",
                options=range(len(st.session_state.gugu_results)),
                format_func=lambda i: f"{st.session_state.gugu_results[i]['text']} (BLEU分数: {st.session_state.gugu_results[i]['score']:.4f})"
            )
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("确认发送", key="bleu_confirm"):
                    save_message(st.session_state.current_user,
                               text=st.session_state.gugu_results[selected_index]["text"])
                    st.session_state.show_bleu_dialog = False
                    st.rerun()
            with col2:
                if st.button("取消", key="bleu_cancel"):
                    st.session_state.show_bleu_dialog = False
                    st.rerun()

    # GuGu配图对话框（新增）
    if st.session_state.show_meme_dialog:
        with st.container():
            st.subheader("GuGu表情包匹配")
            
            # 匹配模式选择
            st.session_state.meme_match_mode = st.radio(
                "选择匹配模式",
                ["内容匹配(BLEU)", "情感匹配"],
                horizontal=True,
                index=0 if st.session_state.meme_match_mode == "内容匹配(BLEU)" else 1
            )
            
            # 获取匹配结果
            if st.session_state.meme_match_mode == "内容匹配(BLEU)":
                results = matcher.bleu_match(st.session_state.text_input, top_n=5)
            else:
                results = matcher.emotion_match(st.session_state.text_input, top_n=5)
            
            # 显示匹配结果
            st.session_state.selected_meme_idx = st.selectbox(
                "选择要发送的表情包",
                range(len(results)),
                format_func=lambda i: f"{results[i]['filename']} (分数: {results[i].get('score', results[i].get('similarity', 0)):.2f})"
            )
            
        # 操作按钮
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("预览"):
                st.session_state.preview_meme = f"./code/emo/{results[st.session_state.selected_meme_idx]['filename']}"
        with col2:
            if st.button("确认发送"):
                # 关键修改点：将image_path改为image
                meme_path = f"./code/emo/{results[st.session_state.selected_meme_idx]['filename']}"
                save_message(
                    st.session_state.current_user,
                    text=st.session_state.text_input,
                    image=meme_path  # 使用image参数传递路径
                )
                st.session_state.show_meme_dialog = False
                st.session_state.preview_meme = None
                st.rerun()
        with col3:
            if st.button("取消"):
                st.session_state.show_meme_dialog = False
                st.session_state.preview_meme = None
                st.rerun()
        
        # 预览区域
        if st.session_state.preview_meme:
            st.image(st.session_state.preview_meme, width=300)
            

    # OCR结果对话框（简化版）
    if st.session_state.show_ocr_dialog:
        with st.container():
            st.subheader("文字识别结果")
            
            # 显示识别文本（可复制）
            st.text_area(
                "识别到的文字",
                st.session_state.current_ocr_text,
                height=200,
                key="ocr_result_area"
            )
            
            if st.button("关闭", key="close_ocr"):
                st.session_state.show_ocr_dialog = False
                st.rerun()

    # GuGu说话确认对话框
    if st.session_state.show_gugu_dialog:
        with st.container():
            st.subheader("请选择要发送的消息")
            
            selected_index = st.radio(
                "GuGu猜你想说:",
                options=range(len(st.session_state.gugu_results)),
                format_func=lambda i: f"{st.session_state.gugu_results[i]['text']} (相似度: {st.session_state.gugu_results[i]['similarity']:.2f})"
            )
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("确认发送"):
                    # 修改这里：使用当前用户身份而不是"GuGu"
                    save_message(st.session_state.current_user, 
                               text=st.session_state.gugu_results[selected_index]["text"])
                    st.session_state.show_gugu_dialog = False
                    st.rerun()
            with col2:
                if st.button("取消"):
                    st.session_state.show_gugu_dialog = False
                    st.rerun()
    
    # 翻译对话框
    if st.session_state.show_translation_dialog:
        with st.container():
            st.subheader("翻译结果")
            
            try:
                translated_text = translate_text(st.session_state.text_input)
                st.markdown(f"**原文**: {st.session_state.text_input}")
                st.markdown(f"**翻译**: {translated_text}")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("发送翻译结果"):
                        # 修改这里：使用当前用户身份而不是"GuGu"
                        save_message(st.session_state.current_user,
                                   text=f"翻译: {translated_text}")
                        st.session_state.show_translation_dialog = False
                        st.rerun()
                with col2:
                    if st.button("取消翻译"):
                        st.session_state.show_translation_dialog = False
                        st.rerun()
            except Exception as e:
                st.error(f"翻译出错: {str(e)}")
                if st.button("关闭"):
                    st.session_state.show_translation_dialog = False
                    st.rerun()

    # 图片描述对话框
    if st.session_state.show_description_dialog and st.session_state.current_description:
        with st.container():
            st.subheader("GuGu图片描述")
            
            desc = st.session_state.current_description
            st.markdown(f"**英文描述**: {desc['english']}")
            st.markdown(f"**中文翻译**: {desc['chinese']}")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("发送描述"):
                    # 修改这里：使用当前用户身份而不是"GuGu"
                    save_message(st.session_state.current_user,
                               text=f"图片描述: {desc['chinese']}")
                    st.session_state.show_description_dialog = False
                    st.rerun()
            with col2:
                if st.button("取消"):
                    st.session_state.show_description_dialog = False
                    st.rerun()

    # 情感分析对话框（移动到这里！）
    if st.session_state.show_sentiment_analysis and st.session_state.selected_items:
        with st.container():
            st.subheader("多模态情感分析结果")
            
            # 分析选中的内容
            negative_logits = []
            positive_logits = []
            descriptions = []
            
            for item in st.session_state.selected_items:
                if item["type"] == "text":
                    # 纯文本分析
                    neg, pos = analyze_multimodal_sentiment(text=item["content"])
                    descriptions.append(f"文本: {item['content']}")
                else:
                    # 图片分析 - 生成描述
                    desc = generate_image_description(item["content"])
                    if desc:
                        # 使用图片描述和图片本身进行多模态分析
                        neg, pos = analyze_multimodal_sentiment(
                            text=desc["english"], 
                            image=item["content"]
                        )
                        descriptions.append(f"图片描述: {desc['chinese']}")
                    else:
                        # 仅使用图片分析
                        neg, pos = analyze_multimodal_sentiment(image=item["content"])
                        descriptions.append("图片(无法生成描述)")
                
                negative_logits.append(neg)
                positive_logits.append(pos)
            
            # 绘制曲线图（图像内部英文）
            st.write("情感趋势图：")
            fig = plot_sentiment_curve(
                range(len(st.session_state.selected_items)),
                negative_logits,
                positive_logits
            )
            st.pyplot(fig)
            
            # 显示详细数据（中文界面）
            st.write("详细分析结果：")
            for i, (desc, neg, pos) in enumerate(zip(descriptions, negative_logits, positive_logits)):
                st.markdown(f"""
                **内容 {i+1}**:  
                {desc}  
                负面情感强度: {neg:.2f}  
                正面情感强度: {pos:.2f}
                """)
            
            if st.button("返回", help="返回聊天界面"):
                st.session_state.show_sentiment_analysis = False
                st.rerun()


if __name__ == "__main__":
    # 设置环境变量
    os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 忽略特定警告
    import warnings
    warnings.filterwarnings("ignore", message="Using a slow image processor")
    
    main()