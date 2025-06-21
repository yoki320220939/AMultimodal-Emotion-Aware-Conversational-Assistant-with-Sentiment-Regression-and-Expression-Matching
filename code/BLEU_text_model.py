## 通过BLUE进行文本匹配

import json
import numpy as np
from collections import Counter

epsilon = 1e-10
# 计算 n-gram 精确度
def ngram_precision(candidate, reference, n):
    candidate_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)]
    reference_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]
    
    candidate_ngrams_count = Counter(candidate_ngrams)
    reference_ngrams_count = Counter(reference_ngrams)

    # 计算重叠的 n-gram 数量
    overlap = sum(min(candidate_ngrams_count[ngram], reference_ngrams_count[ngram]) for ngram in candidate_ngrams_count)
    
    # 精确度 = 重叠的 n-gram 数量 / 待匹配句子中 n-gram 的数量
    precision = overlap / len(candidate_ngrams) if candidate_ngrams else 0
    return precision

# 计算 BLEU 分数
def calculate_bleu(candidate, reference, max_n=4):
    # 计算每个 n 的精确度
    precisions = [ngram_precision(candidate, reference, n) for n in range(1, max_n+1)]
    
    # 计算加权平均精确度
    p_avg = np.mean(precisions)
    
    # 计算简洁惩罚（Brevity Penalty）
    candidate_length = len(candidate)
    reference_length = len(reference)
    
    if candidate_length > reference_length:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - reference_length / candidate_length) if candidate_length > 0 else 0
    
    # BLEU = BP * exp( sum( p_n ) )
    bleu_score = brevity_penalty * np.exp(np.sum(np.log(np.maximum(p_avg, epsilon))))
    
    return bleu_score

# 加载数据集
with open("quotes_with_features.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 提取数据集中的文本（参考句子）
dataset_texts = [item["quote"] for item in dataset]

# 输入待匹配的句子
input_sentence = "世上有个什么路来着"  # 替换为你要匹配的句子

# 计算 BLEU 相似度
bleu_scores = []
for quote in dataset_texts:
    reference = quote  # 参考句子
    candidate = input_sentence  # 待匹配句子
    # 计算 BLEU 分数
    bleu_score = calculate_bleu(candidate, reference, max_n=4)
    bleu_scores.append(bleu_score)

# 找到最相似的句子
most_similar_index = np.argmax(bleu_scores)
most_similar_sentence = dataset_texts[most_similar_index]

print(f"最相似的句子: {most_similar_sentence}")
print(f"BLEU 相似度: {bleu_scores[most_similar_index]}")
