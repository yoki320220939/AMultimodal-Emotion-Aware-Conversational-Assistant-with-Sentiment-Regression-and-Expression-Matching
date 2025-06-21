import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """跨模态注意力融合层（修正维度匹配）"""
    def __init__(self, text_dim=768, image_dim=512, hidden_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, text_feat, image_feat):
        # 投影到相同维度空间
        q = self.text_proj(text_feat).unsqueeze(1)  # [bs, 1, hidden]
        k = v = self.image_proj(image_feat).unsqueeze(1)
        
        # 跨模态注意力
        attn_output, _ = self.attention(q, k, v)
        return attn_output.squeeze(1)  # [bs, hidden]

class FeatureNormalizer(nn.Module):
    """自适应特征归一化层"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mu) / (sigma + 1e-6) + self.beta

class HierarchicalFusion(nn.Module):
    """层级特征融合网络（统一输出维度）"""
    def __init__(self, input_dim, hidden_dims=[512, 512, 512]):  # 保持统一维度
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            self.layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.GELU(),
                FeatureNormalizer(dims[i+1]),
                nn.Dropout(0.3)
            ))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AdvancedMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 跨模态交互组件（输出256维）
        self.cross_attn = CrossModalAttention()
        
        # 双模态特征处理流（统一输出512维）
        self.text_branch = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.image_branch = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        # 多层级融合网络（保持512维）
        self.hierarchical_fusion = HierarchicalFusion(
            input_dim=512 * 2,  # 拼接后的维度
            hidden_dims=[512, 512, 512]  # 统一维度
        )
        
        # 动态门控融合
        self.gate = nn.Linear(512 * 2, 2)
        
        # 特征统一投影层
        self.unify_proj = nn.Linear(512+256, 512)  # 融合512+256维
        
        # 多任务输出头
        self.regression_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 2)  # 回归输出
        )
        self.classification_head = nn.Sequential(
            nn.Linear(512, 5)  # 分类输出
        )

    def forward(self, bert_feat, clip_feat):
        # 跨模态注意力（输出256维）
        attended_feat = self.cross_attn(bert_feat, clip_feat)
        
        # 双模态特征处理（输出512维）
        text_feat = self.text_branch(bert_feat)
        image_feat = self.image_branch(clip_feat)
        
        # 特征拼接
        concat_feat = torch.cat([text_feat, image_feat], dim=-1)
        
        # 层级融合（输出512维）
        fused_feat = self.hierarchical_fusion(concat_feat)
        
        # 门控融合（输出512维）
        gate_weights = torch.softmax(self.gate(concat_feat), dim=-1)
        gated_feat = gate_weights[:, 0:1] * text_feat + gate_weights[:, 1:2] * image_feat
        
        # 统一特征维度
        combined = torch.cat([fused_feat, attended_feat], dim=-1)  # 512+256
        unified_feat = self.unify_proj(combined) + gated_feat  # 最终512维
        
        # 多任务输出
        return {
            "regression": self.regression_head(unified_feat),
            "classification": self.classification_head(unified_feat),
            "attention_weights": gate_weights
        }

def export_onnx():
    model = AdvancedMultimodalModel().eval()
    
    # 创建虚拟输入
    dummy_bert = torch.randn(1, 768)
    dummy_clip = torch.randn(1, 512)
    
    # 导出ONNX
    torch.onnx.export(
        model,
        (dummy_bert, dummy_clip),
        "advanced_fusion.onnx",
        input_names=["bert_features", "clip_features"],
        output_names=["reg_output", "cls_output", "gate_weights"],
        dynamic_axes={
            "bert_features": {0: "batch_size"},
            "clip_features": {0: "batch_size"},
            "reg_output": {0: "batch_size"},
            "cls_output": {0: "batch_size"},
            "gate_weights": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print("ONNX导出成功！")

if __name__ == "__main__":
    export_onnx()