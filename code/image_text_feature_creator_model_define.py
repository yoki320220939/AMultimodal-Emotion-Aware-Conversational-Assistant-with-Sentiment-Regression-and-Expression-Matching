##构建文字和图像特征预测情感向量（通过多模态预测情感）

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import torch.optim as optim

# === 模型定义 ===
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

        self.residual = nn.Linear(512 + 768, hidden_size)  # 残差分支

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_dim)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        image_feat = self.image_encoder.get_image_features(pixel_values=pixel_values)

        fused = torch.cat((text_feat, image_feat), dim=1)
        x = self.fusion(fused) + self.residual(fused)  # 残差连接
        output = self.regressor(x)
        return output

# === 数据集定义 ===
class MemeDataset(Dataset):
    def __init__(self, data, clip_processor, tokenizer, max_length=128):
        self.data = data
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        text = item["text"]
        label = torch.tensor(item["logits"], dtype=torch.float)

        image_inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        text_inputs = self.tokenizer(text, padding="max_length", truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "label": label
        }

# === 训练函数 ===
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()

    for batch in tqdm(dataloader, desc="训练中"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, pixel_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# === 验证函数 ===
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, pixel_values)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# === 主程序入口 ===
def main():
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models/bert-base-chinese")
    CLIP_DIR = os.path.join(os.path.dirname(__file__), "models/clip-vit-base-patch32")

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    text_encoder = BertModel.from_pretrained(MODEL_DIR)
    clip_model = CLIPModel.from_pretrained(CLIP_DIR)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_DIR)

    model = MultimodalRegressor(text_encoder, clip_model, output_dim=2).cuda()

    with open("train_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("val_data.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_dataset = MemeDataset(train_data, clip_processor, tokenizer)
    val_dataset = MemeDataset(val_data, clip_processor, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(300):
        print(f"\n===== Epoch {epoch + 1} =====")
        train_loss = train(model, train_loader, optimizer, device="cuda")
        val_loss = evaluate(model, val_loader, device="cuda")

        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

        if val_loss < best_val_loss:
            print("🔄 验证集损失下降，保存模型。")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            print(f"⚠️ 验证集损失无提升（{patience_counter}/{patience}）")

        if patience_counter >= patience:
            print("⏹️ 早停触发，训练结束。")
            break

if __name__ == "__main__":
    main()
