## 通过图片生成描述

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 设置模型路径为你下载的本地路径
model_path = "./models/blip-image-captioning-base"  # 这里替换成你自己的模型路径

# 加载处理器和模型
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

# 将模型加载到设备（CPU 或 GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 加载图片
img_path = "test.jpg"  # 替换为你本地图片的路径
img = Image.open(img_path).convert("RGB")

# 生成描述
inputs = processor(images=img, return_tensors="pt").to(device)
out = model.generate(**inputs)
description = processor.decode(out[0], skip_special_tokens=True)

# 输出图像描述
print("生成的图像描述：", description)
