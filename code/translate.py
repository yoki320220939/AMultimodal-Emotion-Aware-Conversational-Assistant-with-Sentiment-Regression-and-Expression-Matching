## 翻译功能部分

from transformers import MarianMTModel, MarianTokenizer

def load_model(model_path):
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    return tokenizer, model

def translate(text, tokenizer, model):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def main():
    model_path = "./models/opus-mt-en-zh_model"  # 指向本地文件夹
    tokenizer, model = load_model(model_path)

    input_text = input("请输入要翻译的英文句子：")
    output_text = translate(input_text, tokenizer, model)

    print(f"翻译结果：{output_text}")

if __name__ == "__main__":
    main()
