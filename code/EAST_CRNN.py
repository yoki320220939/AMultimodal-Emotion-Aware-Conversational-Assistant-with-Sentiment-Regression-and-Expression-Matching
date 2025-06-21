## 文字识别OCR


import easyocr
import cv2

def ocr_process_and_output(img_path):
    """
    该函数接受图像路径作为输入，使用 EasyOCR 识别图像中的文本，
    绘制文本框，并输出识别到的所有文字，最终将处理后的图像保存并返回。
    
    参数:
    - img_path: 图像的路径
    
    返回:
    - result_text: 识别到的所有文字（以字符串形式返回）
    - processed_img: 处理后的图像（包含文本框和识别的文字）
    """
    
    # 初始化 EasyOCR Reader，选择英文和简体中文
    reader = easyocr.Reader(['en', 'ch_sim'])

    # 加载图像
    img = cv2.imread(img_path)

    # 进行 OCR 识别
    result = reader.readtext(img)

    # 存储所有识别到的文本
    result_text = ""

    # 在图像上绘制识别到的文本框
    for (bbox, text, prob) in result:
        # 只保留文本内容，不包含 "Detected text" 和 "Confidence"
        result_text += f"{text}\n"  # 每个识别到的文本占一行
        
        # 绘制文本框
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        
        # 在文本框内显示识别的文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (top_left[0], top_left[1] - 10), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # 保存处理后的图像
    output_path = 'output_image.png'  # 设置保存路径
    cv2.imwrite(output_path, img)

    print(f"处理后的图像已保存为: {output_path}")

    return result_text, img

# 示例：调用函数并输出识别的文本
img_path = 'test.jpg'  # 替换为你的图片路径
result_text, processed_img = ocr_process_and_output(img_path)

# 输出所有识别到的文本（去除 "Detected text" 和 "Confidence"）
print("识别到的所有文字：")
print(result_text)
