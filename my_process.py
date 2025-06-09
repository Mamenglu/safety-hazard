import os
import time
import json
import http.client
import socket
from threading import Thread
from PIL import Image
import uuid
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM
import torch
import re
import cv2
import base64
from distortion_correction import Insta360InspProcessor  # 畸变校正类
import datetime

model_dir = "/home/ubuntu/GLM-Edge/glm-edge-v-5b"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your setup.")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

distortion_processor = Insta360InspProcessor()

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def compress_image(image_path, output_path, quality=50):
    """
    压缩图片并保存到指定路径
    :param image_path: 原始图片路径
    :param output_path: 压缩后的图片保存路径
    :param quality: 压缩质量（范围：1-100，数值越小压缩率越高）
    """
    # 打开原始图片
    with Image.open(image_path) as img:
        # 如果不是 RGB 模式，转换为 RGB 以兼容 JPEG
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 保存压缩后的图片
        img.save(output_path, format="JPEG", quality=quality)
        print(f"图片已压缩并保存到 {output_path}")

def get_name():
    # 获取当前时间
    current_time = datetime.datetime.now()

    # 格式化时间字符串，例如：20250601_123456
    time_str = current_time.strftime('%Y%m%d_%H%M%S')

    # 定义文件名和扩展名
    file_name = f'{time_str}'  # 你可以将 .txt 替换为其他扩展名，如 .csv、.log 等
    return file_name


def convert_to_json(text, image_path, ans="_1.jpg", is_pano=False, lang="ch"):
    """
    从模型文本输出中提取结构化字段，并附加 image 字段。
    支持多语言，不依赖行首数字或 '-' 符号，具备更强鲁棒性。
    """
    import re

    # 1. 压缩图片并转 base64
    original_image_path = image_path
    compressed_image_path = "image/" + get_name() + ans
    compress_image(original_image_path, compressed_image_path, quality=40)
    image_b64 = image_to_base64(compressed_image_path)

    # 2. 多语言字段标签
    field_map = {
        "ch": {
            "dangerTopic": ["关键字"],
            "dangerType": ["安全隐患类型"],
            "dangerContent": ["安全隐患内容"],
            "dangerLocation": ["安全隐患位置"],
            "measure": ["措施"]
        },
        "en": {
            "dangerTopic": ["Keyword"],
            "dangerType": ["Hazard Type"],
            "dangerContent": ["Hazard Description"],
            "dangerLocation": ["Hazard Location"],
            "measure": ["Measures"]
        },
        "ja": {
            "dangerTopic": ["キーワード"],
            "dangerType": ["危険の種類"],
            "dangerContent": ["危険の内容"],
            "dangerLocation": ["危険の場所"],
            "measure": ["対策"]
        },
        "ar": {
            "dangerTopic": ["الكلمة المفتاحية"],
            "dangerType": ["نوع الخطر"],
            "dangerContent": ["وصف الخطر"],
            "dangerLocation": ["موقع الخطر"],
            "measure": ["الإجراءات"]
        }
    }

    if lang not in field_map:
        lang = "ch"
    fm = field_map[lang]

    # 3. 去除多余前缀，如编号或短横线
    # 替换 Markdown 风格序号
    text = re.sub(r'^\s*\d+[\.\、]?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\-•]\s*', '', text, flags=re.MULTILINE)

    # 4. 拆分条目
    entries = re.split(r'\n{2,}', text.strip())
    results = []

    for entry in entries:
        item = {k: "" for k in fm}
        item["image"] = image_b64

        for line in entry.strip().splitlines():
            clean = line.strip()
            for key, labels in fm.items():
                for label in labels:
                    # 匹配 “标签：内容” 或 “标签: 内容”，允许前缀存在空格或破折号
                    m = re.search(rf'{re.escape(label)}\s*[：:]\s*(.+)', clean)
                    if m:
                        item[key] = m.group(1).strip()

        if all(item[k] for k in fm):
            results.append(item)

    return results[:1] if is_pano else results[:2]

def analyze_image_security_risks(image_path, lang="ch"):
    image = Image.open(image_path)

    prompts = {
        "ch": "请对图片中的主要安全隐患进行分析，每个隐患请严格按照以下格式逐条输出，换行分隔，不要添加多余内容：\n关键字：xxx\n安全隐患类型：xxx\n安全隐患内容：xxx\n安全隐患位置：xxx\n措施：xxx\n\n如果有多个隐患请依次列出，每组之间空一行。",
        "en": "Please analyze the main safety hazards in the image. For each hazard, strictly follow this format, one line per item, no extra content:\nKeyword: xxx\nHazard Type: xxx\nHazard Description: xxx\nHazard Location: xxx\nMeasures: xxx\n\nList multiple hazards in order, separated by a blank line.",
        # "ja": "画像に含まれる主な安全上のリスクを分析してください。\n\n【出力形式の厳守】\n- 各リスクについて、以下の形式に従ってください。\n- 項目名の前には記号や番号を付けず、「キーワード：」のようにそのまま書いてください。\n- 「-」や「1.」「①」などの記号は絶対に使わないでください。\n- 各リスクは空行で区切って列挙してください。\n- 出力は日本語でお願いします。\n\n【フォーマット】\nキーワード：xxx\n危険の種類：xxx\n危険の内容：xxx\n危険の場所：xxx\n対策：xxx",
        "ja": "画像に含まれる主な安全上のリスクを分析してください。各リスクについて、以下の形式に従ってください（各項目ごとに改行、余計な内容は不要）：\nキーワード：xxx\n危険の種類：xxx\n危険の内容：xxx\n危険の場所：xxx\n対策：xxx\n\n複数ある場合は、空行で区切って順に列挙してください。すべて日本語で記述してください。",
        "ar": "يرجى تحليل المخاطر الرئيسية في الصورة. لكل خطر، يرجى اتباع التنسيق التالي بدقة، سطر لكل عنصر، دون محتوى إضافي:\nالكلمة المفتاحية: xxx\nنوع الخطر: xxx\nوصف الخطر: xxx\nموقع الخطر: xxx\nالإجراءات: xxx\n\nإذا كانت هناك مخاطر متعددة، يرجى سردها واحدة تلو الأخرى، مفصولة بسطر فارغ."
    }

    prompt = prompts.get(lang, prompts["ch"])

    messages = [({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
    })]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors="pt"
    ).to(next(model.parameters()).device)

    pixel_values = torch.tensor(processor(image).pixel_values).to(next(model.parameters()).device)
    generate_kwargs = {**inputs, "pixel_values": pixel_values}
    output = model.generate(**generate_kwargs, max_new_tokens=1000)
    result = tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    # 保存结果到本地
    image_name = os.path.basename(image.filename)
    folder_path = "./results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    random_filename = str(uuid.uuid4())
    file_path = os.path.join(folder_path, f"{random_filename}_result.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result)

    return result.strip()

def process_image_and_callback(image_url, callback_url, data):
    try:
        image_type = data.get('image_type', '01')
        lang = data.get('lang', 'ch')  # 默认中文

        results = []

        if image_type == "02" or image_type == 2:
            print("检测到全景图，进行畸变矫正处理...")
            img_raw = distortion_processor.load_insp_image(image_url)
            front, back, _, _ = distortion_processor.process_image(img_raw)

            front_path = f"/tmp/front_{uuid.uuid4()}.jpg"
            back_path = f"/tmp/back_{uuid.uuid4()}.jpg"
            cv2.imwrite(front_path, front)
            cv2.imwrite(back_path, back)

            result_front = analyze_image_security_risks(front_path, lang)
            result_back = analyze_image_security_risks(back_path, lang)

            json1 = convert_to_json(result_front.replace(":", "："), front_path, "_1.jpg", is_pano=True, lang=lang)
            json2 = convert_to_json(result_back.replace(":", "："), back_path, "_2.jpg", is_pano=True, lang=lang)

            final_result = json1 + json2

        else:
            result = analyze_image_security_risks(image_url, lang)
            final_result = convert_to_json(result.replace(":", "："), image_url, is_pano=False, lang=lang)

        # 回调逻辑不变...

        response_data = json.dumps(final_result, ensure_ascii=False)
        # print("回调数据：", response_data)
        debug_result = []
        for entry in final_result:
            debug_entry = entry.copy()
            if 'image' in debug_entry:
                debug_entry['image'] = f"<{type(debug_entry['image']).__name__}>"
            debug_result.append(debug_entry)

        print("回调数据：", json.dumps(debug_result, ensure_ascii=False, indent=2))

        headers = {'Content-Type': 'application/json'}
        ip_address = callback_url.split("//")[1].split(":")[0]
        port = int(callback_url.split(":")[2].split("/")[0])
        callback_path = "/" + "/".join(callback_url.split("/")[3:])

        conn = http.client.HTTPConnection(ip_address, port)
        try:
            conn.request("POST", callback_path, body=response_data.encode('utf-8'), headers=headers)
            res = conn.getresponse()
            print("回调成功:", res.read().decode("utf-8"))
        finally:
            conn.close()

    except Exception as e:
        print(f"错误: {str(e)}")

# 监听并处理新文件
def process_new_json_files():
    folder_path = "./ans"
    processed_files = set()
    print(f"开始监听 {folder_path} 文件夹...")

    while True:
        try:
            files = os.listdir(folder_path)
            json_files = [f for f in files if f.endswith('.json')]

            for json_file in json_files:
                if json_file in processed_files:
                    continue

                file_path = os.path.join(folder_path, json_file)
                print(f"正在处理文件: {json_file}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                image_url = data.get('image_path')
                callback_url = data.get('callback_url')
                # lang = data.get('lang')
                if image_url and callback_url:
                    process_image_and_callback(image_url, callback_url, data)
                    processed_files.add(json_file)

                os.remove(file_path)
                print(f"文件 {json_file} 处理完毕，已删除")

        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
        time.sleep(1)

def start_file_watcher():
    print("启动文件夹监控线程...")
    thread = Thread(target=process_new_json_files)
    thread.daemon = True
    thread.start()
    while True:
        time.sleep(100)

if __name__ == '__main__':
    start_file_watcher()
