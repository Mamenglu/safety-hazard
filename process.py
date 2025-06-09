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

def convert_to_json(text, image_path, ans="_1.jpg", is_pano=False):
    entries = re.split(r'\n\s*\n', text.strip())
    json_data_list = []

    base64_image = image_to_base64(image_path)
    print(f"原图片Base64字符串长度: {len(base64_image)}")

    original_image_path = image_path
    compressed_image_path = "image/" + get_name() + ans
    compression_quality = 40
    compress_image(original_image_path, compressed_image_path, quality=compression_quality)

    base64_image = image_to_base64(compressed_image_path)
    print(f"压缩后图片Base64字符串长度: {len(base64_image)}")

    # 只保留需要数量的 entries
    if is_pano:
        entries = entries[:1]
    else:
        entries = entries[:2]

    for entry in entries:
        lines = entry.split('\n')
        json_data = {
            "dangerTopic": "",
            "dangerType": "",
            "dangerContent": "",
            "dangerLocation": "",
            "measure": "",
            "image": base64_image
        }

        for line in lines:
            if line.startswith('1. 关键字'):
                json_data["dangerTopic"] = line.split('：')[1].strip()
            elif line.startswith('2. 安全隐患类型'):
                json_data["dangerType"] = line.split('：')[1].strip()
            elif line.startswith('3. 安全隐患内容'):
                json_data["dangerContent"] = line.split('：')[1].strip()
            elif line.startswith('4. 安全隐患位置'):
                json_data["dangerLocation"] = line.split('：')[1].strip()
            elif line.startswith('5. 措施'):
                json_data["measure"] = line.split('：')[1].strip()

        if all(json_data[k] for k in ["dangerTopic", "dangerType", "dangerContent", "dangerLocation", "measure"]):
            json_data_list.append(json_data)

    return json_data_list

def analyze_image_security_risks(image_path):
    image = Image.open(image_path)
    messages = [({
        "role": "user",
        "content": [
            {"type": "image"},
            # {"type": "text", "text": "请识别图片中的安全隐患，并提供针对每个隐患的整改建议。每个隐患请严格按照以下格式逐条输出，换行分隔，不要返回重复内容：\n 1. 关键字：xxx\n 2. 安全隐患类型：xxx\n 3. 安全隐患内容：xxx\n 4. 安全隐患位置：xxx\n 5. 措施：xxx\n,每个隐患之间空一行。"}
            {"type": "text", "text": "请对图片中的主要安全隐患进行分析，每个隐患请严格按照以下格式逐条输出，换行分隔，不要添加多余内容：\n1. 关键字：xxx\n2. 安全隐患类型：xxx\n3. 安全隐患内容：xxx\n4. 安全隐患位置：xxx\n5. 措施：xxx\n\n如果有多个隐患请依次列出，每组之间空一行。"}
        ]
    })]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors="pt"
    ).to(next(model.parameters()).device)

    pixel_values = torch.tensor(processor(image).pixel_values).to(next(model.parameters()).device)
    generate_kwargs = {**inputs, "pixel_values": pixel_values}
    output = model.generate(**generate_kwargs, max_new_tokens=1000)
    result = tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    # 获取图片文件名并生成保存结果的路径
    image_name = os.path.basename(image.filename)  # 获取图片文件名
    image_name_without_ext = os.path.splitext(image_name)[0]  # 去掉扩展名
    folder_path = "./results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 生成一个随机的唯一文件名
    random_filename = str(uuid.uuid4())  # 使用uuid生成唯一的文件名
    file_path = os.path.join(folder_path, f"{random_filename}_result.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result)
    return result.strip()

def process_image_and_callback(image_url, callback_url, data):
    try:
        image_type = data.get('image_type', '01')
        results = []

        if image_type == "02" or image_type == 2:
            print("检测到全景图，进行畸变矫正处理...")
            img_raw = distortion_processor.load_insp_image(image_url)
            front, back, _, _ = distortion_processor.process_image(img_raw)

            front_path = f"/tmp/front_{uuid.uuid4()}.jpg"
            back_path = f"/tmp/back_{uuid.uuid4()}.jpg"
            cv2.imwrite(front_path, front)
            cv2.imwrite(back_path, back)

            result_front = analyze_image_security_risks(front_path)
            result_back = analyze_image_security_risks(back_path)

            json1 = convert_to_json(result_front.replace(":", "："), front_path, "_1.jpg", is_pano=True)
            json2 = convert_to_json(result_back.replace(":", "："), back_path, "_2.jpg", is_pano=True)

            final_result = json1 + json2

        else:
            result = analyze_image_security_risks(image_url)
            final_result = convert_to_json(result.replace(":", "："), image_url, is_pano=False)

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

