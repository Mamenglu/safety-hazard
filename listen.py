import os
import json
import uuid
import requests
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image


app = Flask(__name__)


# 保存请求数据为 JSON 文件
def save_request_data(data):
    try:
        folder_path = "./ans"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 使用 uuid 生成一个唯一的文件名
        random_filename = str(uuid.uuid4())  # 使用 uuid 生成唯一文件名
        filepath = os.path.join(folder_path, f"{random_filename}.json")

        # 将数据写入 JSON 文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        return {"error": str(e)}

    return filepath


# 下载图片并保存到本地文件夹
def download_image(image_url):
    try:
        # 确保图片保存文件夹存在
        picture_folder = "./picture"
        if not os.path.exists(picture_folder):
            os.makedirs(picture_folder)

        # 下载图片
        response = requests.get(image_url)
        if response.status_code == 200:
            # 获取图片的文件名
            image_name = os.path.basename(image_url)
            image_path = os.path.join(picture_folder, image_name)

            # 保存图片
            with open(image_path, 'wb') as f:
                f.write(response.content)

            return image_path  # 返回图片的保存路径
        else:
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


# 接收HTTP请求并处理图片的接口
# http://43.163.124.119:8080//process-image
@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        data = request.json
        image_url = data.get('image_url')
        callback_url = data.get('callback_url')

        if not image_url or not callback_url:
            return jsonify({'error': 'image_url and callback_url are required'}), 400

        # 下载图片并获取保存的路径
        image_path = download_image(image_url)
        if not image_path:
            return jsonify({'error': 'Failed to download image'}), 500

        # 将图片路径添加到数据
        data['image_path'] = image_path

        # 保存数据为 JSON 文件
        save_request_data(data)

        return jsonify({ "status": "Image processed and result sent to callback"}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
