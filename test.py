import requests

# 使用你提供的图片URL
image_url = 'https://mml-1352988614.cos.ap-guangzhou.myqcloud.com/IMG_20201026_154628_00_069.insp'
# 回调URL，这里假设为一个示例URL，你可以根据实际情况修改
callback_url = 'http://127.0.0.1:8899//receive_callback'

data = {
    'image_url': image_url,
    'callback_url': callback_url,
    "image_type": "02"
}

response = requests.post('http://101.33.66.24:8080//process-image', json=data)
print(response.json())