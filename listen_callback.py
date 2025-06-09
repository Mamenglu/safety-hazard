import http.server
import socketserver
import json

# 定义请求处理类
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            # 解析接收到的 JSON 数据
            data = json.loads(post_data.decode('utf-8'))
            print("接收到的数据:")
            print(json.dumps(data, indent=4, ensure_ascii=False))

            # 发送响应
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({"message": "数据已接收"}, ensure_ascii=False).encode('utf-8')
            self.wfile.write(response)

        except json.JSONDecodeError:
            # 处理 JSON 解析错误
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({"error": "无效的 JSON 数据"}, ensure_ascii=False).encode('utf-8')
            self.wfile.write(response)

# 定义服务器的端口
PORT = 8899

# 创建服务器实例
with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"开始监听端口 {PORT}...")
    try:
        # 启动服务器
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("服务器已停止")    