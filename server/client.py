import websocket
import pathlib
import requests
import os
import time
from datetime import datetime
from PIL import Image  # 用于获取图片宽高

# http 头（C++ 客户端参考格式）
# std::map<std::string, std::string> HttpService::GenerateHeaders(const RequestContext& context) const {
#     return {
#         {"User-Agent", "HarmonyOS-RCP-Client/1.0"},
#         {"Accept-Encoding", "gzip, deflate"},
#         {"Accept", "*/*"},
#         {"Connection", "keep-alive"},
#         {"Content-Type", "application/png"},# application/jpeg
#         {"type", "person"},
#         {"photoid", "10001"},  
#         {"width", std::to_string(context.imageInfo.width)},
#         {"height", std::to_string(context.imageInfo.height)},
#         {"Index", "10001"},
#         {"X-Timestamp", GetCurrentTimestamp()}, "YYYY-MM-DD HH:MM:SS"
#         {"Content-Length", std::to_string(context.imageInfo.data.size())}
#     };
# }

# 配置
SERVER_WS_URL = "ws://localhost:8000/ws"
SERVER_HTTP_URL = "http://localhost:8000/upload"
IMAGE_PATH = "./sourceData/test.png"  # 请确保该文件存在


def upload_image(ws):
    """读取图片并通过 HTTP 上传"""
    if not pathlib.Path(IMAGE_PATH).exists():
        print(f"[Client] 错误: 找不到图片文件 {IMAGE_PATH}")
        ws.send("upload_failed")
        return

    try:
        # 1. 获取图片宽高
        with Image.open(IMAGE_PATH) as img:
            width, height = img.size
            print(f"[Client] 图片尺寸: {width}x{height}")

        time.sleep(2)

        # 2. 准备上传
        file_path = pathlib.Path(IMAGE_PATH)
        file_data = file_path.read_bytes()
        
        # 根据文件扩展名确定 Content-Type
        ext = file_path.suffix.lower()
        content_type = "application/jpeg" if ext in [".jpg", ".jpeg"] else "application/png"
        
        # 生成时间戳 (格式: YYYY-MM-DD HH:MM:SS)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建 HTTP 头（模拟 C++ 客户端格式）
        headers = {
            "User-Agent": "HarmonyOS-RCP-Client/1.0",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "*/*",
            "Connection": "keep-alive",
            "Content-Type": content_type,
            "type": "person",
            "photoid": "10001",
            "width": str(width),
            "height": str(height),
            "Index": "10001",
            "X-Timestamp": timestamp,
        }

        print("[Client] 开始通过 HTTP 上传图片...")
        print(f"[Client] Headers: Content-Type={content_type}, width={width}, height={height}, X-Timestamp={timestamp}")
        print(f"[Client] 文件前10字节: {list(file_data[:10])}")
        response = requests.post(SERVER_HTTP_URL, data=file_data, headers=headers)

        # 3. 处理响应
        if response.status_code == 200:
            print(f"[Client] HTTP 上传成功: {response.json()}")
            ws.send("upload_complete")
        else:
            print(f"[Client] HTTP 上传失败: {response.status_code}")
            ws.send("upload_failed")

    except Exception as e:
        print(f"[Client] 发生异常: {e}")
        ws.send("upload_failed")


def on_message(ws, message):
    print(f"[Client] 收到 Server 指令: {message}")
    if message == "upload_request":
        upload_image(ws)


def on_error(ws, error):
    print(f"[Client] WebSocket 错误: {error}")


def on_close(ws, close_status_code, close_msg):
    print("[Client] 连接已关闭")


def on_open(ws):
    print("[Client] 已连接到服务器，等待指令...")


if __name__ == "__main__":
    # 检查图片是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"【警告】请先在 client_data 文件夹下放入 {os.path.basename(IMAGE_PATH)} 图片！")
    else:
        # 开启 WebSocket 长连接
        ws = websocket.WebSocketApp(
            SERVER_WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        # 运行长连接
        ws.run_forever()