import websocket
import requests
import os
import time
from PIL import Image  # 用于获取图片宽高

# 配置
SERVER_WS_URL = "ws://localhost:8000/ws"
SERVER_HTTP_URL = "http://localhost:8000/upload"
IMAGE_PATH = "./sourceData/test.png"  # 请确保该文件存在


def upload_image(ws):
    """读取图片并通过 HTTP 上传"""
    if not os.path.exists(IMAGE_PATH):
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
        headers = {
            "X-Image-Width": str(width),
            "X-Image-Height": str(height)
        }

        files = {
            'file': (os.path.basename(IMAGE_PATH), open(IMAGE_PATH, 'rb'), 'image/png')
        }

        print("[Client] 开始通过 HTTP 上传图片...")
        response = requests.post(SERVER_HTTP_URL, files=files, headers=headers)

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