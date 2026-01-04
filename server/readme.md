# 通知SA上传图片服务器

## 服务器启动
python server.py

## 模拟客户端（模拟SA）
python client.py

## 如何通知服务器请求SA上传文件
通过/api/trigger_client 请求 通知服务器申请图片上传
接受目录为DEST_DIR = "./destineData" （自定义修改）
```python
import requests

TRIGGER_URL = "http://localhost:8000/api/trigger_client"


def run_test():
    print("[Test] 正在尝试触发服务器...")
    try:
        response = requests.get(TRIGGER_URL)
        print(f"[Test] 服务器响应: {response.json()}")

        if response.json().get("status") == "error":
            print("[Test] 提示：请确保 client.py 已经运行并连接到服务器。")

    except requests.exceptions.ConnectionError:
        print("[Test] 错误：无法连接到服务器。请确保 server.py 正在运行。")


if __name__ == "__main__":
    run_test()
```
