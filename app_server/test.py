"""
测试脚本 - 向服务器发送动作推荐数据

使用方法:
    python test.py
"""

import requests
import time

# 服务器地址
SERVER_URL = "http://localhost:8080/api/recommendations"

# 测试数据
test_data = {
    "id": f"rec_{int(time.time())}_001",
    "type": "recommend",
    "action": "transit_QR_code",
    "timestamp": int(time.time()),
    "metadata": {
        "reward": 0.8,
        "description": "前方100米有地铁站，建议打开乘车码。"
    }
}

# 发送请求
response = requests.post(SERVER_URL, json=test_data)
print(f"状态码: {response.status_code}")
print(f"响应: {response.json()}")
