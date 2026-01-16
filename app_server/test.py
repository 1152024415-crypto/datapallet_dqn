"""
测试脚本 - 向服务器发送动作推荐数据

使用方法:
    python test.py
"""

import requests
import time

# 服务器地址
SERVER_URL = "http://localhost:8080/api/recommendations"

# 测试数据 - 选择其中一个发送
test_data = {
    "id": f"rec_{int(time.time())}_001",
    "timestamp": int(time.time()),
    "active_module": "dqn",
    "rule_engine": {"category": "transit", "decision": "open_qr_code"},
    "dqn": {"action": "transit_QR_code", "type": "recommend"},
    "vlm": {
        "scene_category": "transit",
        "description": "前方100米有地铁站，建议打开乘车码。",
    },
}

# Rule Engine 测试数据
test_data_rule = {
    "id": f"rec_{int(time.time())}_001",
    "timestamp": int(time.time()),
    "active_module": "rule_engine",
    "rule_engine": {
        "category": "transit",
        "decision": "open_qr_code",
        "description": "建议打开乘车码",
    },
}

# DQN 测试数据
test_data_dqn = {
    "id": f"rec_{int(time.time())}_001",
    "timestamp": int(time.time()),
    "active_module": "dqn",
    "rule_engine": {"category": "transit", "decision": "open_qr_code"},
    "dqn": {"action": "transit_QR_code", "type": "recommend"},
}

# VLM 测试数据
test_data_vlm = {
    "id": f"rec_{int(time.time())}_001",
    "timestamp": int(time.time()),
    "active_module": "vlm",
    "rule_engine": {"category": "transit", "decision": "open_qr_code"},
    "dqn": {"action": "transit_QR_code", "type": "recommend"},
    "vlm": {
        "scene_category": "transit",
        "description": "前方100米有地铁站，建议打开乘车码。",
    },
}

# 发送请求
response = requests.post(SERVER_URL, json=test_data_vlm)
print(f"状态码: {response.status_code}")
print(f"响应: {response.json()}")
