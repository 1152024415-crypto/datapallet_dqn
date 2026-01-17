"""
测试脚本 - 通过HTTP POST更新推荐数据

使用方法:
    python test.py
"""

import requests
import time

# 服务器地址 - 使用本地IP或localhost
HAP_SERVER_URL = "http://127.0.0.1:8080/update-recommendation"

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


# 发送HTTP POST请求
def send_data(data):
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(HAP_SERVER_URL, json=data, headers=headers, timeout=5)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return False


# 连续发送多条测试数据
if __name__ == "__main__":
    print(f"正在发送数据到: {HAP_SERVER_URL}")
    print("=" * 50)

    # 发送Rule Engine数据
    print("\n1. 发送 Rule Engine 数据:")
    success = send_data(test_data_rule)
    if success:
        print("✓ Rule Engine 数据发送成功")
    else:
        print("✗ Rule Engine 数据发送失败")

    time.sleep(3)

    # 发送DQN数据
    print("\n2. 发送 DQN 数据:")
    success = send_data(test_data_dqn)
    if success:
        print("✓ DQN 数据发送成功")
    else:
        print("✗ DQN 数据发送失败")

    time.sleep(3)

    # 发送VLM数据
    print("\n3. 发送 VLM 数据:")
    success = send_data(test_data_vlm)
    if success:
        print("✓ VLM 数据发送成功")
    else:
        print("✗ VLM 数据发送失败")

    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n请检查HAP应用的Widget卡片是否正确显示数据")
    print("请查看hilog日志确认HTTP请求是否被正确处理")
