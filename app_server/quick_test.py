"""
快速测试HTTP轮询服务器

使用方法:
    python quick_test.py
"""

import requests
import json
import time

SERVER_URL = "http://127.0.0.1:8080"

# 测试数据
test_data = {
    "id": f"rec_{int(time.time())}_001",
    "timestamp": int(time.time()),
    "active_module": "dqn",
    "rule_engine": {
        "category": "transit",
        "decision": "open_qr_code",
        "description": "建议打开乘车码",
    },
    "dqn": {"action": "transit_QR_code", "type": "recommend"},
}

print("=" * 60)
print("HTTP轮询服务器快速测试")
print("=" * 60)

# 1. 测试POST更新数据
print("\n1. 测试 POST /update-recommendation")
try:
    response = requests.post(
        f"{SERVER_URL}/update-recommendation", json=test_data, timeout=5
    )
    print(f"   状态码: {response.status_code}")
    print(f"   响应: {response.text}")
    if response.status_code == 200:
        print("   ✓ POST 成功")
    else:
        print("   ✗ POST 失败")
except Exception as e:
    print(f"   ✗ POST 错误: {e}")
    exit(1)

# 2. 测试GET获取数据
print("\n2. 测试 GET /latest-recommendation")
try:
    response = requests.get(f"{SERVER_URL}/latest-recommendation", timeout=5)
    print(f"   状态码: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   数据: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}...")
        print("   ✓ GET 成功")
    elif response.status_code == 204:
        print("   暂无数据 (204)")
    else:
        print("   ✗ GET 失败")
except Exception as e:
    print(f"   ✗ GET 错误: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
