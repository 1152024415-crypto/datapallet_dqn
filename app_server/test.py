"""
简单测试脚本 - 展示如何封装数据并发送到服务器

使用方法:
    python test.py

功能:
    1. 随机生成一个动作数据 (probe或recommend类型)
    2. 使用test.png图片生成base64数据
    3. 发送数据到HTTP服务器

服务器地址: http://127.0.0.1:8080/update-recommendation
"""

import requests
import time
import json
import random
from util import create_test_image_data

# 服务器地址
SERVER_URL = "http://127.0.0.1:8080/update-recommendation"

# 动作定义列表
PROBE_ACTIONS = [
    "QUERY_LOC_NET",  # 通过网络获取位置信息
    "QUERY_LOC_GPS",  # 通过GPS获取高精度位置信息
    "QUERY_VISUAL",  # 查询视觉/图像信息
    "QUERY_SOUND_INTENSITY",  # 查询环境声音强度
    "QUERY_LIGHT_INTENSITY",  # 查询环境光照强度
]

RECOMMEND_ACTIONS = [
    "transit_QR_code",  # 显示公共交通二维码
    "step_count_and_map",  # 显示步数和地图
    "payment_QR_code",  # 显示支付二维码
    "navigation",  # 导航建议
    "audio_record",  # 音频录制
    "relax",  # 放松内容
    "arrived",  # 到达提醒
    "parking",  # 停车管理
]

# 场景类别
SCENE_CATEGORIES = [
    "transportation",  # 交通
    "food",  # 餐饮
    "shopping",  # 购物
    "home",  # 家庭
    "work",  # 工作
]


def create_random_action_data():
    """
    创建随机动作数据

    返回:
        格式化的动作数据字典
    """
    # 随机选择动作类型 (probe或recommend)
    action_type = random.choice(["probe", "recommend"])

    # 根据类型选择具体动作
    if action_type == "probe":
        action_name = random.choice(PROBE_ACTIONS)
    else:  # recommend
        action_name = random.choice(RECOMMEND_ACTIONS)

    # 随机选择场景
    scene_category = random.choice(SCENE_CATEGORIES)

    # 获取当前时间
    current_time = int(time.time())

    # 创建数据
    data = {
        "id": f"rec_{current_time}_{random.randint(100, 999)}",
        "timestamp": current_time,
        "action_type": action_type,
        "action_name": action_name,
        "scene_category": scene_category,
        "image": create_test_image_data("test.png"),  # 使用test.png图片
    }

    return data


def send_data(data, url=SERVER_URL):
    """
    发送数据到服务器

    参数:
        data: 要发送的数据字典
        url: 服务器URL

    返回:
        成功返回True，失败返回False
    """
    try:
        headers = {"Content-Type": "application/json"}

        # 打印发送的数据（简化版，不显示长base64）
        debug_data = data.copy()
        if debug_data.get("image") and len(debug_data["image"]) > 100:
            debug_data["image"] = debug_data["image"][:100] + "...[truncated]"

        print(f"正在发送数据到: {url}")
        print(f"数据内容:")
        print(json.dumps(debug_data, ensure_ascii=False, indent=2))

        # 发送请求
        response = requests.post(url, json=data, headers=headers, timeout=5)

        print(f"服务器响应: {response.status_code} {response.text}")

        if response.status_code == 200:
            print("✓ 数据发送成功")
            return True
        else:
            print(f"✗ 请求失败，状态码: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ 网络请求失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 未知错误: {e}")
        return False


def main():
    """主函数 - 简单发送一个随机动作数据"""
    print("=" * 60)
    print("简单测试脚本 - 动作数据发送")
    print("=" * 60)

    # 创建随机数据
    data = create_random_action_data()

    # 发送数据
    success = send_data(data)

    print("=" * 60)
    if success:
        print("测试完成！请检查HarmonyOS卡片是否收到数据。")
    else:
        print("测试失败！请检查服务器是否运行。")
        print(f"启动服务器: python server.py")


if __name__ == "__main__":
    main()
