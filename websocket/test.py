"""
简化的触发脚本 - 只发送控制消息，不干扰正常客户端
"""

import asyncio
import websockets
import json
import time


async def trigger_notifications():
    """触发服务器向所有客户端发送通知"""
    print("触发服务器向所有客户端发送通知...")

    try:
        # 连接到服务器
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("已连接到服务器")

            # 发送控制消息
            control_msg = {
                "type": "control",
                "command": "notify_all",
                "timestamp": time.time(),
                "source": "trigger_script"
            }

            print("发送控制消息...")
            await websocket.send(json.dumps(control_msg))

            # 等待服务器响应
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                response_data = json.loads(response)
                print(f"服务器响应: {response_data}")

                if response_data.get("status") == "ok":
                    print("✓ 触发成功！服务器已向所有客户端发送通知")
                else:
                    print("✗ 触发失败")

            except asyncio.TimeoutError:
                print("✗ 服务器未响应（可能不支持控制消息）")

            # 保持连接一小会儿
            await asyncio.sleep(1)

    except ConnectionRefusedError:
        print("✗ 无法连接到服务器，请确保服务器正在运行")
    except Exception as e:
        print(f"✗ 触发失败: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("WebSocket 通知触发器")
    print("=" * 50)
    asyncio.run(trigger_notifications())