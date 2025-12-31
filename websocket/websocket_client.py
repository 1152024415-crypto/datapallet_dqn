import asyncio
import websockets
import json
import time
import random
from typing import Optional


class WebSocketClient:
    def __init__(self, server_url: str = "ws://localhost:8765", client_id: str = "client"):
        self.server_url = server_url
        self.client_id = client_id
        self.websocket: Optional[websockets] = None
        self.running = False

    async def connect(self):
        """连接到服务器"""
        try:
            print(f"正在连接到 {self.server_url} ...")
            self.websocket = await websockets.connect(self.server_url)

            # 发送注册消息
            register_msg = {
                "type": "register",
                "client_id": self.client_id
            }
            await self.websocket.send(json.dumps(register_msg))

            # 等待欢迎消息
            response = await self.websocket.recv()
            data = json.loads(response)

            if data.get("type") == "welcome":
                print(f"✓ {data['message']}")
                self.running = True
                return True
            else:
                print("注册失败")
                return False

        except Exception as e:
            print(f"连接失败: {e}")
            return False

    async def process_upload_request(self):
        """处理上传请求"""
        print(f"收到上传请求，准备上传数据...")

        # 模拟数据处理延迟
        await asyncio.sleep(2)
        # 发送上传响应
        upload_response = {
            "type": "upload_response",
            "client_id": self.client_id,
            "data": "upload data",
            "timestamp": time.time()
        }

        await self.websocket.send(json.dumps(upload_response))
        print("数据上传完成，等待服务器确认...")

        # 等待服务器确认
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5)
            data = json.loads(response)
            if data.get("type") == "upload_ack":
                print(f"服务器确认: {data.get('message')}")
        except asyncio.TimeoutError:
            print("未收到服务器确认")

    async def listen(self):
        """监听服务器消息"""
        print(f"客户端 {self.client_id} 已就绪，等待服务器指令...")
        print("按 Ctrl+C 退出")

        try:
            while self.running and self.websocket:
                try:
                    # 接收消息
                    message = await self.websocket.recv()
                    data = json.loads(message)

                    if data.get("type") == "upload_request":
                        print(f"\n收到上传请求: {data.get('message')}")
                        await self.process_upload_request()
                    else:
                        print(f"收到消息: {data.get('type')}")

                except websockets.exceptions.ConnectionClosed:
                    print("连接已关闭")
                    break

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            await self.close()

    async def run(self):
        """运行客户端"""
        if not await self.connect():
            return

        await self.listen()

    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            self.running = False
            print("连接已关闭")


# 客户端运行示例
def run_client(client_id: str = "client1"):
    """运行单个客户端"""
    client = WebSocketClient(client_id=client_id)
    asyncio.run(client.run())


if __name__ == "__main__":
    run_client()