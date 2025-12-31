import asyncio
from time import sleep

import websockets
import json
from typing import Set
import time


class WebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.client_handlers = {}  # 存储客户端处理任务

    async def send_notification_to_clients(self, exclude_client=None):
        """向所有客户端发送上传通知（可排除特定客户端）"""
        if not self.clients:
            print("没有客户端连接")
            return False

        notification = {
            "type": "upload_request",
            "message": "请上传文本",
            "timestamp": time.time()
        }

        message = json.dumps(notification)
        success_count = 0

        for client in list(self.clients):
            # 排除特定客户端（比如触发脚本）
            if client == exclude_client:
                continue

            try:
                await client.send(message)
                success_count += 1
                print(f"已向客户端发送通知")
            except Exception as e:
                print(f"发送通知失败: {e}")
                continue

        print(f"已向 {success_count}/{len(self.clients)} 个客户端发送通知")
        return success_count > 0

    async def handle_client(self, websocket):
        """处理客户端连接"""
        client_id = id(websocket)
        self.clients.add(websocket)
        print(f"新客户端连接，ID: {client_id}，当前连接数: {len(self.clients)}")

        try:
            # 监听客户端消息
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # 如果是控制消息
                    if data.get("type") == "control":
                        print(f"收到控制消息: {data}")
                        if data.get("command") == "notify_all":
                            await self.send_notification_to_clients(exclude_client=websocket)
                            sleep(1)
                            await websocket.send(json.dumps({"status": "ok", "message": "已触发通知"}))
                        continue
                except:
                    pass

                # 普通文本消息
                print(f"收到客户端 {client_id} 消息，长度: {len(message)} 字符")

                # 发送确认响应
                response = {
                    "status": "ok",
                    "message": f"已接收 {len(message)} 字符",
                    "received_at": time.time()
                }
                await websocket.send(json.dumps(response))

        except websockets.exceptions.ConnectionClosed:
            print(f"客户端 {client_id} 断开连接")
        except Exception as e:
            print(f"处理客户端 {client_id} 时出错: {e}")
        finally:
            self.clients.remove(websocket)
            print(f"客户端 {client_id} 移除，剩余连接数: {len(self.clients)}")

    async def start(self):
        """启动服务器"""
        print(f"WebSocket 服务器启动在 ws://{self.host}:{self.port}")
        print("等待客户端连接...")

        # 启动WebSocket服务器
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    # 单独运行服务端
    server = WebSocketServer()
    asyncio.run(server.start())