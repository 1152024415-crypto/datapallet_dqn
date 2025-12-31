import asyncio
import websockets
import json
import time


class WebSocketClient:
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.running = False

    async def connect(self):
        """连接服务器"""
        print(f"正在连接到 {self.server_url} ...")
        try:
            self.websocket = await websockets.connect(self.server_url)
            print("连接成功!")
            self.running = True
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    async def listen_for_notifications(self):
        """监听服务器通知"""
        if not self.websocket:
            print("未连接到服务器")
            return

        print("开始监听服务器通知...")

        while self.running:
            try:
                # 等待服务器通知
                print("等待服务器通知...")
                notification = await self.websocket.recv()
                print(f"收到原始消息: {notification[:100]}...")

                try:
                    data = json.loads(notification)
                    if data.get("type") == "upload_request":
                        print(f"\n收到服务器通知: {data['message']}")

                        # 发送文本
                        text_to_send = f"这是客户端在 {time.strftime('%H:%M:%S')} 发送的测试文本。内容包含一些测试数据用于验证通信。"
                        await self.websocket.send(text_to_send)
                        print(f"已发送文本，长度: {len(text_to_send)} 字符")

                        # 等待服务器响应
                        response = await self.websocket.recv()
                        response_data = json.loads(response)
                        print(f"服务器响应: {response_data['message']}\n")
                except json.JSONDecodeError:
                    print(f"收到非JSON消息: {notification[:100]}...")

            except websockets.exceptions.ConnectionClosed:
                print("连接已关闭")
                self.running = False
                break
            except Exception as e:
                print(f"接收消息时出错: {e}")
                await asyncio.sleep(1)  # 出错时等待1秒

    async def run(self):
        """运行客户端（持续运行）"""
        if not await self.connect():
            return

        print("客户端已启动，等待服务器通知...")
        print("按 Ctrl+C 退出")

        try:
            await self.listen_for_notifications()
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            await self.close()

    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            print("连接已关闭")


if __name__ == "__main__":
    client = WebSocketClient()
    asyncio.run(client.run())