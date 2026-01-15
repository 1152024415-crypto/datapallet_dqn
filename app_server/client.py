"""
模拟HarmonyOS客户端 - 接收服务器推送的动作数据

使用方法:
    python client.py
"""

import asyncio
import websockets

SERVER_URL = "ws://localhost:8080/recommendation-stream"

async def main():
    print(f"连接到 {SERVER_URL}")
    async with websockets.connect(SERVER_URL) as ws:
        print("已连接，等待数据...")
        async for message in ws:
            print(f"收到数据: {message}")

if __name__ == "__main__":
    asyncio.run(main())
