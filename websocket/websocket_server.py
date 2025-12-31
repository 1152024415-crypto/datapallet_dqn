import asyncio
import websockets
import json
from typing import Dict, Set, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor


class WebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets] = {}
        self.connected_client_ids: Set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def start(self, run_async: bool = False):
        """
        启动服务器

        Args:
            run_async: 如果为True，返回异步任务，不阻塞
                     如果为False，阻塞运行直到服务器停止
        """
        if run_async:
            # 异步启动，不阻塞
            asyncio.create_task(self._start_async())
        else:
            # 同步启动，阻塞当前线程
            asyncio.run(self._start_async())

    async def _start_async(self):
        """异步启动服务器"""
        self._loop = asyncio.get_running_loop()
        print(f"WebSocket 服务器启动在 ws://{self.host}:{self.port}")
        print("等待客户端连接...")

        async with websockets.serve(self._handle_client, self.host, self.port):
            await asyncio.Future()

    async def _handle_client(self, websocket):
        """处理客户端连接"""
        client_id = None

        try:
            # 等待客户端注册
            message = await websocket.recv()
            data = json.loads(message)

            if data.get("type") != "register":
                await websocket.close(code=1008, reason="First message must be registration")
                return

            client_id = data.get("client_id")
            if not client_id:
                await websocket.close(code=1008, reason="Client ID required")
                return

            # 检查是否已连接
            if client_id in self.clients:
                await websocket.close(code=1008, reason="Client ID already connected")
                return

            # 注册客户端
            self.clients[client_id] = websocket
            self.connected_client_ids.add(client_id)
            print(f"客户端 {client_id} 连接成功，当前连接数: {len(self.clients)}")

            # 发送欢迎消息
            welcome_msg = {
                "type": "welcome",
                "client_id": client_id,
                "message": f"客户端 {client_id} 已连接",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(welcome_msg))

            # 监听客户端消息
            async for message in websocket:
                data = json.loads(message)

                if data.get("type") == "upload_response":
                    print(f"收到客户端 {client_id} 上传的数据: {data.get('data', {})}")

                    # 发送确认消息
                    ack_msg = {
                        "type": "upload_ack",
                        "message": "数据接收成功",
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(ack_msg))

                elif data.get("type") == "heartbeat":
                    # 心跳包，保持连接活跃
                    pass

        except websockets.exceptions.ConnectionClosed:
            print(f"客户端 {client_id or 'unknown'} 断开连接")
        except Exception as e:
            print(f"处理客户端 {client_id or 'unknown'} 时出错: {e}")
        finally:
            # 清理客户端连接
            if client_id:
                self.clients.pop(client_id, None)
                self.connected_client_ids.discard(client_id)
                print(f"客户端 {client_id} 已移除，剩余连接数: {len(self.clients)}")

    async def request_upload_async(self, client_id: str) -> bool:
        """
        异步请求客户端上传数据

        Args:
            client_id: 客户端ID

        Returns:
            bool: 请求是否成功发送
        """
        if not self._loop:
            print("错误: 服务器未启动")
            return False

        if client_id not in self.clients:
            print(f"警告: 客户端 {client_id} 未连接")
            return False

        websocket = self.clients[client_id]

        upload_request = {
            "type": "upload_request",
            "message": "请上传数据",
            "timestamp": time.time(),
            "request_id": str(time.time())  # 简单请求ID
        }

        try:
            await websocket.send(json.dumps(upload_request))
            print(f"已向客户端 {client_id} 发送上传请求")
            return True
        except Exception as e:
            print(f"向客户端 {client_id} 发送请求失败: {e}")
            # 清理断开连接的客户端
            self.clients.pop(client_id, None)
            self.connected_client_ids.discard(client_id)
            return False

    def request_upload(self, client_id: str) -> bool:
        """
        同步请求客户端上传数据（线程安全）

        Args:
            client_id: 客户端ID

        Returns:
            bool: 请求是否成功发送
        """
        if not self._loop:
            print("错误: 服务器未启动")
            return False

        # 在线程池中执行异步调用
        future = asyncio.run_coroutine_threadsafe(
            self.request_upload_async(client_id),
            self._loop
        )

        try:
            return future.result(timeout=5)  # 5秒超时
        except Exception as e:
            print(f"请求客户端 {client_id} 上传数据时出错: {e}")
            return False

    def get_connected_clients(self) -> list:
        """获取已连接的客户端ID列表"""
        return list(self.connected_client_ids)

    def is_client_connected(self, client_id: str) -> bool:
        """检查客户端是否已连接"""
        return client_id in self.connected_client_ids

    async def stop(self):
        """停止服务器"""
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        print("服务器已停止")


class Trigger:
    """触发服务器向客户端请求上传数据的简单接口"""

    def __init__(self, server: WebSocketServer):
        self.server = server

    def trigger_upload(self, client_id: str) -> bool:
        """
        触发客户端上传数据

        Args:
            client_id: 客户端ID

        Returns:
            bool: 触发是否成功
        """
        print(f"触发客户端 {client_id} 上传数据...")
        return self.server.request_upload(client_id)

    def trigger_all(self) -> Dict[str, bool]:
        """
        触发所有已连接的客户端上传数据

        Returns:
            Dict[str, bool]: 每个客户端的触发结果
        """
        results = {}
        clients = self.server.get_connected_clients()

        print(f"触发 {len(clients)} 个客户端上传数据...")

        for client_id in clients:
            results[client_id] = self.trigger_upload(client_id)

        return results