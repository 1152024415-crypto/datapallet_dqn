#test.py
from websocket_server import WebSocketServer, Trigger
import time
import threading
#=================================================
# 使用示例
def main():
    # 创建服务器
    server = WebSocketServer()

    def run_server():
        server.start(run_async=False)

    # 启动服务器线程
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    # 等待服务器启动
    time.sleep(2)

    # 创建触发器
    trigger = Trigger(server)

    print("服务器已启动，等待客户端连接...")
    print("按 Ctrl+C 停止")

    try:
        # 示例：定期检查并触发客户端

        while True:
            # 检查连接状态
            clients = server.get_connected_clients()
            if clients:
                print(f"当前连接客户端: {clients}")

                # 触发第一个客户端上传数据
                if clients:
                    success = trigger.trigger_upload(clients[0])
                    if success:
                        print(f"已成功触发客户端 {clients[0]}")
            else:
                print("等待客户端连接...")

            # 每10秒检查一次
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n正在停止服务器...")
        # 在实际应用中，这里应该优雅地停止服务器
        print("服务器已停止")

#=================================================
if __name__ == "__main__":
    main()