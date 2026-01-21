"""
HTTP轮询服务器 - 提供动作推荐数据的GET接口

启动方法:
    python http_poll_server.py

API接口:
    GET /latest-recommendation - 获取最新推荐数据
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
from typing import Optional


class RecommendationData:
    """推荐数据存储"""

    def __init__(self):
        self.lock = threading.Lock()
        self.data: Optional[dict] = None

    def update(self, data: dict):
        """更新推荐数据"""
        with self.lock:
            self.data = data

    def get(self) -> Optional[dict]:
        """获取最新推荐数据"""
        with self.lock:
            return self.data.copy() if self.data else None


# 全局数据存储
recommendation_data = RecommendationData()


class HTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP请求处理器"""

    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[{self.address_string()}] {format % args}")

    def do_GET(self):
        """处理GET请求"""
        if self.path == "/latest-recommendation":
            self.handle_get_recommendation()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        """处理POST请求 - 用于更新推荐数据"""
        if self.path == "/update-recommendation":
            self.handle_post_recommendation()
        else:
            self.send_error(404, "Not Found")

    def handle_get_recommendation(self):
        """处理获取推荐数据请求"""
        data = recommendation_data.get()

        if data:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = json.dumps(data, ensure_ascii=False)
            self.wfile.write(response.encode("utf-8"))
            self.log_message(
                "GET /latest-recommendation - 200 OK - Data length: %d", len(response)
            )
        else:
            self.send_response(204)  # No Content
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.log_message("GET /latest-recommendation - 204 No Content")

    def handle_post_recommendation(self):
        """处理更新推荐数据请求"""
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            # 验证数据格式 - 新统一格式
            required_fields = [
                "id",
                "timestamp",
                "action_type",
                "action_name",
                "scene_category",
            ]
            if not all(field in data for field in required_fields):
                self.send_error(400, "Missing required fields")
                self.log_message(
                    "POST /update-recommendation - 400 Bad Request - Missing required fields"
                )
                return

            # 验证action_type字段
            valid_action_types = ["probe", "recommend", "none"]
            if data["action_type"] not in valid_action_types:
                self.send_error(
                    400, f"Invalid action_type. Must be one of: {valid_action_types}"
                )
                self.log_message(
                    "POST /update-recommendation - 400 Bad Request - Invalid action_type: %s",
                    data["action_type"],
                )
                return

            # 确保image字段存在（可为null）
            if "image" not in data:
                data["image"] = None

            # 更新推荐数据
            recommendation_data.update(data)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = json.dumps({"status": "success", "message": "Data updated"})
            self.wfile.write(response.encode("utf-8"))
            self.log_message(
                "POST /update-recommendation - 200 OK - Action: %s (%s)",
                data.get("action_type"),
                data.get("action_name"),
            )

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            self.log_message(
                "POST /update-recommendation - 400 Bad Request - Invalid JSON"
            )
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")
            self.log_message("POST /update-recommendation - 500 - Error: %s", str(e))


def run_server(host="0.0.0.0", port=8002):
    """启动HTTP服务器"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, HTTPRequestHandler)

    print("=" * 60)
    print("HTTP轮询服务器启动 (新统一格式)")
    print("=" * 60)
    print(f"监听地址: http://{host}:{port}")
    print(f"GET 接口: http://{host}:{port}/latest-recommendation")
    print(f"POST 接口: http://{host}:{port}/update-recommendation")
    print("=" * 60)
    print("数据格式 (POST /update-recommendation):")
    print("```json")
    print("{")
    print('  "id": "rec_<timestamp>_001",')
    print('  "timestamp": <unix_timestamp>,')
    print('  "action_type": "probe|recommend|none",')
    print('  "action_name": "动作名称",')
    print('  "scene_category": "场景类别",')
    print('  "image": "data:image/jpeg;base64,..."  # 或 null')
    print("}")
    print("```")
    print("=" * 60)
    print("=" * 60)
    print("服务器运行中... (按 Ctrl+C 停止)")
    print("=" * 60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
