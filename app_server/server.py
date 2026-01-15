"""
HarmonyOS智能动作推荐后端服务

功能:
- WebSocket端点供HarmonyOS客户端连接
- HTTP API接收外部系统的动作数据并实时转发给WebSocket客户端
"""

import json
import logging
from aiohttp import web, WSMsgType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 预定义动作列表
PROBE_ACTIONS = {
    "QUERY_LOC_NET",
    "QUERY_LOC_GPS",
    "QUERY_VISUAL",
    "QUERY_SOUND_INTENSITY",
    "QUERY_LIGHT_INTENSITY",
}

RECOMMEND_ACTIONS = {
    "NONE",
    "step_count_and_map",
    "transit_QR_code",
    "train_information",
    "flight_information",
    "payment_QR_code",
    "preferred_APP",
    "glasses_snapshot",
    "identify_person",
    "silent_DND",
    "navigation",
    "audio_record",
    "relax",
    "arrived",
    "parking",
}

ALL_ACTIONS = PROBE_ACTIONS | RECOMMEND_ACTIONS

# 全局WebSocket连接（单客户端）
ws_client = None


async def websocket_handler(request):
    """WebSocket连接处理器 - 单客户端模式，新连接替代旧连接"""
    global ws_client

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # 关闭旧连接
    if ws_client is not None:
        logger.info("新连接到达，关闭旧连接")
        await ws_client.close()

    ws_client = ws
    logger.info("客户端已连接")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.CLOSE:
                break
    finally:
        if ws_client is ws:
            ws_client = None
            logger.info("客户端已断开")

    return ws


async def recommendations_handler(request):
    """HTTP API: 接收动作数据并转发给WebSocket客户端"""
    global ws_client

    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"status": "error", "message": "无效的JSON格式"}, status=400)

    # 验证action字段
    action = data.get("action")
    if action not in ALL_ACTIONS:
        return web.json_response(
            {"status": "error", "message": f"无效的action: {action}"},
            status=400,
        )

    # 检查客户端连接状态
    client_connected = ws_client is not None and not ws_client.closed
    forwarded = False

    if client_connected:
        try:
            await ws_client.send_str(json.dumps(data, ensure_ascii=False))
            forwarded = True
            logger.info(f"数据已转发: {action}")
        except Exception as e:
            logger.error(f"转发失败: {e}")

    return web.json_response({
        "status": "success",
        "message": "数据已接收",
        "client_connected": client_connected,
        "forwarded": forwarded,
    })


def create_app():
    app = web.Application()
    app.router.add_get("/recommendation-stream", websocket_handler)
    app.router.add_post("/api/recommendations", recommendations_handler)
    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("服务器启动: http://localhost:8080")
    logger.info("WebSocket: ws://localhost:8080/recommendation-stream")
    logger.info("HTTP API: POST http://localhost:8080/api/recommendations")
    web.run_app(app, host="0.0.0.0", port=8080)
