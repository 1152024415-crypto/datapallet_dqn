import os
import shutil
import asyncio
import pathlib
from typing import Optional
from fastapi import FastAPI, WebSocket, Request, Header, HTTPException
import uvicorn

app = FastAPI()

# 存放接收文件的目录
DEST_DIR = "./destineData"
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# 全局变量，用于存储当前连接的 WebSocket 客户端
# 注意：生产环境中应该使用 ConnectionManager 类来管理多个连接
active_connection: Optional[WebSocket] = None

# 用于等待上传结果的 Future
upload_result_future: Optional[asyncio.Future] = None

DATE_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
_timestamp = None
_image_path = None
_upload_complete_callback = None

def set_upload_complete_callback(callback_func):
    """设置上传完成的回调函数"""
    global _upload_complete_callback
    _upload_complete_callback = callback_func

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_connection, upload_result_future
    await websocket.accept()
    active_connection = websocket
    print(f"[Server] 客户端已连接: {websocket.client}")

    try:
        while True:
            # 持续监听客户端发来的消息 (例如: upload_complete)
            data = await websocket.receive_text()
            print(f"[Server] 收到客户端消息: {data}")

            if data == "upload_complete":
                print("[Server] 流程结束：图片上传成功。")
                # 通知等待的 trigger_client 请求
                if upload_result_future and not upload_result_future.done():
                    upload_result_future.set_result("upload_complete")
            elif data == "upload_failed":
                print("[Server] 流程结束：图片上传失败。")
                # 通知等待的 trigger_client 请求
                if upload_result_future and not upload_result_future.done():
                    upload_result_future.set_result("upload_failed")

    except Exception as e:
        print(f"[Server] 客户端断开连接: {e}")
    finally:
        active_connection = None


@app.post("/upload")
async def upload_file(
        request: Request,
        content_type: Optional[str] = Header(None),
        x_timestamp: Optional[str] = Header(None),
        width: Optional[str] = Header(None),
        height: Optional[str] = Header(None)
):
    """
    HTTP 上传接口（接收二进制数据）
    - Content-Type: application/png 或 application/jpeg
    - X-Timestamp: 时间戳 (格式: YYYY-MM-DD HH:MM:SS)
    - width/height: 图片尺寸
    """
    # 读取请求体的原始二进制数据
    file_data = await request.body()
    
    # 根据 Content-Type 确定文件扩展名
    if content_type and "jpeg" in content_type.lower() or "jpg" in content_type.lower():
        ext = ".jpg"
    else:
        ext = ".png"
    
    # 根据时间戳生成文件名 (IMG_YYYY_MM_DD_HH_MM_SS.ext)
    if x_timestamp:
        # 将 "YYYY-MM-DD HH:MM:SS" 转换为 "YYYY_MM_DD_HH_MM_SS"
        time_str = x_timestamp.replace("-", "_").replace(":", "_").replace(" ", "_")
        filename = f"IMG_{time_str}{ext}"
    else:
        # 使用当前时间作为备用
        from datetime import datetime
        time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"IMG_{time_str}{ext}"
    
    global _image_path
    global _timestamp
    _timestamp = '_'.join(time_str.split('_')[:3]) + ' ' + '_'.join(time_str.split('_')[3:])
     
    print(f"[Server-HTTP] 收到上传请求. 文件名: {filename}, 宽: {width}, 高: {height}, 数据大小: {len(file_data)} bytes")

    try:
        file_location = pathlib.Path(DEST_DIR).joinpath(filename)
        _image_path = file_location

        # 保存二进制数据
        file_location.write_bytes(file_data)

        print(f"[Server-HTTP] 文件已保存至: {file_location}")
        return {"status": "success", "filename": filename}

    except Exception as e:
        print(f"[Server-HTTP] 保存文件失败: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")


@app.get("/api/trigger_client")
async def trigger_client(timeout: float = 30.0):
    """
    供 test.py 调用，触发服务器向客户端发送 upload_request
    会等待上传完成后才返回结果
    
    Args:
        timeout: 等待上传结果的超时时间（秒），默认30秒
    """
    global active_connection, upload_result_future, _image_path, _timestamp, _upload_complete_callback
    
    if not active_connection:
        print("[Server-API] 失败：没有连接的 WebSocket 客户端")
        return {"status": "error", "message": "No active client connected"}
    
    # 创建 Future 用于等待上传结果
    upload_result_future = asyncio.get_event_loop().create_future()
    
    # 向客户端发送上传请求
    await active_connection.send_text("upload_request")
    print("[Server-API] 已向客户端发送 'upload_request' 指令，等待上传结果...")
    
    try:
        # 等待上传结果（带超时）
        result = await asyncio.wait_for(upload_result_future, timeout=timeout)
        print(f"[Server-API] 收到上传结果: {result}")
        if _image_path and _timestamp and _upload_complete_callback:
            _upload_complete_callback(_image_path, _timestamp)
            _image_path, _timestamp = None, None
        return {"status": result, "message": "Upload completed" if result == "upload_complete" else "Upload failed"}
    except asyncio.TimeoutError:
        print(f"[Server-API] 等待上传结果超时（{timeout}秒）")
        return {"status": "timeout", "message": f"Upload result timeout after {timeout}s"}
    finally:
        upload_result_future = None

def is_client_connected() -> bool:
    """检查客户端是否已连接"""
    return active_connection is not None


@app.get("/api/client_status")
async def client_status():
    """查询客户端连接状态"""
    connected = is_client_connected()
    return {"connected": connected}


def run_server():
    """运行 FastAPI 服务器"""
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    # 启动服务，端口 8001
    run_server()