import os
import shutil
import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, UploadFile, File, Header, HTTPException
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
        file: UploadFile = File(...),
        x_image_width: Optional[str] = Header(None),
        x_image_height: Optional[str] = Header(None)
):
    """
    HTTP 上传接口
    """
    print(f"[Server-HTTP] 收到上传请求. 文件名: {file.filename}, 宽: {x_image_width}, 高: {x_image_height}")

    try:
        file_location = os.path.join(DEST_DIR, f"received_{file.filename}")

        # 保存文件
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[Server-HTTP] 文件已保存至: {file_location}")
        return {"status": "success", "filename": file.filename}

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
    global active_connection, upload_result_future
    
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
        return {"status": result, "message": "Upload completed" if result == "upload_complete" else "Upload failed"}
    except asyncio.TimeoutError:
        print(f"[Server-API] 等待上传结果超时（{timeout}秒）")
        return {"status": "timeout", "message": f"Upload result timeout after {timeout}s"}
    finally:
        upload_result_future = None


if __name__ == "__main__":
    # 启动服务，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)