import os
import shutil
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_connection
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
            elif data == "upload_failed":
                print("[Server] 流程结束：图片上传失败。")

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
async def trigger_client():
    """
    供 test.py 调用，触发服务器向客户端发送 upload_request
    """
    global active_connection
    if active_connection:
        await active_connection.send_text("upload_request")
        print("[Server-API] 已向客户端发送 'upload_request' 指令")
        return {"status": "command_sent"}
    else:
        print("[Server-API] 失败：没有连接的 WebSocket 客户端")
        return {"status": "error", "message": "No active client connected"}


if __name__ == "__main__":
    # 启动服务，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)