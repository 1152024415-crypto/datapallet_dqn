import json
import os
import threading
import time
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta
from enum import Enum
# from sensor_server.mcp_client import MCPClientSession
import shutil
import queue
from sensor_server.wgs84_to_gcj02 import wgs84_to_gcj02
import numpy as np
import pandas as pd
import librosa
from sensor_server.bitmap_to_png import bitmap_to_png
import base64
from typing import List, Dict, Any, Optional, Tuple
import pathlib
# from models import SCENEPHOTO_DIR, PERSON_PICTURE_DIR
# from scene_summary import SceneDivisionProcessor
# from llm_client import LLMClient
from pathlib import Path
import struct
import urllib.parse
# from server import app_server
import glob
from sensor_server.sensor_server import process_sensor_data, DATE_TIME_FORMAT
from datapallet.datapallet import DataPallet, create_datapallet
from datapallet.testbed import TestBed

print("~~~~~~~~~~~~000")

dp = create_datapallet()

# 定义回调函数
def test_callback(data_id: str, value: Any):
    formatted = dp.format_value(data_id, value)
    print(f"回调: {data_id} = {formatted}")

# 设置回调
dp.setup(test_callback)

tb = TestBed(dp)

class SensorHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, code=200):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        
        # 解析headers 若为image则独立处理
        print(self.headers)
        print(f"HTTP headers : {self.headers}")
        content_type = self.headers.get('Content-Type', '')
        # if content_type.startswith('image/'):
        # if content_type == 'application/jpeg':
        #     parse_picture_data(self)
        #     return
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data)
            # print(data) # 调整成save to file
            print(f"Sensor data : {data}")

            if isinstance(data, list):
                swings = {item.get("swing") for item in data}
                if not swings or swings <= {None, ""}:
                    for item in data:
                        processed = process_sensor_data([item])

                        data_id = processed.get("data_id")
                        value = processed.get("value")
                        timestamp_str = processed.get("timestamp")
                        if data_id and value and timestamp_str:
                            timestamp = datetime.strptime(timestamp_str, DATE_TIME_FORMAT)
                            tb.receive_and_transmit_data(data_id, value, timestamp)

                        response = {'status': 'received', 'processed': 'ok'}
                        self._set_headers(200)
                # else:
                #     # 处理swing：faceid & photoid; 并更新到活动日志
                #     process_person_data_from_sensorhub(self, data)
                #     return
            else:
                response = {'status': 'received', 'processed': 'not json list'}
                self._set_headers(400)
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=SensorHTTPRequestHandler, port=8100):
    

    def run_http_server():
        server_address = ('', port)
        print(f"Server address: {server_address}")
        try:
            httpd = server_class(server_address, handler_class)
            print(f'Starting http server on port {port}...')
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt received, shutting down server...")
            finally:
                httpd.shutdown()
                httpd.server_close()
                print("Http server closed\n")
        except Exception as e:
            print(f"Error starting server: {e}")

    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    try:
        http_thread.join()  # 确保主线程等待线程完成
    except KeyboardInterrupt:
        print("\nMain thread KeyboardInterrupt received, shutting down...")
        http_thread.join()

if __name__ == '__main__':
    run()