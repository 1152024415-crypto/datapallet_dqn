import os
import pathlib
import random
import threading
import time
from datetime import datetime
from tkinter.font import names
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import signal
import sys
import threading
import time

import requests

from sensor_server.sensor_server import set_on_get_sensor_data_callback, run_sensor_server
from datapallet.datapallet import create_datapallet
from datapallet.testbed import TestBed, set_on_image_request_callback
from server.test import run_test
from server.server import set_upload_complete_callback, run_server
from sceneclassify.deploy_server import run_scene_classify_server
from sceneclassify.predict_scene_type import predict_scene
from dqn_engine.dqn_realtime_adapter import DQNEngineAdapter
from app_server.util import create_test_image_data
from dqn_engine.constants import PROBE_ACTIONS
from datapallet.enums import SceneType, SceneData
from app_server.server import run_server as run_app_server

APP_UI_SERVER_URL = "http://127.0.0.1:8002/update-recommendation"


class ApplicationManager:
    """应用管理器，统一管理所有服务和组件"""
    
    def __init__(self):
        self.dp = None
        self.tb = None
        self.threads = []
        self.running = False
        self.callbacks_registered = False
        self.dqn_engine = None
        self.inference_trigger_event = threading.Event()
        self.last_activity_mode = None
        self.is_querying_visual = False
        self.visual_query_lock = threading.Lock()

    def _map_scene_to_category(self, scene_val) -> str:
        if not scene_val:
            return "unknown"

        sType = scene_val.scene_type if isinstance(scene_val, SceneData) else scene_val

        mapping = {
            SceneType.MEETINGROOM: "meetingroom",
            SceneType.WORKSPACE: "work",
            SceneType.DINING: "food",
            SceneType.OUTDOOR_PARK: "transportation",
            SceneType.SUBWAY_STATION: "transportation",
            SceneType.OTHER: "home",
            SceneType.NULL: "unknown"
        }
        return mapping.get(sType, "meetingroom")

    def send_to_app_server(self, action_name: str):
        action_type = "probe" if action_name in PROBE_ACTIONS else "recommend"
        current_time = int(time.time())

        success, scene_val = self.dp.get("Scene")
        scene_category = self._map_scene_to_category(scene_val)

        image_data = None

        if success and isinstance(scene_val, SceneData) and scene_val.has_image:
            try:
                raw_path = scene_val.image_path
                if raw_path and not os.path.isabs(raw_path):
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    filename = os.path.basename(raw_path)
                    possible_paths = [
                        raw_path,
                        os.path.join(base_dir, raw_path),
                        os.path.join(base_dir, "destineData", filename),
                        os.path.join(base_dir, "destineData", raw_path),
                        os.path.join(base_dir, "datapallet", raw_path),
                        os.path.join(base_dir, "sceneclassify", raw_path)
                    ]
                else:
                    possible_paths = [raw_path]

                real_path = None
                for p in possible_paths:
                    if p and os.path.exists(p):
                        real_path = p
                        break

                if real_path:
                    image_data = create_test_image_data(real_path)
                else:
                    print(f"[AppPush-Error] DataPallet有记录但文件找不到: {possible_paths}")

            except Exception as e:
                print(f"[AppPush-Error] 真实图片读取异常: {e}")

        if image_data is None and action_name == "QUERY_VISUAL":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            test_img_path = os.path.join(base_dir, "app_server", "test.png")

            if os.path.exists(test_img_path):
                image_data = create_test_image_data(str(test_img_path))
                print("[AppPush] 暂无真实照片，使用默认测试图片 (QUERY_VISUAL)")
            else:
                print(f"[AppPush-Error] 默认测试图片不存在: {test_img_path}")

        payload = {
            "id": f"rec_{current_time}_{random.randint(100, 999)}",
            "timestamp": current_time,
            "action_type": action_type,
            "action_name": action_name,
            "scene_category": scene_category,
            "image": image_data
        }

        try:
            response = requests.post(APP_UI_SERVER_URL, json=payload, headers={"Content-Type": "application/json"},
                                     timeout=0.5)
            if response.status_code != 200:
                print(f"[AppPush] 推送失败: {response.status_code}")
        except Exception as e:
            print(f"[AppPush] 连接 App Server 失败: {e}")


    def execute_visual_query(self):
        if self.is_querying_visual:
            print("[VisualQuery] 上一次拍照任务尚未完成，跳过本次请求")
            return

        with self.visual_query_lock:
            self.is_querying_visual = True

        try:
            print("[VisualQuery] DQN 触发了视觉查询请求 (QUERY_VISUAL)")

            if not self.wait_for_client_connection(max_wait=30.0):
                print("[VisualQuery] 错误：客户端未连接，无法拍照")
                return

            self.request_image_capture()


        except Exception as e:
            print(f"[VisualQuery] 执行异常: {e}")
        finally:
            with self.visual_query_lock:
                self.is_querying_visual = False
        
    def register_callbacks(self):
        """注册所有回调函数"""
        if self.callbacks_registered:
            return
            
        # WS服务器图片上传完成回调
        def on_upload_complete(img_path, img_timestamp):
            scene_data = self.process_scene_data(img_path, img_timestamp)
            if self.tb:
                self.tb.receive_and_transmit_data("Scene", scene_data, img_timestamp)
            print("[Callback] 正在将新图片推送到 APP...")
            self.send_to_app_server("QUERY_VISUAL")
                
        set_upload_complete_callback(on_upload_complete)
        
        # Testbed拍照请求回调
        def on_image_request():
            self.request_image_capture()
            
        set_on_image_request_callback(on_image_request)
        
        # 传感器数据回调
        def on_get_sensor_data(data_id, value, timestamp):
            if self.tb:
                self.tb.receive_and_transmit_data(data_id, value, timestamp)
                # === 检测姿态变化触发推理 ===
                if data_id == "activity_mode" and self.dqn_engine:
                    current_act = value
                    if isinstance(value, tuple):
                        current_act = value[1]

                    if current_act != self.last_activity_mode:
                        print(f"[Trigger] 姿态变化: {self.last_activity_mode} -> {current_act}")
                        self.last_activity_mode = current_act
                        self.inference_trigger_event.set()  # 立即触发事件
                
        set_on_get_sensor_data_callback(on_get_sensor_data)
        
        # Datapallet回调
        def test_callback(data_id: str, value: Any):
            if self.dp:
                formatted = self.dp.format_value(data_id, value)
                print(f"回调: {data_id} = {formatted}")
                
        self.test_callback = test_callback
        self.callbacks_registered = True
        
    def process_scene_data(self, img_path: str, img_timestamp) -> Dict[str, Any]:
        """处理场景数据"""
        scene_data = {
            "image_path": img_path,
            "scene_type": predict_scene(img_path)
        }
        print(f"[Scene处理] 图片路径: {img_path}")
        print(f"[Scene处理] 时间戳: {img_timestamp}")
        print(f"[Scene处理] scene_type: {scene_data['scene_type']}")
        return scene_data
        
    def request_image_capture(self):
        """请求拍照"""
        print("[拍照请求] 正在请求拍照...")
        run_test()
        
    def initialize_components(self):
        """初始化所有组件"""
        print("[初始化] 正在初始化组件...")
        self.dp = create_datapallet()
        self.dp.setup(self.test_callback)
        self.tb = TestBed(self.dp)
        self.dp.connect_testbed(self.tb)
        print("[初始化] 加载 DQN 模型...")
        ckpt_path = r"D:\c00894262\datapallet\dqn_engine\dqn_aod_ckpt_episode_1100.pt"
        try:
            self.dqn_engine = DQNEngineAdapter(
                ckpt_path=Path(ckpt_path),
                history_len=0, # TODO now is 0
                device="cpu"
            )
            print("[初始化] DQN 模型加载完成")
        except Exception as e:
            print(f"[错误] DQN 模型加载失败: {e}")

        print("[初始化] 组件初始化完成")

    def start_dqn_inference_loop(self):
        print("[DQN] 启动推理线程...")
        while self.running:
            is_event_triggered = self.inference_trigger_event.wait(timeout=5.0)

            if is_event_triggered:
                print("[DQN] 触发源: 姿态变化")
                self.inference_trigger_event.clear()
            else:
                print("[DQN] 触发源: 定时周期 (5s)")
                pass

            if self.dqn_engine and self.dp:
                try:
                    action, debug_info = self.dqn_engine.update_and_predict(self.dp)
                    print("-" * 60)
                    print(f"[DQN 推理] {datetime.now().strftime('%H:%M:%S')}")
                    print(f"  ├── {debug_info}")  # 打印物理输入
                    print(f"  └── Output Action: [{action}]")  # 打印输出动作
                    print("-" * 60)
                    if action == "QUERY_VISUAL":
                        print("DQN Action is QUERY_VISUAL, trigger take picture")
                        threading.Thread(
                            target=self.execute_visual_query,
                            name="Visual-Query-Worker",
                            daemon=True
                        ).start()

                    # TODO 要不要过滤NONE的动作？
                    self.send_to_app_server(action)

                except Exception as e:
                    print(f"[DQN 错误] 推理过程异常: {e}")

    def start_services(self):
        """启动所有服务"""
        print("[服务启动] 正在启动服务...")
        
        # 启动传感器服务
        run_sensor_server()
        
        # 启动WS服务器
        server_thread = threading.Thread(
            target=run_server, 
            name="WS-Server-Thread",
            daemon=True
        )
        server_thread.start()
        self.threads.append(server_thread)
        
        # 启动图像分类服务
        scene_classify_thread = threading.Thread(
            target=run_scene_classify_server,
            name="Scene-Classify-Thread",
            daemon=True
        )
        scene_classify_thread.start()
        self.threads.append(scene_classify_thread)

        # === 启动 DQN 推理线程 ===
        dqn_thread = threading.Thread(
            target=self.start_dqn_inference_loop,
            name="DQN-Inference-Thread",
            daemon=True
        )
        dqn_thread.start()
        self.threads.append(dqn_thread)
        
        print("[服务启动] 所有服务已启动")
        
    def wait_for_scene_data(self):
        """等待Scene数据"""
        
        def _wait_in_thread():
            if not self.dp:
                print("[Scene等待] 错误: Datapallet未初始化")
                return
                
            success, value = self.dp.get("Scene")
            if success:
                print(f"[Scene等待] 获取到Scene数据: {value}")
            else:
                print("[Scene等待] 获取Scene数据超时")
                
        wait_thread = threading.Thread(
            target=_wait_in_thread,
            name="Scene-Wait-Thread",
            daemon=True
        )
        wait_thread.start()
        return wait_thread
        
    def wait_for_client_connection(self, check_interval: float = 2.0, max_wait: float = 30.0) -> bool:
        """
        等待客户端连接到服务器
        
        Args:
            check_interval: 检查间隔（秒）
            max_wait: 最大等待时间（秒）
            
        Returns:
            True if client connected, False if timeout
        """
        import requests
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get("http://localhost:8001/api/client_status", timeout=2)
                if response.json().get("connected"):
                    return True
            except requests.exceptions.RequestException:
                pass  # 服务器可能还未完全启动
            
            elapsed = time.time() - start_time
            print(f"[等待连接] 客户端尚未连接，已等待 {elapsed:.0f}秒...")
            time.sleep(check_interval)
        
        return False
    
    def delayed_scene_request(self, delay: int = 10):
        """延迟请求拍照"""
        print(f"[延迟请求] {delay}秒后将开始请求拍照...")
        time.sleep(delay)
        
        # 等待客户端连接
        print("[延迟请求] 正在等待客户端连接...")
        if not self.wait_for_client_connection():
            print("[延迟请求] 错误：等待客户端连接超时，放弃请求拍照")
            return
        
        print("[延迟请求] 客户端已连接，正在请求拍照...")
        self.request_image_capture()
        
        # 启动Scene数据等待
        self.wait_for_scene_data()
        
    def start(self):
        """启动应用"""
        if self.running:
            print("应用已经在运行中")
            return
            
        self.running = True
        print("=== 应用启动 ===")
        
        # 注册回调
        self.register_callbacks()
        
        # 初始化组件
        self.initialize_components()
        
        # 启动服务
        self.start_services()
        
        # 延迟启动Scene请求（模拟）- change to DQN trigger
        # request_thread = threading.Thread(
        #     target=self.delayed_scene_request,
        #     args=(10,),
        #     name="Delayed-Request-Thread",
        #     daemon=True
        # )
        # request_thread.start()
        # self.threads.append(request_thread)

        app_server_thread = threading.Thread(
            target=run_app_server,
            kwargs={
                "host": "0.0.0.0",
                "port": 8002,
            },
            name="App-Server-Thread",
            daemon=True
        )
        app_server_thread.start()
        self.threads.append(app_server_thread)

        print("=== 应用启动完成 ===")
        
    def stop(self):
        """停止应用"""
        print("正在停止应用...")
        self.running = False
        
        # 这里可以添加各个组件的清理代码
        # 注意：daemon线程会在主线程退出时自动结束
        
        print("应用已停止")
        
    def run(self):
        """运行主循环"""
        self.start()
        
        # 设置信号处理
        def signal_handler(sig, frame):
            print("\n接收到停止信号，正在关闭...")
            self.stop()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 主循环
        try:
            while self.running:
                # 可以在这里添加心跳检查、状态监控等
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n用户中断")
            self.stop()
        except Exception as e:
            print(f"发生未预期错误: {e}")
            self.stop()
            raise

def main():
    """主函数"""
    app = ApplicationManager()
    app.run()

if __name__ == '__main__':
    main()