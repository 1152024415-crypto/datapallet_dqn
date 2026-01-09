import threading
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
import signal
import sys
import threading
import time
from sensor_server.sensor_server import set_on_get_sensor_data_callback, run_sensor_server
from datapallet.datapallet import create_datapallet
from datapallet.testbed import TestBed, set_on_image_request_callback
from server.test import run_test
from server.server import set_upload_complete_callback, run_server
from sceneclassify.deploy_server import run_scene_classify_server
from sceneclassify.predict_scene_type import predict_scene

class ApplicationManager:
    """应用管理器，统一管理所有服务和组件"""
    
    def __init__(self):
        self.dp = None
        self.tb = None
        self.threads = []
        self.running = False
        self.callbacks_registered = False
        
    def register_callbacks(self):
        """注册所有回调函数"""
        if self.callbacks_registered:
            return
            
        # WS服务器图片上传完成回调
        def on_upload_complete(img_path, img_timestamp):
            scene_data = self.process_scene_data(img_path, img_timestamp)
            if self.tb:
                self.tb.receive_and_transmit_data("Scene", scene_data, img_timestamp)
                
        set_upload_complete_callback(on_upload_complete)
        
        # Testbed拍照请求回调
        def on_image_request():
            self.request_image_capture()
            
        set_on_image_request_callback(on_image_request)
        
        # 传感器数据回调
        def on_get_sensor_data(data_id, value, timestamp):
            if self.tb:
                self.tb.receive_and_transmit_data(data_id, value, timestamp)
                
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
        print("[初始化] 组件初始化完成")
        
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
        
    def delayed_scene_request(self, delay: int = 10):
        """延迟请求拍照"""
        print(f"[延迟请求] {delay}秒后将开始请求拍照...")
        time.sleep(delay)
        
        # 模拟请求拍照
        print("[延迟请求] 正在请求拍照...")
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
        
        # 延迟启动Scene请求（模拟）
        request_thread = threading.Thread(
            target=self.delayed_scene_request,
            args=(10,),
            name="Delayed-Request-Thread",
            daemon=True
        )
        request_thread.start()
        self.threads.append(request_thread)
        
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