"""
采集数据采集，周期发送数据，以及被动触发获取最新的指定数据
"""

import time
import threading
from typing import Any, Dict, Optional
from datetime import datetime

# 导入数据托盘和测试床模块
from enums import (
    ActivityMode, LightIntensity, SoundIntensity,
    LocationType, SceneType, to_str, SceneData
)
from datapallet import DataPallet, create_datapallet
from testbed import TestBed, create_testbed, PlaybackConfig
from sensor_server import run as run_sensor_server 

# ==================== 测试回调函数 ====================

class TestCallbackHandler:
    """测试回调处理器"""
    
    def __init__(self, name: str = "Default", datapallet: Optional[DataPallet] = None):
        self.name = name
        self.callback_count = 0
        self.received_data: Dict[str, list] = {}
        self.lock = threading.RLock()
        self.datapallet = datapallet
        self.active_fetch_count = 0
    
    def callback(self, data_id: str, value: Any):
        """通用回调函数"""
        with self.lock:
            self.callback_count += 1
            
            # 记录接收到的数据
            if data_id not in self.received_data:
                self.received_data[data_id] = []
            self.received_data[data_id].append({
                "timestamp": datetime.now(),
                "value": value
            })
            
            # 格式化值为中文字符串
            formatted_value = to_str(data_id, value)
            
            # 打印回调信息
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {self.name} 回调 #{self.callback_count}: {data_id} = {formatted_value}")
            
            # 演示主动获取能力：在回调中随机主动获取位置或场景数据
            self._demo_random_active_fetch()
    
    def _demo_random_active_fetch(self):
        """随机主动获取位置或场景数据，展示主动获取能力"""
        if self.datapallet is None:
            return
            
        import random
        
        # 随机决定是否进行主动获取
        if random.random() < 0.9: # 90%概率主动获取
            self.active_fetch_count += 1
            
            # 随机选择要获取的数据类型
            fetch_type = random.choice(["location", "scene", "all"])
            
            if fetch_type == "location":
                print(f"  └─ [主动获取演示 #{self.active_fetch_count}] 随机主动获取位置信息...")
                success, location_value = self.datapallet.get("Location")
                if success:
                    formatted_location = to_str("Location", location_value)
                    print(f"  └─ 主动获取结果: Location = {formatted_location}")
                else:
                    print(f"  └─ 主动获取结果: 无有效位置数据")
                    
            elif fetch_type == "scene":
                print(f"  └─ [主动获取演示 #{self.active_fetch_count}] 随机主动获取场景图像信息...")
                success, scene_value = self.datapallet.get("Scence")
                if success:
                    formatted_scene = to_str("Scence", scene_value)
                    print(f"  └─ 主动获取结果: Scence = {formatted_scene}")
                    
                    # 如果是SceneData且有图像，显示图像信息
                    if isinstance(scene_value, SceneData) and scene_value.has_image:
                        print(f"  └─ 图像详情: 路径={scene_value.image_path}, 大小={scene_value.image_size}字节")
                else:
                    print(f"  └─ 主动获取结果: 无有效场景数据")
                    
            else:  # fetch_type == "all"
                print(f"  └─ [主动获取演示 #{self.active_fetch_count}] 获取托盘全部有效信息...")
                success, all_data = self.datapallet.get(None)
                if success and isinstance(all_data, dict):
                    print(f"  └─ 主动获取结果: 共获取 {len(all_data)} 种有效数据")
                    for data_id, value in all_data.items():
                        formatted_value = to_str(data_id, value)
                        print(f"  └─   - {data_id}: {formatted_value}")
                else:
                    print(f"  └─ 主动获取结果: 无有效数据")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                "name": self.name,
                "total_callbacks": self.callback_count,
                "active_fetches": self.active_fetch_count,
                "data_types": list(self.received_data.keys()),
                "count_by_type": {k: len(v) for k, v in self.received_data.items()}
            }
    
    def clear(self):
        """清除记录"""
        with self.lock:
            self.callback_count = 0
            self.active_fetch_count = 0
            self.received_data.clear()


# ==================== 测试场景 ====================

def test_playback_functionality(filepath: str, duration: int):
    """测试播放功能
    
    Args:
        filepath: 回放文件路径
    """
    print("\n=== 测试播放功能（包含主动获取演示） ===")
    print("说明：在播放期间，当收到回调时，会随机主动获取位置或场景数据，展示主动获取能力")
    print("注意：使用较短的TTL（2秒）以便更快获取最新数据")
    
    dp = create_datapallet(ttl=2)
    handler = TestCallbackHandler("播放测试", datapallet=dp)
    
    # 设置回调
    dp.setup(handler.callback)
    
    # 创建测试床
    tb = create_testbed(dp)
    
    # 加载回放文件
    print(f"\n1. 加载回放文件: {filepath}")
    tb.load_recording(filepath)
    
    # 配置播放参数
    config = PlaybackConfig(
        speed=1.0,  # 正常速度播放
        interval=1.0,  # 1.0秒间隔（实际使用时间戳驱动）
        loop=False
    )
    
    print("\n2. 开始播放测试数据...")
    tb.start_playback(config)
    
    # 播放一段时间（期间callback会触发主动获取演示）
    print(f"\n3. 播放{duration}秒（期间会展示主动获取能力）...")
    time.sleep(float(duration))
    
    # 停止播放
    print("\n4. 停止播放...")
    tb.stop_playback()
    
    # 获取数据托盘状态
    print("\n5. 检查数据托盘状态...")
    data_info = dp.get_data_info()
    for data_id, info in data_info.items():
        status = "有效" if info["valid"] else "过期"
        print(f"  {data_id}: {status}, 剩余时间: {info['time_remaining']:.1f}秒")
    
    # 显示回调统计和主动获取统计
    print("\n6. 测试统计:")
    stats = handler.get_stats()
    print(f"  - 总回调次数: {stats['total_callbacks']}")
    print(f"  - 主动获取演示次数: {stats['active_fetches']}")
    print(f"  - 接收到的数据类型: {', '.join(stats['data_types'])}")
    
    dp.stop()
    return dp, tb, handler


def test_recording_functionality():
    """测试录制功能（LLM模拟）"""
    print("\n=== 测试录制功能 ===")
    
    dp = create_datapallet(ttl=30)
    handler = TestCallbackHandler("录制测试")
    
    # 设置回调
    dp.setup(handler.callback)
    
    # 创建测试床
    tb = create_testbed(dp)
    
    # 测试不同的行为描述
    test_scenarios = [
        # 超过50s需要拆分，因为大模型输出max token有限制，虽然写了拆分输出能力，但是效果感觉不是特别好
        ("开会", "走路去会议室开会，距离20米，会议开始时坐下，然后开始讨论", 30.0, 1.0),
        ("下楼", "从开放的办公区走到楼梯，然后坐电梯到1楼，办公区在5楼，到1楼后穿过大堂，走到办公楼外面", 50.0, 1.0),
    ]
    
    for name, description, duration, interval in test_scenarios:
        print(f"\n# 录制场景: {name}")
        tb.record_data(name, description, duration, interval)
    
    dp.stop()
    return dp, tb, handler

def test_sensor_server_functionality():
    """测试直接接收数据模式（通过 sensor_server 上报数据）"""
    print("\n=== 测试直接接收数据模式 ===")
    print("说明：启动 sensor_server，让其将数据上报给 TestBed，然后由 TestBed 透传给 DataPallet")
    
    # 创建数据托盘实例
    dp = create_datapallet(ttl=30)
    handler = TestCallbackHandler("直接接收测试", datapallet=dp)
    
    # 设置回调
    dp.setup(handler.callback)
    
    # 创建测试床实例
    tb = create_testbed(dp)
    
    # 启动 sensor_server
    print("\n=== 启动 sensor_server ===")
    sensor_server_thread = threading.Thread(target=run_sensor_server, daemon=True)
    sensor_server_thread.start()
    
    # 主线程保持运行，等待用户中断
    try:
        while True:
            time.sleep(1)  # 保持主线程运行
    except KeyboardInterrupt:
        print("\n=== 停止 sensor_server ===")
        dp.stop()
        print("\n=== 测试结束 ===")
        stats = handler.get_stats()
        print(f"  - 总回调次数: {stats['total_callbacks']}")
        print(f"  - 主动获取演示次数: {stats['active_fetches']}")
        print(f"  - 接收到的数据类型: {', '.join(stats['data_types'])}")

# ==================== 命令行接口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据托盘系统测试程序")
    parser.add_argument("--mode", choices=["playback", "recording", "sensor_server"],
                       default="playback", help="测试模式")
    parser.add_argument("--file", help="playback的文件名")
    parser.add_argument("--time", help="playback的时长")
    
    args = parser.parse_args()
    
    if args.mode == "playback": # 回放模拟场景数据
        test_playback_functionality(args.file, args.time)
    elif args.mode == "recording": # 生成模拟场景数据
        test_recording_functionality()
    elif args.mode == "sensor_server":
        test_sensor_server_functionality()
    else:
        print(f"未知模式: {args.mode}")
        print("可用模式: playback, recording")