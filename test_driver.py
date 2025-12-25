"""
测试驱动程序
用于驱动数据托盘获取数据和测试回调功能
"""

import time
import threading
from typing import Any, Dict
from datetime import datetime

# 导入数据托盘和测试床模块
from datapallet import DataPallet, create_datapallet, ActivityMode, LightIntensity, SoundIntensity, LocationType, SceneType
from testbed import TestBed, create_testbed, PlaybackConfig


# ==================== 测试回调函数 ====================

class TestCallbackHandler:
    """测试回调处理器"""
    
    def __init__(self, name: str = "Default"):
        self.name = name
        self.callback_count = 0
        self.received_data: Dict[str, list] = {}
        self.lock = threading.RLock()
    
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
            
            # 打印回调信息
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {self.name} 回调 #{self.callback_count}: {data_id} = {value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                "name": self.name,
                "total_callbacks": self.callback_count,
                "data_types": list(self.received_data.keys()),
                "count_by_type": {k: len(v) for k, v in self.received_data.items()}
            }
    
    def clear(self):
        """清除记录"""
        with self.lock:
            self.callback_count = 0
            self.received_data.clear()


# ==================== 测试场景 ====================

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 创建数据托盘
    dp = create_datapallet(ttl=30)
    
    # 创建回调处理器
    handler = TestCallbackHandler("基本测试")
    
    # 设置回调
    print("1. 设置回调函数...")
    dp.setup(handler.callback)
    
    # 创建测试床并连接
    print("2. 创建并连接测试床...")
    tb = create_testbed(dp)
    
    # 发送一些测试数据
    print("3. 发送测试数据...")
    tb.send_data("activity_mode", (ActivityMode.STROLLING, ActivityMode.RUNNING))
    tb.send_data("Light_Intensity", LightIntensity.BRIGHT)
    tb.send_data("Sound_Intensity", SoundIntensity.NORMAL_SOUND)
    
    # 等待数据处理
    time.sleep(0.5)
    
    # 测试get函数
    print("4. 测试get函数...")
    
    # 获取单个数据
    success, value = dp.get("activity_mode")
    if success:
        formatted = dp.format_value("activity_mode", value)
        print(f"  获取 activity_mode: {formatted}")
    
    # 获取所有数据
    success, all_data = dp.get(None)
    if success:
        print(f"  获取所有数据: {len(all_data)} 项")
        for data_id, val in all_data.items():
            formatted = dp.format_value(data_id, val)
            print(f"    {data_id}: {formatted}")
    
    # 显示回调统计
    stats = handler.get_stats()
    print(f"5. 回调统计: {stats['total_callbacks']} 次回调")
    
    # 停止数据托盘
    dp.stop()
    
    return dp, tb, handler


def test_subscription_patterns():
    """测试订阅模式"""
    print("\n=== 测试订阅模式 ===")
    
    dp = create_datapallet(ttl=30)
    tb = create_testbed(dp)
    
    # 创建不同的回调处理器
    all_handler = TestCallbackHandler("全部数据")
    activity_handler = TestCallbackHandler("仅姿态")
    light_handler = TestCallbackHandler("仅亮度")
    
    # 测试不同的订阅模式
    print("1. 订阅所有数据...")
    dp.setup(all_handler.callback)
    
    print("2. 仅订阅activity_mode...")
    dp.setup(activity_handler.callback, "activity_mode")
    
    print("3. 仅订阅Light_Intensity...")
    dp.setup(light_handler.callback, "Light_Intensity")
    
    # 发送测试数据
    print("4. 发送测试数据...")
    tb.send_all_data()
    
    # 等待回调
    time.sleep(1.0)
    
    # 显示统计
    print("5. 回调统计:")
    for handler in [all_handler, activity_handler, light_handler]:
        stats = handler.get_stats()
        print(f"  {stats['name']}: {stats['total_callbacks']} 次回调")
    
    dp.stop()
    return dp, tb


def test_playback_functionality():
    """测试播放功能"""
    print("\n=== 测试播放功能 ===")
    
    dp = create_datapallet(ttl=10)
    handler = TestCallbackHandler("播放测试")
    
    # 设置回调
    dp.setup(handler.callback)
    
    # 创建测试床
    tb = create_testbed(dp)
    
    # 配置播放参数
    config = PlaybackConfig(
        speed=2.0,  # 2倍速播放
        interval=0.5,  # 0.5秒间隔
        random_order=True  # 随机顺序
    )
    
    print("1. 开始播放测试数据...")
    tb.start_playback(config)
    
    # 播放一段时间
    print("2. 播放5秒...")
    time.sleep(5.0)
    
    # 停止播放
    print("3. 停止播放...")
    tb.stop_playback()
    
    # 获取数据托盘状态
    print("4. 检查数据托盘状态...")
    data_info = dp.get_data_info()
    for data_id, info in data_info.items():
        status = "有效" if info["valid"] else "过期"
        print(f"  {data_id}: {status}, 剩余时间: {info['time_remaining']:.1f}秒")
    
    # 显示回调统计
    stats = handler.get_stats()
    print(f"5. 回调统计: {stats['total_callbacks']} 次回调")
    
    dp.stop()
    return dp, tb, handler


def test_get_blocking():
    """测试get函数的阻塞行为"""
    print("\n=== 测试get函数阻塞行为 ===")
    
    dp = create_datapallet(ttl=5)  # 短TTL
    tb = create_testbed(dp)
    
    # 先获取一次数据（应该返回默认值）
    print("1. 获取初始数据...")
    success, value = dp.get("activity_mode", timeout=1.0)
    print(f"   结果: {success}, 值: {value}")
    
    # 在另一个线程中发送数据
    def send_data_after_delay():
        time.sleep(2.0)
        print("   [后台线程] 发送数据...")
        tb.send_data("activity_mode", (ActivityMode.RUNNING, ActivityMode.CYCLING))
    
    print("2. 启动后台线程发送数据...")
    thread = threading.Thread(target=send_data_after_delay, daemon=True)
    thread.start()
    
    print("3. 调用get函数（应该会阻塞等待数据）...")
    start_time = time.time()
    success, value = dp.get("activity_mode", timeout=5.0)
    elapsed = time.time() - start_time
    
    if success:
        formatted = dp.format_value("activity_mode", value)
        print(f"   获取成功! 等待时间: {elapsed:.2f}秒")
        print(f"   获取到的值: {formatted}")
    else:
        print(f"   获取失败，等待时间: {elapsed:.2f}秒")
    
    dp.stop()
    return dp, tb


def test_integration_scenario():
    """测试集成场景"""
    print("\n=== 测试集成场景 ===")
    
    # 创建数据托盘
    dp = create_datapallet(ttl=60)
    
    # 创建多个回调处理器
    monitor_handler = TestCallbackHandler("监控器")
    logger_handler = TestCallbackHandler("日志器")
    
    # 设置回调
    dp.setup(monitor_handler.callback)  # 监控所有数据
    dp.setup(logger_handler.callback, "activity_mode")  # 仅记录姿态
    
    # 创建测试床
    tb = create_testbed(dp)
    
    # 录制一些测试数据
    print("1. 录制测试数据...")
    tb.record_data(duration=3.0, interval=0.3)
    
    # 播放录制的数据
    print("2. 播放录制的数据...")
    config = PlaybackConfig(speed=1.0, interval=0.1)
    tb.start_playback(config)
    time.sleep(2.0)
    tb.stop_playback()
    
    # 测试各种get操作
    print("3. 测试各种get操作...")
    
    # 获取单个数据
    success, value = dp.get("Light_Intensity")
    if success:
        formatted = dp.format_value("Light_Intensity", value)
        print(f"   Light_Intensity: {formatted}")
    
    # 获取所有数据
    success, all_data = dp.get(None)
    if success:
        print(f"   所有有效数据: {len(all_data)} 项")
    
    # 获取数据详细信息
    print("4. 获取数据详细信息...")
    data_info = dp.get_data_info()
    for data_id, info in data_info.items():
        if info["valid"]:
            formatted = dp.format_value(data_id, info["value"])
            print(f"   {data_id}: {formatted} (剩余: {info['time_remaining']:.1f}秒)")
    
    # 显示统计
    print("5. 回调统计:")
    for handler in [monitor_handler, logger_handler]:
        stats = handler.get_stats()
        print(f"   {stats['name']}: {stats['total_callbacks']} 次回调")
        print(f"     按类型: {stats['count_by_type']}")
    
    # 保存录制数据
    print("6. 保存录制数据...")
    tb.save_recording("test_recording.json")
    
    dp.stop()
    return dp, tb


# ==================== 主测试函数 ====================

def run_all_tests():
    """运行所有测试"""
    print("开始运行数据托盘系统测试")
    print("=" * 50)
    
    results = []
    
    try:
        # 测试1: 基本功能
        dp1, tb1, handler1 = test_basic_functionality()
        results.append(("基本功能测试", "通过"))
        
        # 测试2: 订阅模式
        dp2, tb2 = test_subscription_patterns()
        results.append(("订阅模式测试", "通过"))
        
        # 测试3: 播放功能
        dp3, tb3, handler3 = test_playback_functionality()
        results.append(("播放功能测试", "通过"))
        
        # 测试4: get阻塞行为
        dp4, tb4 = test_get_blocking()
        results.append(("get阻塞测试", "通过"))
        
        # 测试5: 集成场景
        dp5, tb5 = test_integration_scenario()
        results.append(("集成场景测试", "通过"))
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        results.append(("错误处理", f"失败: {e}"))
    
    # 打印测试总结
    print("\n" + "=" * 50)
    print("测试总结:")
    for test_name, result in results:
        print(f"  {test_name}: {result}")
    
    print(f"\n共运行 {len(results)} 个测试")
    print("测试完成!")


def simple_demo():
    """简单演示"""
    print("=== 数据托盘系统简单演示 ===")
    
    # 创建数据托盘
    dp = DataPallet(default_ttl=30)
    dp.start()
    
    # 定义回调函数
    def demo_callback(data_id: str, value: Any):
        formatted = dp.format_value(data_id, value)
        print(f"  数据更新: {data_id} -> {formatted}")
    
    # 设置回调
    dp.setup(demo_callback)
    
    # 创建测试床
    tb = TestBed(dp)
    
    print("1. 发送一些测试数据...")
    tb.send_data("activity_mode", (ActivityMode.STROLLING, ActivityMode.RUNNING))
    tb.send_data("Location", LocationType.PARK)
    
    time.sleep(0.5)
    
    print("2. 获取数据...")
    success, value = dp.get("activity_mode")
    if success:
        print(f"   当前姿态: {dp.format_value('activity_mode', value)}")
    
    print("3. 开始自动播放...")
    tb.start_playback(PlaybackConfig(interval=1.0))
    
    print("4. 运行5秒...")
    time.sleep(5.0)
    
    print("5. 停止系统...")
    tb.stop_playback()
    dp.stop()
    
    print("演示完成!")


# ==================== 命令行接口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据托盘系统测试程序")
    parser.add_argument("--mode", choices=["all", "demo", "basic", "subscription", "playback", "blocking", "integration"],
                       default="demo", help="测试模式")
    
    args = parser.parse_args()
    
    if args.mode == "all":
        run_all_tests()
    elif args.mode == "demo":
        simple_demo()
    elif args.mode == "basic":
        test_basic_functionality()
    elif args.mode == "subscription":
        test_subscription_patterns()
    elif args.mode == "playback":
        test_playback_functionality()
    elif args.mode == "blocking":
        test_get_blocking()
    elif args.mode == "integration":
        test_integration_scenario()
    else:
        print(f"未知模式: {args.mode}")
        print("可用模式: all, demo, basic, subscription, playback, blocking, integration")