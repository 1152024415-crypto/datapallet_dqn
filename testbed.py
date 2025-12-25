"""
测试床 (TestBed) 模块
负责按照时间戳播放录制的数据或人工编写的数据流
"""

import threading
import time
import json
import random
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import IntEnum

# 导入数据托盘的数据枚举
try:
    from datapallet import (
        ActivityMode, LightIntensity, SoundIntensity, 
        LocationType, SceneType, DataPallet
    )
except ImportError:
    # 定义本地枚举（如果datapallet不可用）
    class ActivityMode(IntEnum):
        NULL = 0; RIDING = 1; CYCLING = 2; SITTING = 3; STROLLING = 4
        RUNNING = 5; STANDING = 6; BUS = 7; CAR = 8; SUBWAY = 9
        HIGH_SPEED_TRAIN = 10; TRAIN = 11; HIKING = 12; BRISK_WALKING = 13
        VEHICLE_BRAKING = 14; ELEVATOR = 15; GARAGE_PARKING = 16
    
    class LightIntensity(IntEnum):
        NULL = 0; EXTREMELY_DARK = 1; DIM = 2; MODERATE_BRIGHTNESS = 3
        BRIGHT = 4; HARSH_LIGHT = 5
    
    class SoundIntensity(IntEnum):
        NULL = 0; VERY_QUIET = 1; SOFT_SOUND = 2; NORMAL_SOUND = 3
        NOISY = 4; VERY_NOISY = 5
    
    class LocationType(IntEnum):
        NULL = 0; OTHER = 1; DESTINATION = 2; HOME = 3; WORK = 4
        BUS_STATION = 5; SUBWAY_STATION = 6; TRAIN_STATION = 7; AIRPORT = 8
        ACCOMMODATION = 9; RESIDENTIAL = 10; COMMERCIAL = 11; SCHOOL = 12
        HEALTH = 13; GOVERNMENT = 14; ENTERTAINMENT = 15; DINING = 16
        SHOPPING = 17; SPORT = 18; ATTRACTION = 19; PARK = 20; STREET = 21
    
    class SceneType(IntEnum):
        NULL = 0; OTHER = 1; MEETINGROOM = 2


# ==================== 数据结构定义 ====================

@dataclass
class DataRecord:
    """数据记录，包含时间戳和数据"""
    timestamp: datetime
    data_id: str
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "id": self.data_id,
            "value": self.value
        }


@dataclass
class PlaybackConfig:
    """播放配置"""
    speed: float = 1.0  # 播放速度倍数
    loop: bool = False  # 是否循环播放
    random_order: bool = False  # 是否随机顺序播放
    interval: float = 1.0  # 数据发送间隔（秒）


# ==================== 测试床核心类 ====================

class TestBed:
    """
    测试床核心类
    负责生成和播放测试数据
    """
    
    def __init__(self, datapallet: Optional[Any] = None):
        """
        初始化测试床
        
        Args:
            datapallet: 数据托盘实例（可选）
        """
        self.datapallet = datapallet
        
        # 数据记录存储
        self.recorded_data: List[DataRecord] = []
        
        # 播放状态
        self.playing = False
        self.playback_thread: Optional[threading.Thread] = None
        self.playback_config = PlaybackConfig()
        
        # 线程同步
        self.lock = threading.RLock()
        
        # 数据生成器
        self.data_generators: Dict[str, Callable[[], Any]] = {
            "activity_mode": self._generate_activity_mode,
            "Light_Intensity": self._generate_light_intensity,
            "Sound_Intensity": self._generate_sound_intensity,
            "Location": self._generate_location,
            "Scence": self._generate_scene
        }
        
        # 当前数据缓存（用于get_latest_data）
        self.current_data: Dict[str, Any] = {}
        
        # 连接到数据托盘
        if datapallet:
            self.connect_datapallet(datapallet)
    
    def connect_datapallet(self, datapallet):
        """连接到数据托盘"""
        self.datapallet = datapallet
        if hasattr(datapallet, 'connect_testbed'):
            datapallet.connect_testbed(self)
    
    # ==================== 数据生成方法 ====================
    
    def _generate_activity_mode(self) -> Tuple[int, int]:
        """生成用户姿态数据"""
        # 随机选择两个不同的姿态
        activities = list(ActivityMode)
        last = random.choice(activities)
        current = random.choice([a for a in activities if a != last])
        return (last.value, current.value)
    
    def _generate_light_intensity(self) -> int:
        """生成环境亮度数据"""
        intensities = list(LightIntensity)
        return random.choice(intensities).value
    
    def _generate_sound_intensity(self) -> int:
        """生成环境声音强度数据"""
        intensities = list(SoundIntensity)
        return random.choice(intensities).value
    
    def _generate_location(self) -> int:
        """生成位置POI类型数据"""
        locations = list(LocationType)
        return random.choice(locations).value
    
    def _generate_scene(self) -> int:
        """生成图像场景分类数据"""
        scenes = list(SceneType)
        return random.choice(scenes).value
    
    def generate_data(self, data_id: str) -> Any:
        """
        生成指定类型的数据
        
        Args:
            data_id: 数据ID
            
        Returns:
            生成的数据值
        """
        if data_id in self.data_generators:
            return self.data_generators[data_id]()
        else:
            raise ValueError(f"不支持的数据ID: {data_id}")
    
    def generate_all_data(self) -> Dict[str, Any]:
        """生成所有类型的数据"""
        data = {}
        for data_id in self.data_generators.keys():
            data[data_id] = self.generate_data(data_id)
        return data
    
    # ==================== 数据发送方法 ====================
    
    def send_data(self, data_id: str, value: Any, timestamp: Optional[datetime] = None):
        """
        发送数据到数据托盘
        
        Args:
            data_id: 数据ID
            value: 数据值
            timestamp: 时间戳（如果为None则使用当前时间）
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 记录数据
        record = DataRecord(timestamp, data_id, value)
        with self.lock:
            self.recorded_data.append(record)
            self.current_data[data_id] = value
        
        # 发送到数据托盘
        if self.datapallet:
            if hasattr(self.datapallet, 'receive_data'):
                self.datapallet.receive_data(data_id, value)
            else:
                print(f"警告: 数据托盘没有receive_data方法")
        else:
            print(f"测试数据已生成: {data_id} = {value} (未连接到数据托盘)")
    
    def send_all_data(self):
        """发送所有类型的数据"""
        data = self.generate_all_data()
        for data_id, value in data.items():
            self.send_data(data_id, value)
    
    # ==================== 播放控制方法 ====================
    
    def start_playback(self, config: Optional[PlaybackConfig] = None):
        """
        开始播放数据
        
        Args:
            config: 播放配置（如果为None则使用默认配置）
        """
        if self.playing:
            print("播放已在运行")
            return
        
        if config:
            self.playback_config = config
        
        self.playing = True
        self.playback_thread = threading.Thread(
            target=self._playback_loop,
            daemon=True
        )
        self.playback_thread.start()
        print(f"测试床播放已启动 (速度: {self.playback_config.speed}x)")
    
    def stop_playback(self):
        """停止播放"""
        self.playing = False
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        print("测试床播放已停止")
    
    def _playback_loop(self):
        """播放循环"""
        data_ids = list(self.data_generators.keys())
        index = 0
        
        while self.playing:
            try:
                # 确定要发送的数据ID
                if self.playback_config.random_order:
                    data_id = random.choice(data_ids)
                else:
                    data_id = data_ids[index]
                    index = (index + 1) % len(data_ids)
                
                # 生成并发送数据
                value = self.generate_data(data_id)
                self.send_data(data_id, value)
                
                # 等待下一个发送间隔
                time.sleep(self.playback_config.interval / self.playback_config.speed)
                
            except Exception as e:
                print(f"播放循环出错: {e}")
                time.sleep(1.0)
    
    # ==================== 录制和回放方法 ====================
    
    def record_data(self, duration: float = 10.0, interval: float = 1.0):
        """
        录制测试数据
        
        Args:
            duration: 录制时长（秒）
            interval: 录制间隔（秒）
        """
        print(f"开始录制数据，时长: {duration}秒，间隔: {interval}秒")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # 生成并记录所有数据
            data = self.generate_all_data()
            timestamp = datetime.now()
            
            with self.lock:
                for data_id, value in data.items():
                    record = DataRecord(timestamp, data_id, value)
                    self.recorded_data.append(record)
                    self.current_data[data_id] = value
            
            # 等待下一个录制点
            time.sleep(interval)
        
        print(f"录制完成，共录制 {len(self.recorded_data)} 条数据记录")
    
    def playback_recorded(self, config: Optional[PlaybackConfig] = None):
        """播放已录制的数据"""
        if not self.recorded_data:
            print("没有录制的数据可供播放")
            return
        
        if config:
            self.playback_config = config
        
        print(f"开始播放录制的数据，共 {len(self.recorded_data)} 条记录")
        
        # 按时间戳排序
        sorted_records = sorted(self.recorded_data, key=lambda r: r.timestamp)
        
        # 计算时间基准
        if sorted_records:
            start_time = time.time()
            first_timestamp = sorted_records[0].timestamp
            
            for record in sorted_records:
                if not self.playing:
                    break
                
                # 计算应该发送的时间
                time_diff = (record.timestamp - first_timestamp).total_seconds()
                target_time = start_time + time_diff / self.playback_config.speed
                
                # 等待到目标时间
                current_time = time.time()
                if target_time > current_time:
                    time.sleep(target_time - current_time)
                
                # 发送数据
                self.send_data(record.data_id, record.value, record.timestamp)
    
    # ==================== 数据获取方法 ====================
    
    def get_latest_data(self, data_id: str) -> Tuple[bool, Any]:
        """
        获取最新的数据（供数据托盘调用）
        
        Args:
            data_id: 数据ID
            
        Returns:
            (success, value): 成功标志和数据值
        """
        with self.lock:
            if data_id in self.current_data:
                return True, self.current_data[data_id]
            else:
                return False, f"没有可用的数据: {data_id}"
    
    def get_all_data(self) -> Dict[str, Any]:
        """获取所有最新的数据"""
        with self.lock:
            return self.current_data.copy()
    
    # ==================== 文件操作方法 ====================
    
    def save_recording(self, filepath: str):
        """
        保存录制数据到文件
        
        Args:
            filepath: 文件路径
        """
        with self.lock:
            records_dict = [record.to_dict() for record in self.recorded_data]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(records_dict, f, indent=2, ensure_ascii=False)
        
        print(f"录制数据已保存到: {filepath}")
    
    def load_recording(self, filepath: str):
        """
        从文件加载录制数据
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                records_dict = json.load(f)
            
            with self.lock:
                self.recorded_data = []
                for record_dict in records_dict:
                    timestamp = datetime.fromisoformat(record_dict["timestamp"])
                    data_id = record_dict["id"]
                    value = record_dict["value"]
                    record = DataRecord(timestamp, data_id, value)
                    self.recorded_data.append(record)
            
            print(f"录制数据已从 {filepath} 加载，共 {len(self.recorded_data)} 条记录")
            
        except Exception as e:
            print(f"加载录制数据失败: {e}")
    
    # ==================== 工具方法 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取测试床状态"""
        with self.lock:
            return {
                "playing": self.playing,
                "recorded_count": len(self.recorded_data),
                "current_data": self.current_data,
                "config": {
                    "speed": self.playback_config.speed,
                    "loop": self.playback_config.loop,
                    "random_order": self.playback_config.random_order,
                    "interval": self.playback_config.interval
                }
            }
    
    def clear_recording(self):
        """清除所有录制数据"""
        with self.lock:
            self.recorded_data.clear()
        print("录制数据已清除")


# ==================== 工具函数 ====================

def create_testbed(datapallet: Optional[Any] = None) -> TestBed:
    """创建测试床实例"""
    return TestBed(datapallet)


def create_test_scenario() -> Dict[str, List[Dict[str, Any]]]:
    """创建测试场景数据"""
    scenario = {
        "morning_commute": [
            {"time": "08:00", "activity_mode": (ActivityMode.SITTING, ActivityMode.STANDING)},
            {"time": "08:15", "Light_Intensity": LightIntensity.BRIGHT},
            {"time": "08:30", "Sound_Intensity": SoundIntensity.NOISY},
            {"time": "08:45", "Location": LocationType.SUBWAY_STATION},
        ],
        "office_work": [
            {"time": "09:00", "activity_mode": (ActivityMode.SITTING, ActivityMode.SITTING)},
            {"time": "10:00", "Light_Intensity": LightIntensity.MODERATE_BRIGHTNESS},
            {"time": "11:00", "Sound_Intensity": SoundIntensity.NORMAL_SOUND},
            {"time": "12:00", "Location": LocationType.WORK},
        ]
    }
    return scenario


if __name__ == "__main__":
    # 简单的自测试
    print("=== 测试床模块自测试 ===")
    
    # 创建测试床（不连接数据托盘）
    tb = create_testbed()
    
    # 测试数据生成
    print("\n1. 测试数据生成:")
    for data_id in tb.data_generators.keys():
        value = tb.generate_data(data_id)
        print(f"  {data_id}: {value}")
    
    # 测试录制
    print("\n2. 测试录制数据:")
    tb.record_data(duration=2.0, interval=0.5)
    print(f"  录制了 {len(tb.recorded_data)} 条数据")
    
    # 测试播放
    print("\n3. 测试播放控制:")
    tb.start_playback(PlaybackConfig(speed=2.0, interval=0.5))
    time.sleep(2.0)
    tb.stop_playback()
    
    print("\n测试床自测试完成")