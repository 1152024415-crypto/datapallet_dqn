"""
重构后的测试床 (TestBed) 模块
支持数据录制、LLM生成用户行为的数据序列、时间戳驱动的数据发送
"""

import threading
import time
import json
import random
import re
import os
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import IntEnum

from enums import (
    ActivityMode, LightIntensity, SoundIntensity,
    LocationType, SceneType, SceneData, to_str, from_str
)
from datapallet import DataPallet

from llm_simulator import LLMSimulator, DataRecord
from image_generator import ImageGenerator, create_image_generator


# ==================== 数据结构定义 ====================


@dataclass
class PlaybackConfig:
    """播放配置"""
    speed: float = 1.0  # 播放速度倍数
    loop: bool = False  # 是否循环播放
    interval: float = 1.0  # 数据发送间隔（秒）


@dataclass
class RecordingConfig:
    """录制配置"""
    duration: float = 10.0  # 录制时长（秒）
    interval: float = 1.0  # 录制间隔（秒）
    description: str = ""  # 用户行为描述


# ==================== LLM模拟器（已移至独立文件） ====================
# 现在从 llm_simulator 模块导入 LLMSimulator


# ==================== 测试床核心类 ====================

class TestBed:
    def __init__(self, datapallet: Optional[DataPallet] = None, image_generator: Optional[ImageGenerator] = None):
        """
        初始化测试床
        
        Args:
            datapallet: 数据托盘实例
            image_generator: 图像生成器实例（可选）
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
        
        # 当前数据状态（每个ID的最新值）
        self.current_state: Dict[str, Any] = {}
        
        # LLM模拟器
        self.llm_simulator = LLMSimulator()
        
        # 图像生成器
        self.image_generator = image_generator
        if self.image_generator is None:
            # 创建默认的图像生成器
            self.image_generator = create_image_generator()
        
        # 连接到数据托盘
        if datapallet:
            self.connect_datapallet(datapallet)
    
    def connect_datapallet(self, datapallet):
        """连接到数据托盘"""
        self.datapallet = datapallet
        if hasattr(datapallet, 'connect_testbed'):
            datapallet.connect_testbed(self)
    
    # ==================== 数据发送方法 ====================
    
    def send_data(self, data_id: str, value: Any, timestamp: datetime):
        """
        发送数据到数据托盘
        
        Args:
            data_id: 数据ID
            value: 数据值
            timestamp: 时间戳（如果为None则使用当前时间）
        """
        # 发送到数据托盘（仅发送姿态信息，其他信息由datapallet主动获取）
        if self.datapallet:
            if hasattr(self.datapallet, 'receive_data'):
                self.datapallet.receive_data(data_id, value, timestamp)
            else:
                print(f"警告: 数据托盘没有receive_data方法")
        else:
            print(f"测试数据已生成: {data_id} = {value} (未连接到数据托盘)")
    
    # ==================== 数据录制功能 ====================
    
    def record_data(self, name: str, description: str, duration: float = 10.0, interval: float = 1.0):
        """
        录制数据（LLM模拟）
        
        Args:
            description: 用户行为自然语言描述
            duration: 录制时长（秒）
            interval: 录制间隔（秒）
        """
        print(f"开始录制数据: {description}")
        print(f"时长: {duration}秒，间隔: {interval}秒")
        
        # 使用LLM模拟器生成数据序列
        records = self.llm_simulator.generate_data_sequence(description, duration, interval)
        
        with self.lock:
            self.recorded_data = records
        
        # 自动保存文件
        filename = self._generate_filename(name)
        self.save_recording(filename)
        
        print(f"录制完成，共录制 {len(self.recorded_data)} 条数据记录")
        print(f"数据已保存到: {filename}")
    
    def _generate_filename(self, name: str) -> str:
        """生成包含时间的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"recording_{timestamp}_{name}.json"
    
    # ==================== 时间戳驱动的数据发送 ====================
    
    def start_playback(self, config: Optional[PlaybackConfig] = None):
        """
        开始播放数据（严格按照时间戳）
        
        Args:
            config: 播放配置（如果为None则使用默认配置）
        """
        if self.playing:
            print("播放已在运行")
            return
        
        if not self.recorded_data:
            print("没有录制的数据可供播放")
            return
        
        if config:
            self.playback_config = config
        
        self.playing = True
        self.playback_thread = threading.Thread(
            target=self._timestamp_driven_playback,
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
    
    def _timestamp_driven_playback(self):
        """时间戳驱动的数据播放"""
        if not self.recorded_data:
            return
        
        # 按时间戳排序
        sorted_records = sorted(self.recorded_data, key=lambda r: r.timestamp)
        
        # 计算时间基准
        start_time = time.time()
        first_timestamp = sorted_records[0].timestamp
        
        index = 0
        while self.playing and index < len(sorted_records):
            record = sorted_records[index]
            
            # 计算应该发送的时间（考虑播放速度）
            time_diff = (record.timestamp - first_timestamp).total_seconds()
            target_time = start_time + time_diff / self.playback_config.speed
            
            # 等待到目标时间
            current_time = time.time()
            if target_time > current_time:
                time.sleep(target_time - current_time)

            # 更新当前状态
            with self.lock:
                self.current_state[record.data_id] = record.value
            
            # 发送数据（仅发送姿态信息，其他信息由datapallet主动获取）
            if record.data_id == "activity_mode":
                self.send_data(record.data_id, record.value, datetime.now())
            
            index += 1
            
            # 如果循环播放，重置索引
            if self.playback_config.loop and index >= len(sorted_records):
                index = 0
                start_time = time.time()
                first_timestamp = sorted_records[0].timestamp
    
    # ==================== 文件加载和保存功能 ====================
    
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
                    
                    # 对于Scence数据，处理SceneData
                    if data_id == "Scence":
                        parsed_value = self._parse_scene_data(value, filepath)
                    else:
                        parsed_value = self._parse_chinese_value(data_id, value)
                    
                    record = DataRecord(timestamp, data_id, parsed_value)
                    self.recorded_data.append(record)
            
            print(f"录制数据已从 {filepath} 加载，共 {len(self.recorded_data)} 条记录")

        except Exception as e:
            print(f"加载录制数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_scene_data(self, value: Any, recording_filepath: str) -> SceneData:
        """解析SceneData值"""
        # 如果value已经是SceneData对象，直接返回
        if isinstance(value, SceneData):
            return value
        
        # 如果value是字典，表示包含图像信息
        if isinstance(value, dict) and "scene_type" in value:
            scene_type_value = value.get("scene_type", 0)
            scene_type = SceneType(scene_type_value)
            
            # 获取图像路径
            image_path = value.get("image_path")
            
            # 如果image_path是相对路径，转换为绝对路径（相对于录制文件）
            if image_path and not os.path.isabs(image_path):
                recording_dir = os.path.dirname(recording_filepath)
                # 尝试几种可能的路径
                possible_paths = [
                    os.path.join(recording_dir, image_path),
                    os.path.join(recording_dir, "scene_images", os.path.basename(image_path)),
                    os.path.join(os.path.dirname(recording_dir), image_path)
                ]
                
                for possible_path in possible_paths:
                    if os.path.exists(possible_path):
                        image_path = possible_path
                        break
            
            # 创建SceneData（不立即加载图像数据）
            return SceneData(scene_type=scene_type, image_path=image_path)
        
        # 如果value是整数或SceneType，转换为SceneData
        elif isinstance(value, int):
            scene_type = SceneType(value)
            return SceneData.from_scene_type(scene_type)
        elif isinstance(value, SceneType):
            return SceneData.from_scene_type(value)
        else:
            # 默认返回未知场景
            return SceneData.from_scene_type(SceneType.NULL)
    
    def _parse_chinese_value(self, data_id: str, value: Any) -> Any:
        """
        将中文字符串值解析为对应的枚举值
        
        Args:
            data_id: 数据ID
            value: 值（中文字符串）

        Returns:
            解析后的值（整数）
        """
        # 使用datapallet中的from_str函数
        return from_str(data_id, value)
    
    # ==================== 数据获取方法 ====================
    
    def get_latest_data(self, data_id: str) -> Tuple[bool, Any]:
        """
        获取最新的数据（供数据托盘调用）
        
        注意：当datapallet需要最新数据时，主动调用此方法
        此时testbed仅需将最新状态返回即可
        
        Args:
            data_id: 数据ID
            
        Returns:
            (success, value): 成功标志和数据值
        """
        with self.lock:
            if data_id in self.current_state:
                return True, self.current_state[data_id]
            else:
                # 如果没有当前状态，返回失败
                return False, None
    
    def get_all_latest_data(self) -> Dict[str, Any]:
        """获取所有最新的数据"""
        with self.lock:
            return self.current_state.copy()
    
    # ==================== 工具方法 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取测试床状态"""
        with self.lock:
            return {
                "playing": self.playing,
                "recorded_count": len(self.recorded_data),
                "current_state": self.current_state,
                "config": {
                    "speed": self.playback_config.speed,
                    "loop": self.playback_config.loop,
                    "interval": self.playback_config.interval
                }
            }

# ==================== 工具函数 ====================

def create_testbed(datapallet: Optional[DataPallet] = None,
                  image_generator: Optional[ImageGenerator] = None) -> TestBed:
    """创建测试床实例"""
    return TestBed(datapallet, image_generator)

if __name__ == "__main__":
    # 简单的自测试
    print("=== 测试床模块自测试 ===")
    
    # 创建测试床（不连接数据托盘）
    tb = create_testbed()
    
    # 测试录制功能
    print("\n2. 测试LLM模拟录制:")
    tb.record_data("早上通勤上班", "早上通勤上班", duration=10.0, interval=1.0)
    print(f"  录制了 {len(tb.recorded_data)} 条数据")
    
    # 测试文件保存和加载
    print("\n4. 测试文件操作:")
    tb.save_recording("test_recording_new.json")
    tb.load_recording("test_recording_new.json")
    
    # 测试播放
    print("\n5. 测试播放控制:")
    tb.start_playback(PlaybackConfig(speed=2.0))
    time.sleep(2.0)
    tb.stop_playback()
    
    print("\n重构后的测试床自测试完成")