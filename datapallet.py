"""
数据托盘 (DataPallet) 模块
负责处理和提供传感器数据，提供订阅和获取接口
"""

import threading
import time
import queue
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta

# 从enums模块导入枚举定义和工具函数
from enums import (
    ActivityMode, LightIntensity, SoundIntensity,
    LocationType, SceneType, SceneData, to_str, from_str
)

# ==================== 数据结构定义 ====================

@dataclass
class DataItem:
    """数据项，包含值、时间戳和TTL"""
    value: Any
    timestamp: datetime
    ttl: timedelta  # 数据有效时间
    
    def is_valid(self) -> bool:
        """检查数据是否仍然有效（未过期）"""
        return datetime.now() < self.timestamp + self.ttl
    
    def time_remaining(self) -> float:
        """返回剩余有效时间（秒），负数表示已过期"""
        remaining = (self.timestamp + self.ttl) - datetime.now()
        return remaining.total_seconds()

# ==================== 数据托盘核心类 ====================

class DataPallet:
    """
    数据托盘核心类
    负责管理传感器数据，提供订阅和获取接口
    """
    
    def __init__(self, default_ttl: int = 60):
        """
        初始化数据托盘
        
        Args:
            default_ttl: 默认数据有效时间（秒）
        """
        self.default_ttl = timedelta(seconds=default_ttl)
        
        # 数据存储：id -> DataItem
        self.data_store: Dict[str, DataItem] = {}
        
        # 订阅回调函数
        self.callbacks: Dict[str, List[Callable[[str, Any], None]]] = {}
        
        # 消息队列（用于接收测试床数据）
        self.message_queue = queue.Queue()
        
        # 线程同步
        self.lock = threading.RLock()
        
        # 工作线程
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.running = False
        
        # 连接到测试床的引用（可选）
        self.testbed = None
        
        # 初始化数据存储
        self._initialize_data_store()
    
    def _initialize_data_store(self):
        """初始化数据存储，设置默认值（立即过期）"""
        with self.lock:
            # 使用过去的时间戳（1天前），使初始数据立即过期
            expired_timestamp = datetime.now() - timedelta(days=1)
            # 初始化所有数据ID
            # activity_mode存储为tuple格式：(前一个状态, 当前状态)
            self.data_store["activity_mode"] = DataItem(
                value=(ActivityMode.NULL, ActivityMode.NULL),
                timestamp=expired_timestamp,
                ttl=self.default_ttl
            )
            self.data_store["Light_Intensity"] = DataItem(
                value=LightIntensity.NULL,
                timestamp=expired_timestamp,
                ttl=self.default_ttl
            )
            self.data_store["Sound_Intensity"] = DataItem(
                value=SoundIntensity.NULL,
                timestamp=expired_timestamp,
                ttl=self.default_ttl
            )
            self.data_store["Location"] = DataItem(
                value=LocationType.NULL,
                timestamp=expired_timestamp,
                ttl=self.default_ttl
            )
            self.data_store["Scence"] = DataItem(
                value=SceneData.from_scene_type(SceneType.NULL),
                timestamp=expired_timestamp,
                ttl=self.default_ttl
            )
    
    def start(self):
        """启动数据托盘工作线程"""
        if not self.running:
            self.running = True
            self.worker_thread.start()
            print("数据托盘已启动")
    
    def stop(self):
        """停止数据托盘"""
        self.running = False
        # 发送停止信号到消息队列
        self.message_queue.put(None)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        print("数据托盘已停止")
    
    def _worker_loop(self):
        """工作线程主循环，处理消息队列"""
        while self.running:
            try:
                # 从消息队列获取数据（阻塞，但可超时）
                message = self.message_queue.get(timeout=1.0)
                if message is None:  # 停止信号
                    break
                
                # 处理消息
                self._process_message(message)
                
            except queue.Empty:
                # 超时，继续循环
                continue
            except Exception as e:
                print(f"工作线程处理消息时出错: {e}")
    
    def _process_message(self, message):
        """处理来自测试床的消息"""
        if not isinstance(message, dict):
            return
        
        data_id = message.get("id")
        value = message.get("value")
        timestamp = message.get("timestamp")
        
        if data_id is None or value is None:
            return
        
        # 更新时间戳
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            # 如果是字符串，尝试解析
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
        
        # 更新数据
        self._update_data(data_id, value, timestamp)

        # 触发回调（在锁外执行，避免死锁）
        self._trigger_callbacks(data_id, value)
    
    def _update_data(self, data_id: str, value: Any, timestamp: datetime):
        """
        更新数据并触发回调
        
        Args:
            data_id: 数据ID
            value: 数据值
            timestamp: 时间戳
        """
        with self.lock:
            # 对于activity_mode，创建tuple格式：(前一个状态, 当前状态)
            if data_id == "activity_mode":
                # 获取当前值作为前一个状态
                previous_value = self.data_store[data_id].value[1]
                
                # 创建tuple值
                value = (previous_value, value)
            # 对于Scence，确保值是SceneData类型
            elif data_id == "Scence":
                # 如果value已经是SceneData对象，直接使用
                if isinstance(value, SceneData):
                    pass  # 已经是SceneData，不需要转换
                # 如果value是字典，使用from_dict转换
                elif isinstance(value, dict):
                    value = SceneData.from_dict(value)
                # 如果value是整数或SceneType，转换为SceneData
                elif isinstance(value, int) or isinstance(value, SceneType):
                    scene_type = SceneType(value) if isinstance(value, int) else value
                    value = SceneData.from_scene_type(scene_type)
                else:
                    # 其他情况，创建默认的SceneData
                    value = SceneData.from_scene_type(SceneType.NULL)
            
            # 创建新的数据项
            data_item = DataItem(
                value=value,
                timestamp=timestamp,
                ttl=self.default_ttl
            )
            
            # 存储数据
            self.data_store[data_id] = data_item
    
    def _trigger_callbacks(self, data_id: str, value: Any):
        """触发指定数据ID的所有回调函数"""
        if data_id in self.callbacks:
            for callback in self.callbacks[data_id]:
                try:
                    callback(data_id, value)
                except Exception as e:
                    print(f"回调函数执行出错 (id={data_id}): {e}")
    
    def setup(self, callback: Callable[[str, Any], None], data_id: Optional[str] = None) -> Optional[Callable]:
        """
        设置数据订阅回调函数
        
        Args:
            callback: 回调函数，参数为 (data_id, value)
            data_id: 要订阅的数据ID，如果为None则订阅所有数据
            
        Returns:
            返回之前设置的回调函数（如果有），否则返回None
        """
        with self.lock:
            if data_id is None:
                # 订阅所有数据
                data_ids = list(self.data_store.keys())
                last_callbacks = []
                for did in data_ids:
                    last_cb = self._setup_single(did, callback)
                    if last_cb:
                        last_callbacks.append(last_cb)
                return last_callbacks[0] if last_callbacks else None
            else:
                # 订阅单个数据
                return self._setup_single(data_id, callback)
    
    def _setup_single(self, data_id: str, callback: Callable[[str, Any], None]) -> Optional[Callable]:
        """为单个数据ID设置回调"""
        if data_id not in self.callbacks:
            self.callbacks[data_id] = []
        
        # 保存之前的回调（如果有）
        last_callback = self.callbacks[data_id][-1] if self.callbacks[data_id] else None
        
        # 添加新回调
        self.callbacks[data_id].append(callback)
        
        return last_callback
    
    def get(self, data_id: Optional[str] = None) -> Tuple[bool, Any]:
        """
        获取数据
        
        Args:
            data_id: 数据ID，如果为None则获取所有数据（包括尝试从测试床获取无效数据）
            
        Returns:
            (success, value): 成功标志和数据值
        """
        with self.lock:
            if data_id is None:
                # 获取所有数据，对于无效的数据尝试从测试床获取
                result = {}
                
                for did, item in self.data_store.items():
                    if item.is_valid():
                        # 数据有效，直接使用
                        result[did] = item.value
                    elif self.testbed is not None:
                        # 数据无效，尝试从测试床获取最新数据
                        try:
                            success, value = self.testbed.get_latest_data(did)
                            if success:
                                # 更新数据托盘并返回获取到的值
                                self._update_data(did, value, datetime.now())
                                result[did] = value
                        except Exception as e:
                            print(f"从测试床获取数据失败 ({did}): {e}")
                
                # 如果至少获取到一些数据，返回成功
                if result:
                    return True, result
                else:
                    return False, None
            
            # 检查数据是否存在
            if data_id not in self.data_store:
                return False, f"未知的数据ID: {data_id}"
            
            # 获取指定数据
            data_item = self.data_store[data_id]
            
            # 如果数据有效，直接返回
            if data_item.is_valid():
                return True, data_item.value
            
            # 数据无效，尝试从测试床获取（如果已连接）
            if self.testbed is not None:
                try:
                    # 从测试床获取最新数据
                    success, value = self.testbed.get_latest_data(data_id)
                    if success:
                        # 更新数据托盘
                        self._update_data(data_id, value, datetime.now())
                        return True, value
                except Exception as e:
                    print(f"从测试床获取数据失败: {e}")
            
            # 返回False，表明无法获取有效数据
            return False, None
    
    def connect_testbed(self, testbed):
        """连接到测试床"""
        self.testbed = testbed
    
    def receive_data(self, data_id: str, value: Any, timestamp: datetime):
        """
        接收数据（供测试床调用）
        
        Args:
            data_id: 数据ID
            value: 数据值
        """
        # 将数据放入消息队列，由工作线程处理
        message = {
            "id": data_id,
            "value": value,
            "timestamp": datetime.now()
        }
        self.message_queue.put(message)
    
    # 可视化所有数据
    def get_data_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有数据的详细信息"""
        with self.lock:
            result = {}
            for data_id, item in self.data_store.items():
                result[data_id] = {
                    "value": self.format_value(data_id, item.value),
                    "timestamp": item.timestamp,
                    "valid": item.is_valid(),
                    "time_remaining": item.time_remaining()
                }
            return result
        
    # 可视化指定数据
    def format_value(self, data_id: str, value: Any) -> str:
        """格式化数据值为可读字符串"""
        return to_str(data_id, value)

# ==================== 工具函数 ====================

def create_datapallet(ttl: int = 60) -> DataPallet:
    """创建并启动数据托盘实例"""
    dp = DataPallet(default_ttl=ttl)
    dp.start()
    return dp


if __name__ == "__main__":
    # 简单的自测试
    dp = create_datapallet()
    
    # 定义回调函数
    def test_callback(data_id: str, value: Any):
        formatted = dp.format_value(data_id, value)
        print(f"回调: {data_id} = {formatted}")
    
    # 设置回调
    dp.setup(test_callback)
    
    # 模拟接收数据
    dp.receive_data("activity_mode", ActivityMode.STROLLING, datetime.now())
    dp.receive_data("Light_Intensity", LightIntensity.BRIGHT, datetime.now())
    
    # 等待数据处理
    time.sleep(0.5)
    
    # 获取数据
    success, value = dp.get("activity_mode")
    if success:
        print(f"获取到数据: activity_mode = {dp.format_value('activity_mode', value)}")
    
    # 停止数据托盘
    dp.stop()