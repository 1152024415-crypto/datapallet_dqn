# 数据托盘和测试床系统架构设计

## 1. 系统概述

### 1.1 项目背景
设计一个用于移动设备的数据托盘系统，用于采集和处理手机用户的各种传感器信息，并提供数据订阅服务。同时设计一个解耦的测试床模块，用于播放录制的数据或人工编写的数据流。

### 1.2 技术栈选择
- **编程语言**: Python 3.8+
- **并发模型**: 多线程 + 消息队列
- **数据存储**: 内存存储 + TTL管理
- **通信方式**: 进程内消息队列 (queue.Queue)
- **部署环境**: 移动设备本地服务

### 1.3 核心需求
1. 数据托盘提供北向数据订阅服务（回调触发）
2. 数据托盘提供立即获取数据接口
3. 测试床与数据托盘解耦，按时间戳播放数据
4. 数据托盘有独立运行线程和消息队列
5. get函数需要阻塞，但不能阻塞数据托盘线程

## 2. 整体架构图

### 2.1 组件架构
```
┌─────────────────────────────────────────────────────────────┐
│                     数据托盘系统 (Data Pallet)                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  数据采集器  │    │  数据处理    │    │  数据存储    │     │
│  │ (Collector) │───▶│ (Processor) │───▶│  (Storage)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                         │              │          │
│         ▼                         ▼              ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  传感器接口  │    │  TTL管理器   │    │  订阅管理器   │     │
│  │ (Sensors)   │    │ (TTL Manager)│    │(Subscription│     │
│  └─────────────┘    └─────────────┘    │   Manager)  │     │
│                                         └─────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                     消息队列系统 (Message Queue)             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 queue.Queue (线程安全)                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         ▲
         │
┌────────┴─────────────────────────────────────────────────────┐
│                     测试床系统 (Test Bed)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  数据加载器  │    │  时间调度器  │    │  数据发送器  │     │
│  │ (DataLoader)│───▶│ (Scheduler) │───▶│ (Sender)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │  配置文件    │    │  数据文件    │                        │
│  │ (Config)    │    │ (Data Files)│                        │
│  └─────────────┘    └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流架构
```
传感器/测试床 → 消息队列 → 数据托盘主线程 → 数据处理 → 数据存储
                                    ↓
                            触发订阅回调 → 客户端
                                    ↓
                              get()请求 → 返回数据
```

## 3. 模块划分和接口设计

### 3.1 数据托盘模块

#### 3.1.1 DataPallet 核心类
```python
class DataPallet:
    """
    数据托盘核心类，管理数据采集、存储和订阅
    """
    def __init__(self, config: Dict = None):
        self.message_queue = queue.Queue()
        self.data_storage = DataStorage()
        self.subscription_manager = SubscriptionManager()
        self.ttl_manager = TTLManager()
        self.running = False
        self.worker_thread = None
        
    def setup(self, callback: Callable[[str, Any], None]) -> Callable:
        """
        设置数据更新回调函数
        
        Args:
            callback: 回调函数，参数为 (data_id, value)
            
        Returns:
            上一个回调函数（用于链式调用）
        """
        pass
        
    def get(self, data_id: str = None) -> Tuple[bool, Any]:
        """
        获取数据
        
        Args:
            data_id: 数据ID，为空时获取所有TTL有效的数据
            
        Returns:
            (success, value): 成功标志和数据值
        """
        pass
        
    def start(self):
        """启动数据托盘"""
        pass
        
    def stop(self):
        """停止数据托盘"""
        pass
        
    def put_data(self, data_id: str, value: Any, timestamp: float = None):
        """
        向消息队列放入数据（供测试床或传感器调用）
        """
        pass
```

#### 3.1.2 数据存储模块
```python
class DataStorage:
    """
    数据存储管理，支持TTL
    """
    def __init__(self):
        self.data = {}  # data_id -> DataItem
        
    class DataItem:
        def __init__(self, value: Any, timestamp: float, ttl: int = 60):
            self.value = value
            self.timestamp = timestamp
            self.ttl = ttl  # 生存时间（秒）
            
    def update(self, data_id: str, value: Any, ttl: int = 60):
        """更新数据"""
        pass
        
    def get(self, data_id: str) -> Optional[Any]:
        """获取单个数据"""
        pass
        
    def get_all_valid(self) -> Dict[str, Any]:
        """获取所有有效数据"""
        pass
        
    def cleanup_expired(self):
        """清理过期数据"""
        pass
```

#### 3.1.3 订阅管理模块
```python
class SubscriptionManager:
    """
    订阅管理，支持多个回调函数
    """
    def __init__(self):
        self.callbacks = []
        
    def add_callback(self, callback: Callable[[str, Any], None]) -> Callable:
        """添加回调函数"""
        pass
        
    def remove_callback(self, callback: Callable):
        """移除回调函数"""
        pass
        
    def notify_all(self, data_id: str, value: Any):
        """通知所有订阅者"""
        pass
```

### 3.2 测试床模块

#### 3.2.1 TestBed 核心类
```python
class TestBed:
    """
    测试床，按时间戳播放数据
    """
    def __init__(self, data_pallet: DataPallet = None):
        self.data_pallet = data_pallet
        self.playback_thread = None
        self.running = False
        self.schedule = []
        
    def load_data_file(self, filepath: str, format: str = "json"):
        """
        加载数据文件
        
        Args:
            filepath: 文件路径
            format: 数据格式 (json, csv, yaml)
        """
        pass
        
    def add_data_point(self, timestamp: float, data_id: str, value: Any):
        """
        添加数据点
        
        Args:
            timestamp: 时间戳（相对时间或绝对时间）
            data_id: 数据ID
            value: 数据值
        """
        pass
        
    def start_playback(self, speed: float = 1.0):
        """
        开始播放
        
        Args:
            speed: 播放速度倍数
        """
        pass
        
    def stop_playback(self):
        """停止播放"""
        pass
        
    def connect_to_pallet(self, data_pallet: DataPallet):
        """连接到数据托盘"""
        pass
```

## 4. 数据流设计

### 4.1 数据类型定义
```python
# 数据ID常量
DATA_IDS = {
    "activity_mode": "用户姿态",
    "light_intensity": "环境亮度", 
    "sound_intensity": "环境声音强度",
    "location": "位置POI类型",
    "scene": "图像场景分类"
}

# 数据值格式
DataValue = Union[int, float, str, Dict, List]
```

### 4.2 消息格式
```python
@dataclass
class DataMessage:
    """数据消息格式"""
    data_id: str
    value: Any
    timestamp: float  # 时间戳（秒）
    source: str  # 数据来源（sensor/testbed）
    ttl: int = 60  # 默认TTL（秒）
```

### 4.3 数据流序列
1. **数据产生**: 传感器或测试床产生数据
2. **消息入队**: 数据封装为DataMessage放入消息队列
3. **数据处理**: 数据托盘工作线程从队列取出消息
4. **数据存储**: 更新数据存储，应用TTL
5. **订阅通知**: 触发所有订阅者的回调函数
6. **数据查询**: 客户端通过get()接口查询数据

## 5. 线程/并发模型

### 5.1 线程设计
```
主线程 (Main Thread)
├── 数据托盘工作线程 (DataPallet Worker Thread)
│   ├── 消息队列消费
│   ├── 数据处理
│   ├── 订阅通知
│   └── TTL清理
├── 测试床播放线程 (TestBed Playback Thread)
│   ├── 时间调度
│   └── 数据发送
└── 客户端线程 (Client Threads)
    ├── get()调用（阻塞）
    └── 回调函数执行
```

### 5.2 并发控制
1. **消息队列**: 使用`queue.Queue`实现线程安全的生产者-消费者模式
2. **数据存储**: 使用`threading.Lock`保护共享数据
3. **订阅管理**: 回调函数在独立线程中执行，避免阻塞工作线程
4. **get()阻塞**: 使用条件变量(`threading.Condition`)实现等待-通知机制

### 5.3 get函数详细逻辑设计

根据需求，get函数的行为逻辑如下：

1. **数据检查阶段**：
   - 首先检查数据托盘中是否有请求数据的有效版本（未过期）
   - 如果有有效数据，立即返回该数据
   - 如果没有有效数据，进入数据获取阶段

2. **数据获取阶段**：
   - 向测试床发送数据请求
   - 测试床根据当前时间线提供最新数据
   - 将获取的数据放入消息队列
   - 等待数据托盘处理并存储

3. **阻塞等待阶段**：
   - 使用条件变量等待数据更新
   - 设置合理的超时时间避免永久阻塞
   - 收到数据更新通知后返回数据

4. **返回结果**：
   - 返回获取的数据（单个或全部）
   - 如果超时或失败，返回错误信息

### 5.4 关键实现细节
```python
class DataPallet:
    def __init__(self, testbed: TestBed = None):
        self.message_queue = queue.Queue(maxsize=1000)
        self.data_lock = threading.Lock()
        self.data_condition = threading.Condition(self.data_lock)
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.testbed = testbed  # 可选的测试床连接
        
    def _worker_loop(self):
        """工作线程主循环"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=0.1)
                self._process_message(message)
                # 处理完成后通知等待的get调用
                with self.data_condition:
                    self.data_condition.notify_all()
            except queue.Empty:
                continue
                
    def get(self, data_id: str = None, timeout: float = 2.0) -> Tuple[bool, Any]:
        """
        获取数据（阻塞但不阻塞数据托盘线程）
        
        逻辑：
        1. 首先检查是否有有效数据
        2. 如果没有，尝试从测试床获取一次
        3. 等待数据更新或超时
        
        Args:
            data_id: 数据ID，为空时获取所有TTL有效的数据
            timeout: 超时时间（秒）
            
        Returns:
            (success, value): 成功标志和数据值
        """
        # 阶段1：检查现有数据
        with self.data_lock:
            if data_id:
                # 检查单个数据
                data = self.data_storage.get(data_id)
                if data is not None and not self.data_storage.is_expired(data_id):
                    return True, data.value
            else:
                # 检查所有数据
                all_data = self.data_storage.get_all_valid()
                if all_data:
                    return True, all_data
        
        # 阶段2：从测试床获取数据（如果连接了测试床）
        if self.testbed and data_id:
            # 请求测试床提供最新数据
            testbed_data = self.testbed.get_latest_data(data_id)
            if testbed_data:
                # 将数据放入消息队列
                self.put_data(data_id, testbed_data.value, testbed_data.timestamp)
        
        # 阶段3：等待数据更新
        with self.data_condition:
            # 等待数据更新通知
            if not self.data_condition.wait(timeout=timeout):
                return False, "Timeout waiting for data"
            
            # 再次检查数据
            if data_id:
                data = self.data_storage.get(data_id)
                if data is not None and not self.data_storage.is_expired(data_id):
                    return True, data.value
                else:
                    return False, "Data not available after update"
            else:
                all_data = self.data_storage.get_all_valid()
                if all_data:
                    return True, all_data
                else:
                    return False, "No valid data available"
                    
    def _process_message(self, message: DataMessage):
        """处理数据消息"""
        # 更新数据存储
        self.data_storage.update(
            message.data_id,
            message.value,
            message.timestamp,
            message.ttl
        )
        
        # 触发订阅回调
        self.subscription_manager.notify_all(message.data_id, message.value)
```

### 5.5 测试床数据获取接口
```python
class TestBed:
    def get_latest_data(self, data_id: str) -> Optional[DataMessage]:
        """
        根据当前时间线获取最新数据
        
        Args:
            data_id: 请求的数据ID
            
        Returns:
            DataMessage对象或None
        """
        # 根据当前播放位置获取数据
        current_time = self.get_current_playback_time()
        
        # 查找最接近当前时间的数据点
        data_point = self.find_nearest_data_point(data_id, current_time)
        
        if data_point:
            return DataMessage(
                data_id=data_id,
                value=data_point.value,
                timestamp=data_point.timestamp,
                source="testbed",
                ttl=self.get_ttl_for_data_id(data_id)
            )
        return None
```

## 6. 文件结构建议

```
data_pallet/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── datapallet/
│   │   ├── __init__.py
│   │   ├── core.py           # DataPallet核心类
│   │   ├── storage.py        # 数据存储模块
│   │   ├── subscription.py   # 订阅管理模块
│   │   ├── ttl_manager.py    # TTL管理模块
│   │   └── message.py        # 消息格式定义
│   ├── testbed/
│   │   ├── __init__.py
│   │   ├── core.py           # TestBed核心类
│   │   ├── scheduler.py      # 时间调度器
│   │   ├── loader.py         # 数据加载器
│   │   └── formats/          # 数据格式支持
│   │       ├── json_loader.py
│   │       ├── csv_loader.py
│   │       └── yaml_loader.py
│   └── sensors/
│       ├── __init__.py
│       ├── base.py           # 传感器基类
│       ├── mock_sensor.py    # 模拟传感器
│       └── android/          # Android传感器实现
│           ├── __init__.py
│           └── sensor_bridge.py
├── config/
│   ├── default.yaml
│   ├── development.yaml
│   └── production.yaml
├── data/
│   ├── samples/              # 示例数据文件
│   │   ├── sample_data.json
│   │   ├── sample_data.csv
│   │   └── playback_sequence.yaml
│   └── recordings/           # 录制数据
├── tests/
│   ├── __init__.py
│   ├── test_datapallet.py
│   ├── test_testbed.py
│   ├── test_integration.py
│   └── fixtures/             # 测试数据
├── examples/
│   ├── basic_usage.py
│   ├── subscription_demo.py
│   ├── testbed_demo.py
│   └── sensor_integration.py
└── docs/
    ├── architecture.md
    ├── api_reference.md
    └── user_guide.md
```

## 7. 配置管理

### 7.1 默认配置
```yaml
# config/default.yaml
datapallet:
  queue_size: 1000
  worker_interval: 0.01  # 工作线程检查间隔（秒）
  cleanup_interval: 5    # TTL清理间隔（秒）
  
  ttl_settings:
    activity_mode: 300     # 5分钟
    light_intensity: 60    # 1分钟
    sound_intensity: 60    # 1分钟
    location: 300          # 5分钟
    scene: 300             # 5分钟
    
testbed:
  playback_speed: 1.0
  realtime_mode: true
  loop_playback: false
  
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 8. get函数行为详细设计

### 8.1 get函数核心逻辑流程

根据用户需求，get函数的完整行为逻辑如下：

```mermaid
graph TD
    A[调用get(data_id)] --> B{检查数据托盘是否有<br>有效数据?}
    B -->|有| C[立即返回有效数据]
    B -->|无| D{是否连接了测试床?}
    D -->|是| E[向测试床请求最新数据]
    D -->|否| F[直接进入等待阶段]
    E --> G[测试床根据时间线<br>提供最新数据]
    G --> H[数据放入消息队列]
    H --> I[数据托盘处理并存储]
    F --> J[等待数据更新]
    I --> J
    J --> K{等待超时?}
    K -->|是| L[返回超时错误]
    K -->|否| M[检查数据是否有效]
    M --> N[返回获取的数据]
    C --> O[结束]
    L --> O
    N --> O
```

### 8.2 数据有效性检查规则

1. **TTL检查**：
   - 每个数据项都有独立的TTL配置
   - 默认TTL为1分钟，可通过配置修改
   - 检查公式：`当前时间 - 数据时间戳 < TTL`

2. **数据优先级**：
   - 数据托盘中的有效数据优先返回
   - 只有当数据托盘中没有有效数据时，才向测试床请求
   - 测试床数据作为临时补充，不替代正常的数据流

### 8.3 阻塞机制设计

1. **非阻塞数据托盘线程**：
   - get函数的阻塞只影响调用线程
   - 数据托盘工作线程继续正常运行
   - 使用条件变量实现线程间通信

2. **超时控制**：
   - 默认超时时间：2秒
   - 可配置超时参数
   - 超时后返回错误，避免永久阻塞

3. **通知机制**：
   - 数据更新时通知所有等待的get调用
   - 使用`threading.Condition.notify_all()`
   - 避免虚假唤醒问题

### 8.4 测试床集成模式

1. **按时间线获取数据**：
   - 测试床维护一个时间线调度器
   - 根据当前播放位置提供数据
   - 支持实时模式和加速播放

2. **数据格式一致性**：
   - 测试床数据与传感器数据格式相同
   - 使用统一的`DataMessage`格式
   - 确保数据处理的兼容性

## 9. 关键设计决策

### 9.1 解耦设计
- 测试床与数据托盘通过消息队列解耦
- 传感器接口通过统一的消息格式解耦
- 订阅系统通过回调函数解耦
- get函数与数据采集线程解耦

### 9.2 性能考虑
- 内存存储：适合移动设备环境
- 异步处理：避免阻塞数据采集
- 批量通知：优化回调触发性能
- 条件变量：减少CPU空转

### 9.3 可扩展性
- 插件式传感器支持
- 可配置的数据格式
- 模块化的架构设计
- 灵活的TTL配置

### 9.4 可靠性设计
- 数据有效性检查
- 超时机制防止死锁
- 错误处理和日志记录
- 资源清理和释放

## 10. 数据流优化设计

### 10.1 高效的数据查询
1. **缓存机制**：有效数据缓存在内存中，快速响应查询
2. **懒加载**：只有在需要时才从测试床获取数据
3. **批量处理**：多个get请求可以共享同一个数据更新

### 10.2 内存管理
1. **TTL自动清理**：定期清理过期数据，释放内存
2. **队列大小限制**：防止消息队列无限增长
3. **数据压缩**：对于大型数据值，考虑压缩存储

### 10.3 并发安全
1. **锁粒度优化**：细粒度锁减少竞争
2. **无锁数据结构**：在适当场景使用无锁队列
3. **线程局部存储**：减少共享数据访问

## 11. 实施路线图

### 11.1 第一阶段：核心功能实现（1-2周）
- 实现数据托盘基本框架
- 实现消息队列系统
- 实现数据存储和TTL管理
- 实现get函数基本逻辑

### 11.2 第二阶段：测试床集成（1周）
- 实现测试床数据加载和播放
- 实现时间线调度器
- 集成测试床与数据托盘
- 实现按需数据获取

### 11.3 第三阶段：功能完善（1周）
- 配置文件支持
- 日志系统集成
- 错误处理机制
- 性能优化

### 11.4 第四阶段：测试验证（1周）
- 单元测试覆盖
- 集成测试
- 性能测试
- 内存泄漏检测

## 12. 风险评估与缓解

### 12.1 技术风险
1. **并发死锁**：通过超时机制和锁顺序规范缓解
2. **内存泄漏**：定期内存检查和资源清理
3. **性能瓶颈**：性能测试和优化迭代

### 12.2 业务风险
1. **数据延迟**：优化消息队列处理速度
2. **数据不一致**：加强数据验证和同步机制
3. **系统稳定性**：完善的错误恢复机制

### 12.3 缓解措施
- 代码审查和测试覆盖
- 监控和日志系统
- 渐进式部署策略
- 回滚机制

## 13. 总结

本架构设计为数据托盘和测试床系统提供了一个完整、可扩展的解决方案，具有以下特点：

1. **清晰的模块划分**：数据托盘、测试床、传感器接口明确分离
2. **高效的并发模型**：多线程+消息队列，确保系统响应性
3. **灵活的数据获取**：支持立即返回和按需从测试床获取
4. **可靠的数据管理**：TTL机制保证数据新鲜度
5. **良好的扩展性**：插件式设计支持未来功能扩展

该设计充分考虑了移动设备环境的限制，同时满足了实时数据采集、处理和分发的需求。实施过程中可以根据实际情况进行适当调整和优化。