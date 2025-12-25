# 数据托盘系统 (Data Pallet System)

一个用于采集、处理和提供手机传感器数据的Python系统，包含数据托盘和测试床两个解耦模块。

## 功能特性

### 数据托盘 (DataPallet)
- **数据采集**: 采集并处理手机用户的各种传感器信息
- **订阅服务**: 提供北向数据订阅服务，数据更新时立即触发回调函数
- **数据获取**: 提供立即获取数据接口，支持单个或批量获取
- **TTL管理**: 支持数据有效期(TTL)管理，自动清理过期数据
- **线程安全**: 多线程设计，确保并发安全

### 测试床 (TestBed)
- **数据模拟**: 按照时间戳播放录制的数据或人工编写的数据流
- **灵活配置**: 支持播放速度、间隔、随机顺序等配置
- **录制回放**: 支持数据录制和回放功能
- **解耦设计**: 与数据托盘通过消息队列完全解耦

## 支持的数据类型

1. **用户姿态** (`activity_mode`)
   - 值类型: `(last_activity, current_activity)`
   - 枚举: 行走、跑步、骑行、乘坐等16种姿态

2. **环境亮度** (`Light_Intensity`)
   - 值类型: `int` (枚举)
   - 枚举: 极暗、昏暗、中等亮度、明亮、刺眼光线

3. **环境声音强度** (`Sound_Intensity`)
   - 值类型: `int` (枚举)
   - 枚举: 非常安静、柔和声音、正常声音、嘈杂、非常嘈杂

4. **位置POI类型** (`Location`)
   - 值类型: `int` (枚举)
   - 枚举: 家、工作、公交站、地铁站、餐厅、商场等21种类型

5. **图像场景分类** (`Scence`)
   - 值类型: `int` (枚举)
   - 枚举: 空、其他、会议室等

## 核心接口

### 数据托盘接口

```python
# 创建数据托盘
dp = DataPallet(default_ttl=60)
dp.start()

# 设置订阅回调
def callback(data_id: str, value: Any):
    print(f"数据更新: {data_id} = {value}")

dp.setup(callback)  # 订阅所有数据
dp.setup(callback, "activity_mode")  # 仅订阅特定数据

# 获取数据
success, value = dp.get("activity_mode")  # 获取单个数据
success, all_data = dp.get(None)  # 获取所有有效数据

# 停止数据托盘
dp.stop()
```

### 测试床接口

```python
# 创建测试床并连接数据托盘
tb = TestBed(dp)

# 发送数据
tb.send_data("activity_mode", (ActivityMode.WALKING, ActivityMode.RUNNING))

# 开始自动播放
config = PlaybackConfig(speed=1.0, interval=1.0, random_order=True)
tb.start_playback(config)

# 录制数据
tb.record_data(duration=10.0, interval=0.5)

# 停止播放
tb.stop_playback()
```

## 快速开始

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd data_pallet

# 安装依赖（本项目使用标准库，无需额外安装）
# 确保Python版本 >= 3.8
```

### 运行演示

```bash
# 运行简单演示
python test_driver.py --mode demo

# 运行所有测试
python test_driver.py --mode all

# 运行特定测试
python test_driver.py --mode basic
python test_driver.py --mode playback
python test_driver.py --mode integration
```

### 基本使用示例

```python
from datapallet import DataPallet, ActivityMode
from testbed import TestBed, PlaybackConfig
import time

# 创建数据托盘
dp = DataPallet(default_ttl=30)
dp.start()

# 设置回调
def my_callback(data_id: str, value):
    print(f"[回调] {data_id}: {value}")

dp.setup(my_callback)

# 创建测试床
tb = TestBed(dp)

# 发送测试数据
tb.send_data("activity_mode", (ActivityMode.WALKING, ActivityMode.RUNNING))
tb.send_data("Light_Intensity", 4)  # 明亮

# 获取数据
success, value = dp.get("activity_mode")
if success:
    print(f"获取到的姿态: {value}")

# 运行一段时间
time.sleep(2.0)

# 清理
dp.stop()
```

## 项目结构

```
data_pallet/
├── datapallet.py          # 数据托盘核心模块
├── testbed.py            # 测试床模块
├── test_driver.py        # 测试驱动程序
├── requirements.txt      # 项目依赖
├── README.md            # 项目说明
└── plans/
    └── architecture_design.md  # 架构设计文档
```

## 架构设计

### 组件图
```
┌─────────────────┐    消息队列    ┌─────────────────┐
│                 │ ────────────> │                 │
│    测试床       │                │   数据托盘      │
│  (TestBed)      │ <──────────── │  (DataPallet)   │
│                 │   数据获取     │                 │
└─────────────────┘                └─────────────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────┐                ┌─────────────────┐
│  数据生成器     │                │  订阅回调       │
│  录制/回放      │                │  数据存储       │
└─────────────────┘                └─────────────────┘
```

### 数据流
1. 测试床生成或播放数据
2. 数据通过消息队列发送到数据托盘
3. 数据托盘工作线程处理消息
4. 更新数据存储并触发回调
5. 北向应用通过get接口获取数据

### 并发模型
- **数据托盘**: 主线程 + 工作线程 + 消息队列
- **测试床**: 独立的播放线程
- **线程安全**: 使用锁和条件变量确保并发安全
- **非阻塞**: get函数可阻塞等待，但不阻塞数据托盘工作线程

## 配置选项

### 数据托盘配置
- `default_ttl`: 默认数据有效期（秒），默认60秒
- 可通过`DataPallet(default_ttl=30)`构造函数配置

### 测试床配置
- `PlaybackConfig`: 播放配置对象
  - `speed`: 播放速度倍数（默认1.0）
  - `interval`: 数据发送间隔（秒，默认1.0）
  - `random_order`: 是否随机顺序播放（默认False）
  - `loop`: 是否循环播放（默认False）

## 测试覆盖

测试驱动程序包含以下测试场景：

1. **基本功能测试**: 测试数据托盘和测试床的基本功能
2. **订阅模式测试**: 测试不同的订阅模式（全部/单个数据）
3. **播放功能测试**: 测试测试床的播放功能
4. **get阻塞测试**: 测试get函数的阻塞等待行为
5. **集成场景测试**: 测试完整的数据流和集成场景

## 扩展指南

### 添加新的数据类型

1. 在`datapallet.py`中添加新的枚举类
2. 在`DataPallet._initialize_data_store()`中初始化新数据
3. 在`TestBed.data_generators`中添加数据生成器
4. 在`DataPallet.format_value()`中添加格式化支持

### 自定义数据源

1. 继承`TestBed`类并重写数据生成方法
2. 实现自定义的数据播放逻辑
3. 通过`send_data()`方法将数据发送到数据托盘

### 集成到现有系统

1. 将数据托盘作为服务运行
2. 通过回调接口接收数据更新
3. 通过get接口按需获取数据
4. 根据业务需求调整TTL和并发配置

## 性能考虑

- **内存使用**: 数据存储使用字典，内存占用较小
- **并发性能**: 使用线程和队列，适合中等并发场景
- **扩展性**: 模块化设计，易于扩展新的数据类型和功能
- **实时性**: 消息队列确保数据及时处理，回调机制保证实时通知

## 故障排除

### 常见问题

1. **回调函数不触发**
   - 检查数据ID是否正确
   - 确认数据托盘是否已启动
   - 检查回调函数签名是否正确

2. **get函数超时**
   - 检查数据TTL是否已过期
   - 确认测试床是否在发送数据
   - 调整timeout参数

3. **线程阻塞**
   - 确保回调函数执行时间较短
   - 避免在回调函数中进行耗时操作
   - 使用异步处理复杂逻辑

### 调试建议

1. 启用详细日志
2. 使用测试驱动程序验证功能
3. 检查数据托盘状态信息
4. 验证消息队列是否正常工作

## 许可证

本项目采用MIT许可证。详见LICENSE文件（如有）。

## 贡献指南

欢迎提交Issue和Pull Request来改进本项目。

1. Fork本仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送电子邮件到[邮箱地址]

---

**注意**: 本项目为原型系统，适用于学习和研究目的。在生产环境中使用时，请根据具体需求进行适当的修改和测试。