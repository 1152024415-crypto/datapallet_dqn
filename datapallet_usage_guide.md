# DataPallet 使用指南

## 快速启动

执行如下命令可以快速启动场景回放：
```
python test_driver.py --mode playback --file 开会.json --time 30
```

执行如下命令可以快速启动批量场景录制
```
python test_driver.py --mode recording
```

通过修改数组`test_scenarios`，以扩展不同的场景，如希望补充录制“在公园散步场景”，修改如下：
```
    # 测试不同的行为描述
    test_scenarios = [
        # 超过50s需要拆分，因为大模型输出max token有限制，虽然写了拆分输出能力，但是效果感觉不是特别好
        # (场景名字, 场景描述, 持续时长, 录制间隔)
        ("开会", "走路去会议室开会，距离20米，会议开始时坐下，然后开始讨论", 30.0, 1.0),
        ("下楼", "从开放的办公区走到楼梯，然后坐电梯到1楼，办公区在5楼，到1楼后穿过大堂，走到办公楼外面", 50.0, 1.0),
        ("公园散步", "持续在公园散步",20.0, 1.0)
    ]
```

## 概述

DataPallet（数据托盘）是一个传感器数据管理模块，负责处理和提供传感器数据，提供订阅和获取接口。它支持：

1. **数据订阅**：通过回调函数接收数据更新
2. **主动获取**：随时获取当前有效数据
3. **数据缓存**：自动管理数据有效期（TTL）
4. **测试床集成**：与测试床（TestBed）配合，支持数据回放和模拟

## 核心概念

### 数据ID
DataPallet 管理以下类型的数据：
- `activity_mode`：用户姿态（站立、慢走、坐着等）
- `Light_Intensity`：环境亮度（极暗、昏暗、明亮等）
- `Sound_Intensity`：环境声音强度（安静、正常、嘈杂等）
- `Location`：位置POI类型（工作场所、家、商业区等）
- `Scence`：场景图像数据（包含图像路径和场景类型）

### 数据有效期（TTL）
每个数据都有有效期，默认60秒。数据过期后，DataPallet 会尝试从测试床获取最新数据。

## 基本使用

### 1. 创建和启动 DataPallet

```python
from datapallet import create_datapallet

# 创建数据托盘实例（默认TTL=60秒）
dp = create_datapallet(ttl=60)

# 数据托盘会自动启动工作线程
# 使用完毕后需要停止
dp.stop()
```

### 2. 设置数据订阅回调

```python
def my_callback(data_id: str, value: Any):
    """自定义回调函数"""
    print(f"收到数据更新: {data_id} = {value}")

# 订阅所有数据
dp.setup(my_callback)

# 或只订阅特定数据
dp.setup(my_callback, data_id="activity_mode")
```

### 3. 主动获取数据

```python
# 获取单个数据
success, value = dp.get("activity_mode")
if success:
    print(f"当前姿态: {value}")
else:
    print("获取数据失败，可能数据无效或不存在")

# 获取所有数据（包括尝试从测试床获取无效数据）
success, all_data = dp.get(None)
if success:
    print(f"成功获取 {len(all_data)} 种数据:")
    for data_id, value in all_data.items():
        print(f"  {data_id}: {value}")
else:
    print("无法获取任何数据")
```

**重要改进**：`get(None)` 现在不仅返回有效数据，对于无效数据还会尝试从测试床获取最新值。这意味着即使数据在托盘中已过期，只要测试床有最新数据，仍然可以获取到。

### 4. 获取数据详细信息

```python
# 获取所有数据的详细信息（包括有效期状态）
data_info = dp.get_data_info()
for data_id, info in data_info.items():
    status = "有效" if info["valid"] else "过期"
    print(f"{data_id}: {status}, 剩余时间: {info['time_remaining']:.1f}秒")
```

## 完整示例

### 示例1：基本数据订阅和获取

```python
import time
from datapallet import create_datapallet
from enums import ActivityMode

def simple_callback(data_id: str, value: Any):
    """简单回调示例"""
    print(f"[回调] {data_id} = {value}")

# 创建数据托盘
dp = create_datapallet(ttl=10)

# 设置回调
dp.setup(simple_callback)

# 模拟接收数据（实际由测试床发送）
dp.receive_data("activity_mode", ActivityMode.STROLLING)

# 等待数据处理
time.sleep(0.5)

# 主动获取数据
success, value = dp.get("activity_mode")
if success:
    print(f"主动获取到的姿态: {value}")

# 停止数据托盘
dp.stop()
```

## 最佳实践

1. **合理设置TTL**：
   - 实时性要求高的场景：设置较短的TTL（如5-10秒）
   - 节省资源的场景：设置较长的TTL（如30-60秒）

2. **错误处理**：
   ```python
   try:
       success, value = dp.get("some_data")
       if not success:
           print("获取数据失败，可能数据无效或不存在")
   except Exception as e:
       print(f"获取数据时出错: {e}")
   ```

3. **资源清理**：
   ```python
   # 确保使用完毕后停止数据托盘
   try:
       # 业务逻辑
       pass
   finally:
       dp.stop()
   ```

4. **回调函数设计**：
   - 保持回调函数简洁，避免长时间阻塞
   - 在回调中避免调用可能死锁的方法
   - 考虑使用队列将数据处理移到其他线程

## 故障排除

### 常见问题

1. **获取不到数据**：
   - 检查数据是否已过期（使用 `get_data_info()`）
   - 确认测试床是否已连接并发送数据
   - 检查数据ID是否正确

2. **回调不触发**：
   - 确认已正确调用 `setup()` 方法
   - 检查测试床是否在发送数据
   - 确认数据托盘已启动

3. **SceneData 处理错误**：
   - 确保正确处理 `SceneData` 对象类型
   - 检查图像文件路径是否存在

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据托盘状态
print("数据托盘状态:")
info = dp.get_data_info()
for k, v in info.items():
    print(f"  {k}: {v}")
```
