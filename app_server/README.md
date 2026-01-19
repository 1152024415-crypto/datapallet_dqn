# 智能动作推荐系统 - 服务器端

## 系统概述
这是一个智能动作推荐系统的服务器端，为HarmonyOS卡片应用提供动作推荐数据。系统采用HTTP轮询架构，支持实时数据更新和卡片展示。

## 快速开始

### 1. 启动服务器
```bash
# 进入服务器目录
cd app_server

# 启动HTTP服务器（默认端口8080）
python server.py
```

服务器启动后显示：
```
============================================================
HTTP轮询服务器启动 (新统一格式)
============================================================
服务器地址: http://0.0.0.0:8080
API接口:
  GET  /latest-recommendation  - 获取最新推荐数据
  POST /update-recommendation  - 更新推荐数据
============================================================
```

### 2. 启动HarmonyOS应用
1. 克隆应用代码仓：
   ```bash
   git clone https://gitee.com/xiaxingyu/datapallet_hap
   ```

2. 配置应用IP地址：
   - 文件位置：`entry/src/main/ets/common/CommonConstants.ets`
   - 修改第7行：
     ```typescript
     static readonly SERVER_URL: string = 'http://127.0.0.1:8080';
     ```
   - 如果服务器在其他设备运行，将`127.0.0.1`改为服务器IP地址

### 3. 发送测试数据
```bash
# 在服务器目录执行测试脚本
python test.py
```

测试脚本会：
1. 随机生成动作数据（probe/recommend类型）
2. 使用test.png图片生成base64数据
3. 发送到服务器`http://127.0.0.1:8080/update-recommendation`

### 4. 应用展示
应用启动后：
1. 卡片每2秒自动轮询服务器获取数据
2. 数据变化时卡片会振动反馈（800ms）
3. 显示动作名称、场景分类和图片预览
4. 支持三种动作类型：
   - 🔍 **Probe**（蓝色）：信息查询类动作
   - 💡 **Recommend**（绿色）：推荐执行类动作
   - ⚪ **None**（灰色）：无动作状态

## IP地址配置

### 服务器端
- **默认配置**：`0.0.0.0:8080`（server.py第143行）
- **修改方法**：编辑`server.py`文件：
  ```python
  def run_server(host="0.0.0.0", port=8080):
  ```

### HarmonyOS应用端
- **配置文件**：`entry/src/main/ets/common/CommonConstants.ets`
- **配置项**：第7行`SERVER_URL`
- **示例**：
  ```typescript
  // 本地服务器
  static readonly SERVER_URL: string = 'http://127.0.0.1:8080';
  
  // 局域网服务器（替换为实际IP）
  static readonly SERVER_URL: string = 'http://192.168.1.100:8080';
  ```

## API接口

### 1. 获取最新数据
```
GET /latest-recommendation
```
**响应示例**：
```json
{
  "action_type": "probe",
  "action_name": "QUERY_LOC_GPS",
  "scene_category": "transportation",
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### 2. 更新数据
```
POST /update-recommendation
Content-Type: application/json
```
**请求体**：
```json
{
  "action_type": "recommend",
  "action_name": "transit_QR_code",
  "scene_category": "transportation",
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## 数据格式说明

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| action_type | string | 动作类型 | `"probe"`, `"recommend"`, `"none"` |
| action_name | string | 动作名称 | `"QUERY_LOC_GPS"`, `"transit_QR_code"` |
| scene_category | string | 场景分类 | `"transportation"`, `"food"`, `"shopping"` |
| image | string/null | Base64图片 | `"data:image/png;base64,..."` 或 `null` |

## 测试工具

### test.py
- 随机生成测试数据
- 自动发送到服务器
- 支持自定义数据：
  ```python
  # 修改test.py中的action_type
  action_type = "probe"  # 或 "recommend", "none"
  ```

## 文件说明

| 文件 | 说明 |
|------|------|
| server.py | HTTP服务器主程序 |
| test.py | 测试数据发送工具 |
| util.py | 工具函数（图片处理等） |
| test.png | 测试图片文件 |

## 注意事项
服务器默认监听所有网络接口（0.0.0.0）
