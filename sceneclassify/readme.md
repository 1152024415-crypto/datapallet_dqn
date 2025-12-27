# 场景分类服务 (Scene Classification Service)

项目基于 [FastAPI](https://fastapi.tiangolo.com/) 和 [Birder](https://github.com/birder-project/birder) 构建的场景分类服务。
项目使用 `rope_vit_reg4_b14_capi-places365` 预训练模型（86M 参数量）对上传的图片进行场景识别（如：会议室、办公室、公园等），并将识别结果返回和标注在图片上保存。

## 功能特性

- **REST API 接口**：提供 HTTP 接口接收图片上传。
- **高性能推理**：使用 Birder 库加载 Efficient ViT 模型进行推理（已验证该模型可以在端侧NPU推理，为了Server部署的统一性，迁移到Server侧部署）。
- **结果可视化**：自动在图片上绘制 Top-5 预测类别和置信度，并保存到服务器。
- **异步处理**：基于 FastAPI 的异步特性，支持高并发请求。

## 环境要求

- Python 3.10

## 安装指南

1**安装依赖库**:
   请在项目根目录下运行以下命令安装所需依赖：
   ```bash
   pip install fastapi uvicorn python-multipart numpy birder pillow requests
   ```
2**模型下载**
本项目[ViT模型下载路径](https://huggingface.co/birder-project/rope_vit_reg4_b14_capi-places365/tree/main)，下载后与推理脚本同一文件夹下新建models文件，将模型放置其中。

## 项目结构

建议的项目文件结构如下：

```text
sceneclassify/
├── model/
│   ├── rope_vit_reg4_b14_capi-places365.pt    # 模型权重
├── result/
│   ├── 9a5dbaa7fdae4f32ae4c68a5c374e6bf.png   # 推理后生成的图片(原始图片+分类结果新生成的图片)
│── scene_images /                             # 测试图片
│── deploy_server.py                           # Server 部署程序
│── labels.txt                                 # 分类模型标签  
│── sceneclassify.py                           # 未服务化的测试程序   
│── test_client.py                             # Server 部署后的client测试程序   
└── README.md               # 项目说明文档
```

## 快速开始

### 1. 启动服务器

运行服务端脚本：

```bash
python eploy_server.py
```

当看到如下日志时，说明服务启动成功：
```text
INFO:     Started server process [1936]
INFO:     Waiting for application startup.
--- 正在初始化模型 ---
[27/Dec/2025 11:05:19.0393 INFO fs_ops.py:540 ] Loading model from models/rope_vit_reg4_b14_capi-places365.pt on device cpu...
--- 模型加载完成，服务已就绪 ---
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 2. 运行客户端测试

打开一个新的终端窗口，运行客户端脚本发送请求：

```bash
python client.py
```

### 3. 查看结果

- **控制台输出**：客户端会打印预测的 Top-5 类别和置信度。
- **结果图片**：处理后的图片（带有文字标注）会保存在 `results/` 目录下。

## API 接口文档

### 1. 场景预测接口

- **URL**: `/predict/`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### 请求参数

| 参数名 | 类型 | 必选 | 描述 |
| :--- | :--- | :--- | :--- |
| `file` | File | 是 | 需要识别的图片文件 (jpg, png 等) |

#### 成功响应 (JSON)

```json
{
  "message": "预测成功",
  "predictions": [
    {
      "label": "conference_room",
      "probability": "85.13%"
    },
    {
      "label": "office",
      "probability": "5.21%"
    }
    // ... 其他 Top 5 结果
  ],
  "result_image_path": "/absolute/path/to/server/results/uuid.png"
}
```
