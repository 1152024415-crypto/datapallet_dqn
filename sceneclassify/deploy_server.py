import os
import uuid
import io
import numpy as np
import birder
from birder.inference.classification import infer_image
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn


def load_labels(file_path):
    class_map = {}
    if not os.path.exists(file_path):
        print(f"找不到标签文件 {file_path}，将使用数字索引。")
        return {i: str(i) for i in range(365)}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    label_path, idx = parts[0], int(parts[1])
                    clean_name = label_path.split('/')[-1]
                    class_map[idx] = clean_name
    return class_map


ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = "rope_vit_reg4_b14_capi-places365"
    labels_path = "labels.txt"

    print("--- 正在初始化模型 ---")

    (net, model_info) = birder.load_pretrained_model(model_name, inference=True)

    size = birder.get_size_from_signature(model_info.signature)
    transform = birder.classification_transform(size, model_info.rgb_stats)

    class_map = load_labels(labels_path)

    ml_models["net"] = net
    ml_models["transform"] = transform
    ml_models["class_map"] = class_map

    print("--- 模型加载完成，服务已就绪 ---")
    yield
    ml_models.clear()


app = FastAPI(title="场景分类 API", lifespan=lifespan)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "文件类型错误，请上传图片。"})

    net = ml_models.get("net")
    transform = ml_models.get("transform")
    class_map = ml_models.get("class_map")

    if not net:
        return JSONResponse(status_code=500, content={"message": "模型尚未加载完成"})

    contents = await file.read()
    try:
        original_img = Image.open(io.BytesIO(contents)).convert("RGB")
        (out, _) = infer_image(net, original_img, transform)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"处理图片或模型推理出错: {str(e)}"})

    probs = out[0]
    top5_indices = np.argsort(probs)[::-1][:5]

    draw = ImageDraw.Draw(original_img)
    font_size = max(20, int(original_img.height * 0.03))
    margin_top = 20
    line_height = font_size + 10

    predictions = []
    print("\nTop 5 预测结果:")
    for i, idx in enumerate(top5_indices):
        label = class_map.get(idx, f"Unknown({idx})")
        prob_text = f"{label}: {probs[idx] * 100:.2f}%"
        print(f" - {prob_text}")

        predictions.append({"label": label, "probability": f"{probs[idx] * 100:.2f}%"})

        x, y = 20, margin_top + i * line_height
        draw.text((x + 1, y + 1), prob_text, fill=(0, 0, 0))
        draw.text((x, y), prob_text, fill=(0, 255, 0))

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    unique_filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join(results_dir, unique_filename)
    original_img.save(output_path)

    abs_output_path = os.path.abspath(output_path)
    print(f"结果已保存: {abs_output_path}")

    return JSONResponse(
        status_code=200,
        content={
            "message": "预测成功",
            "predictions": predictions,
            "result_image_path": abs_output_path,
        }
    )


@app.get("/")
def read_root():
    return {"message": "欢迎使用场景分类 API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)