import os
import numpy as np
import birder
from birder.inference.classification import infer_image
from PIL import Image, ImageDraw


def load_labels(file_path):
    class_map = {}
    if not os.path.exists(file_path):
        print(f"找不到标签文件 {file_path}")
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


def main():
    model_name = "rope_vit_reg4_b14_capi-places365"
    image_path = "/Users/cannkit/PycharmProjects/AO/sceneclassify/scene_images/meeting_room.png"
    labels_path = "labels.txt"

    print("正在加载模型...")
    (net, model_info) = birder.load_pretrained_model(model_name, inference=True)
    size = birder.get_size_from_signature(model_info.signature)
    transform = birder.classification_transform(size, model_info.rgb_stats)

    class_map = load_labels(labels_path)

    print(f"正在处理图片: {image_path}...")
    original_img = Image.open(image_path).convert('RGB')

    try:
        (out, _) = infer_image(net, original_img, transform)
    except TypeError:
        temp_path = "temp_rgb_image.jpg"
        original_img.save(temp_path)
        (out, _) = infer_image(net, temp_path, transform)
        os.remove(temp_path)
    probs = out[0]

    top5_indices = np.argsort(probs)[::-1][:5]

    original_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original_img)

    font_size = max(20, int(original_img.height * 0.03))
    padding = 10
    margin_top = 20
    line_height = font_size + 10

    print("\nTop 5 预测结果:")
    for i, idx in enumerate(top5_indices):
        label = class_map.get(idx, f"Unknown({idx})")
        prob_text = f"{label}: {probs[idx] * 100:.2f}%"
        print(f" - {prob_text}")

        x, y = 20, margin_top + i * line_height
        draw.text((x + 1, y + 1), prob_text, fill=(0, 0, 0))
        draw.text((x, y), prob_text, fill=(0, 255, 0))

    base_name, ext = os.path.splitext(image_path)
    output_path = f"{base_name}_predict{ext}"

    original_img.save(output_path)
    print(f"\n结果图片已保存至: {output_path}")


if __name__ == "__main__":
    main()