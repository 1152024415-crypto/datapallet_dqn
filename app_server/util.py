"""
图片处理工具 - 将图片转换为base64字符串

功能:
    1. 加载图片文件
    2. 缩放图片到指定尺寸
    3. 转换为base64编码的data URL

注意: 需要安装Pillow库: pip install Pillow
"""

import base64
import os
from io import BytesIO


def image_to_base64(image_path, size=(120, 120)):
    """
    将图片文件转换为base64 data URL

    参数:
        image_path: 图片文件路径
        size: 目标尺寸 (宽, 高)，默认120×120

    返回:
        base64 data URL字符串，失败时返回None
    """
    try:
        # 动态导入PIL，避免全局依赖
        from PIL import Image
    except ImportError:
        print("错误: PIL库未安装，无法处理图片。")
        print("请安装: pip install Pillow")
        return None

    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return None

        # 打开图片
        img = Image.open(image_path)

        # 转换为RGB（处理透明度）
        if img.mode in ("RGBA", "LA", "P"):
            # 创建白色背景
            background = Image.new("RGB", img.size, (255, 255, 255))

            # 处理透明度
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode == "RGBA":
                background.paste(img, mask=img.split()[3])  # 使用alpha通道
                img = background
            elif img.mode == "LA":
                background.paste(img, mask=img.split()[1])
                img = background
        else:
            img = img.convert("RGB")

        # 缩放图片（保持宽高比）
        img.thumbnail(size, Image.Resampling.LANCZOS)

        # 转换为base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 创建data URL
        result = f"data:image/jpeg;base64,{img_base64}"

        print(f"图片处理成功: {image_path} -> {size[0]}x{size[1]}")
        return result

    except Exception as e:
        print(f"图片处理失败: {e}")
        return None


def create_test_image_data(image_path="test.png"):
    """
    创建测试图片数据

    参数:
        image_path: 图片文件路径，默认为"test.png"

    返回:
        如果图片存在则返回base64，否则返回None
    """
    if os.path.exists(image_path):
        return image_to_base64(image_path)
    else:
        print(f"测试图片不存在: {image_path}")
        print("将使用null作为image字段")
        return None


if __name__ == "__main__":
    # 测试代码
    test_path = "test.png" if os.path.exists("test.png") else "server.py"
    print(f"测试图片处理: {test_path}")
    result = image_to_base64(test_path, size=(100, 100))
    if result:
        print(f"处理成功，数据长度: {len(result)}")
        print(f"前缀: {result[:50]}...")
    else:
        print("处理失败")
