from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

BitmapLike = Union[bytes, np.ndarray, Image.Image, str, Path]

def bitmap_to_png(
    src: BitmapLike,
    dst: Union[str, Path, None] = None,
    *,
    size: tuple[int, int] | None = None,
    mode: str = "RGBA",
) -> Image.Image:
    """
    将位图（bitmap）保存为 PNG 文件，或仅返回 PIL.Image 对象。

    参数
    ----
    src : bytes | np.ndarray | PIL.Image | str | Path
        源位图数据。支持多种输入：
        - bytes：原始像素数据（需同时提供 size）
        - np.ndarray：H×W×C 或 H×W 的数组
        - PIL.Image：直接保存
        - str/Path：磁盘上的任意图片路径，先读取再转 PNG
    dst : str | Path | None, optional
        输出 PNG 文件名。如果为 None，则只返回 Image 对象，不保存。
    size : (int, int), optional
        当 src 为原始字节串时必须指定，表示 (width, height)。
    mode : str, default "L"
        当 src 为原始字节串时的像素模式，如 "L"、"RGB"、"RGBA"。

    返回
    ----
    PIL.Image.Image
        转换后的 PIL 图像对象，方便继续处理。
    """
    # 1. 统一转成 PIL.Image ----------------------------------------------------
    if isinstance(src, (str, Path)):
        img = Image.open(src)
    elif isinstance(src, Image.Image):
        img = src
    elif isinstance(src, np.ndarray):
        # NumPy 数组 -> PIL
        if src.dtype != np.uint8:
            src = src.astype(np.uint8)
        img = Image.fromarray(src)
    elif isinstance(src, bytes):
        if size is None:
            raise ValueError("当 src 为 bytes 时必须指定 size=(w, h)")
        print("isinstance(src, bytes) true")
        w, h = size
        print(f"size {size}")
        img = Image.frombytes("RGBA", (w, h), src)
    else:
        raise TypeError(f"不支持的输入类型：{type(src)}")

    # 逆时针旋转 90 度 : sensorhub目前传输过来的图片为顺时针旋转了90度的，需要修正回去--------------------------------------------------------
    # img = img.rotate(90, expand=True)

    # 2. 保存为 PNG -----------------------------------------------------------
    if dst is not None:
        img.save(dst, format="PNG")

    return img


# ------------------- DEMO -------------------
if __name__ == "__main__":
    # 2) 从原始字节流生成灰度 PNG
    w, h = 640, 480
    raw_bytes = bytes(i % 256 for i in range(w * h))  # 0-255 灰度渐变
    bitmap_to_png(raw_bytes, "gradient.png", size=(w, h), mode="L")