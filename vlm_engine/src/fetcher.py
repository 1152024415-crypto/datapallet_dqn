# src/fetcher.py
import base64
import json
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from .data_models import FrameData, SceneBatch
from .logger import get_logger

logger = get_logger(__name__)


class SceneDataFetcher:
    """场景数据获取器 - 从文件夹获取多帧数据"""

    def __init__(self, image_dir: str, metadata_list: List[Dict[str, Any]] = None):
        """
        初始化场景数据获取器

        Args:
            image_dir: 图像文件夹路径
            metadata_list: 元数据列表，每个元素对应一帧的元数据（可选）
        """
        self.image_dir = Path(image_dir)

        # 获取文件夹中的所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        self.image_files = []

        if self.image_dir.is_file():
            if self.image_dir.suffix.lower() in [ext.lower() for ext in image_extensions]:
                self.image_files.append(self.image_dir)
        else:
            for ext in image_extensions:
                self.image_files.extend(list(self.image_dir.glob(f"*{ext}")))

        # 按文件名排序，确保顺序一致
        self.image_files.sort(key=lambda x: x.name)

        # 如果没有元数据列表，创建默认元数据
        if metadata_list is None:
            self.metadata_list = self._create_default_metadata(len(self.image_files))
        else:
            self.metadata_list = metadata_list

        logger.info(f"在 {image_dir} 中找到 {len(self.image_files)} 张图片")

        # 检查图像和元数据数量是否匹配
        if len(self.image_files) < len(self.metadata_list):
            logger.warning(f"图片数量({len(self.image_files)})少于元数据数量({len(self.metadata_list)})")
        elif len(self.image_files) > len(self.metadata_list):
            logger.warning(
                f"图片数量({len(self.image_files)})多于元数据数量({len(self.metadata_list)})，将为额外图片使用默认元数据")
            # 补充默认元数据
            extra_count = len(self.image_files) - len(self.metadata_list)
            self.metadata_list.extend(self._create_default_metadata(extra_count))

    def _create_default_metadata(self, count: int) -> List[Dict[str, Any]]:
        """创建默认元数据"""
        default_metadata = []
        base_time = datetime.now()

        for i in range(count):
            metadata = {
                "activity_mode": "unknown",
                "Light_Intensity": "unknown",
                "Sound_Intensity": "unknown",
                "Location": "unknown",
                "scene_type": "unknown",
            }
            default_metadata.append(metadata)

        return default_metadata

    def fetch_scene_batch(self, start_idx: int = 0, batch_size: int = None) -> SceneBatch:
        """
        获取一批帧数据用于场景分析

        Args:
            start_idx: 起始索引
            batch_size: 批处理大小，None表示使用所有数据

        Returns:
            SceneBatch: 一批帧数据
        """
        # 确定结束索引
        if batch_size is None:
            end_idx = len(self.image_files)
        else:
            end_idx = min(start_idx + batch_size, len(self.image_files))

        if start_idx >= len(self.image_files):
            logger.error(f"起始索引 {start_idx} 超出范围（总共 {len(self.image_files)} 张图片）")
            raise ValueError(f"起始索引 {start_idx} 超出范围")

        frames = []
        loaded_count = 0

        for idx in range(start_idx, end_idx):
            img_path = self.image_files[idx]

            try:
                # 读取图像并编码为base64
                logger.debug(f"加载图片: {img_path.name}")
                with open(img_path, "rb") as f:
                    image_data = f.read()
                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                
                # 根据文件扩展名确定MIME类型
                ext = img_path.suffix.lower()
                if ext == '.jpg' or ext == '.jpeg':
                    mime_type = 'image/jpeg'
                elif ext == '.png':
                    mime_type = 'image/png'
                elif ext == '.gif':
                    mime_type = 'image/gif'
                elif ext == '.bmp':
                    mime_type = 'image/bmp'
                else:
                    mime_type = 'image/jpeg'  # 默认
                
                # 创建完整的data URI
                image_b64 = f"data:{mime_type};base64,{image_b64}"

                # 获取对应的元数据
                frame_metadata = self.metadata_list[idx].copy()  # 使用副本避免修改

                # 确保frame_id正确
                frame_metadata["frame_id"] = idx
                frame_metadata["image_name"] = img_path.name
                frame_metadata["image_path"] = str(img_path)

                # 创建FrameData对象
                timestamp = frame_metadata.get("timestamp", datetime.now().isoformat())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)

                frame = FrameData(
                    frame_id=idx,
                    timestamp=timestamp,
                    image_base64=image_b64,
                    metadata=frame_metadata
                )
                frames.append(frame)
                loaded_count += 1

            except Exception as e:
                logger.error(f"加载图片失败 {img_path}: {e}")
                # 可以继续加载其他图片

        logger.info(f"成功加载 {loaded_count} 帧数据（从 {start_idx} 到 {end_idx - 1}）")

        if loaded_count == 0:
            raise ValueError("没有成功加载任何图片数据")

        return SceneBatch(frames=frames)

    def get_total_frames(self) -> int:
        """获取总帧数"""
        return len(self.image_files)

    def list_images(self) -> List[str]:
        """列出所有图片文件名"""
        return [img.name for img in self.image_files]