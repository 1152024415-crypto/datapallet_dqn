# src/data_models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class FrameData:
    """单帧数据：图像 + 对应元数据"""
    frame_id: int  # 帧编号
    timestamp: datetime  # 时间戳
    image_base64: str  # Base64编码的图像
    metadata: Dict[str, Any]  # 该帧的元数据


@dataclass
class SceneBatch:
    """一批需要分析的帧数据"""
    frames: List[FrameData]  # 多个帧数据
    batch_id: str = ""  # 批次ID

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = f"batch_{int(datetime.now().timestamp())}"


@dataclass
class SceneSegment:
    """划分出的一个场景段"""
    scene_id: int  # 场景编号
    description: str  # 场景描述
    main_activity: str  # 主要活动

@dataclass
class SceneAnalysisResult:
    """场景分析最终结果"""
    batch_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    total_frames: int = 0
    total_scenes: int = 0
    scenes: List[SceneSegment] = field(default_factory=list)  # 划分出的多个场景
    processing_time: float = 0.0  # 处理时间