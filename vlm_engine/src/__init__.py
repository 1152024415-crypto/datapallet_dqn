from datetime import datetime
from importlib.metadata import metadata

from .module import ScenePerceptionModule
from .vlm_client import VLMClient
from .data_models import SceneBatch, SceneAnalysisResult
from .fetcher import SceneDataFetcher
from .logger import get_logger
import os
from typing import Callable, Optional, Tuple, Dict, Any, List
from datapallet.datapallet import DataPallet
from datapallet.enums import SceneData, to_str

def _get_scene_data(datapallet: DataPallet) -> Dict[str, Any]:
    """从DataPallet获取场景数据"""
    scene_success, scene_value = datapallet.get("Scence")
    if not scene_success or not scene_value:
        raise ValueError("无法从DataPallet获取Scence数据")

    if not isinstance(scene_value, SceneData):
        raise ValueError("Scence数据不是SceneData类型")

    if not scene_value.image_path:
        raise ValueError("SceneData中没有图片路径")

    if not os.path.exists(scene_value.image_path):
        raise ValueError(f"图片文件不存在: {scene_value.image_path}")

    return {
        "scene_data": scene_value,
        "image_dir": scene_value.image_path,
        "scene_type": scene_value.scene_type
    }

def _get_metadata_from_datapallet(datapallet: DataPallet, scene_type) -> List[Dict[str, Any]]:
    """从DataPallet获取所有元数据"""
    metadata = {}

    # 获取各个元数据字段
    data_fields = ["activity_mode", "Light_Intensity", "Sound_Intensity", "Location"]

    for data_id in data_fields:
        success, value = datapallet.get(data_id)
        if success and value is not None:
            res = to_str(data_id, value)
            metadata[data_id] = res
        else:
            metadata[data_id] = "unknown"

    # 添加scene_type（从SceneData获取）
    metadata["scene_type"] = to_str("Scence", scene_type)

    return [metadata]

def initialize_vlm_service(api_type: str, datapallet: DataPallet) -> Tuple[ScenePerceptionModule, Callable]:
    """
    初始化VLM服务并返回模块和场景分析回调

    Args:
        image_dir: 图像文件夹路径
        metadata_list: 元数据列表

    Returns:
        Tuple[ScenePerceptionModule, Callable]: (VLM模块实例, 场景分析回调函数)
    """
    scene_info = _get_scene_data(datapallet)
    metadata_list = _get_metadata_from_datapallet(datapallet, scene_info["scene_type"])

    # metadata_list = [
    #     {
    #         "activity_mode": "sitting",
    #         "Light_Intensity": "bright",
    #         "Sound_Intensity": "normal_sound",
    #         "Location": "work",
    #         "scene_type": "meeting"
    #     }
    # ]
    _fetcher = SceneDataFetcher(scene_info["image_dir"], metadata_list)
    _vlm_client = VLMClient(api_type=api_type)
    _vlm_module = ScenePerceptionModule(_fetcher, _vlm_client)
    analysis_callback = _vlm_module.get_scene_analysis_callback()

    return _vlm_module, analysis_callback


__all__ = [
    "ScenePerceptionModule",
    "VLMClient",
    "SceneBatch",
    "SceneAnalysisResult",
    "SceneDataFetcher",
    "get_logger",
    "initialize_vlm_service"
]