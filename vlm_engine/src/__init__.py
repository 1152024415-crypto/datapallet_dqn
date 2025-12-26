from .module import ScenePerceptionModule
from .vlm_client import VLMClient
from .data_models import SceneBatch, SceneAnalysisResult
from .fetcher import SceneDataFetcher
from .logger import get_logger
from typing import Callable, Optional, Tuple


def initialize_vlm_service(image_dir: str, metadata_list: list) -> Tuple[ScenePerceptionModule, Callable]:
    """
    初始化VLM服务并返回模块和场景分析回调

    Args:
        image_dir: 图像文件夹路径
        metadata_list: 元数据列表

    Returns:
        Tuple[ScenePerceptionModule, Callable]: (VLM模块实例, 场景分析回调函数)
    """
    fetcher = SceneDataFetcher(image_dir, metadata_list)
    vlm_client = VLMClient()
    vlm_module = ScenePerceptionModule(fetcher, vlm_client)
    analysis_callback = vlm_module.get_scene_analysis_callback()

    return vlm_module, analysis_callback


__all__ = [
    "ScenePerceptionModule",
    "VLMClient",
    "SceneBatch",
    "SceneAnalysisResult",
    "SceneDataFetcher",
    "get_logger",
    "initialize_vlm_service"
]