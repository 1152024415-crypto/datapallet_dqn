"""
VLM场景感知模块
提供视觉语言模型场景分析的核心功能
"""

from src.module import ScenePerceptionModule
from src.vlm_client import VLMClient
from src.data_models import SceneBatch, SceneAnalysisResult
from src.fetcher import SceneDataFetcher
from src.logger import get_logger
from src import initialize_vlm_service

# 导入配置
import config

# 重新导出所有核心组件
__all__ = [
    "ScenePerceptionModule",
    "VLMClient",
    "SceneBatch",
    "SceneAnalysisResult",
    "SceneDataFetcher",
    "get_logger",
    "initialize_vlm_service",
]