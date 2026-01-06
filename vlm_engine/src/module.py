import threading
import json
import time
from typing import Optional, Callable, Dict, Any
from .data_models import SceneBatch, SceneAnalysisResult, SceneSegment
from .fetcher import SceneDataFetcher
from .vlm_client import VLMClient
from .logger import get_logger

logger = get_logger(__name__)

class ScenePerceptionModule:
    """场景感知主模块"""
    
    def __init__(self, data_fetcher: SceneDataFetcher, vlm_client: VLMClient):
        """
        初始化场景感知模块
        
        Args:
            data_fetcher: 数据获取器
            vlm_client: VLM客户端
        """
        self.data_fetcher = data_fetcher
        self.vlm_client = vlm_client
        self.lock = threading.Lock()
        self.is_processing = False
        self.last_result: Optional[SceneAnalysisResult] = None

        logger.info("场景感知模块初始化完成")

    def get_scene_analysis_callback(self) -> Callable[[int, Optional[int]], bool]:
        """
        获取场景分析回调函数

        Returns:
            Callable: 场景分析触发函数，接收(start_idx, batch_size)参数
        """

        def analyze_scenes(start_idx: int = 0, batch_size: int = None) -> bool:
            """
            外部调用的场景分析函数

            Args:
                start_idx: 起始帧索引
                batch_size: 批处理大小

            Returns:
                bool: 是否成功启动分析
            """
            return self._trigger_scene_analysis(start_idx, batch_size)

        return analyze_scenes

    def _trigger_scene_analysis(self, start_idx: int = 0, batch_size: int = None) -> bool:
        """
        触发场景分析

        Args:
            start_idx: 起始帧索引
            batch_size: 批处理大小

        Returns:
            bool: True表示成功启动，False表示有任务正在运行
        """
        with self.lock:
            if self.is_processing:
                logger.warning("有任务正在运行，丢弃本次触发")
                return False

            self.is_processing = True

        # 在新线程中处理
        thread = threading.Thread(
            target=self._process_scene_analysis,
            args=(start_idx, batch_size),
            daemon=True
        )
        thread.start()

        return True

    def _process_scene_analysis(self, start_idx: int, batch_size: int):
        """工作线程：处理场景分析"""
        start_time = time.time()

        try:
            # 1. 获取批量数据
            logger.info(f"开始场景分析，起始帧: {start_idx}, 批大小: {batch_size}")
            scene_batch = self.data_fetcher.fetch_scene_batch(start_idx, batch_size)

            if not scene_batch.frames:
                logger.error("没有获取到有效帧数据")
                self._set_error_result("无有效帧数据", start_time)
                return

            logger.info(f"获取到 {len(scene_batch.frames)} 帧数据")

            # 2. 调用VLM进行场景分析
            response_text = self.vlm_client.analyze_scenes(scene_batch)

            # 3. 清洗VLM返回结果
            cleaned_response = self._clean_vlm_response(response_text)

            # 4. 解析结果
            vlm_result = json.loads(cleaned_response)

            # 5. 转换为SceneAnalysisResult
            self.last_result = self._convert_to_scene_analysis_result(
                vlm_result, scene_batch, start_time
            )

            elapsed = time.time() - start_time
            logger.info(f"场景分析完成，总耗时: {elapsed:.2f}秒")
            logger.info(f"划分出 {len(self.last_result.scenes)} 个场景")

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            self._set_error_result(f"JSON解析失败: {str(e)}", start_time)
        except Exception as e:
            logger.error(f"场景分析处理失败: {e}")
            self._set_error_result(str(e), start_time)
        finally:
            with self.lock:
                self.is_processing = False

    def _convert_to_scene_analysis_result(self, vlm_result: dict, scene_batch: SceneBatch,
                                          start_time: float) -> SceneAnalysisResult:
        """将VLM原始结果转换为SceneAnalysisResult"""
        scenes = []

        if "scenes" in vlm_result:
            for scene_data in vlm_result["scenes"]:
                scene = SceneSegment(
                    scene_id=scene_data.get("scene_id", 0),
                    description=scene_data.get("description", ""),
                    main_activity=scene_data.get("main_activity", "")
                )
                scenes.append(scene)

        # 计算处理时间
        processing_time = time.time() - start_time

        return SceneAnalysisResult(
            batch_id=scene_batch.batch_id,
            total_frames=len(scene_batch.frames),
            total_scenes=len(scenes),
            scenes=scenes,
            processing_time=processing_time
        )

    def _set_error_result(self, error_msg: str, start_time: float):
        """设置错误结果"""
        self.last_result = SceneAnalysisResult(
            batch_id=f"error_{int(time.time())}",
            processing_time=time.time() - start_time
        )

    def _clean_vlm_response(self, response_text: str) -> str:
        """
        清理VLM返回的文本，移除Markdown代码块标记
        """
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        elif response_text.startswith('```'):
            response_text = response_text[3:]

        if response_text.endswith('```'):
            response_text = response_text[:-3]

        return response_text.strip()

    def get_last_result(self) -> Optional[SceneAnalysisResult]:
        """获取最近一次感知结果"""
        return self.last_result

    def is_busy(self) -> bool:
        """检查是否正在处理"""
        with self.lock:
            return self.is_processing