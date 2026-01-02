"""
VLM场景感知模块测试 - 简化版
"""
import time
import sys
import os
from typing import Any
import time
from datapallet.datapallet import DataPallet, create_datapallet
from datapallet.enums import ActivityMode, SceneData, SceneType, LightIntensity, SoundIntensity, LocationType
from datetime import datetime, timedelta
# 添加vlm_engine目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vlm_engine'))

from vlm_engine import initialize_vlm_service
from vlm_engine import SceneAnalysisResult

# DQN / 规则引擎
class TestEngine:
    """简化测试引擎"""

    def __init__(self):
        self.scene_analysis_callback = None

    def setup_callback(self, callback):
        """
        设置场景分析回调

        Args:
            callback: 场景分析触发函数
        """
        self.scene_analysis_callback = callback
        print("TestEngine: 已接收VLM场景分析回调")

    def trigger_analysis(self, start_idx: int = 0, batch_size: int = None) -> bool:
        """
        触发场景分析

        Args:
            start_idx: 起始帧索引
            batch_size: 批处理大小

        Returns:
            bool: 是否成功触发
        """
        if self.scene_analysis_callback:
            print(f"触发场景分析: start_idx={start_idx}, batch_size={batch_size}")
            return self.scene_analysis_callback(start_idx, batch_size)
        else:
            print("错误: 尚未设置场景分析回调")
            return False


def display_results(result: SceneAnalysisResult):
    """显示分析结果"""
    print(f"\n场景分析结果 (批次: {result.batch_id})")
    print(f"处理时间: {result.processing_time:.2f}秒")

    for i, scene in enumerate(result.scenes, 1):
        print(f"\n场景 {i}:")
        print(f"  主要活动: {scene.main_activity}")
        print(f"  描述: {scene.description[:80]}..." if len(scene.description) > 80 else f"  描述: {scene.description}")


def main():
    def simple_callback(data_id: str, value: Any):
        """简单回调示例"""
        print(f"[回调] {data_id} = {value}")

    # 创建数据托盘
    dp = create_datapallet(ttl=10)

    # 设置回调
    dp.setup(simple_callback)
    # 模拟接收数据（实际由测试床发送）
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datapallet/scene_images/会议室.png")
    sd = SceneData(
        scene_type=SceneType.MEETINGROOM,
        image_path=image_path
    )
    dp.receive_data("Scence", sd, datetime.now())
    print(f"✓ 添加 Scence: SceneType.MEETINGROOM, 图片路径: {image_path}")
    success, value = dp.get("Scence")
    if success:
        print("Scence value get success")
    dp.receive_data("activity_mode", ActivityMode.SITTING, datetime.now())
    print(f"✓ 添加 activity_mode: ActivityMode.SITTING")
    success, value = dp.get("ActivityMode")
    if success:
        print("ActivityMode value get success")
    dp.receive_data("Light_Intensity", LightIntensity.BRIGHT, datetime.now())
    print(f"✓ 添加 Light_Intensity: LightIntensity.BRIGHT")
    success, value = dp.get("Light_Intensity")
    if success:
        print("Light_Intensity value get success")
    dp.receive_data("Sound_Intensity", SoundIntensity.NORMAL_SOUND, datetime.now())
    print(f"✓ 添加 Sound_Intensity: SoundIntensity.NORMAL_SOUND")
    success, value = dp.get("Sound_Intensity")
    if success:
        print("Sound_Intensity value get success")
    dp.receive_data("Location", LocationType.WORK, datetime.now())
    print(f"✓ 添加 Location: LocationType.WORK")
    success, value = dp.get("Location")
    if success:
        print("Location value get success")

    # """主测试函数 - 简化版本"""
    print("VLM场景感知模块测试")
    print("=" * 60)

#=============================================================================================================
    # 1. 创建测试引擎
    test_engine = TestEngine()

    # 2. 使用初始化接口创建VLM服务
    print("初始化VLM服务...")
    vlm_module, analysis_callback = initialize_vlm_service(api_type="siliconflow", datapallet=dp)

    # 3. 设置回调
    test_engine.setup_callback(analysis_callback)

    print("\nVLM服务初始化完成，准备开始测试...")

    # 测试: 正常触发
    print("\n测试1: 触发场景分析")
    success1 = test_engine.trigger_analysis()
    print("test trigger success: ", success1)
    success2 = test_engine.trigger_analysis()
    print("test trigger success: ", success2)
    if success1 or success2:
        print("分析已启动，等待完成...")
        # 等待分析完成
        while vlm_module.is_busy():
            time.sleep(0.5)

        # 获取结果
        result = vlm_module.get_last_result()
        if result:
            display_results(result)
# =============================================================================================================

    print("\n测试完成！")
    # 停止数据托盘
    dp.stop()

if __name__ == "__main__":
    main()