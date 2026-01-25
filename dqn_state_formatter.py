# dqn_state_formatter.py
"""
DQN状态格式化工具
将DQN的输入状态统一格式化为中文显示格式
"""

from typing import Dict, Any
from datapallet.enums import ActivityMode, LocationType, LightIntensity, SceneType


class DQNStateFormatter:
    """格式化DQN输入状态为中文显示格式"""

    @staticmethod
    def format_state(data_palette) -> Dict[str, str]:
        """
        从DataPalette提取DQN状态并格式化为中文

        Args:
            data_palette: DataPallet实例

        Returns:
            格式化后的状态字典
        """
        state_keys = ["activity_mode", "Location", "Light_Intensity", "Scene"]
        state_dict = {}

        for key in state_keys:
            success, val = data_palette.get(key, only_valid=True)
            if success:
                # 调用DataPallet的format_value转为中文
                formatted_val = data_palette.format_value(key, val)

                # 特殊处理Scene类型
                if key == "Scene":
                    formatted_val = DQNStateFormatter._clean_scene_text(formatted_val)

                state_dict[key] = formatted_val
            else:
                state_dict[key] = "未知"

        return {
            "activity": state_dict.get("activity_mode", "未知"),
            "location": state_dict.get("Location", "未知"),
            "light": state_dict.get("Light_Intensity", "未知"),
            "scene": state_dict.get("Scene", "未知")
        }

    @staticmethod
    def _clean_scene_text(scene_text: str) -> str:
        """清理Scene枚举的显示文本"""
        if not scene_text:
            return "未知"

        # 移除前缀 "SceneType: "
        if scene_text.startswith("SceneType: "):
            scene_text = scene_text[11:]

        # 移除括号中的说明
        if " (" in scene_text:
            scene_text = scene_text.split(" (")[0]

        return scene_text

    @staticmethod
    def get_state_icons() -> Dict[str, str]:
        """获取状态字段对应的图标"""
        return {
            "activity": "🏃",  # 姿态图标
            "location": "📍",  # 位置图标
            "light": "☀️",  # 光照图标
            "scene": "👁️"  # 场景图标
        }