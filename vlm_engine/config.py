# 简单配置，无需复杂yaml
import os

class Config:
    # 数据获取配置
    IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    MOCK_METADATA_LIST = [
        {
            "timestamp": "2024-01-15T10:00:00",
            "object_classes": ["monitor"],
            "activity_type": ["working"],
            "specific_address": "路口A",
            "weather": "sunny"
            # 亮度：
        },
        {
            "timestamp": "2024-01-15T10:00:00",
            "object_classes": ["monitor", "keyboard"],
            "activity_type": ["working", "typing"],
            "specific_address": "路口A",
            "weather": "sunny"
        },
        {
            "timestamp": "2024-01-15T11:00:02",
            "object_classes": ["basketball"],
            "activity_type": ["play basketball"],
            "specific_address": "路口B",
            "weather": "sunny"
        },
        # ... 更多帧的元数据
    ]

    MAX_PROCESSING_TIME = 60.0  # 最大处理时间（秒）