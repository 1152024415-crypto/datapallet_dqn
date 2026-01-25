"""
数据枚举定义模块
包含所有数据类型的枚举定义
使用简单直接的方法，避免复杂的元编程
"""

from enum import IntEnum
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass


# ==================== 辅助函数 ====================

def add_string_methods(enum_cls, string_map):
    """为枚举类添加字符串转换方法"""
    
    @classmethod
    def to_string(cls, value: int) -> str:
        """将枚举值转换为可读字符串"""
        return string_map.get(value, "未知")
    
    @classmethod
    def from_string(cls, chinese_str: str) -> int:
        """将中文字符串转换为枚举值"""
        # 创建反向映射
        reverse_map = {v: k for k, v in string_map.items()}
        return reverse_map.get(chinese_str, getattr(cls, 'NULL', 0))
    
    # 添加方法
    enum_cls.to_string = to_string
    enum_cls.from_string = from_string
    
    return enum_cls


# ==================== 枚举定义 ====================

# ActivityMode枚举
class ActivityMode(IntEnum):
    """用户姿态枚举"""
    NULL = 0                    # 未知
    RIDING = 1                  # 骑摩托车
    CYCLING = 2                 # 骑自行车
    SITTING = 3                 # 坐着
    STROLLING = 4               # 慢走
    RUNNING = 5                 # 跑步
    STANDING = 6                # 站立
    BUS = 7                     # 公交车
    CAR = 8                     # 小汽车
    SUBWAY = 9                  # 地铁
    HIGH_SPEED_TRAIN = 10       # 高铁
    TRAIN = 11                  # 火车
    HIKING = 12                 # 登山
    BRISK_WALKING = 13          # 快走
    VEHICLE_BRAKING = 14        # 刹车
    ELEVATOR = 15               # 电梯
    GARAGE_PARKING = 16         # 车库停车
    TILT = 17                   # 摔倒

# 添加字符串方法
ActivityMode = add_string_methods(ActivityMode, {
    ActivityMode.NULL: "未知",
    ActivityMode.RIDING: "骑摩托车",
    ActivityMode.CYCLING: "骑自行车",
    ActivityMode.SITTING: "坐着",
    ActivityMode.STROLLING: "慢走",
    ActivityMode.RUNNING: "跑步",
    ActivityMode.STANDING: "静止",
    ActivityMode.BUS: "公交车",
    ActivityMode.CAR: "小汽车",
    ActivityMode.SUBWAY: "地铁",
    ActivityMode.HIGH_SPEED_TRAIN: "高铁",
    ActivityMode.TRAIN: "火车",
    ActivityMode.HIKING: "登山",
    ActivityMode.BRISK_WALKING: "快走",
    ActivityMode.VEHICLE_BRAKING: "刹车",
    ActivityMode.ELEVATOR: "电梯",
    ActivityMode.GARAGE_PARKING: "车库停车",
    ActivityMode.TILT: "摔倒",

})


# LightIntensity枚举
class LightIntensity(IntEnum):
    """环境亮度枚举"""
    NULL = 0                     # 未知
    EXTREMELY_DARK = 1           # 极暗
    DIM = 2                      # 昏暗
    MODERATE_BRIGHTNESS = 3      # 正常
    BRIGHT = 4                   # 明亮
    HARSH_LIGHT = 5              # 刺眼

# 添加字符串方法
LightIntensity = add_string_methods(LightIntensity, {
    LightIntensity.NULL: "未知",
    LightIntensity.EXTREMELY_DARK: "极暗",
    LightIntensity.DIM: "昏暗",
    LightIntensity.MODERATE_BRIGHTNESS: "正常",
    LightIntensity.BRIGHT: "明亮",
    LightIntensity.HARSH_LIGHT: "刺眼",
})


# SoundIntensity枚举
class SoundIntensity(IntEnum):
    """环境声音强度枚举"""
    NULL = 0                     # 未知
    VERY_QUIET = 1               # 安静
    SOFT_SOUND = 2               # 轻柔
    NORMAL_SOUND = 3             # 正常
    NOISY = 4                    # 嘈杂
    VERY_NOISY = 5               # 非常嘈杂

# 添加字符串方法
SoundIntensity = add_string_methods(SoundIntensity, {
    SoundIntensity.NULL: "未知",
    SoundIntensity.VERY_QUIET: "安静",
    SoundIntensity.SOFT_SOUND: "轻柔",
    SoundIntensity.NORMAL_SOUND: "正常",
    SoundIntensity.NOISY: "嘈杂",
    SoundIntensity.VERY_NOISY: "非常嘈杂",
})


# LocationType枚举
class LocationType(IntEnum):
    """位置POI类型枚举"""
    NULL = 0                     # 未知
    OTHER = 1                    # 其他
    DESTINATION = 2              # 目的地
    HOME = 3                     # 家
    WORK = 4                     # 工作场所
    BUS_STATION = 5              # 公交车站
    SUBWAY_STATION = 6           # 地铁站
    TRAIN_STATION = 7            # 火车站
    AIRPORT = 8                  # 机场
    ACCOMMODATION = 9            # 住宿
    RESIDENTIAL = 10             # 住宅区
    COMMERCIAL = 11              # 商业区
    SCHOOL = 12                  # 学校
    HEALTH = 13                  # 医疗机构
    GOVERNMENT = 14              # 政府机构
    ENTERTAINMENT = 15           # 娱乐场所
    DINING = 16                  # 餐饮场所
    SHOPPING = 17                # 购物场所
    SPORT = 18                   # 运动场所
    ATTRACTION = 19              # 旅游景点
    PARK = 20                    # 公园
    STREET = 21                  # 街道
    Research_Institution = 22    # 科研机构

# 添加字符串方法
LocationType = add_string_methods(LocationType, {
    LocationType.NULL: "未知",
    LocationType.OTHER: "其他",
    LocationType.DESTINATION: "目的地",
    LocationType.HOME: "家",
    LocationType.WORK: "工作场所",
    LocationType.BUS_STATION: "公交车站",
    LocationType.SUBWAY_STATION: "地铁站",
    LocationType.TRAIN_STATION: "火车站",
    LocationType.AIRPORT: "机场",
    LocationType.ACCOMMODATION: "住宿",
    LocationType.RESIDENTIAL: "住宅区",
    LocationType.COMMERCIAL: "商业区",
    LocationType.SCHOOL: "学校",
    LocationType.HEALTH: "医疗机构",
    LocationType.GOVERNMENT: "政府机构",
    LocationType.ENTERTAINMENT: "娱乐场所",
    LocationType.DINING: "餐饮场所",
    LocationType.SHOPPING: "购物场所",
    LocationType.SPORT: "运动场所",
    LocationType.ATTRACTION: "旅游景点",
    LocationType.PARK: "公园",
    LocationType.STREET: "街道",
    LocationType.Research_Institution: "科研机构",
})


# SceneType枚举
class SceneType(IntEnum):
    """图像场景分类枚举"""
    NULL = 0                     # 未知
    OTHER = 1                    # 其他
    MEETINGROOM = 2              # 会议室
    WORKSPACE = 3                # 工位(办公)
    DINING = 4                   # 餐厅(就餐)
    OUTDOOR_PARK = 5             # 室外园区(散步)
    SUBWAY_STATION = 6           # 地铁站

# 添加字符串方法
SceneType = add_string_methods(SceneType, {
    SceneType.NULL: "未知",
    SceneType.OTHER: "其他",
    SceneType.MEETINGROOM: "会议室",
    SceneType.WORKSPACE: "工位办公",
    SceneType.DINING: "餐厅",
    SceneType.OUTDOOR_PARK: "室外园区",
    SceneType.SUBWAY_STATION: "地铁站",
})


@dataclass
class SceneData:
    """场景数据，包含枚举值和对应的图像路径"""
    scene_type: SceneType
    image_path: Optional[str] = None  # 图像文件路径
    image_data: Optional[bytes] = None  # 图像字节数据（懒加载）
    
    def __post_init__(self):
        # 确保scene_type是SceneType枚举
        if isinstance(self.scene_type, int):
            self.scene_type = SceneType(self.scene_type)
    
    @property
    def image_size(self) -> int:
        """获取图像大小"""
        if self.image_data:
            return len(self.image_data)
        elif self.image_path:
            # 尝试从文件获取大小
            import os
            try:
                return os.path.getsize(self.image_path) if os.path.exists(self.image_path) else 0
            except:
                return 0
        return 0
    
    @property
    def has_image(self) -> bool:
        """检查是否有图像（有image_path就表示有图像）"""
        return self.image_path is not None and self.image_path != ""
    
    def load_image_data(self) -> Optional[bytes]:
        """从文件加载图像数据（懒加载）"""
        if self.image_data:
            return self.image_data
        
        if self.image_path:
            import os
            try:
                if os.path.exists(self.image_path):
                    with open(self.image_path, 'rb') as f:
                        self.image_data = f.read()
                    return self.image_data
            except Exception as e:
                print(f"加载图像文件失败 {self.image_path}: {e}")
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "scene_type": self.scene_type.value,
            "scene_type_str": SceneType.to_string(self.scene_type.value), # type: ignore
            "has_image": self.has_image,
            "image_size": self.image_size,
            "image_path": self.image_path
            # 注意：不保存image_data，只保存路径
        }
    
    @classmethod
    def from_scene_type(cls, scene_type: SceneType, image_path: Optional[str] = None) -> 'SceneData':
        """从SceneType创建SceneData"""
        return cls(scene_type=scene_type, image_path=image_path)
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'SceneData':
        """从字典创建SceneData"""
        scene_type_value = data_dict.get("scene_type", 0)
        scene_type = SceneType(scene_type_value)
        
        image_path = data_dict.get("image_path")
        
        return cls(scene_type=scene_type, image_path=image_path)


# ==================== 工具函数 ====================

def to_str(data_id: str, value: Any) -> str:
    """格式化数据值为可读字符串"""
    if data_id == "activity_mode":
        # tuple格式：(前一个状态, 当前状态)
        if isinstance(value, tuple) and len(value) == 2:
            previous_str = ActivityMode.to_string(value[0]) # type: ignore
            current_str = ActivityMode.to_string(value[1]) # type: ignore
            return f"({previous_str}, {current_str})"
        else:
            # 单个姿态
            return ActivityMode.to_string(value if isinstance(value, int) else ActivityMode.NULL) # type: ignore
    elif data_id == "Light_Intensity":
        return LightIntensity.to_string(value) # type: ignore
    elif data_id == "Sound_Intensity":
        return SoundIntensity.to_string(value) # type: ignore
    elif data_id == "Location":
        return LocationType.to_string(value) # type: ignore
    elif data_id == "Scene":
        # 处理SceneData或SceneType
        if isinstance(value, SceneData):
            # 对于SceneData，返回包含图像信息的字符串
            scene_info = value.to_dict()
            return f"SceneType: {scene_info['scene_type_str']} (has_image: {scene_info['has_image']}, image_size: {scene_info['image_size']} bytes)"
        elif isinstance(value, SceneType):
            # 对于SceneType枚举，直接返回字符串表示
            return SceneType.to_string(value.value) # type: ignore
        elif isinstance(value, int):
            # 对于整数值，转换为SceneType再处理
            return SceneType.to_string(value) # type: ignore
        else:
            # 其他情况
            return str(value)
    
    # 默认返回字符串表示
    return str(value)


def from_str(data_id: str, value: Any) -> Any:
    """将可读字符串值解析为对应的枚举值"""
    # 如果已经是整数，直接返回
    if isinstance(value, int):
        return value
    
    # 使用枚举类from_string方法
    if data_id == "activity_mode":
        if isinstance(value, str):
            return ActivityMode.from_string(value) # type: ignore
    elif data_id == "Light_Intensity":
        if isinstance(value, str):
            return LightIntensity.from_string(value) # type: ignore
    elif data_id == "Sound_Intensity":
        if isinstance(value, str):
            return SoundIntensity.from_string(value) # type: ignore
    elif data_id == "Location":
        if isinstance(value, str):
            return LocationType.from_string(value) # type: ignore
    elif data_id == "Scene":
        if isinstance(value, str):
            return SceneType.from_string(value) # type: ignore
    
    # 如果无法解析，返回0-未知
    return 0