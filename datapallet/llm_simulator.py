"""
LLM 模拟器模块 - 连接到 DeepSeek API 生成用户行为数据序列
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

from datapallet.enums import (
    ActivityMode, LightIntensity, SoundIntensity,
    LocationType, SceneType, SceneData, to_str
)

# 导入图像生成器
from datapallet.image_generator import ImageGenerator, create_image_generator


@dataclass
class DataRecord:
    """数据记录，包含时间戳和数据"""
    timestamp: datetime
    data_id: str
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        value_to_save = self.value
        
        # 如果是SceneData，特殊处理
        if self.data_id == "Scence" and isinstance(value_to_save, SceneData):
            # 保存SceneData的信息（包含image_path）
            scene_dict = value_to_save.to_dict()
            
            return {
                "timestamp": self.timestamp.isoformat(),
                "id": self.data_id,
                "value": scene_dict
            }
        
        # 如果是整数，尝试转换为中文字符串
        if isinstance(value_to_save, int):
            value_to_save = to_str(self.data_id, value_to_save)
        
        return {
            "timestamp": self.timestamp.isoformat(),
            "id": self.data_id,
            "value": value_to_save
        }


class LLMSimulator:
    """LLM模拟器，连接到DeepSeek API生成用户行为数据序列"""
    
    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com",
                 image_generator: Optional[ImageGenerator] = None):
        """
        初始化LLM模拟器
        
        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
            image_generator: 图像生成器实例（可选）
        """
        if api_key is None:
            api_key = os.getenv("LLM_API_KEY")
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 缓存常见行为模式的结果，减少API调用
        self.behavior_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # 图像生成器
        self.image_generator = image_generator
        if self.image_generator is None:
            # 创建默认的图像生成器
            self.image_generator = create_image_generator()
        
        # 枚举值映射表（用于LLM理解）
        self.enum_mappings = {
            "activity_mode": {
                "未知": ActivityMode.NULL,
                "骑摩托车": ActivityMode.RIDING,
                "骑自行车": ActivityMode.CYCLING,
                "坐着": ActivityMode.SITTING,
                "坐": ActivityMode.SITTING,  # 简写形式
                "慢走": ActivityMode.STROLLING,
                "跑步": ActivityMode.RUNNING,
                "站立": ActivityMode.STANDING,
                "公交车": ActivityMode.BUS,
                "小汽车": ActivityMode.CAR,
                "地铁": ActivityMode.SUBWAY,
                "高铁": ActivityMode.HIGH_SPEED_TRAIN,
                "火车": ActivityMode.TRAIN,
                "登山": ActivityMode.HIKING,
                "快走": ActivityMode.BRISK_WALKING,
                "刹车": ActivityMode.VEHICLE_BRAKING,
                "电梯": ActivityMode.ELEVATOR,
                "车库停车": ActivityMode.GARAGE_PARKING,
            },
            "Light_Intensity": {
                "未知": LightIntensity.NULL,
                "极暗": LightIntensity.EXTREMELY_DARK,
                "昏暗": LightIntensity.DIM,
                "正常": LightIntensity.MODERATE_BRIGHTNESS,
                "明亮": LightIntensity.BRIGHT,
                "刺眼": LightIntensity.HARSH_LIGHT,
            },
            "Sound_Intensity": {
                "未知": SoundIntensity.NULL,
                "安静": SoundIntensity.VERY_QUIET,
                "轻柔": SoundIntensity.SOFT_SOUND,
                "正常": SoundIntensity.NORMAL_SOUND,
                "嘈杂": SoundIntensity.NOISY,
                "非常嘈杂": SoundIntensity.VERY_NOISY,
            },
            "Location": {
                "未知": LocationType.NULL,
                "其他": LocationType.OTHER,
                "目的地": LocationType.DESTINATION,
                "家": LocationType.HOME,
                "工作场所": LocationType.WORK,
                "办公室": LocationType.WORK,  # 同义词
                "公交车站": LocationType.BUS_STATION,
                "地铁": LocationType.SUBWAY_STATION, # 同义词
                "地铁站": LocationType.SUBWAY_STATION,
                "火车站": LocationType.TRAIN_STATION,
                "机场": LocationType.AIRPORT,
                "住宿": LocationType.ACCOMMODATION,
                "住宅区": LocationType.RESIDENTIAL,
                "商业区": LocationType.COMMERCIAL,
                "学校": LocationType.SCHOOL,
                "医疗机构": LocationType.HEALTH,
                "政府机构": LocationType.GOVERNMENT,
                "娱乐场所": LocationType.ENTERTAINMENT,
                "餐饮场所": LocationType.DINING,
                "购物场所": LocationType.SHOPPING,
                "运动场所": LocationType.SPORT,
                "旅游景点": LocationType.ATTRACTION,
                "公园": LocationType.PARK,
                "街道": LocationType.STREET,
            },
            "Scence": {
                "未知": SceneType.NULL,
                "其他": SceneType.OTHER,
                "会议室": SceneType.MEETINGROOM,
                # DEMO新增演示场景
                "工位办公": SceneType.WORKSPACE,
                "餐厅": SceneType.DINING,
                "室外园区": SceneType.OUTDOOR_PARK,
                "地铁站": SceneType.SUBWAY_STATION,
            }
        }
    
    def generate_data_sequence(self, description: str, duration: float, interval: float) -> List[DataRecord]:
        """
        根据用户行为描述生成数据序列（使用DeepSeek API）
        
        Args:
            description: 用户行为自然语言描述
            duration: 序列时长（秒）
            interval: 数据间隔（秒）
            
        Returns:
            数据记录列表
        """
        # 检查缓存
        cache_key = f"{description}_{duration}_{interval}"
        if cache_key in self.behavior_cache:
            print("使用缓存数据")
            cached_data = self.behavior_cache[cache_key]
            return self._convert_to_records(cached_data, duration, interval)
        
        # 使用DeepSeek API生成行为序列
        try:
            behavior_plan = self._generate_behavior_plan_with_llm(description, duration, interval)
            
            # 缓存结果
            self.behavior_cache[cache_key] = behavior_plan
            
            # 转换为数据记录
            records = self._convert_to_records(behavior_plan, duration, interval)
            
            print(f"成功生成 {len(records)} 条数据记录")
            return records
            
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return []
    
    def _generate_behavior_plan_with_llm(self, description: str, duration: float, interval: float) -> List[Dict[str, Any]]:
        """
        使用DeepSeek API生成行为计划（支持分块生成）
        
        Args:
            description: 用户行为描述
            duration: 时长（秒）
            interval: 间隔（秒）
            
        Returns:
            行为计划列表
        """
        num_points = int(duration / interval)
        
        # 如果数据点数量较少，直接生成
        if num_points <= 50:
            return self._generate_single_chunk(description, duration, interval, num_points)
        else:
            # 分块生成：每块最多50个数据点
            return self._generate_chunked_plan(description, duration, interval, num_points)
    
    def _generate_single_chunk(self, description: str, duration: float, interval: float, num_points: int) -> List[Dict[str, Any]]:
        """生成单个数据块"""
        prompt = self._build_prompt(description, duration, interval, num_points)
        response = self._call_deepseek_api(prompt, num_points)
        return self._parse_llm_response(response, num_points, max_retries=2)
    
    def _generate_chunked_plan(self, description: str, duration: float, interval: float, num_points: int) -> List[Dict[str, Any]]:
        """分块生成行为计划（支持连续性）"""
        print(f"数据点数量较多({num_points}个)，启用分块生成策略")
        
        # 每块最多50个数据点
        max_chunk_size = 50
        chunks = []
        
        # 计算需要多少块
        num_chunks = (num_points + max_chunk_size - 1) // max_chunk_size
        points_per_chunk = num_points // num_chunks
        remainder = num_points % num_chunks
        
        print(f"将{num_points}个数据点分为{num_chunks}块生成，确保连续性")
        
        # 跟踪前一个块的结束状态和行为摘要
        previous_end_state = None
        previous_behavior_summary = None
        
        for chunk_idx in range(num_chunks):
            # 计算当前块的数据点数量
            chunk_points = points_per_chunk
            if chunk_idx < remainder:
                chunk_points += 1
            
            # 计算当前块的起始时间和时长
            chunk_start_idx = sum(chunk_points for i in range(chunk_idx))
            chunk_duration = chunk_points * interval
            
            # 计算进度
            progress_start = chunk_start_idx / num_points
            progress_end = (chunk_start_idx + chunk_points) / num_points
            
            # 构建分块上下文信息（包含行为摘要）
            chunk_context = {
                "chunk_idx": chunk_idx,
                "total_chunks": num_chunks,
                "previous_end_state": previous_end_state,
                "previous_behavior_summary": previous_behavior_summary,
                "progress_start": progress_start,
                "progress_end": progress_end,
                "is_first_chunk": chunk_idx == 0,
                "is_last_chunk": chunk_idx == num_chunks - 1
            }
            
            print(f"生成第{chunk_idx+1}/{num_chunks}块: {chunk_points}个数据点，进度{progress_start:.0%}-{progress_end:.0%}")
            
            try:
                # 生成当前块（使用分块上下文）
                chunk_prompt = self._build_prompt(
                    description,
                    chunk_duration,
                    interval,
                    chunk_points,
                    chunk_context=chunk_context
                )
                chunk_response = self._call_deepseek_api(chunk_prompt, chunk_points)
                chunk_data = self._parse_llm_response(chunk_response, chunk_points, max_retries=1)
                
                # 保存当前块的结束状态，供下一个块使用
                if chunk_data:
                    previous_end_state = chunk_data[-1].copy()
                    print(f"第{chunk_idx+1}块结束状态已保存: {previous_end_state}")
                    
                    # 如果不是最后一个块，为当前块生成行为摘要，供下一个块使用
                    if chunk_idx < num_chunks - 1:
                        print(f"为第{chunk_idx+1}块生成行为摘要...")
                        previous_behavior_summary = self._generate_behavior_summary(chunk_data, description)
                        print(f"行为摘要生成完成: {previous_behavior_summary[:100]}...")
                
                chunks.extend(chunk_data)
                print(f"第{chunk_idx+1}块生成成功，共{len(chunk_data)}个数据点")
                
                # 添加小块之间的延迟，避免API速率限制
                if chunk_idx < num_chunks - 1:
                    time.sleep(0.5)
                
            except Exception as e:
                print(f"第{chunk_idx+1}块生成失败: {e}")
                # 如果某块失败，使用默认数据填充
                default_chunk = self._create_default_chunk(chunk_points)
                chunks.extend(default_chunk)
                
                # 更新前一个状态为默认数据的最后一个点
                if default_chunk:
                    previous_end_state = default_chunk[-1].copy()
                    # 为默认数据生成简单摘要
                    if chunk_idx < num_chunks - 1:
                        previous_behavior_summary = "使用默认数据填充，行为无变化。"
                
                print(f"使用默认数据填充第{chunk_idx+1}块")
        
        # 确保总数据点数量正确
        if len(chunks) > num_points:
            chunks = chunks[:num_points]
            print(f"裁剪多余数据点，保留{len(chunks)}个")
        elif len(chunks) < num_points:
            # 补充默认数据
            print(f"数据点不足，补充{num_points - len(chunks)}个默认数据点")
            while len(chunks) < num_points:
                chunks.append(self._create_default_data_point())
        
        return chunks
    
    def _create_default_chunk(self, num_points: int) -> List[Dict[str, Any]]:
        """创建默认数据块"""
        default_point = {
            "activity_mode": "未知",
            "Light_Intensity": "正常",
            "Sound_Intensity": "正常",
            "Location": "其他",
            "Scence": "其他"
        }
        
        return [default_point.copy() for _ in range(num_points)]
    
    def _create_default_data_point(self) -> Dict[str, Any]:
        """创建单个默认数据点"""
        return {
            "activity_mode": "未知",
            "Light_Intensity": "正常",
            "Sound_Intensity": "正常",
            "Location": "其他",
            "Scence": "其他"
        }
    
    def _build_prompt(self, description: str, duration: float, interval: float, num_points: int,
                     chunk_context: Optional[Dict[str, Any]] = None) -> str:
        """构建LLM提示词（支持分块上下文和行为摘要）
        
        Args:
            description: 用户行为描述
            duration: 时长（秒）
            interval: 间隔（秒）
            num_points: 数据点数量
            chunk_context: 分块上下文信息，包含：
                - chunk_idx: 当前块索引（从0开始）
                - total_chunks: 总块数
                - previous_end_state: 前一个块的结束状态（字典，包含所有5个字段）
                - previous_behavior_summary: 前一个块的行为摘要
                - progress_start: 进度开始百分比
                - progress_end: 进度结束百分比
                - is_first_chunk: 是否是第一个块
                - is_last_chunk: 是否是最后一个块
        """
        # 简化的枚举信息，只列出主要选项
        simplified_enum_info = """
activity_mode 可选值: 未知, 骑摩托车, 骑自行车, 坐着, 坐, 慢走, 跑步, 站立, 公交车, 小汽车, 地铁, 高铁, 火车, 登山, 快走, 刹车, 电梯, 车库停车
Light_Intensity 可选值: 未知, 极暗, 昏暗, 正常, 明亮, 刺眼
Sound_Intensity 可选值: 未知, 安静, 轻柔, 正常, 嘈杂, 非常嘈杂
Location 可选值: 未知, 其他, 目的地, 家, 工作场所, 办公室, 公交车站, 地铁站, 火车站, 机场, 住宿, 住宅区, 商业区, 学校, 医疗机构, 政府机构, 娱乐场所, 餐饮场所, 购物场所, 运动场所, 旅游景点, 公园, 街道
Scence 可选值: 未知, 其他, 会议室, 工位办公, 餐厅, 室外园区, 地铁站
"""
        
        # 基础prompt
        base_prompt = f"""你是一个用户行为模拟专家。请根据以下用户行为描述，生成一个合理的数据序列。

用户行为描述：{description}
总时长：{duration}秒
数据间隔：{interval}秒
数据点数量：{num_points}个

可用的数据枚举值：
{simplified_enum_info}

请生成一个包含{num_points}个时间点的数据序列。每个时间点应包含以下5个数据字段：
1. activity_mode: 用户姿态
2. Light_Intensity: 环境亮度
3. Sound_Intensity: 环境声音强度
4. Location: 位置类型
5. Scence: 图像场景分类

要求：
1. 数据变化要符合实际行为逻辑
2. 时间序列要有连贯性
3. 环境数据（亮度、声音）要与位置和活动相匹配
4. 使用中文枚举值
5. 需要在duraton内完成所有的行为

请以JSON数组格式返回，每个元素是一个对象，包含5个字段的中文值。
示例格式：
[
  {{
    "activity_mode": "坐着",
    "Light_Intensity": "明亮",
    "Sound_Intensity": "正常",
    "Location": "工作场所",
    "Scence": "会议室"
  }},
  // ... 更多数据点
]
"""
        
        # 添加分块上下文信息
        if chunk_context:
            chunk_info = f"""
重要：这是整个连续行为的第{chunk_context['chunk_idx']+1}/{chunk_context['total_chunks']}部分。
时间进度：{chunk_context['progress_start']:.0%} - {chunk_context['progress_end']:.0%}
"""
            
            if chunk_context['is_first_chunk']:
                chunk_info += "\n这是行为的开始部分。请从合理的初始状态开始。"
            elif chunk_context['is_last_chunk']:
                chunk_info += "\n这是行为的最后部分。请确保行为自然结束。"
            else:
                chunk_info += "\n这是行为的中间部分。请继续正在进行的行为。"
            
            # 如果有前一个块的行为摘要，使用它
            if chunk_context.get('previous_behavior_summary'):
                chunk_info += f"""
                
前一个部分的行为摘要：
{chunk_context['previous_behavior_summary']}

请基于这个摘要继续生成数据序列。你的序列应该自然地从前一个部分结束的地方开始。
"""
            elif chunk_context['previous_end_state']:
                # 如果没有摘要，只提供结束状态
                prev_state_str = json.dumps(chunk_context['previous_end_state'], ensure_ascii=False, indent=2)
                chunk_info += f"""
                
前一个部分的结束状态：
{prev_state_str}

请确保你的序列从这个状态自然过渡。
"""
            else:
                chunk_info += "\n没有前一个部分的信息（这是第一个部分）。"
            
            # 组合prompt
            prompt = chunk_info + "\n\n" + base_prompt + f"\n\n现在请生成符合\"{description}\"行为的数据序列（第{chunk_context['chunk_idx']+1}/{chunk_context['total_chunks']}部分）："
        else:
            # 单块生成
            prompt = base_prompt + f"\n\n现在请生成符合\"{description}\"行为的数据序列："
        
        return prompt
    
    def _call_deepseek_api(self, prompt: str, num_points: int = 10) -> str:
        """调用DeepSeek API
        
        Args:
            prompt: 提示词
            num_points: 需要生成的数据点数量，用于计算max_tokens
            
        Returns:
            API响应内容
        """
        url = f"{self.base_url}/chat/completions"
        
        # 根据数据点数量动态计算max_tokens
        # 每个数据点大约需要150个token，加上基础token
        base_tokens = 500  # 基础token用于系统消息和格式
        tokens_per_point = 150
        calculated_tokens = base_tokens + (tokens_per_point * num_points)
        
        # 限制在合理范围内：最小1000，最大16000（DeepSeek API限制）
        max_tokens = max(1000, min(calculated_tokens, 16000))
        
        print(f"API调用: 生成{num_points}个数据点，使用max_tokens={max_tokens}")
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个用户行为数据生成专家，请严格按照要求生成JSON格式的数据序列。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens
        }
        
        response = None
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=1500)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 提取JSON部分（处理可能的markdown代码块）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"API请求错误: {e}")
            if response is not None:
                try:
                    print(f"响应状态码: {response.status_code}")
                    print(f"响应内容: {response.text[:200]}...")
                except:
                    pass
            raise
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"API响应解析错误: {e}")
            if response is not None:
                try:
                    print(f"响应内容: {response.text[:500]}...")
                except:
                    pass
            raise
     
    def _parse_llm_response(self, response_text: str, expected_points: int, max_retries: int = 2) -> List[Dict[str, Any]]:
        """解析LLM响应为行为计划（带重试机制）
        
        Args:
            response_text: API响应文本
            expected_points: 期望的数据点数量
            max_retries: 最大重试次数
            
        Returns:
            行为计划列表
        """
        for attempt in range(max_retries + 1):
            try:
                # 尝试清理响应文本
                cleaned_text = self._clean_response_text(response_text)
                
                # 解析JSON
                data = json.loads(cleaned_text)
                
                if not isinstance(data, list):
                    raise ValueError(f"响应不是JSON数组，类型: {type(data)}")
                
                # 验证数据结构
                validated_data = self._validate_data_structure(data, expected_points)
                
                print(f"成功解析{len(validated_data)}个数据点")
                return validated_data
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt < max_retries:
                    print(f"解析失败，第{attempt+1}次重试: {e}")
                    # 如果是JSON解析错误，尝试修复
                    if isinstance(e, json.JSONDecodeError):
                        response_text = self._attempt_fix_json(response_text)
                    continue
                else:
                    print(f"解析失败，已达到最大重试次数: {e}")
                    print(f"响应文本前500字符: {response_text[:500]}...")
                    # 返回默认数据
                    return self._create_default_chunk(expected_points)
        
        # 不应该到达这里
        return self._create_default_chunk(expected_points)
    
    def _clean_response_text(self, text: str) -> str:
        """清理响应文本，提取JSON部分"""
        # 移除可能的markdown代码块
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # 移除可能的开头和结尾的空白字符
        text = text.strip()
        
        # 如果文本以"["开头，以"]"结尾，直接返回
        if text.startswith("[") and text.endswith("]"):
            return text
        
        # 尝试找到JSON数组的开始和结束
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx+1]
        
        return text
    
    def _attempt_fix_json(self, text: str) -> str:
        """尝试修复JSON格式"""
        # 简单的修复：确保括号匹配
        open_brackets = text.count("[")
        close_brackets = text.count("]")
        
        if open_brackets > close_brackets:
            text += "]" * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            text = "[" * (close_brackets - open_brackets) + text
        
        return text
    
    def _validate_data_structure(self, data: List[Dict[str, Any]], expected_points: int) -> List[Dict[str, Any]]:
        """验证数据结构并确保数据点数量正确"""
        validated_data = []
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"警告: 第{i+1}个数据点不是字典，跳过")
                continue
            
            # 确保包含所有必需的字段
            validated_item = {
                "activity_mode": item.get("activity_mode", "未知"),
                "Light_Intensity": item.get("Light_Intensity", "正常"),
                "Sound_Intensity": item.get("Sound_Intensity", "正常"),
                "Location": item.get("Location", "其他"),
                "Scence": item.get("Scence", "其他")
            }
            
            validated_data.append(validated_item)
        
        # 确保有足够的数据点
        if len(validated_data) < expected_points:
            print(f"警告: 验证后只有{len(validated_data)}个数据点，需要{expected_points}个")
            # 使用最后一个有效数据点填充
            last_item = validated_data[-1] if validated_data else self._create_default_data_point()
            while len(validated_data) < expected_points:
                validated_data.append(last_item.copy())
        elif len(validated_data) > expected_points:
            validated_data = validated_data[:expected_points]
        
        return validated_data
     
    def _convert_to_records(self, behavior_plan: List[Dict[str, Any]], duration: float, interval: float) -> List[DataRecord]:
        """将行为计划转换为数据记录"""
        records = []
        start_time = datetime.now()
        
        for i, data_point in enumerate(behavior_plan):
            timestamp = start_time + timedelta(seconds=i * interval)
            
            # 为每个数据字段创建记录
            for data_id, chinese_value in data_point.items():
                # 将中文值转换为枚举值
                enum_value = self._chinese_to_enum(data_id, chinese_value)
                
                # 对于Scence数据，生成SceneData（包含图像）
                if data_id == "Scence":
                    scene_type = SceneType(enum_value)
                    scene_data = self._generate_scene_data(scene_type)
                    record = DataRecord(timestamp, data_id, scene_data)
                else:
                    record = DataRecord(timestamp, data_id, enum_value)
                
                records.append(record)
        
        return records
    
    def _generate_scene_data(self, scene_type: SceneType) -> SceneData:
        """为场景类型生成SceneData（包含图像路径）"""
        # 使用图像生成器生成图像
        if self.image_generator:
            generated_image = self.image_generator.generate_scene_image(scene_type)
            
            if generated_image:
                # 如果有图像路径，创建包含图像路径的SceneData
                return SceneData(
                    scene_type=scene_type,
                    image_path=generated_image.image_path
                )
        
        # 如果没有图像生成器或生成失败，创建不包含图像的SceneData
        return SceneData.from_scene_type(scene_type)
     
    def _chinese_to_enum(self, data_id: str, chinese_value: str) -> int:
        """将中文字符串转换为枚举值"""
        if data_id in self.enum_mappings:
            mapping = self.enum_mappings[data_id]
            if chinese_value in mapping:
                return mapping[chinese_value].value
        
        # 如果找不到映射，返回未知/默认值
        print(f"警告: 未知的中文值 '{chinese_value}' 对于数据ID '{data_id}'")
        
        if data_id == "activity_mode":
            return ActivityMode.NULL
        elif data_id == "Light_Intensity":
            return LightIntensity.NULL
        elif data_id == "Sound_Intensity":
            return SoundIntensity.NULL
        elif data_id == "Location":
            return LocationType.NULL
        elif data_id == "Scence":
            return SceneType.NULL
        else:
            return 0
    
    def _generate_behavior_summary(self, behavior_data: List[Dict[str, Any]], description: str) -> str:
        """生成行为摘要
        
        调用LLM API生成前一个块的行为摘要
        
        Args:
            behavior_data: 前一个块的行为数据
            description: 原始行为描述
            
        Returns:
            行为摘要字符串
        """
        if not behavior_data or len(behavior_data) < 3:
            return "行为数据不足，无法生成摘要"
        
        try:
            # 提取关键数据点：开始、中间、结束
            start_point = behavior_data[0]
            mid_idx = len(behavior_data) // 2
            mid_point = behavior_data[mid_idx]
            end_point = behavior_data[-1]
            
            summary_prompt = f"""请根据以下行为数据生成一个简短的摘要（最多3句话），描述这个时间段内发生了什么：

原始行为描述：{description}
数据点数量：{len(behavior_data)}个

关键数据点：
1. 开始状态：{json.dumps(start_point, ensure_ascii=False)}
2. 中间状态：{json.dumps(mid_point, ensure_ascii=False)}
3. 结束状态：{json.dumps(end_point, ensure_ascii=False)}

请用中文生成一个简洁的摘要，描述：
- 用户的主要活动变化
- 环境的变化（亮度、声音）
- 位置和场景的变化

摘要格式示例："用户从慢走开始，逐渐加速到跑步，环境从明亮变为正常亮度，位置从街道移动到公园。"
"""
            
            # 调用API生成摘要
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个行为分析专家，请根据数据生成简洁的行为摘要。"},
                    {"role": "user", "content": summary_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            summary = result["choices"][0]["message"]["content"].strip()

            return summary

        except Exception as e:
            print(f"生成行为摘要失败: {e}")
            # 生成一个简单的摘要作为后备
            if behavior_data:
                start_activity = behavior_data[0].get('activity_mode', '未知')
                end_activity = behavior_data[-1].get('activity_mode', '未知')
                return f"用户从{start_activity}变化到{end_activity}。"
            return "行为摘要生成失败，使用默认摘要。"

# 工具函数
def create_llm_simulator(api_key: Optional[str] = None,
                        image_generator: Optional[ImageGenerator] = None) -> LLMSimulator:
    """创建LLM模拟器实例"""
    if api_key is None:
        api_key = os.getenv("LLM_API_KEY")
    
    return LLMSimulator(api_key=api_key, image_generator=image_generator)


if __name__ == "__main__":
    # 简单的自测试
    print("=== LLM模拟器模块自测试 ===")
    
    simulator = create_llm_simulator()
    
    # 测试数据生成
    print("\n测试通勤场景数据生成:")
    try:
        records = simulator.generate_data_sequence("早上通勤上班", duration=5.0, interval=1.0)
        print(f"生成了 {len(records)} 条数据记录")
        
        # 显示前几条记录
        for i in range(len(records)):
            record = records[i]
            value_str = record.value
            # 如果是枚举值，尝试转换为中文
            if isinstance(value_str, int):
                value_str = to_str(record.data_id, value_str)
            
            print(f"  记录{i+1}: {record.data_id} = {value_str} @ {record.timestamp.strftime('%H:%M:%S')}")
            
    except Exception as e:
        print(f"测试失败: {e}")
    
    print("\nLLM模拟器自测试完成")