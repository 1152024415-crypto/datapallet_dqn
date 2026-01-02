# src/vlm_client.py
from typing import List, Dict, Any, Optional, Tuple
from .data_models import SceneBatch
from .logger import get_logger
import base64
import json
import requests
from datetime import datetime
import warnings
import ssl
logger = get_logger(__name__)

class VLMClient:
    def __init__(self,
                 api_type: str = "openai",  # "openai" 或 "siliconflow"
                 max_images_per_request: int = 8,
                 model: Optional[str] = None):
        """
        初始化VLM客户端，支持两种完全不同的API调用方式

        Args:
            api_type: API类型，"openai"或"siliconflow"
            max_images_per_request: 单次请求最大图片数
            model: 指定模型
        """
        self.api_type = api_type
        self.max_images_per_request = max_images_per_request

        # 根据API类型设置默认模型
        if model:
            self.model = model
        elif api_type == "siliconflow":
            self.model = "Qwen/Qwen3-VL-8B-Instruct"
        else:  # openai格式
            self.model = "qwen2.5vl:7b"

        logger.info(f"初始化VLM客户端，API类型: {api_type}, 模型: {self.model}")

        # 根据API类型配置不同的客户端
        if api_type == "siliconflow":
            self._init_siliconflow_client()
        else:
            self._init_openai_client()

    def _init_openai_client(self):
        """初始化OpenAI格式客户端（使用OpenAI库）"""
        from openai import OpenAI
        import httpx

        # 配置HTTP客户端
        custom_client = httpx.Client(
            verify=False,
            trust_env=False,
            timeout=60.0,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=300
            ),
            http2=True,
        )

        # OpenAI格式配置
        self.api_base = "http://10.123.183.18:11434/v1/"
        self.api_key = "ollama"

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            http_client=custom_client
        )
        self._call_api = self._call_openai_api

    def _init_siliconflow_client(self):
        """初始化SiliconFlow客户端（使用requests/httpx）"""
        # SiliconFlow API配置
        self.api_base = "https://api.siliconflow.cn/v1/chat/completions"
        self.api_key = "sk-qrxwstofijzxgtrsypbihgvixxlbyelexdaymbkbnjlexoxi"  # 请替换为您的真实API密钥

        # 设置请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 设置超时
        self.timeout = 60.0
        self._call_api = self._call_siliconflow_api

        # 禁用SSL警告
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # ==================== 公共方法 ====================

    def analyze_scenes(self, scene_batch: SceneBatch) -> str:
        """
        分析场景并划分

        Args:
            scene_batch: SceneBatch对象

        Returns:
            str: VLM返回的场景划分结果
        """
        # 如果图片太多，分批处理
        if len(scene_batch.frames) > self.max_images_per_request:
            logger.warning(f"图片数量({len(scene_batch.frames)})超过限制，将分批处理")
            return self._analyze_in_batches(scene_batch)

        # 构建消息
        if self.api_type == "siliconflow":
            messages = self._build_siliconflow_messages(scene_batch)
        else:
            messages = self._build_openai_messages(scene_batch)

        # 调用API
        logger.info(f"调用{self.model}进行场景分析，API类型: {self.api_type}")

        try:
            response = self._call_api(messages)
            return response

        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise RuntimeError(f"API调用失败: {str(e)}")

    def _analyze_in_batches(self, scene_batch: SceneBatch) -> str:
        """分批处理"""
        logger.warning(f"分批处理功能待实现，将使用前{self.max_images_per_request}张图片")
        truncated_batch = SceneBatch(frames=scene_batch.frames[:self.max_images_per_request])
        return self.analyze_scenes(truncated_batch)

    # ==================== 公共辅助方法 ====================

    def _extract_frame_descriptions(self, scene_batch: SceneBatch, max_frames: Optional[int] = None) -> Tuple[
        List[str], List[str]]:
        """
        提取帧描述和图片数据

        Args:
            scene_batch: SceneBatch对象
            max_frames: 最大帧数限制

        Returns:
            Tuple[List[str], List[str]]: (帧描述列表, 图片base64列表)
        """
        if max_frames is None:
            max_frames = self.max_images_per_request

        frames = scene_batch.frames[:max_frames]

        frame_descriptions = []
        image_bases = []

        for i, frame in enumerate(frames):
            time_str = frame.timestamp.strftime("%H:%M:%S")
            desc = f"图片{i + 1} (帧ID:{frame.frame_id}, 时间:{time_str}): "

            meta = frame.metadata
            available_metadata = []

            if "activity_mode" in meta:
                available_metadata.append(f"姿态模式: {meta['activity_mode']}")
            if "Light_Intensity" in meta:
                available_metadata.append(f"环境光亮度: {meta['Light_Intensity']}")
            if "Sound_Intensity" in meta:
                available_metadata.append(f"背景音强度: {meta['Sound_Intensity']}")
            if "Location" in meta:
                available_metadata.append(f"位置: {meta['Location']}")
            if "scene_type" in meta:
                available_metadata.append(f"图像场景分类: {meta['scene_type']}")

            if available_metadata:
                desc += " | " + " | ".join(available_metadata)

            frame_descriptions.append(desc)
            image_bases.append(frame.image_base64)

        return frame_descriptions, image_bases

    def _build_user_prompt_text(self, frame_descriptions: List[str], num_frames: int):
        """
        构建用户提示文本

        Args:
            frame_descriptions: 帧描述列表
            num_frames: 总帧数

        Returns:
            str: 用户提示文本
        """
        system_prompt = """你是一个专业的场景分析AI。你需要分析单张图片及其相关的元数据信息，识别当前场景的活动和状态。

你的任务：
1. 分析图片的视觉内容
2. 结合提供的元数据信息（可能不完整）进行综合分析
3. 识别当前场景的主要活动、状态和环境特征
4. 输出详细且准确的场景分析结果

重要说明：
- 元数据可能不完整：只提供存在的元信息，缺失的信息不要假设
- 结合视觉内容：优先根据图片内容分析，元数据作为补充信息
- main_activity要尽量简单：用一个词语概括主要活动
- description要详细：提供场景的详细描述，结合视觉内容和元数据

输出要求：
- 你必须且只能输出一个纯粹的JSON对象
- 不要包含任何Markdown格式（不要使用```json或```）
- 不要包含任何额外的文本、解释或说明
- 直接输出JSON对象，以{开始，以}结束

输出JSON必须包含以下字段：
- scenes: 场景列表，每个场景包含：
  - scene_id: (int) 场景编号，从1开始
  - description: (string) 对该场景的详细描述，结合视觉内容和元数据
  - main_activity: (string) 该场景的主要活动类型，一个词语概括"""

        analysis_guidelines = """场景分析要点：
1. 根据视觉内容确定主要活动或状态
2. 结合姿态模式（如有）推断人物活动
3. 根据环境光亮度（如有）判断光照条件
4. 根据背景音强度（如有）推断环境嘈杂度
5. 根据位置信息（如有）确定场景地点
6. 根据场景分类（如有）了解场景类型
7. 结合所有可用信息给出综合判断
8. main_activity要简单，一个词语概括（如：办公、会议、休息、行走等）
9. description要详细，包括场景环境、人物活动、环境条件等"""

        # SiliconFlow格式直接将系统提示合并到用户提示中
        user_prompt = f"""请分析以下{num_frames}张图片及其相关的元数据信息，给出详细的场景分析。

图片元数据信息：
{chr(10).join(frame_descriptions)}

重要说明：
1. 元数据信息来自同一张图片的不同维度感知
2. 元数据可能不完整，只提供已知信息
3. 请结合图片视觉内容和所有可用元数据进行综合分析

场景分析要点：
1. 根据视觉内容确定主要活动或状态
2. 结合姿态模式（如有）推断人物活动
3. 根据环境光亮度（如有）判断光照条件
4. 根据背景音强度（如有）推断环境嘈杂度
5. 根据位置信息（如有）确定场景地点
6. 根据场景分类（如有）了解场景类型
7. 结合所有可用信息给出综合判断
8. main_activity要简单，一个词语概括（如：办公、会议、休息、行走等）
9. description要详细，包括场景环境、人物活动、环境条件等

请基于以上信息，输出详细且准确的场景分析结果："""

        return system_prompt, user_prompt

    # ==================== API特定消息构建方法 ====================

    def _build_siliconflow_messages(self, scene_batch: SceneBatch) -> Dict[str, Any]:
        """构建SiliconFlow格式的请求数据"""
        # 提取帧描述和图片数据
        frame_descriptions, image_bases = self._extract_frame_descriptions(scene_batch)

        # 构建用户提示文本
        system_prompt, user_prompt = self._build_user_prompt_text(frame_descriptions, len(scene_batch.frames))

        # 构建SiliconFlow格式的content数组
        # 为每张图片添加image_url条目
        user_array = [{
            "type": "text",
            "text": user_prompt
        }]
        for image_base64 in image_bases:
            user_array.append({
                "type": "image_url",
                "image_url": {
                    "url": image_base64,
                    "detail": "auto"
                }
            })
        # 构建完整的SiliconFlow请求体
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": user_array
                }
            ],
            "stream": False,
            "max_tokens": 4096,
            "min_p": 0.05,
            "stop": [],
            "temperature": 0.1,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"},
            "tools": []
        }

        return payload

    def _build_openai_messages(self, scene_batch: SceneBatch) -> List[Dict[str, Any]]:
        """构建OpenAI格式的消息（用于Ollama）"""
        # 提取帧描述和图片数据
        frame_descriptions, image_bases = self._extract_frame_descriptions(scene_batch)

        # 构建系统提示词
        system_prompt, user_prompt = self._build_user_prompt_text(frame_descriptions, len(scene_batch.frames))

        # 构建Ollama格式的消息（使用images字段）
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
                "images": image_bases
            }
        ]

        return messages

    # ==================== API调用方法 ====================

    def _call_siliconflow_api(self, payload: Dict[str, Any]) -> str:
        """调用SiliconFlow API（使用requests）"""
        try:
            response = requests.post(
                self.api_base,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
                verify=False
            )

            # 检查响应状态
            response.raise_for_status()

            # 解析响应
            result = response.json()

            # 提取回复内容
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise RuntimeError(f"API响应格式错误: {result}")


        except requests.exceptions.HTTPError as e:
            logger.error(f"SiliconFlow API请求失败，状态码: {e.response.status_code}")
            logger.error(f"响应内容: {e.response.text}")
            raise RuntimeError(f"API请求失败，状态码: {e.response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"SiliconFlow API网络请求失败: {e}")
            raise RuntimeError(f"网络请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"API响应JSON解析失败: {e}")
            raise RuntimeError(f"响应解析失败: {str(e)}")
    def _call_openai_api(self, messages: List[Dict[str, Any]]) -> str:
        """调用OpenAI格式API（使用OpenAI客户端）"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            raise


# ==================== 测试代码 ====================

class MockFrame:
    """模拟帧对象，用于测试"""

    def __init__(self, frame_id, image_data):
        self.frame_id = frame_id
        self.image_base64 = image_data
        self.timestamp = datetime.now()
        self.metadata = {
            "activity_mode": "sitting",
            "Light_Intensity": 0.8,
            "scene_type": "office"
        }


class MockSceneBatch:
    """模拟SceneBatch对象，用于测试"""

    def __init__(self, frames):
        self.frames = frames


def test_vlm_client():
    """测试VLMClient的两种API格式"""
    from datetime import datetime
    import base64

    # 创建虚拟的base64图片数据
    import base64
    img_path1 = "../data/meeting_room.png"
    with open(img_path1, "rb") as image_file:
        test_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # 测试两种API格式
    for api_type in ["siliconflow", "openai"]:
        print(f"\n{'=' * 60}")
        print(f"测试 {api_type.upper()} 格式API")
        print('=' * 60)

        try:
            # 创建客户端
            client = VLMClient(api_type=api_type, max_images_per_request=2)

            # 创建模拟帧
            mock_frames = [
                MockFrame(1, test_image_base64),
                MockFrame(2, test_image_base64)
            ]

            # 创建模拟SceneBatch
            scene_batch = MockSceneBatch(mock_frames)

            # 调用API（注意：这里可能会因为API密钥问题而失败）
            # 实际使用时请取消注释并替换正确的API密钥
            response = client.analyze_scenes(scene_batch)
            print(f"API响应: {response[:500]}...")

            # 仅测试消息构建
            if api_type == "siliconflow":
                messages = client._build_siliconflow_messages(scene_batch)
                print("构建的SiliconFlow消息结构:")
                print(json.dumps(messages, indent=2, ensure_ascii=False)[:1000] + "...")
            else:
                messages = client._build_openai_messages(scene_batch)
                print("构建的OpenAI消息结构:")
                print(f"系统消息长度: {len(messages[0]['content'])}")
                print(f"用户消息文本长度: {len(messages[1]['content'])}")
                print(f"图片数量: {len(messages[1]['images'])}")

            print(f"✓ {api_type.upper()} API客户端初始化成功")

        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_vlm_client()