# src/vlm_client.py
from typing import List, Dict, Any
from .data_models import SceneBatch
from .logger import get_logger
logger = get_logger(__name__)

from openai import OpenAI
import httpx
# from dotenv import load_dotenv

# load_dotenv()

class VLMClient:
    def __init__(self, model: str = "qwen2.5vl:7b", max_images_per_request: int = 8):
        self.model = model
        self.max_images_per_request = max_images_per_request
        logger.info(f"初始化VLM客户端，模型: {model}")
        custom_client = httpx.Client(
            trust_env=False,         # 忽略系统代理环境变量
            timeout=30.0,            # 请求超时时间
            limits=httpx.Limits(     # 连接池配置
                max_connections=100,  # 最大连接数
                max_keepalive_connections=20,  # 保持活跃的连接数
                keepalive_expiry=300  # 保持连接时间(秒)
            ),
            http2=True,              # 启用HTTP/2
        )
        api_base = "http://10.123.183.18:11434/v1/"
        api_key = "ollama"
        self.cli = OpenAI(base_url=api_base, api_key=api_key, http_client=custom_client)

    def chat(self, messages, temperature=0.1):
        """messages: list[dict] -> assistant reply str"""
        logger.info(f"调用VLM，消息数量: {len(messages)}")
        resp = self.cli.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def analyze_scenes(self, scene_batch) -> str:
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

        # 构建批量消息
        messages = self._build_scene_messages(scene_batch)

        # 调用VLM
        logger.info(f"调用VLM进行场景分析")
        try:
            response = self.cli.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"VLM调用失败: {e}")
            raise

    def _build_scene_messages(self, scene_batch) -> List[Dict[str, Any]]:
        """构建场景分析的消息"""
        # 提取所有图片
        images = [frame.image_base64 for frame in scene_batch.frames]

        # 构建帧描述文本
        frame_descriptions = []
        for frame in scene_batch.frames:
            time_str = frame.timestamp.strftime("%H:%M:%S")

            # 构建该帧的描述
            desc = f"帧{frame.frame_id} (时间: {time_str}): "

            # 添加元数据中的关键信息
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

            desc += "  " + "\n  ".join(available_metadata)
            frame_descriptions.append(desc)

        # 构建系统提示词
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
  - main_activity: (string) 该场景的主要活动类型，一个词语概括
  
示例输出格式：
{
  "scenes": [
    {
      "scene_id": 1,
      "description": "详细的场景描述...",
      "main_activity": "一个词语概括"
    }
  ]
}
"""

        # 构建用户提示词
        user_prompt = f"""请分析以下图片及其相关的元数据信息，给出详细的场景分析。

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

        # 构建消息（使用Ollama格式）
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
                "images": images  # Ollama格式：直接将图片列表放在images键中
            }
        ]

        return messages

    def _analyze_in_batches(self, scene_batch):
        """分批处理（如果需要的话）"""
        logger.warning("分批处理功能待实现，将使用前{self.max_images_per_request}张图片")
        # 只取前max_images_per_request张
        truncated_batch = SceneBatch(frames=scene_batch.frames[:self.max_images_per_request])
        return self.analyze_scenes(truncated_batch)


if __name__ == "__main__":
    import base64
    img_path1 = "../data/1.png"
    img_path2 = "../data/2.png"
    with open(img_path1, "rb") as image_file:
        base64_image1 = base64.b64encode(image_file.read()).decode('utf-8')
    with open(img_path2, "rb") as image_file:
        base64_image2 = base64.b64encode(image_file.read()).decode('utf-8')

    custom_client = httpx.Client(
        trust_env=False,  # 忽略系统代理环境变量
        timeout=30.0,  # 请求超时时间
        limits=httpx.Limits(  # 连接池配置
            max_connections=100,  # 最大连接数
            max_keepalive_connections=20,  # 保持活跃的连接数
            keepalive_expiry=300  # 保持连接时间(秒)
        ),
        http2=True,  # 启用HTTP/2
    )
    client = OpenAI(
        api_key="ollama",
        base_url="http://10.123.183.18:11434/v1/",
        http_client=custom_client
    )
    completion = client.chat.completions.create(
        model="qwen2.5vl:7b",
        messages=[
            {
                'role': 'user',
                'content': 'What is in this image? Be concise.',
                'images': [base64_image1, base64_image2],
            }
        ],
        stream=True,
        stream_options={"include_usage": True}
    )
    for chunk in completion:
        print(chunk.model_dump())