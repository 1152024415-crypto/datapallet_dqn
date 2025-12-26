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
        logger.info(f"调用VLM进行场景分析，共{len(scene_batch.frames)}帧")
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
            if "object_classes" in meta:
                desc += f"物体: {', '.join(meta['object_classes'][:3])}"
                if len(meta["object_classes"]) > 3:
                    desc += f" 等{len(meta['object_classes'])}个物体"
            if "activity_type" in meta:
                desc += f"，活动: {meta['activity_type']}"
            if "specific_address" in meta:
                desc += f"，位置: {meta['specific_address']}"

            frame_descriptions.append(desc)

        # 构建系统提示词
        system_prompt = """你是一个专业的场景分析AI。你需要分析一系列按时间顺序排列的图片帧（每张图片都有关联的元数据），识别连续的相关活动，并将它们划分为不同的场景。

你的任务：
1. 分析所有图片帧的内容和关联元数据
2. 识别每帧中的活动、物体、人物和状态变化
3. 基于时间连续性和活动相关性将帧分组为不同的场景
4. 输出详细的场景划分结果

输出要求：
- 你必须且只能输出一个纯粹的JSON对象
- 不要包含任何Markdown格式（不要使用```json或```）
- 不要包含任何额外的文本、解释或说明
- 直接输出JSON对象，以{开始，以}结束

输出JSON必须包含以下字段：
- scenes: 场景列表，每个场景包含：
  - scene_id: (int) 场景编号，从1开始
  - start_frame: (int) 起始帧编号
  - end_frame: (int) 结束帧编号（包含）
  - description: (string) 对该场景的详细描述
  - main_activity: (string) 该场景的主要活动类型
  - confidence: (float) 划分的置信度，0-1之间
  - tags: (list) 场景标签列表
- summary: (string) 整体场景分析的总结"""

        # 构建用户提示词
        user_prompt = f"""请分析以下连续的 {len(scene_batch.frames)} 帧图片，并将它们划分为不同的场景。

图片按时间顺序排列（从早到晚）：
{chr(10).join(frame_descriptions)}

重要说明：
1. 每张图片都与上方的描述一一对应
2. 帧编号从0开始，依次递增
3. 请基于视觉内容和元数据信息进行场景划分

场景划分原则：
1. 一个场景应该包含时间上连续的若干帧
2. 同一场景内的帧应该具有一致的主要活动或主题
3. 场景之间的切换应该有明显的活动变化或主题变化
4. 尽可能将相关活动合并到同一场景中

请输出划分结果："""

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
        #TODO 这里实现分批逻辑
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