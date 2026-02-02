"""
图像生成模块 - 使用硅基流动的Qwen/Qwen-Image模型生成场景照片
支持本地文件缓存，减少API调用开销
"""

import os
import json
import base64
import requests
import hashlib
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

from datapallet.enums import SceneType, to_str


@dataclass
class GeneratedImage:
    """生成的图像数据"""
    scene_type: SceneType
    image_data: bytes  # 原始图像字节数据
    image_path: Optional[str] = None  # 图像文件路径（如果已保存）
    prompt: str = ""  # 生成图像的提示词
    timestamp: Optional[datetime] = None  # 生成时间戳
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def save_to_file(self, filepath: str) -> bool:
        """保存图像到文件"""
        try:
            with open(filepath, 'wb') as f:
                f.write(self.image_data)
            self.image_path = filepath  # 更新image_path
            return True
        except Exception as e:
            print(f"保存图像失败: {e}")
            return False
    
    def to_base64(self) -> str:
        """转换为base64字符串"""
        return base64.b64encode(self.image_data).decode('utf-8')
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "scene_type": self.scene_type.value,
            "scene_type_str": to_str("Scence", self.scene_type),
            "image_size": len(self.image_data),
            "image_path": self.image_path,
            "prompt": self.prompt,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class ImageGenerator:
    """图像生成器，使用硅基流动的Qwen模型API，支持本地文件缓存"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str = "https://api.siliconflow.cn/v1",
                 cache_dir: str = "scene_images"):
        """
        初始化图像生成器
        
        Args:
            api_key: 硅基流动API密钥
            base_url: API基础URL
            cache_dir: 本地缓存目录
        """
        if api_key is None:
            api_key = os.getenv("VLM_SILICONFLOW_API_KEY")
        self.api_key = api_key
        self.base_url = base_url
        self.cache_dir = cache_dir
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 模型名称
        self.model = "Qwen/Qwen-Image"
        
        # 缓存生成的图像（scene_type -> GeneratedImage）
        self.image_cache: Dict[int, GeneratedImage] = {}
        
        # 场景描述映射
        self.scene_descriptions = {
            SceneType.NULL: "一个未知的室内场景，光线昏暗，没有明显特征",
            SceneType.OTHER: "一个普通的室内或室外场景，具有日常环境特征",
            SceneType.MEETINGROOM: "一个现代化的会议室，有长桌、椅子、投影屏幕和窗户，会议室坐着几个男人和女人，光线明亮，适合商务会议",
            # DEMO新增演示场景
            SceneType.WORKSPACE: "现代化的开放式办公区域，有电脑显示器、办公桌椅、隔断，员工正在专注工作，典型的办公室环境",
            SceneType.DINING: "繁忙的餐厅或食堂区域，有餐桌、餐椅和正在用餐的人群，光线温暖",
            SceneType.OUTDOOR_PARK: "风景优美的室外园区，有绿树、草坪、铺装步道，阳光明媚，适合散步",
            SceneType.SUBWAY_STATION: "地铁站站台层，有屏蔽门、列车轨道、指示牌和候车座椅，典型的公共交通场景",
        }
        
        # 默认图像尺寸
        self.default_size = "640x480"
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载已存在的缓存文件
        self._load_cached_images()
    
    def _load_cached_images(self):
        """从缓存目录加载已存在的图像"""
        print(f"从目录 {self.cache_dir} 加载缓存图像...")
        
        for scene_type in SceneType:
            scene_name = to_str("Scence", scene_type).replace(" ", "_")
            filename = f"{scene_name}.png"
            filepath = os.path.join(self.cache_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        image_data = f.read()
                    
                    generated_image = GeneratedImage(
                        scene_type=scene_type,
                        image_data=image_data,
                        image_path=filepath,
                        prompt=f"从缓存加载: {scene_name}",
                        timestamp=datetime.fromtimestamp(os.path.getmtime(filepath))
                    )
                    
                    self.image_cache[scene_type.value] = generated_image
                    print(f"  已加载缓存图像: {scene_name}")
                    
                except Exception as e:
                    print(f"  加载缓存图像失败 {filename}: {e}")
    
    def generate_scene_image(self, scene_type: SceneType,
                            prompt_override: Optional[str] = None,
                            size: Optional[str] = None,
                            force_regenerate: bool = False) -> Optional[GeneratedImage]:
        """
        为指定场景类型生成图像
        
        Args:
            scene_type: 场景类型枚举
            prompt_override: 可选的提示词覆盖
            size: 图像尺寸，格式为"宽x高"，如"1024x1024"
            force_regenerate: 是否强制重新生成（忽略缓存）
            
        Returns:
            GeneratedImage对象，如果生成失败则返回None
        """
        # 检查缓存（除非强制重新生成）
        if not force_regenerate and scene_type.value in self.image_cache:
            cached_image = self.image_cache[scene_type.value]
            print(f"使用缓存的图像 for scene type: {to_str('Scence', scene_type)}")
            return cached_image
        
        # 检查本地文件缓存
        if not force_regenerate:
            cached_image = self._load_from_file_cache(scene_type)
            if cached_image:
                self.image_cache[scene_type.value] = cached_image
                print(f"从文件缓存加载图像 for scene type: {to_str('Scence', scene_type)}")
                return cached_image
        
        # 构建提示词
        if prompt_override:
            prompt = prompt_override
        else:
            prompt = self._build_scene_prompt(scene_type)
        
        # 调用API生成图像
        image_data = self._call_qwen_api(prompt, size or self.default_size)
        
        if image_data:
            # 先保存到文件缓存，获取文件路径
            filepath = self._get_scene_filepath(scene_type)
            try:
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                print(f"图像已保存到文件缓存: {filepath}")
            except Exception as e:
                print(f"保存到文件缓存失败: {e}")
                filepath = None
            
            # 创建GeneratedImage对象，包含image_path
            generated_image = GeneratedImage(
                scene_type=scene_type,
                image_data=image_data,
                image_path=filepath,
                prompt=prompt
            )
            
            # 缓存结果
            self.image_cache[scene_type.value] = generated_image
            
            print(f"成功生成图像 for scene type: {to_str('Scence', scene_type)}")
            return generated_image
        else:
            print(f"生成图像失败 for scene type: {to_str('Scence', scene_type)}")
            return None
    
    def _get_scene_filepath(self, scene_type: SceneType) -> str:
        """获取场景图像的文件路径"""
        scene_name = to_str("Scence", scene_type).replace(" ", "_")
        filename = f"{scene_name}.png"
        return os.path.join(self.cache_dir, filename)
    
    def _load_from_file_cache(self, scene_type: SceneType) -> Optional[GeneratedImage]:
        """从文件缓存加载图像"""
        filepath = self._get_scene_filepath(scene_type)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    image_data = f.read()
                
                return GeneratedImage(
                    scene_type=scene_type,
                    image_data=image_data,
                    image_path=filepath,
                    prompt=f"从文件缓存加载: {to_str('Scence', scene_type)}",
                    timestamp=datetime.fromtimestamp(os.path.getmtime(filepath))
                )
            except Exception as e:
                print(f"从文件缓存加载失败 {filepath}: {e}")
        
        return None
    
    def _save_to_file_cache(self, scene_type: SceneType, image: GeneratedImage):
        """保存图像到文件缓存"""
        scene_name = to_str("Scence", scene_type).replace(" ", "_")
        filename = f"{scene_name}.png"
        filepath = os.path.join(self.cache_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                f.write(image.image_data)
            print(f"图像已保存到文件缓存: {filepath}")
        except Exception as e:
            print(f"保存到文件缓存失败: {e}")
    
    def _build_scene_prompt(self, scene_type: SceneType) -> str:
        """构建场景描述提示词"""
        base_description = self.scene_descriptions.get(scene_type, "一个普通的场景")
        
        # 根据场景类型添加更多细节
        if scene_type == SceneType.MEETINGROOM:
            prompt = f"{base_description}。高清摄影，专业构图，自然光线，现代办公室风格，细节丰富，写实风格。"
        elif scene_type == SceneType.OTHER:
            prompt = f"{base_description}。日常环境，写实风格，自然光线，细节丰富，高清摄影。"
        else:
            prompt = f"{base_description}。写实风格，自然光线，细节丰富。"
        
        return prompt
    
    def _call_qwen_api(self, prompt: str, size: str = "1024x1024") -> Optional[bytes]:
        """
        调用硅基流动的Qwen图像生成API
        
        Args:
            prompt: 图像生成提示词
            size: 图像尺寸，如"1024x1024"
            
        Returns:
            图像字节数据，如果失败则返回None
        """
        url = f"{self.base_url}/images/generations"
        
        # 解析尺寸
        try:
            width, height = map(int, size.split('x'))
        except:
            width, height = 640, 480
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "size": f"{width}x{height}",
            "batch_size": 1
        }
        
        try:
            print(f"调用Qwen图像生成API: {prompt[:50]}...")
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # 解析响应，获取图像URL
            if "data" in result and len(result["data"]) > 0:
                image_url = result["data"][0].get("url")
                if image_url:
                    print(f"图像生成成功，URL: {image_url}")
                    # 下载图像
                    return self._download_image(image_url)
            
            print(f"API响应格式异常: {result}")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"API请求错误: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print(f"响应状态码: {e.response.status_code}")
                    print(f"响应内容: {e.response.text[:200]}...")
                except:
                    pass
            return None
        except Exception as e:
            print(f"图像生成处理错误: {e}")
            return None
    
    def _download_image(self, image_url: str) -> Optional[bytes]:
        """从URL下载图像"""
        try:
            print(f"下载图像: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            image_data = response.content
            print(f"图像下载成功，大小: {len(image_data)} 字节")
            return image_data
            
        except Exception as e:
            print(f"下载图像失败: {e}")
            return None
    
    def get_cached_image(self, scene_type: SceneType) -> Optional[GeneratedImage]:
        """获取缓存的图像"""
        return self.image_cache.get(scene_type.value)
    
    def pregenerate_all_scenes(self, force_regenerate: bool = False) -> Dict[SceneType, GeneratedImage]:
        """预生成所有场景类型的图像"""
        generated = {}
        
        for scene_type in SceneType:
            print(f"预生成场景图像: {to_str('Scence', scene_type)}")
            image = self.generate_scene_image(scene_type, force_regenerate=force_regenerate)
            if image:
                generated[scene_type] = image
        
        return generated
    
    def save_all_images(self, directory: Optional[str] = None) -> Dict[SceneType, str]:
        """保存所有缓存的图像到指定目录"""
        if directory is None:
            directory = self.cache_dir
        
        saved_paths = {}
        
        # 创建目录
        os.makedirs(directory, exist_ok=True)
        
        for scene_type_value, image in self.image_cache.items():
            scene_type = SceneType(scene_type_value)
            scene_name = to_str("Scence", scene_type).replace(" ", "_")
            filename = f"{scene_name}.png"
            filepath = os.path.join(directory, filename)
            
            if image.save_to_file(filepath):
                saved_paths[scene_type] = filepath
                print(f"图像已保存: {filepath}")
        
        return saved_paths
    
    def clear_cache(self):
        """清除内存缓存"""
        self.image_cache.clear()
        print("内存缓存已清除")
    
    def clear_file_cache(self):
        """清除文件缓存"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"文件缓存目录已清除: {self.cache_dir}")


# 工具函数
def create_image_generator(api_key: Optional[str] = None, 
                          cache_dir: Optional[str] = None) -> ImageGenerator:
    """创建图像生成器实例"""
    if api_key is None:
        api_key = os.getenv("VLM_SILICONFLOW_API_KEY")
    
    if cache_dir is None:
        cache_dir = "scene_images"
    
    return ImageGenerator(api_key=api_key, cache_dir=cache_dir)


if __name__ == "__main__":
    # 简单的自测试
    print("=== 图像生成模块自测试 ===")
    
    generator = create_image_generator()
    
    # 测试生成会议室图像
    print("\n1. 测试生成会议室图像:")
    from enums import SceneType
    
    meeting_image = generator.generate_scene_image(SceneType.MEETINGROOM)
    if meeting_image:
        print(f"  生成成功！图像大小: {len(meeting_image.image_data)} 字节")
        print(f"  提示词: {meeting_image.prompt[:50]}...")
        
        # 保存测试图像
        meeting_image.save_to_file("test_meetingroom.png")
        print("  图像已保存为 test_meetingroom.png")
    else:
        print("  生成失败")
    
    # 测试缓存
    print("\n2. 测试缓存功能:")
    cached_image = generator.get_cached_image(SceneType.MEETINGROOM)
    if cached_image:
        print("  成功从缓存获取图像")
    
    # 测试预生成所有场景
    print("\n3. 测试预生成所有场景:")
    generated = generator.pregenerate_all_scenes()
    print(f"  成功生成/加载 {len(generated)} 个场景图像")
    
    # 测试保存所有图像
    print("\n4. 测试保存所有图像:")
    saved = generator.save_all_images("test_images")
    print(f"  保存了 {len(saved)} 个图像文件")
    
    print("\n图像生成模块自测试完成")