"""
AI Chat Service - AI聊天服务
"""

import base64
import requests
import json
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from openai import AzureOpenAI

from ....config import get_config

class ModelType(Enum):
    GPT = "GPT"
    LM_STUDIO = "LM_STUDIO"

class AIChatTool:
    """
    AI聊天工具类，用于与各种AI模型进行交互
    """
    def __init__(self):
        self.config = get_config()
        self.llm_studio_url = self.config.ai.llm_studio_url
        self._setup_clients()

    def _setup_clients(self):
        """设置AI客户端"""
        try:
            # 设置OpenAI API密钥和基础URL
            import openai
            openai.api_key = self.config.ai.openai_api_key
            openai.base_url = self.config.ai.openai_base_url
        except Exception as e:
            print(f"初始化AI客户端失败: {e}")

    @staticmethod
    def image_to_base64(file_path: str) -> str:
        """
        将图像文件转换为base64编码
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            str: base64编码的字符串
        """
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

    def build_messages(self,
                       prompt: str,
                       content: Optional[str] = None,
                       image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        构建消息列表
        
        Args:
            prompt: 提示词
            content: 额外内容
            image_path: 图像路径
            
        Returns:
            List[Dict[str, Any]]: 消息列表
        """
        if image_path:
            # 文本 + 图
            b64 = self.image_to_base64(image_path)
            return [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }]

        # 纯文本
        messages = [{"role": "user", "content": prompt}]
        if content:
            messages.append({"role": "user", "content": content})

        return messages

    def call_chatai(self,
                    model_type: Union[str, ModelType],
                    prompt: str,
                    content: Optional[str] = None,
                    image_path: Optional[str] = None) -> str:
        """
        调用AI模型
        
        Args:
            model_type: 模型类型
            prompt: 提示词
            content: 额外内容
            image_path: 图像路径
            
        Returns:
            str: AI模型返回的结果
        """
        try:
            # 转换为枚举类型
            if isinstance(model_type, str):
                model_type = ModelType[model_type.upper().replace(' ', '_')]

            messages = self.build_messages(prompt, content, image_path)

            if model_type == ModelType.GPT:
                return self._call_gpt(messages)
            elif model_type == ModelType.LM_STUDIO:
                return self._call_local_llm(messages)
            else:
                raise Exception(f"不支持的模型类型: {model_type}")

        except Exception as e:
            return f"模型调用失败: {str(e)}"

    def _call_gpt(self, messages: List[Dict[str, Any]], model: str = 'gpt-4o') -> str:
        """
        调用GPT模型
        
        Args:
            messages: 消息列表
            model: 模型名称
            
        Returns:
            str: GPT模型返回的结果
        """
        try:
            import openai
            response = openai.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"GPT调用失败: {e}")

    def _call_local_llm(self, messages: List[Dict[str, Any]]) -> str:
        """
        调用本地LLM Studio
        
        Args:
            messages: 消息列表
            
        Returns:
            str: 本地LLM返回的结果
        """
        payload = {
            "model": "local",
            "messages": messages,
            "max_tokens": 2500,
            "temperature": 0.3
        }

        try:
            response = requests.post(
                url=f"{self.llm_studio_url}/chat/completions",
                json=payload,
                timeout=60
            )

            response.raise_for_status()  # 抛出HTTP错误
            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        except requests.exceptions.RequestException as e:
            raise Exception(f"本地LLM请求失败: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"本地LLM响应格式错误: {e}")