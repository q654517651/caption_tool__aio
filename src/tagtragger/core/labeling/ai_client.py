"""
AI Client - AI服务调用客户端
"""

import base64
import json
import requests
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from ...utils.logger import log_error, log_info
from ...utils.exceptions import AIServiceError
from ...config import get_config

class ModelType(Enum):
    """AI模型类型"""
    GPT = "GPT"
    LM_STUDIO = "LM_STUDIO"
    CLAUDE = "CLAUDE"
    LOCAL = "LOCAL"

class AIClient:
    """AI服务客户端"""
    
    def __init__(self):
        self.config = get_config()
        self._setup_clients()
    
    def _setup_clients(self):
        """设置各种AI客户端"""
        # GPT客户端设置
        self.gpt_config = {
            'api_key': '',
            'base_url': 'https://api.openai.com/v1',
            'model': 'gpt-4o'
        }
        
        # LM Studio配置
        self.lm_studio_config = {
            'base_url': 'http://127.0.0.1:1234/v1',
            'model': 'local-model'
        }
        
        # 从配置文件加载API密钥
        self._load_api_keys()
    
    def _load_api_keys(self):
        """加载API密钥"""
        try:
            # 尝试从gpt_key.json加载（兼容旧版本）
            try:
                with open('gpt_key.json', 'r') as f:
                    keys = json.load(f)
                    if 'OPENAI_API_KEY' in keys:
                        self.gpt_config['api_key'] = keys['OPENAI_API_KEY']
                    if 'AZURE_ENDPOINT' in keys:
                        self.gpt_config['base_url'] = keys['AZURE_ENDPOINT']
            except FileNotFoundError:
                log_info("未找到gpt_key.json，使用默认配置")
        except Exception as e:
            log_error(f"加载API密钥失败: {str(e)}")
    
    def call_ai(self, 
                model_type: Union[str, ModelType],
                prompt: str,
                content: Optional[str] = None,
                image_path: Optional[str] = None,
                **kwargs) -> str:
        """调用AI服务"""
        try:
            # 转换模型类型
            if isinstance(model_type, str):
                model_type = ModelType[model_type.upper().replace(' ', '_')]
            
            # 构建消息
            messages = self._build_messages(prompt, content, image_path)
            
            # 调用相应的服务
            if model_type == ModelType.GPT:
                return self._call_gpt(messages, **kwargs)
            elif model_type == ModelType.LM_STUDIO:
                return self._call_lm_studio(messages, **kwargs)
            elif model_type == ModelType.CLAUDE:
                return self._call_claude(messages, **kwargs)
            elif model_type == ModelType.LOCAL:
                return self._call_local(messages, **kwargs)
            else:
                raise AIServiceError(str(model_type), "不支持的模型类型")
                
        except Exception as e:
            error_msg = f"AI调用失败: {str(e)}"
            log_error(error_msg)
            raise AIServiceError(str(model_type), error_msg)
    
    def _build_messages(self, 
                       prompt: str, 
                       content: Optional[str] = None, 
                       image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """构建消息列表"""
        if image_path:
            # 多模态消息（文本+图像）
            try:
                base64_image = self._image_to_base64(image_path)
                return [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }]
            except Exception as e:
                log_error(f"图像编码失败: {str(e)}")
                # 回退到纯文本模式
                return [{"role": "user", "content": f"{prompt}\n[图像加载失败]"}]
        
        # 纯文本消息
        messages = [{"role": "user", "content": prompt}]
        if content:
            messages.append({"role": "user", "content": content})
        
        return messages
    
    def _image_to_base64(self, file_path: str) -> str:
        """将图像转换为base64编码"""
        try:
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                return encoded_string.decode("utf-8")
        except Exception as e:
            raise Exception(f"图像编码失败: {str(e)}")
    
    def _call_gpt(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """调用GPT服务"""
        try:
            import openai
            
            # 设置API配置
            openai.api_key = self.gpt_config['api_key']
            openai.base_url = self.gpt_config['base_url']
            
            response = openai.chat.completions.create(
                model=kwargs.get('model', self.gpt_config['model']),
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise AIServiceError("GPT", f"GPT调用失败: {str(e)}")
    
    def _call_lm_studio(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """调用LM Studio本地服务"""
        try:
            url = f"{self.lm_studio_config['base_url']}/chat/completions"
            
            payload = {
                "model": kwargs.get('model', self.lm_studio_config['model']),
                "messages": messages,
                "max_tokens": kwargs.get('max_tokens', 2000),
                "temperature": kwargs.get('temperature', 0.7),
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.RequestException as e:
            raise AIServiceError("LM_Studio", f"LM Studio连接失败: {str(e)}")
        except Exception as e:
            raise AIServiceError("LM_Studio", f"LM Studio调用失败: {str(e)}")
    
    def _call_claude(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """调用Claude服务"""
        # TODO: 实现Claude API调用
        raise AIServiceError("Claude", "Claude服务暂未实现")
    
    def _call_local(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """调用本地模型服务"""
        # TODO: 实现本地模型调用
        raise AIServiceError("Local", "本地模型服务暂未实现")
    
    def test_connection(self, model_type: Union[str, ModelType]) -> bool:
        """测试AI服务连接"""
        try:
            if isinstance(model_type, str):
                model_type = ModelType[model_type.upper().replace(' ', '_')]
            
            test_messages = [{"role": "user", "content": "Hello"}]
            
            if model_type == ModelType.GPT:
                self._call_gpt(test_messages, max_tokens=10)
            elif model_type == ModelType.LM_STUDIO:
                self._call_lm_studio(test_messages, max_tokens=10)
            
            return True
        except Exception as e:
            log_error(f"AI服务连接测试失败 ({model_type}): {str(e)}")
            return False