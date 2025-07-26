import base64
import requests
from openai import AzureOpenAI
import json
from enum import Enum
from typing import Optional, List, Dict, Any, Union


class ModelType(Enum):
    GPT = "GPT"
    LLM_Studio = "LLM_Studio"


class AIChatTool:
    def __init__(self,
                 config_path: str = 'gpt_key.json',
                 llm_studio_url: str = "http://localhost:1234/v1"
                 ):

        self.llm_studio_url = llm_studio_url
        self.openapi_client = self._init_azure_client(config_path)

    @staticmethod
    def _init_azure_client(config_path: str) -> AzureOpenAI:
        """初始化Azure OpenAI客户端"""
        try:
            with open(config_path, 'r') as f:
                gpt_key = json.load(f)

            return AzureOpenAI(
                api_key=gpt_key["OPENAI_API_KEY"],
                api_version="2024-08-01-preview",
                azure_endpoint=gpt_key["AZURE_ENDPOINT"]
            )
        except Exception as e:
            raise Exception(f"初始化Azure客户端失败: {e}")

    @staticmethod
    def image_to_base64(file_path):
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

    def build_messages(self,
                       prompt: str,
                       content: Optional[str] = None,
                       image_path: Optional[str] = None
                       ):
        if image_path:
            # 文本 + 图
            b64 = self.image_to_base64(image_path)
            return [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
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
                    image_path: Optional[str] = None,
                    ) -> str:
        try:
            # 转换为枚举类型
            if isinstance(model_type, str):
                model_type = ModelType[model_type]

            messages = self.build_messages(prompt, content, image_path)

            if model_type == ModelType.GPT:
                return self._call_gpt(messages)
            elif model_type == ModelType.LLM_Studio:
                return self._call_local_llm(messages)
            else:
                raise Exception(f"<UNK>{model_type}<UNK>")

        except Exception as e:
            return f"模型调用失败: {str(e)}"

    def _call_gpt(self, messages: List[Dict[str, Any]], model: str = 'Design-4o-mini') -> str:
        try:
            response = self.openapi_client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"GPT调用失败: {e}")

    def _call_local_llm(self, messages: List[Dict[str, Any]]) -> str:
        """调用本地LLM Studio"""
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


if __name__ == "__main__":
    ai = AIChatTool()
    text = ai.call_chatai(model_type="LLM_Studio", prompt="你好")
    print(text)
