import base64
import requests
from openai import AzureOpenAI
import json


class AIChatTool:
    def __init__(self):
        self.llm_studio_url = "http://localhost:1234/v1"

        with open('gpt_key.json', 'r') as f:
            gpt_key = json.load(f)

        # 使用实际的gpt key
        self.openapi_client = AzureOpenAI(
            api_key=gpt_key["OPENAI_API_KEY"],
            api_version="2024-08-01-preview",
            azure_endpoint=gpt_key["AZURE_ENDPOINT"]
        )

    @staticmethod
    def image_to_base64(file_path):
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

    def build_messages(self, prompt: str, content: str = None, image_path: str = None):
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
        elif content is not None:
            # 纯文本对话
            return [
                {"role": "user", "content": prompt},
                {"role": "user", "content": content},
            ]
        elif content is None:
            return [{"role": "user", "content": prompt}]
        else:
            raise ValueError("content 和 image_path 二选一")

    def call_chatai(self, model_type: str, prompt: str, content=None, image_path: str = None) -> str:
        try:
            if model_type == "GPT":
                return self._call_gpt(prompt=prompt, content=content, image_path=image_path)
            else:  # 本地 LLM
                return self._call_local_llm(prompt=prompt, content=content, image_path=image_path)
        except Exception as e:
            # log_error(f"模型调用失败: {str(e)}")
            return f"错误: {str(e)}"

    def _call_gpt(self, prompt: str, content=None, image_path=None) -> str:
        model = 'Design-4o-mini'
        messages = self.build_messages(prompt, content=content, image_path=image_path)
        print(messages)
        try:
            response = self.openapi_client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"发生错误: {str(e)}")
            return "发生错误"

    def _call_local_llm(self, prompt: str, content=None, image_path: str = None) -> str:
        """调用本地LLM Studio"""
        messages = self.build_messages(prompt, content=content, image_path=image_path)
        payload = {
            "model": "gpt-4-vision-preview",
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

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return f"本地LLM调用失败: {response.status_code}"

        except Exception as e:
            # log_error(f"本地LLM调用失败: {e}")
            return f"错误: {str(e)}"


if __name__ == "__main__":
    ai = AIChatTool()
    text = ai.call_chatai(model_type="GPT", prompt="你好")
    print(text)
