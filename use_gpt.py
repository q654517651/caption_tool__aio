from unittest import case
import base64

from openai import AzureOpenAI
import json
with open('gpt_key.json', 'r') as f:
    gpt_key = json.load(f)

OPENAI_API_KEY = gpt_key.OPENAI_API_KEY
AZURE_ENDPOINT = gpt_key.AZURE_ENDPOINT

openapi_client = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version="2024-08-01-preview",
    azure_endpoint=AZURE_ENDPOINT
)


def get_completion(control, prompt, content=None):
    model = 'Design-4o-mini'
    match control:
        case 'gpt':
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in stable diffusion prompt crafting. "
                },
                {
                    "role": "user",
                    "content": "请将当前提示词翻译成英文，保持原来的意思不变，直接输出优化后的提示词并翻译成英文，逗号连接，不要任何你的描述："
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        case 'image_generate':
            img_type = 'image/png'
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": '我正在训练一个AI图像模型，我需要你帮我使用自然语言对这个ip形象进行精炼的描述，语句之间用逗号连接不要换行，不要附带任何你个人的描述，并用中文发给我：'},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{content}"},
                        },
                    ]
                }
            ]
        case 'customize':
            # encoded_string = base64.b64encode(content).decode('utf-8')
            img_type = 'image/png'
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{content}"},
                        },
                    ]
                }
            ]
        case 'optimize':
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的翻译员，擅长将中文翻译成英文。"
                },
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        case _:
            messages = []
    try:
        response = openapi_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return None


# if __name__ == "__main__":
#
#     # a = get_completion()
#     # print(a)
