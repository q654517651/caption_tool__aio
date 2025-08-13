"""
Application constants
"""

# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

# 支持的视频格式
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

# 数据集类型
DATASET_TYPES = {
    'image': '图像',
    'video': '视频', 
    'image_control': '图像+控制图像'
}

# 训练任务类型
TRAINING_TYPES = {
    'qwen_image_lora': 'Qwen-Image LoRA',
    'kontext_lora': 'Kontext LoRA',
    'wan22_lora': 'WAN2.2 LoRA'
}

# 训练状态
TRAINING_STATES = {
    'pending': '待开始',
    'running': '训练中',
    'completed': '已完成',
    'failed': '失败',
    'cancelled': '已取消'
}

# AI模型类型
AI_MODEL_TYPES = {
    'lm_studio': 'LM Studio',
    'gpt': 'OpenAI GPT',
    'claude': 'Anthropic Claude',
    'local': '本地模型'
}

# 默认提示词
DEFAULT_LABELING_PROMPT = """你是一名图像理解专家，请根据以下图片内容，生成自然流畅、具体清晰的图像描述。要求如下：
1. 使用简洁准确的中文句子，使用逗号进行连接；
2. 避免使用"图中"、"这是一张图片"等冗余措辞；
3. 语言风格自然、具象，不使用抽象形容词或主观感受；
4. 描述的内容不要重复
5. 将描述结构划分为以下模块，并标明模块标题；

【输出格式】
请按以下模块生成描述：
【主体与外貌】
【服饰与道具】
【动作与姿态】
【环境与场景】
【氛围与光效】
【镜头视角信息】

开始生成"""

DEFAULT_TRANSLATION_PROMPT = """【FLUX LoRA 图像打标专用翻译 Prompt】
将下方中文描述翻译为英文，严格遵守以下硬性规则：

1.准确传达原意，不得加入任何主观润色或感情色彩修饰。
2.全句仅使用主动语态，每句动作锚点必须前置，静态锚点必须具备可视化实体描述。
3.每个视觉锚点必须拆解成一句独立短句，禁止在同一句出现多个动作、道具、服饰或背景信息。
4.句子顺序固定为：主体外貌 → 动作姿态 → 服饰道具 → 场景背景 → 光效氛围，严禁顺序颠倒。
5.句子之间仅使用英文逗号, 连接，不允许句子内部使用逗号。
6.禁止使用"and / but / or"等连词，禁止使用被动语态，禁止任何修饰性从句。
7.输出格式为：一整行英文逗号串，最后以英文句号. 结尾。
8.仅输出英文翻译，不要输出任何标签、换行或解释说明。

开始翻译："""