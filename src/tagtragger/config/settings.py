"""
Application configuration management
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class QwenImagePaths:
    """Qwen-Image模型路径配置"""
    dit_path: str = ""
    vae_path: str = ""
    text_encoder_path: str = ""

@dataclass  
class FluxPaths:
    """Flux模型路径配置"""
    dit_path: str = ""
    vae_path: str = ""
    text_encoder_path: str = ""
    clip_path: str = ""

@dataclass
class StableDiffusionPaths:
    """Stable Diffusion模型路径配置"""
    unet_path: str = ""
    vae_path: str = ""
    text_encoder_path: str = ""
    clip_path: str = ""

@dataclass
class ModelPaths:
    """模型路径配置 - 按训练类型分组"""
    qwen_image: QwenImagePaths = None
    flux: FluxPaths = None
    stable_diffusion: StableDiffusionPaths = None
    
    def __post_init__(self):
        if self.qwen_image is None:
            self.qwen_image = QwenImagePaths()
        if self.flux is None:
            self.flux = FluxPaths()
        if self.stable_diffusion is None:
            self.stable_diffusion = StableDiffusionPaths()
    
@dataclass
class LabelingConfig:
    """打标配置"""
    default_prompt: str = ""
    translation_prompt: str = ""
    model_type: str = "LM_STUDIO"
    delay_between_calls: float = 2.0
    
    def __post_init__(self):
        if not self.default_prompt:
            from .constants import DEFAULT_LABELING_PROMPT
            self.default_prompt = DEFAULT_LABELING_PROMPT
        if not self.translation_prompt:
            from .constants import DEFAULT_TRANSLATION_PROMPT
            self.translation_prompt = DEFAULT_TRANSLATION_PROMPT
    
@dataclass
class TrainingConfig:
    """训练配置"""
    default_epochs: int = 16
    default_batch_size: int = 2
    default_learning_rate: float = 1e-4
    default_resolution: str = "1024,1024"
    memory_presets: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.memory_presets is None:
            self.memory_presets = {
                "low": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 45},
                "medium": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 16},
                "high": {"fp8_base": False, "fp8_scaled": False, "blocks_to_swap": 0}
            }

@dataclass
class StorageConfig:
    """存储配置"""
    workspace_root: str = "./workspace"
    datasets_dir: str = "datasets"
    cache_dir: str = "cache"
    models_dir: str = "models"
    medium_max_side: int = 1280
    preview_max_side: int = 512
    
@dataclass
class UIConfig:
    """界面配置"""
    theme_mode: str = "light"
    window_width: int = 1400
    window_height: int = 900
    cards_per_row: int = 4
    
@dataclass
class AppConfig:
    """应用主配置"""
    model_paths: ModelPaths
    labeling: LabelingConfig
    training: TrainingConfig
    storage: StorageConfig
    ui: UIConfig
    
    def __init__(self):
        self.model_paths = ModelPaths()
        self.labeling = LabelingConfig()
        self.training = TrainingConfig()
        self.storage = StorageConfig()
        self.ui = UIConfig()

# 全局配置实例
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """获取全局配置"""
    global _config
    if _config is None:
        _config = load_config()
    return _config

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """加载配置文件"""
    if config_path is None:
        config_path = get_config_path()
    
    config = AppConfig()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 更新配置
            if 'model_paths' in data:
                mp_data = data['model_paths']
                config.model_paths = ModelPaths()
                
                # 处理嵌套的模型路径配置
                if 'qwen_image' in mp_data:
                    config.model_paths.qwen_image = QwenImagePaths(**mp_data['qwen_image'])
                if 'flux' in mp_data:
                    config.model_paths.flux = FluxPaths(**mp_data['flux'])
                if 'stable_diffusion' in mp_data:
                    config.model_paths.stable_diffusion = StableDiffusionPaths(**mp_data['stable_diffusion'])
                
                # 向后兼容：处理旧版本的配置格式
                if 'dit_path' in mp_data:
                    config.model_paths.qwen_image.dit_path = mp_data['dit_path']
                if 'vae_path' in mp_data:
                    config.model_paths.qwen_image.vae_path = mp_data['vae_path']
                if 'text_encoder_path' in mp_data:
                    config.model_paths.qwen_image.text_encoder_path = mp_data['text_encoder_path']
            if 'labeling' in data:
                config.labeling = LabelingConfig(**data['labeling'])
            if 'training' in data:
                config.training = TrainingConfig(**data['training'])
            if 'storage' in data:
                config.storage = StorageConfig(**data['storage'])
            if 'ui' in data:
                config.ui = UIConfig(**data['ui'])
                
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration")
    
    return config

def save_config(config: Optional[AppConfig] = None, config_path: Optional[str] = None):
    """保存配置文件"""
    if config is None:
        config = get_config()
    
    if config_path is None:
        config_path = get_config_path()
    
    # 确保配置目录存在
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # 转换为字典
    config_data = {
        'model_paths': {
            'qwen_image': asdict(config.model_paths.qwen_image),
            'flux': asdict(config.model_paths.flux),
            'stable_diffusion': asdict(config.model_paths.stable_diffusion)
        },
        'labeling': asdict(config.labeling),
        'training': asdict(config.training),
        'storage': asdict(config.storage),
        'ui': asdict(config.ui)
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error: Failed to save config to {config_path}: {e}")

def get_config_path() -> str:
    """获取配置文件路径"""
    return os.path.join(get_app_data_dir(), "config.json")

def get_app_data_dir() -> str:
    """获取应用数据目录"""
    # 使用程序根目录存储配置文件
    program_dir = Path(__file__).parent.parent.parent
    config_dir = program_dir / "config"
    config_dir.mkdir(exist_ok=True)
    return str(config_dir)

def update_config(**kwargs):
    """更新配置"""
    config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            if isinstance(getattr(config, key), dict):
                getattr(config, key).update(value)
            else:
                setattr(config, key, value)
    
    save_config(config)