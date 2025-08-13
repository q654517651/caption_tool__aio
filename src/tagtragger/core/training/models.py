"""
Training models and configurations
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime

class TrainingState(Enum):
    """训练状态"""
    PENDING = "pending"
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrainingType(Enum):
    """训练类型"""
    QWEN_IMAGE_LORA = "qwen_image_lora"
    FLUX_LORA = "flux_lora"
    SD_LORA = "sd_lora"

@dataclass
class QwenImageConfig:
    """Qwen-Image训练特定配置"""
    dit_path: str = ""
    vae_path: str = ""
    text_encoder_path: str = ""
    mixed_precision: str = "bf16"
    timestep_sampling: str = "shift"
    weighting_scheme: str = "none"
    discrete_flow_shift: float = 3.0
    optimizer_type: str = "adamw8bit"
    gradient_checkpointing: bool = True
    fp8_base: bool = False
    fp8_scaled: bool = False
    fp8_vl: bool = False
    blocks_to_swap: int = 0
    split_attn: bool = False
    attention_type: str = "sdpa"  # sdpa, xformers, flash_attn

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    name: str
    training_type: TrainingType  # 保持与UI一致的字段名
    dataset_id: str
    task_id: str = ""  # 向后兼容UI传递的task_id
    
    # 训练参数
    epochs: int = 16
    batch_size: int = 1
    learning_rate: float = 1e-4
    resolution: str = "1024,1024"
    network_dim: int = 32
    network_alpha: int = 16
    
    # 数据集参数
    repeats: int = 1
    dataset_size: int = 0
    enable_bucket: bool = True  # 添加缺失的enable_bucket字段
    
    # 优化器和调度器
    optimizer: str = "adamw8bit"
    scheduler: str = "cosine"
    
    # 采样配置
    sample_prompt: str = ""
    sample_every_n_steps: int = 200
    
    # 保存配置
    save_every_n_epochs: int = 1  # 添加缺失的save_every_n_epochs字段
    
    # GPU配置
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # 数据加载器配置（UI传递的字段）
    max_data_loader_n_workers: int = 2
    persistent_data_loader_workers: bool = True
    
    # 模型特定配置
    qwen_config: QwenImageConfig = field(default_factory=QwenImageConfig)
    flux_config: Optional['FluxConfig'] = None
    sd_config: Optional['StableDiffusionConfig'] = None
    
    # 高级选项
    seed: int = 42
    
    @property
    def type(self) -> TrainingType:
        """向后兼容的type属性"""
        return self.training_type
    
    def __post_init__(self):
        if isinstance(self.training_type, str):
            self.training_type = TrainingType(self.training_type)

@dataclass  
class TrainingTask:
    """训练任务"""
    id: str  # 使用id而不是task_id，保持与TrainingManager一致
    name: str
    config: TrainingConfig
    state: TrainingState = TrainingState.PENDING
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    eta_seconds: Optional[int] = None
    speed: Optional[float] = None  # it/s
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    error_message: str = ""
    output_dir: str = ""
    checkpoint_files: List[str] = field(default_factory=list)
    sample_images: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.state, str):
            self.state = TrainingState(self.state)
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def created_time(self) -> str:
        """向后兼容的created_time属性"""
        if self.created_at:
            return self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        return ""
    
    @property
    def task_id(self) -> str:
        """向后兼容的task_id属性"""
        return self.id


@dataclass
class FluxConfig:
    """Flux模型特定配置"""
    dit_path: str = ""
    vae_path: str = ""
    text_encoder_path: str = ""
    clip_path: str = ""
    mixed_precision: str = "bf16"
    guidance_scale: float = 3.5
    
@dataclass
class StableDiffusionConfig:
    """Stable Diffusion模型特定配置"""
    unet_path: str = ""
    vae_path: str = ""
    text_encoder_path: str = ""
    clip_path: str = ""
    mixed_precision: str = "fp16"


# 训练预设配置模板
TRAINING_PRESETS = {
    TrainingType.QWEN_IMAGE_LORA: {
        "script_path": "src/musubi_tuner/qwen_image_train_network.py",
        "cache_scripts": [
            "src/musubi_tuner/qwen_image_cache_latents.py",
            "src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py"
        ],
        "required_models": ["dit", "vae", "text_encoder"],
        "network_module": "musubi_tuner.networks.lora_qwen_image",
        "default_args": {
            "--mixed_precision": "bf16",
            "--timestep_sampling": "shift", 
            "--weighting_scheme": "none",
            "--discrete_flow_shift": "3.0",
            "--sdpa": None,  # flag参数
            "--network_dim": "32",
            "--network_alpha": "16"
        }
    },
    TrainingType.FLUX_LORA: {
        "script_path": "src/musubi_tuner/flux_train_network.py",
        "required_models": ["dit", "vae", "text_encoder", "clip"],
        "network_module": "networks.lora",
        "default_args": {
            "--mixed_precision": "bf16",
            "--network_dim": "32",
            "--network_alpha": "16"
        }
    },
    TrainingType.SD_LORA: {
        "script_path": "src/musubi_tuner/sd_train_network.py",
        "required_models": ["unet", "vae", "text_encoder"],
        "network_module": "networks.lora",
        "default_args": {
            "--mixed_precision": "fp16",
            "--network_dim": "32",
            "--network_alpha": "16"
        }
    }
}
