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
    task_id: str
    name: str
    training_type: TrainingType
    dataset_id: str
    
    # 训练参数
    epochs: int = 16
    batch_size: int = 1
    learning_rate: float = 1e-4
    resolution: str = "1024,1024"
    network_dim: int = 32
    network_alpha: int = 16
    
    # 数据集参数
    repeats: int = 1
    enable_bucket: bool = False
    min_bucket_reso: int = 512
    max_bucket_reso: int = 2048
    bucket_reso_steps: int = 64
    
    # 优化器和调度器
    optimizer: str = "adamw8bit"
    scheduler: str = "cosine"
    warmup_steps: int = 0
    
    # 采样配置
    sample_prompt: str = ""
    sample_every_n_steps: int = 200
    sample_resolution: str = "1024,1024"
    
    # 保存配置
    save_every_n_epochs: int = 1
    max_train_steps: Optional[int] = None
    
    # GPU配置
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # 特定配置
    qwen_config: QwenImageConfig = field(default_factory=QwenImageConfig)
    
    # 高级选项
    max_data_loader_n_workers: int = 2
    persistent_data_loader_workers: bool = True
    seed: int = 42
    
    def __post_init__(self):
        if isinstance(self.training_type, str):
            self.training_type = TrainingType(self.training_type)

@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
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
    created_time: Optional[str] = None
    started_time: Optional[str] = None
    completed_time: Optional[str] = None
    error_message: str = ""
    
    # 输出路径
    output_dir: str = ""
    checkpoint_files: List[str] = field(default_factory=list)
    sample_images: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.state, str):
            self.state = TrainingState(self.state)
        if self.created_time is None:
            self.created_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'config': {
                'task_id': self.config.task_id,
                'name': self.config.name,
                'training_type': self.config.training_type.value,
                'dataset_id': self.config.dataset_id,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'resolution': self.config.resolution,
                'network_dim': self.config.network_dim,
                'network_alpha': self.config.network_alpha,
                'repeats': self.config.repeats,
                'sample_prompt': self.config.sample_prompt,
                'sample_every_n_steps': self.config.sample_every_n_steps,
                'gpu_ids': self.config.gpu_ids,
                'qwen_config': {
                    'dit_path': self.config.qwen_config.dit_path,
                    'vae_path': self.config.qwen_config.vae_path,
                    'text_encoder_path': self.config.qwen_config.text_encoder_path,
                    'mixed_precision': self.config.qwen_config.mixed_precision,
                    'gradient_checkpointing': self.config.qwen_config.gradient_checkpointing,
                    'fp8_base': self.config.qwen_config.fp8_base,
                    'fp8_scaled': self.config.qwen_config.fp8_scaled,
                    'blocks_to_swap': self.config.qwen_config.blocks_to_swap,
                }
            },
            'state': self.state.value,
            'progress': self.progress,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'current_epoch': self.current_epoch,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'eta_seconds': self.eta_seconds,
            'speed': self.speed,
            'created_time': self.created_time,
            'started_time': self.started_time,
            'completed_time': self.completed_time,
            'error_message': self.error_message,
            'output_dir': self.output_dir,
            'checkpoint_files': self.checkpoint_files,
            'sample_images': self.sample_images,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingTask':
        """从字典创建"""
        config_data = data['config']
        qwen_data = config_data.get('qwen_config', {})
        
        config = TrainingConfig(
            task_id=config_data['task_id'],
            name=config_data['name'],
            training_type=TrainingType(config_data['training_type']),
            dataset_id=config_data['dataset_id'],
            epochs=config_data.get('epochs', 16),
            batch_size=config_data.get('batch_size', 1),
            learning_rate=config_data.get('learning_rate', 1e-4),
            resolution=config_data.get('resolution', '1024,1024'),
            network_dim=config_data.get('network_dim', 32),
            network_alpha=config_data.get('network_alpha', 16),
            repeats=config_data.get('repeats', 1),
            sample_prompt=config_data.get('sample_prompt', ''),
            sample_every_n_steps=config_data.get('sample_every_n_steps', 200),
            gpu_ids=config_data.get('gpu_ids', [0]),
            qwen_config=QwenImageConfig(
                dit_path=qwen_data.get('dit_path', ''),
                vae_path=qwen_data.get('vae_path', ''),
                text_encoder_path=qwen_data.get('text_encoder_path', ''),
                mixed_precision=qwen_data.get('mixed_precision', 'bf16'),
                gradient_checkpointing=qwen_data.get('gradient_checkpointing', True),
                fp8_base=qwen_data.get('fp8_base', False),
                fp8_scaled=qwen_data.get('fp8_scaled', False),
                blocks_to_swap=qwen_data.get('blocks_to_swap', 0),
            )
        )
        
        return cls(
            task_id=data['task_id'],
            config=config,
            state=TrainingState(data.get('state', 'pending')),
            progress=data.get('progress', 0.0),
            current_step=data.get('current_step', 0),
            total_steps=data.get('total_steps', 0),
            current_epoch=data.get('current_epoch', 0),
            loss=data.get('loss', 0.0),
            learning_rate=data.get('learning_rate', 0.0),
            eta_seconds=data.get('eta_seconds'),
            speed=data.get('speed'),
            created_time=data.get('created_time'),
            started_time=data.get('started_time'),
            completed_time=data.get('completed_time'),
            error_message=data.get('error_message', ''),
            output_dir=data.get('output_dir', ''),
            checkpoint_files=data.get('checkpoint_files', []),
            sample_images=data.get('sample_images', []),
        )