# trainers/types.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TrainingBackend(str, Enum):
    MUSUBI_QWEN_IMAGE = "musubi.qwen_image"


@dataclass
class TrainingConfig:
    backend: TrainingBackend
    name: str
    dataset_id: str
    dataset_size: int
    repeats: int = 1
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 1
    resolution: str = "1024,1024"
    base_model: str = ""
    clip_path: str = ""
    vae_path: str = ""
    optimizer: str = "adamw8bit"
    lr: float = 1e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    precision: str = "bf16"
    sample_prompt: str = ""
    sample_every_n_steps: Optional[int] = 200
    sample_resolution: str = "1024,1024"
    sample_num_images: int = 2
    gpu_index: int = 0


BACKEND_PRESETS: dict[TrainingBackend, dict] = {
    TrainingBackend.MUSUBI_QWEN_IMAGE: {
        # 学习率与调度器默认
        "optimizer": "adamw8bit",
        "lr": 1e-4,
        "scheduler": "cosine",
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "precision": "bf16",
        # 采样默认
        "sample_resolution": "1024,1024",
        "sample_num_images": 2,
        "sample_every_n_steps": 200,
        # 数据集默认
        "batch_size": 2,
        "grad_accum": 1,
        "resolution": "1024,1024",
    },
    # 未来新增其它后端：
    # TrainingBackend.FLUX_LORA: {...},
}
