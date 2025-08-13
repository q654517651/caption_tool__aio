"""
Musubi-Tuner 训练器集成
支持 HunyuanVideo 和其他 Musubi 模型的 LoRA 训练
"""

import os
import sys
import time
import subprocess
import signal
import uuid
import tempfile
import toml
import json
import platform
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from dataclasses import asdict

from ...common.events import EventBus, Job
from ....utils.logger import log_info, log_error, log_success
from ....utils.exceptions import TrainingError
from ....config import get_config
from .types import TrainingConfig, TrainingBackend


class MusubiTrainer:
    """Musubi-Tuner 训练器"""
    
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.config = get_config()
        self._proc: Optional[subprocess.Popen] = None
        self._id = uuid.uuid4().hex
        
    def _get_musubi_script_path(self, backend: TrainingBackend) -> str:
        """获取 Musubi 训练脚本路径"""
        # 使用内置的 Musubi-Tuner 子模块
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        musubi_dir = project_root / "third_party" / "musubi-tuner"
        
        script_mapping = {
            TrainingBackend.MUSUBI_HUNYUAN_VIDEO: "src/musubi_tuner/hv_train_network.py",
            TrainingBackend.MUSUBI_QWEN_IMAGE: "src/musubi_tuner/qwen_image_train_network.py",
        }
        
        script_name = script_mapping.get(backend)
        if not script_name:
            raise TrainingError(f"不支持的训练后端: {backend}")
            
        script_path = musubi_dir / script_name
        if not script_path.exists():
            raise TrainingError(f"Musubi 训练脚本不存在: {script_path}\n请确保已正确初始化 git 子模块")
            
        return str(script_path)
    
    def _create_training_workspace(self, cfg: TrainingConfig) -> Path:
        """创建训练工作空间目录"""
        workspace_root = Path(self.config.storage.workspace_root)
        training_dir = workspace_root / "trainings" / cfg.task_id
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (training_dir / "logs").mkdir(exist_ok=True)
        
        return training_dir
    
    def _create_dataset_config(self, cfg: TrainingConfig) -> str:
        """创建 Musubi 数据集配置文件"""
        # 获取数据集路径
        dataset_path = Path(self.config.storage.workspace_root) / "datasets" / cfg.dataset_id / "original"
        
        if not dataset_path.exists():
            raise TrainingError(f"数据集路径不存在: {dataset_path}")
        
        # 创建训练工作空间
        training_dir = self._create_training_workspace(cfg)
        
        # 缓存目录 - 每个训练任务独立的缓存
        cache_dir = training_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        # 创建 TOML 配置
        dataset_config = {
            "general": {
                "resolution": [int(x) for x in cfg.resolution.split(",")],
                "batch_size": cfg.batch_size,
                "enable_bucket": True,
                "bucket_no_upscale": False,
                "bucket_resolution_steps": 64,
                "min_bucket_reso": 256,
                "max_bucket_reso": 1536,
            },
            "datasets": [
                {
                    "image_directory": str(dataset_path),
                    "cache_directory": str(cache_dir), 
                    "caption_extension": ".txt",
                    "num_repeats": cfg.repeats,
                }
            ]
        }
        
        # 保存到训练目录（持久化）
        toml_path = training_dir / "datasets.toml"
        with open(toml_path, 'w', encoding='utf-8') as f:
            toml.dump(dataset_config, f)
        
        log_info(f"数据集配置已保存: {toml_path}")
        return str(toml_path)
    
    def _save_training_config(self, cfg: TrainingConfig, training_dir: Path) -> None:
        """保存训练配置到JSON文件"""
        config_dict = asdict(cfg)
        config_dict['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        config_dict['platform'] = platform.system()
        
        config_path = training_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        log_info(f"训练配置已保存: {config_path}")
    
    def _create_training_scripts(self, cfg: TrainingConfig, dataset_config_path: str, training_dir: Path) -> tuple[str, str]:
        """创建训练脚本（bat和sh）"""
        script_path = self._get_musubi_script_path(cfg.backend)
        
        # 输出目录
        output_dir = Path(self.config.storage.workspace_root) / "models" / cfg.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        log_file = training_dir / "logs" / "training.log"
        
        # 基础命令参数
        base_args = [
            f'--dataset_config "{dataset_config_path}"',
            f'--output_dir "{output_dir}"',
            f'--output_name "{cfg.name}"',
            '--network_module networks.lora',
            '--network_dim 32',
            '--network_alpha 16',
            f'--max_train_epochs {cfg.epochs}',
            f'--learning_rate {cfg.lr}',
            f'--optimizer_type {cfg.optimizer}',
            f'--lr_scheduler {cfg.scheduler}',
            f'--mixed_precision {cfg.precision}',
            '--save_every_n_epochs 1',
            '--seed 42',
        ]
        
        # 模型路径配置
        if hasattr(cfg, 'base_model') and cfg.base_model:
            if cfg.backend == TrainingBackend.MUSUBI_HUNYUAN_VIDEO:
                base_args.append(f'--dit "{cfg.base_model}"')
            else:
                base_args.append(f'--pretrained_model_name_or_path "{cfg.base_model}"')
        
        # 模型组件路径
        if self.config.model_paths.vae_path:
            base_args.append(f'--vae "{self.config.model_paths.vae_path}"')
        if self.config.model_paths.clip_path:
            base_args.append(f'--clip_model "{self.config.model_paths.clip_path}"')
        if self.config.model_paths.t5_path:
            base_args.append(f'--t5_model "{self.config.model_paths.t5_path}"')
        
        # 采样配置
        if cfg.sample_prompt:
            base_args.append(f'--sample_prompts "{cfg.sample_prompt}"')
            base_args.append(f'--sample_every_n_steps {cfg.sample_every_n_steps or 200}')
        
        # GPU配置
        if cfg.gpu_index >= 0:
            base_args.append(f'--gpu_ids {cfg.gpu_index}')
        
        # 高级参数
        if cfg.grad_accum > 1:
            base_args.append(f'--gradient_accumulation_steps {cfg.grad_accum}')
        if cfg.weight_decay > 0:
            base_args.append(f'--weight_decay {cfg.weight_decay}')
        
        # 创建 Windows 批处理脚本
        bat_content = f'''@echo off
echo ===== TagTracker Musubi 训练脚本 =====
echo 任务名称: {cfg.name}
echo 开始时间: %date% %time%
echo =====================================

cd /d "{Path(__file__).parent.parent.parent.parent.parent.parent}"

accelerate launch "{script_path}" {' '.join(base_args)} 2>&1 | tee "{log_file}"

echo =====================================
echo 训练完成时间: %date% %time%
echo =====================================
pause
'''
        
        # 创建 Linux/Mac shell 脚本
        sh_content = f'''#!/bin/bash
echo "===== TagTracker Musubi 训练脚本 ====="
echo "任务名称: {cfg.name}"
echo "开始时间: $(date)"
echo "====================================="

cd "{Path(__file__).parent.parent.parent.parent.parent.parent}"

accelerate launch "{script_path}" {' '.join(base_args)} 2>&1 | tee "{log_file}"

echo "====================================="
echo "训练完成时间: $(date)"
echo "====================================="
'''
        
        # 保存脚本文件
        bat_path = training_dir / "train.bat"
        sh_path = training_dir / "train.sh"
        
        with open(bat_path, 'w', encoding='gbk') as f:  # Windows 用 GBK 编码
            f.write(bat_content)
        
        with open(sh_path, 'w', encoding='utf-8') as f:
            f.write(sh_content)
        
        # Linux/Mac 脚本设置执行权限
        if platform.system() != 'Windows':
            os.chmod(sh_path, 0o755)
        
        log_info(f"训练脚本已生成: {bat_path}, {sh_path}")
        return str(bat_path), str(sh_path)
    
    def _prepare_training_files(self, cfg: TrainingConfig) -> dict:
        """准备训练文件（数据集配置、训练脚本、配置记录）"""
        try:
            # 创建数据集配置文件
            dataset_config_path = self._create_dataset_config(cfg)
            
            # 获取训练目录
            training_dir = self._create_training_workspace(cfg)
            
            # 保存训练配置记录
            self._save_training_config(cfg, training_dir)
            
            # 生成训练脚本
            bat_path, sh_path = self._create_training_scripts(cfg, dataset_config_path, training_dir)
            
            return {
                'training_dir': training_dir,
                'dataset_config': dataset_config_path,
                'bat_script': bat_path,
                'sh_script': sh_path,
                'log_file': training_dir / "logs" / "training.log"
            }
            
        except Exception as e:
            log_error(f"准备训练文件失败: {e}")
            raise TrainingError(f"准备训练文件失败: {e}")
    
    def _compose_cmd(self, cfg: TrainingConfig, total_steps: int) -> list[str]:
        """组装 Musubi 训练命令（兼容旧版本）"""
        try:
            # 准备训练文件
            files = self._prepare_training_files(cfg)
            
            # 根据平台选择执行方式
            if platform.system() == 'Windows':
                # Windows: 直接执行bat文件
                cmd = [files['bat_script']]
            else:
                # Linux/Mac: 执行sh文件
                cmd = ["bash", files['sh_script']]
            
            log_info(f"Musubi 训练命令: {' '.join(cmd)}")
            log_info(f"训练文件已保存到: {files['training_dir']}")
            return cmd
            
        except Exception as e:
            log_error(f"组装训练命令失败: {e}")
            raise TrainingError(f"组装训练命令失败: {e}")
    
    def build_job(self, cfg: TrainingConfig, total_steps: int) -> Job:
        """构建训练任务"""
        return Job(
            id=self._id,
            name=cfg.name,
            run=lambda log_cb, progress_cb: self.run(log_cb, progress_cb, cfg, total_steps),
            cancel=self.cancel
        )
    
    def run(self, log_callback: Callable[[str], None], 
            progress_callback: Callable[[float], None],
            cfg: TrainingConfig, total_steps: int) -> bool:
        """执行训练"""
        try:
            cmd = self._compose_cmd(cfg, total_steps)
            if not cmd:
                log_error("无法组装训练命令")
                return False
            
            log_info(f"开始 Musubi 训练: {cfg.name}")
            log_callback(f"开始训练 {cfg.name}")
            
            # 启动训练进程
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 监控训练进度
            current_step = 0
            while True:
                if self._proc.poll() is not None:
                    break
                
                line = self._proc.stdout.readline()
                if not line:
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                log_callback(line)
                
                # 解析进度信息
                progress = self._parse_progress(line, current_step, total_steps)
                if progress >= 0:
                    current_step = int(progress * total_steps)
                    progress_callback(progress)
            
            # 检查训练结果
            return_code = self._proc.poll()
            if return_code == 0:
                log_success(f"训练完成: {cfg.name}")
                progress_callback(1.0)
                return True
            else:
                log_error(f"训练失败，退出码: {return_code}")
                return False
                
        except Exception as e:
            log_error(f"训练执行失败: {e}")
            return False
        finally:
            self._proc = None
    
    def _parse_progress(self, line: str, current_step: int, total_steps: int) -> float:
        """解析训练进度"""
        # 查找步数信息，如: "step 100/1000" 或 "100/1000"
        import re
        
        # 匹配 "step X/Y" 或 "X/Y" 格式
        step_match = re.search(r'(?:step\s+)?(\d+)/(\d+)', line.lower())
        if step_match:
            current = int(step_match.group(1))
            total = int(step_match.group(2))
            return current / total if total > 0 else 0.0
        
        # 匹配 "epoch X" 格式
        epoch_match = re.search(r'epoch\s+(\d+)', line.lower())
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            # 估算进度 (不够精确，但可以提供反馈)
            return min(current_epoch / 10, 0.9)  # 最多显示90%，等步数信息
        
        return -1  # 无法解析
    
    def cancel(self):
        """取消训练"""
        if self._proc and self._proc.poll() is None:
            log_info("正在取消 Musubi 训练...")
            try:
                if os.name == 'nt':  # Windows
                    self._proc.terminate()
                else:  # Unix/Linux
                    self._proc.send_signal(signal.SIGTERM)
                
                # 等待进程结束
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    
                log_info("Musubi 训练已取消")
            except Exception as e:
                log_error(f"取消训练时出错: {e}")
            finally:
                self._proc = None
    
    def is_available(self) -> bool:
        """检查 Musubi-Tuner 是否可用"""
        try:
            # 检查 Musubi 目录是否存在
            musubi_dir = Path(self.config.model_paths.musubi_dir)
            if not musubi_dir.exists():
                return False
            
            # 检查训练脚本是否存在
            script_path = musubi_dir / "src/musubi_tuner/hv_train_network.py"
            if not script_path.exists():
                return False
            
            # 检查 accelerate 命令是否可用
            result = subprocess.run(
                ["accelerate", "--help"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
            
        except Exception:
            return False