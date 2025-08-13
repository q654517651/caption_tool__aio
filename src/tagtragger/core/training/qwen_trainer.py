"""
Qwen-Image LoRA Trainer
"""

import os
import sys
import shlex
import subprocess
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from .models import TrainingConfig, TrainingTask, TrainingState
from ...utils.logger import log_info, log_error, log_success, log_progress
from ...utils.exceptions import TrainingError, ValidationError
from ...config import get_config

class QwenImageTrainer:
    """Qwen-Image LoRA训练器"""
    
    def __init__(self):
        self.config = get_config()
        self.process: Optional[subprocess.Popen] = None
        self.is_cancelled = False
        
    def create_dataset_config(self, task: TrainingTask) -> str:
        """创建数据集配置文件（TOML格式）"""
        try:
            from ..dataset import DatasetManager
            
            dataset_manager = DatasetManager()
            dataset = dataset_manager.get_dataset(task.config.dataset_id)
            if not dataset:
                raise TrainingError(f"数据集不存在: {task.config.dataset_id}")
            
            # 数据集路径
            dataset_path = dataset_manager.get_dataset_path(task.config.dataset_id)
            original_path = dataset_path / "original"
            
            # 创建输出目录
            output_dir = Path(self.config.storage.workspace_root) / "models" / task.task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成TOML配置
            toml_content = self._generate_toml_config(
                dataset_path=str(original_path),
                resolution=task.config.resolution,
                batch_size=task.config.batch_size,
                repeats=task.config.repeats,
                enable_bucket=task.config.enable_bucket
            )
            
            # 保存配置文件
            config_file = output_dir / "dataset_config.toml"
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(toml_content)
            
            log_info(f"创建数据集配置文件: {config_file}")
            return str(config_file)
            
        except Exception as e:
            raise TrainingError(f"创建数据集配置失败: {str(e)}")
    
    def _generate_toml_config(self, dataset_path: str, resolution: str, batch_size: int, repeats: int, enable_bucket: bool) -> str:
        """生成TOML配置内容"""
        width, height = map(int, resolution.split(','))
        
        toml_content = f'''[general]
enable_bucket = {str(enable_bucket).lower()}

[[datasets]]
resolution = [{width}, {height}]
batch_size = {batch_size}
enable_bucket = {str(enable_bucket).lower()}

  [[datasets.subsets]]
  image_dir = "{dataset_path}"
  num_repeats = {repeats}
'''
        
        if enable_bucket:
            toml_content += f'''
  min_bucket_reso = 512
  max_bucket_reso = 2048
  bucket_reso_steps = 64
'''
        
        return toml_content
    
    def prepare_training(self, task: TrainingTask) -> Dict[str, str]:
        """准备训练环境和文件"""
        try:
            # 验证配置
            self._validate_config(task.config)
            
            # 创建输出目录
            output_dir = Path(self.config.storage.workspace_root) / "models" / task.task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            task.output_dir = str(output_dir)
            
            # 创建数据集配置
            dataset_config = self.create_dataset_config(task)
            
            # 准备训练参数
            training_args = {
                'dataset_config': dataset_config,
                'output_dir': str(output_dir),
                'output_name': f"{task.config.name}_lora"
            }
            
            return training_args
            
        except Exception as e:
            raise TrainingError(f"训练准备失败: {str(e)}")
    
    def _validate_config(self, config: TrainingConfig):
        """验证训练配置"""
        qwen_cfg = config.qwen_config
        
        # 验证必需的模型路径
        if not qwen_cfg.dit_path or not os.path.exists(qwen_cfg.dit_path):
            raise ValidationError("dit_path", qwen_cfg.dit_path, "DiT模型路径无效")
        
        if not qwen_cfg.vae_path or not os.path.exists(qwen_cfg.vae_path):
            raise ValidationError("vae_path", qwen_cfg.vae_path, "VAE模型路径无效")
        
        if not qwen_cfg.text_encoder_path or not os.path.exists(qwen_cfg.text_encoder_path):
            raise ValidationError("text_encoder_path", qwen_cfg.text_encoder_path, "文本编码器路径无效")
    
    def build_training_command(self, task: TrainingTask, training_args: Dict[str, str]) -> List[str]:
        """构建训练命令"""
        config = task.config
        qwen_cfg = config.qwen_config
        
        # 基础命令
        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", qwen_cfg.mixed_precision,
            "src/musubi_tuner/qwen_image_train_network.py"
        ]
        
        # 模型路径
        cmd.extend([
            "--dit", qwen_cfg.dit_path,
            "--vae", qwen_cfg.vae_path,
            "--text_encoder", qwen_cfg.text_encoder_path
        ])
        
        # 数据集配置
        cmd.extend([
            "--dataset_config", training_args['dataset_config']
        ])
        
        # 训练参数
        cmd.extend([
            "--mixed_precision", qwen_cfg.mixed_precision,
            "--timestep_sampling", qwen_cfg.timestep_sampling,
            "--weighting_scheme", qwen_cfg.weighting_scheme,
            "--discrete_flow_shift", str(qwen_cfg.discrete_flow_shift),
            "--optimizer_type", qwen_cfg.optimizer_type,
            "--learning_rate", str(config.learning_rate),
            "--max_data_loader_n_workers", str(config.max_data_loader_n_workers),
            "--network_dim", str(config.network_dim),
            "--max_train_epochs", str(config.epochs),
            "--save_every_n_epochs", str(config.save_every_n_epochs),
            "--seed", str(config.seed),
            "--output_dir", training_args['output_dir'],
            "--output_name", training_args['output_name']
        ])
        
        # 可选参数
        if qwen_cfg.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        
        if qwen_cfg.fp8_base:
            cmd.append("--fp8_base")
        
        if qwen_cfg.fp8_scaled:
            cmd.append("--fp8_scaled")
        
        if qwen_cfg.fp8_vl:
            cmd.append("--fp8_vl")
        
        if qwen_cfg.blocks_to_swap > 0:
            cmd.extend(["--blocks_to_swap", str(qwen_cfg.blocks_to_swap)])
        
        if qwen_cfg.split_attn:
            cmd.append("--split_attn")
        
        # 注意力机制
        if qwen_cfg.attention_type == "sdpa":
            cmd.append("--sdpa")
        elif qwen_cfg.attention_type == "xformers":
            cmd.append("--xformers")
        elif qwen_cfg.attention_type == "flash_attn":
            cmd.append("--flash_attn")
        
        # 采样参数
        if config.sample_prompt:
            cmd.extend(["--sample_prompt", config.sample_prompt])
            cmd.extend(["--sample_every_n_steps", str(config.sample_every_n_steps)])
        
        # 数据集持久化
        if config.persistent_data_loader_workers:
            cmd.append("--persistent_data_loader_workers")
        
        log_info(f"训练命令: {' '.join(cmd)}")
        return cmd
    
    def run_training(self, 
                    task: TrainingTask,
                    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                    log_callback: Optional[Callable[[str], None]] = None) -> bool:
        """运行训练"""
        try:
            # 准备训练
            task.state = TrainingState.PREPARING
            if progress_callback:
                progress_callback({"state": task.state.value})
            
            training_args = self.prepare_training(task)
            
            # 构建命令
            cmd = self.build_training_command(task, training_args)
            
            # 检查musubi-tuner环境
            musubi_dir = self.config.model_paths.musubi_dir
            if not musubi_dir or not os.path.exists(musubi_dir):
                raise TrainingError("Musubi-Tuner目录未配置或不存在")
            
            # 开始训练
            task.state = TrainingState.RUNNING
            if progress_callback:
                progress_callback({"state": task.state.value})
            
            log_info(f"开始Qwen-Image LoRA训练: {task.config.name}")
            
            # 启动训练进程
            self.process = subprocess.Popen(
                cmd,
                cwd=musubi_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时读取输出
            return self._monitor_training(task, progress_callback, log_callback)
            
        except Exception as e:
            task.state = TrainingState.FAILED
            task.error_message = str(e)
            log_error(f"训练失败: {str(e)}")
            if progress_callback:
                progress_callback({
                    "state": task.state.value,
                    "error": task.error_message
                })
            return False
    
    def _monitor_training(self, 
                         task: TrainingTask,
                         progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                         log_callback: Optional[Callable[[str], None]] = None) -> bool:
        """监控训练进度"""
        try:
            if not self.process:
                return False
            
            while True:
                if self.is_cancelled:
                    self.cancel_training()
                    task.state = TrainingState.CANCELLED
                    return False
                
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                
                if output:
                    line = output.strip()
                    if log_callback:
                        log_callback(line)
                    
                    # 解析训练进度
                    progress_info = self._parse_training_output(line)
                    if progress_info:
                        # 更新任务状态
                        task.current_step = progress_info.get('step', task.current_step)
                        task.current_epoch = progress_info.get('epoch', task.current_epoch)
                        task.total_steps = progress_info.get('total_steps', task.total_steps)
                        task.loss = progress_info.get('loss', task.loss)
                        task.learning_rate = progress_info.get('lr', task.learning_rate)
                        task.speed = progress_info.get('speed', task.speed)
                        task.eta_seconds = progress_info.get('eta_seconds', task.eta_seconds)
                        
                        # 计算进度
                        if task.total_steps > 0:
                            task.progress = task.current_step / task.total_steps
                        
                        # 回调更新
                        if progress_callback:
                            progress_callback({
                                "step": task.current_step,
                                "total_steps": task.total_steps,
                                "epoch": task.current_epoch,
                                "loss": task.loss,
                                "lr": task.learning_rate,
                                "speed": task.speed,
                                "eta_seconds": task.eta_seconds,
                                "progress": task.progress
                            })
            
            # 检查训练结果
            return_code = self.process.poll()
            if return_code == 0:
                task.state = TrainingState.COMPLETED
                log_success(f"训练完成: {task.config.name}")
                
                # 扫描输出文件
                self._scan_output_files(task)
                
                if progress_callback:
                    progress_callback({"state": task.state.value})
                return True
            else:
                task.state = TrainingState.FAILED
                task.error_message = f"训练进程退出，代码: {return_code}"
                log_error(task.error_message)
                if progress_callback:
                    progress_callback({
                        "state": task.state.value,
                        "error": task.error_message
                    })
                return False
                
        except Exception as e:
            task.state = TrainingState.FAILED
            task.error_message = str(e)
            log_error(f"训练监控失败: {str(e)}")
            if progress_callback:
                progress_callback({
                    "state": task.state.value,
                    "error": task.error_message
                })
            return False
    
    def _parse_training_output(self, line: str) -> Optional[Dict[str, Any]]:
        """解析训练输出"""
        try:
            progress_info = {}
            
            # 解析步数和轮次 (例: "Epoch 1/16, Step 100/1000")
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                progress_info['epoch'] = int(epoch_match.group(1))
                progress_info['total_epochs'] = int(epoch_match.group(2))
            
            step_match = re.search(r'Step (\d+)/(\d+)', line)
            if step_match:
                progress_info['step'] = int(step_match.group(1))
                progress_info['total_steps'] = int(step_match.group(2))
            
            # 解析loss (例: "loss: 0.1234")
            loss_match = re.search(r'loss:?\s*([\d.]+)', line, re.IGNORECASE)
            if loss_match:
                progress_info['loss'] = float(loss_match.group(1))
            
            # 解析学习率 (例: "lr: 1e-4")
            lr_match = re.search(r'lr:?\s*([\d.e-]+)', line, re.IGNORECASE)
            if lr_match:
                progress_info['lr'] = float(lr_match.group(1))
            
            # 解析速度 (例: "1.2 it/s")
            speed_match = re.search(r'([\d.]+)\s*it/s', line)
            if speed_match:
                progress_info['speed'] = float(speed_match.group(1))
            
            # 解析ETA (例: "ETA: 00:15:30")
            eta_match = re.search(r'ETA:?\s*(\d{2}):(\d{2}):(\d{2})', line)
            if eta_match:
                hours, minutes, seconds = map(int, eta_match.groups())
                progress_info['eta_seconds'] = hours * 3600 + minutes * 60 + seconds
            
            return progress_info if progress_info else None
            
        except Exception as e:
            log_error(f"解析训练输出失败: {str(e)}")
            return None
    
    def _scan_output_files(self, task: TrainingTask):
        """扫描输出文件"""
        try:
            if not task.output_dir or not os.path.exists(task.output_dir):
                return
            
            output_path = Path(task.output_dir)
            
            # 扫描checkpoint文件
            for file in output_path.glob("*.safetensors"):
                if file.is_file():
                    task.checkpoint_files.append(str(file))
            
            # 扫描采样图片
            for file in output_path.glob("*.png"):
                if file.is_file():
                    task.sample_images.append(str(file))
            
            log_info(f"找到 {len(task.checkpoint_files)} 个checkpoint文件，{len(task.sample_images)} 张采样图片")
            
        except Exception as e:
            log_error(f"扫描输出文件失败: {str(e)}")
    
    def cancel_training(self):
        """取消训练"""
        self.is_cancelled = True
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                log_info("训练进程已终止")
            except Exception as e:
                log_error(f"终止训练进程失败: {str(e)}")
    
    def get_memory_usage_estimate(self, config: TrainingConfig) -> Dict[str, Any]:
        """估算显存使用量"""
        qwen_cfg = config.qwen_config
        
        # 基础显存使用
        base_usage = 12000  # MB，基础模型加载
        
        # 根据配置调整
        if qwen_cfg.mixed_precision == "fp16":
            base_usage = int(base_usage * 0.8)
        elif qwen_cfg.mixed_precision == "bf16":
            base_usage = int(base_usage * 0.8)
        
        if qwen_cfg.fp8_base and qwen_cfg.fp8_scaled:
            base_usage = int(base_usage * 0.7)
        
        # 批次大小影响
        batch_multiplier = config.batch_size * 2000
        
        # blocks_to_swap减少显存使用
        if qwen_cfg.blocks_to_swap > 0:
            swap_reduction = min(qwen_cfg.blocks_to_swap * 400, base_usage * 0.6)
            base_usage -= int(swap_reduction)
        
        total_usage = base_usage + batch_multiplier
        
        return {
            "estimated_vram_mb": total_usage,
            "estimated_vram_gb": round(total_usage / 1024, 1),
            "config_preset": self._get_config_preset(total_usage)
        }
    
    def _get_config_preset(self, vram_mb: int) -> str:
        """根据显存使用量推荐配置预设"""
        if vram_mb <= 12000:
            return "high"  # 高端卡
        elif vram_mb <= 20000:
            return "medium"  # 中端卡
        else:
            return "low"  # 低端卡，需要更多优化