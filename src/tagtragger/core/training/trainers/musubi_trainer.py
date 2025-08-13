"""
统一的Musubi-Tuner训练器
支持所有模型类型的LoRA训练（Qwen-Image, Flux, Stable Diffusion等）
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
import re
import shutil
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from dataclasses import asdict

from ...common.events import EventBus, Job
from ....utils.logger import log_info, log_error, log_success, log_progress
from ....utils.exceptions import TrainingError
from ....config import get_config
from ..models import TrainingConfig, TrainingTask, TrainingType, TrainingState, TRAINING_PRESETS


class MusubiTrainer:
    """统一的Musubi-Tuner训练器"""
    
    def __init__(self, bus: EventBus = None):
        self.bus = bus
        self.config = get_config()
        self._proc: Optional[subprocess.Popen] = None
        self._id = uuid.uuid4().hex
        
    def get_musubi_path(self) -> Path:
        """获取内嵌的musubi-tuner路径"""
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        musubi_dir = project_root / "third_party" / "musubi-tuner"
        
        if not musubi_dir.exists():
            raise TrainingError("Musubi-Tuner子模块不存在，请运行: git submodule update --init --recursive")
        
        return musubi_dir
    
    def _get_accelerate_cmd(self) -> List[str]:
        """获取正确的accelerate命令"""
        # 首先尝试直接调用accelerate命令
        if shutil.which("accelerate"):
            return ["accelerate"]
        
        # 如果直接命令不可用，尝试通过Python模块调用
        # 但需要避免使用 -m accelerate.__main__ 
        try:
            import accelerate
            # 尝试找到accelerate脚本的位置
            accelerate_path = Path(accelerate.__file__).parent.parent / "Scripts" / "accelerate.exe"
            if accelerate_path.exists():
                return [str(accelerate_path)]
        except ImportError:
            pass
        
        # 最后尝试在当前Python环境的Scripts目录查找
        python_dir = Path(sys.executable).parent
        accelerate_exe = python_dir / "accelerate.exe"  # Windows
        if accelerate_exe.exists():
            return [str(accelerate_exe)]
        
        accelerate_script = python_dir / "accelerate"  # Linux/Mac
        if accelerate_script.exists():
            return [str(accelerate_script)]
        
        # 如果都找不到，回退到Python模块调用（可能会失败，但这是最后的选择）
        log_error("警告: 无法找到accelerate命令，使用Python模块调用，可能会失败")
        return [sys.executable, "-c", "import accelerate.commands.launch; accelerate.commands.launch.main()"]
        
    def _get_script_path(self, training_type: TrainingType) -> str:
        """获取训练脚本路径"""
        if training_type not in TRAINING_PRESETS:
            raise TrainingError(f"不支持的训练类型: {training_type}")
            
        preset = TRAINING_PRESETS[training_type]
        musubi_dir = self.get_musubi_path()
        script_path = musubi_dir / preset["script_path"]
        
        if not script_path.exists():
            raise TrainingError(f"训练脚本不存在: {script_path}")
            
        return str(script_path)
    
    def _create_training_workspace(self, task: TrainingTask) -> Path:
        """创建训练工作空间目录"""
        workspace_root = Path(self.config.storage.workspace_root)
        training_dir = workspace_root / "trainings" / task.id
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (training_dir / "logs").mkdir(exist_ok=True)
        (training_dir / "cache").mkdir(exist_ok=True)
        
        return training_dir
    
    def _create_dataset_config(self, task: TrainingTask) -> str:
        """创建Musubi数据集配置文件（dataset.toml）"""
        # 获取数据集路径 - 使用绝对路径
        dataset_path = Path(self.config.storage.workspace_root).resolve() / "datasets" / task.config.dataset_id / "original"
        
        if not dataset_path.exists():
            raise TrainingError(f"数据集路径不存在: {dataset_path}")
        
        # 创建训练工作空间
        training_dir = self._create_training_workspace(task)
        
        # 缓存目录 - 每个训练任务独立的缓存，使用绝对路径
        cache_dir = training_dir.resolve() / "cache"
        
        # 解析分辨率
        try:
            width, height = map(int, task.config.resolution.split(","))
        except ValueError:
            raise TrainingError(f"无效的分辨率格式: {task.config.resolution}")
        
        # 创建TOML配置 - 按照正确的格式顺序
        dataset_config = {
            "general": {
                "resolution": [width, height],
                "caption_extension": ".txt",
                "batch_size": task.config.batch_size,
                "enable_bucket": task.config.enable_bucket,
                "bucket_no_upscale": False,
                "bucket_reso_steps": 64,
                "min_bucket_reso": 256,
                "max_bucket_reso": 1536,
            },
            "datasets": [
                {
                    "image_directory": str(dataset_path).replace('\\', '/'),  # 使用正斜杠
                    "cache_directory": str(cache_dir).replace('\\', '/'),     # 使用正斜杠
                    "num_repeats": task.config.repeats,
                }
            ]
        }
        
        # 保存到训练目录（持久化）- 手动控制格式
        toml_path = training_dir / "dataset.toml"
        
        # 手动构建TOML内容，确保正确的格式顺序（只包含musubi支持的字段）
        # 计算相对于musubi工作目录的相对路径
        musubi_path = self.get_musubi_path()
        rel_dataset_path = os.path.relpath(dataset_path, musubi_path).replace(chr(92), chr(47))
        rel_cache_path = os.path.relpath(cache_dir, musubi_path).replace(chr(92), chr(47))
        
        toml_content = f"""# TagTracker 生成的数据集配置文件
[general]
resolution = [{width}, {height}]
caption_extension = ".txt"
batch_size = {task.config.batch_size}
enable_bucket = {str(task.config.enable_bucket).lower()}
bucket_no_upscale = false

[[datasets]]
image_directory = "{rel_dataset_path}"
cache_directory = "{rel_cache_path}"
num_repeats = {task.config.repeats}
"""
        
        with open(toml_path, 'w', encoding='utf-8') as f:
            f.write(toml_content)
        
        # 验证生成的TOML文件
        log_info(f"生成的数据集配置:")
        log_info(f"  图像目录: {dataset_path}")
        log_info(f"  缓存目录: {cache_dir}")
        log_info(f"  分辨率: {width}x{height}")
        log_info(f"  重复次数: {task.config.repeats}")
        
        log_info(f"数据集配置已保存: {toml_path}")
        return str(toml_path)
    
    def _save_training_config(self, task: TrainingTask, training_dir: Path) -> None:
        """保存训练配置到JSON文件（用于记录）"""
        # 手动构建可序列化的配置字典
        config_dict = {
            "task_id": task.id,
            "name": task.config.name,
            "training_type": task.config.training_type.value,
            "dataset_id": task.config.dataset_id,
            "config": self._config_to_dict(task.config),
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "platform": platform.system()
        }
        
        config_path = training_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        log_info(f"训练配置已保存: {config_path}")
    
    def _config_to_dict(self, config: TrainingConfig) -> Dict[str, Any]:
        """将训练配置转换为可 JSON 序列化的字典"""
        try:
            result = {
                "name": config.name,
                "training_type": config.training_type.value,  # 转换枚举为字符串
                "dataset_id": config.dataset_id,
                "task_id": config.task_id,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "resolution": config.resolution,
                "network_dim": config.network_dim,
                "network_alpha": config.network_alpha,
                "repeats": config.repeats,
                "dataset_size": config.dataset_size,
                "enable_bucket": config.enable_bucket,
                "optimizer": config.optimizer,
                "scheduler": config.scheduler,
                "sample_prompt": config.sample_prompt,
                "sample_every_n_steps": config.sample_every_n_steps,
                "save_every_n_epochs": config.save_every_n_epochs,
                "gpu_ids": config.gpu_ids,
                "max_data_loader_n_workers": config.max_data_loader_n_workers,
                "persistent_data_loader_workers": config.persistent_data_loader_workers,
                "seed": config.seed,
                # 模型特定配置
                "qwen_config": asdict(config.qwen_config) if config.qwen_config else None,
                "flux_config": asdict(config.flux_config) if config.flux_config else None,
                "sd_config": asdict(config.sd_config) if config.sd_config else None,
            }
            return result
        except Exception as e:
            log_error(f"配置序列化失败: {e}")
            # 返回简化版本
            return {
                "name": config.name,
                "training_type": config.training_type.value,
                "dataset_id": config.dataset_id,
                "error": "序列化失败"
            }
    
    def _build_model_args(self, config: TrainingConfig) -> List[str]:
        """根据训练类型构建模型路径参数"""
        args = []
        
        if config.training_type == TrainingType.QWEN_IMAGE_LORA:
            qwen = config.qwen_config
            args.extend([
                "--dit", qwen.dit_path,
                "--vae", qwen.vae_path,
                "--text_encoder", qwen.text_encoder_path
            ])
        elif config.training_type == TrainingType.FLUX_LORA:
            flux = config.flux_config
            if flux:
                args.extend([
                    "--dit", flux.dit_path,
                    "--vae", flux.vae_path,
                    "--text_encoder", flux.text_encoder_path,
                    "--clip", flux.clip_path
                ])
        elif config.training_type == TrainingType.SD_LORA:
            sd = config.sd_config  
            if sd:
                args.extend([
                    "--unet", sd.unet_path,
                    "--vae", sd.vae_path,
                    "--text_encoder", sd.text_encoder_path
                ])
        
        return args
    
    def _build_training_command(self, task: TrainingTask, dataset_config_path: str, training_dir: Path) -> List[str]:
        """构建完整的训练命令"""
        config = task.config
        preset = TRAINING_PRESETS[config.training_type]
        script_path = self._get_script_path(config.training_type)
        
        # 输出目录
        output_dir = Path(self.config.storage.workspace_root) / "models" / task.id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 基础命令 - 获取正确的accelerate命令
        accelerate_cmd = self._get_accelerate_cmd()
        cmd = accelerate_cmd + [
            "launch",
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", "bf16",
            script_path
        ]
        
        # 模型路径参数
        cmd.extend(self._build_model_args(config))
        
        # 数据集配置 - 使用相对路径
        try:
            config_path = Path(dataset_config_path).resolve()
            musubi_path = Path(self.get_musubi_path()).resolve()
            relative_config_path = os.path.relpath(config_path, musubi_path)
            # 确保使用正确的路径分隔符
            relative_config_path = relative_config_path.replace('\\', '/')
        except Exception:
            relative_config_path = dataset_config_path
        cmd.extend(["--dataset_config", relative_config_path])
        
        # 输出配置
        cmd.extend([
            "--output_dir", str(output_dir),
            "--output_name", f"{config.name}_lora"
        ])
        
        # 网络配置
        cmd.extend([
            "--network_module", preset["network_module"],
            "--network_dim", str(config.network_dim),
            "--network_alpha", str(config.network_alpha)
        ])
        
        # 训练参数
        cmd.extend([
            "--max_train_epochs", str(config.epochs),
            "--learning_rate", str(config.learning_rate),
            "--optimizer_type", config.optimizer,
            "--lr_scheduler", config.scheduler,
            "--save_every_n_epochs", str(config.save_every_n_epochs),
            "--seed", str(config.seed),
            "--max_data_loader_n_workers", str(config.max_data_loader_n_workers)
        ])
        
        # 预设默认参数
        for key, value in preset["default_args"].items():
            if value is None:  # flag参数
                cmd.append(key)
            else:
                cmd.extend([key, str(value)])
        
        # 模型特有参数
        if config.training_type == TrainingType.QWEN_IMAGE_LORA:
            qwen = config.qwen_config
            
            # 可选参数
            if qwen.gradient_checkpointing:
                cmd.append("--gradient_checkpointing")
            if qwen.fp8_base:
                cmd.append("--fp8_base")
            if qwen.fp8_scaled:
                cmd.append("--fp8_scaled")
            if qwen.fp8_vl:
                cmd.append("--fp8_vl")
            if qwen.blocks_to_swap > 0:
                cmd.extend(["--blocks_to_swap", str(qwen.blocks_to_swap)])
            if qwen.split_attn:
                cmd.append("--split_attn")
        
        # 采样配置
        if config.sample_prompt:
            sample_prompts_file = training_dir / "sample_prompts.txt"
            with open(sample_prompts_file, 'w', encoding='utf-8') as f:
                f.write(config.sample_prompt)
            cmd.extend([
                "--sample_prompts", str(sample_prompts_file),
                "--sample_every_n_epochs", "1",
                "--sample_at_first"
            ])
        
        # 数据加载器
        if config.persistent_data_loader_workers:
            cmd.append("--persistent_data_loader_workers")
        
        # 日志目录
        log_dir = training_dir / "logs"  
        cmd.extend(["--logging_dir", str(log_dir)])
        
        log_info(f"训练命令: {' '.join(cmd)}")
        return cmd
    
    def _create_training_scripts(self, task: TrainingTask, dataset_config_path: str, training_dir: Path) -> Dict[str, str]:
        """创建训练脚本文件（bat和sh）"""
        cmd = self._build_training_command(task, dataset_config_path, training_dir)
        
        # 获取musubi工作目录
        musubi_dir = self.get_musubi_path()
        
        # 使用当前Python解释器执行训练命令
        script_cmd = cmd
        
        # Windows批处理脚本
        bat_content = f'''@echo off
echo ===== TagTracker Musubi 训练脚本 =====
echo 任务名称: {task.config.name}
echo Python解释器: {sys.executable}
echo 开始时间: %date% %time%
echo =====================================

cd /d "{musubi_dir}"
set PYTHONPATH={musubi_dir}\\src;%PYTHONPATH%
set PYTHONIOENCODING=utf-8

{' '.join(f'"{arg}"' if ' ' in arg else arg for arg in script_cmd)} 2>&1 | tee "{training_dir / 'logs' / 'training.log'}"

echo =====================================
echo 训练完成时间: %date% %time%
echo =====================================
pause
'''
        
        # Linux/Mac shell脚本
        sh_content = f'''#!/bin/bash
echo "===== TagTracker Musubi 训练脚本 ====="
echo "任务名称: {task.config.name}"
echo "Python解释器: {sys.executable}"
echo "开始时间: $(date)"
echo "====================================="

cd "{musubi_dir}"
export PYTHONPATH="{musubi_dir}/src:$PYTHONPATH"
export PYTHONIOENCODING=utf-8

{' '.join(f'"{arg}"' if ' ' in arg else arg for arg in script_cmd)} 2>&1 | tee "{training_dir / 'logs' / 'training.log'}"

echo "====================================="
echo "训练完成时间: $(date)"
echo "====================================="
'''
        
        # 保存脚本文件
        bat_path = training_dir / "train.bat"
        sh_path = training_dir / "train.sh"
        
        with open(bat_path, 'w', encoding='gbk') as f:
            f.write(bat_content)
        
        with open(sh_path, 'w', encoding='utf-8') as f:
            f.write(sh_content)
        
        # Linux/Mac脚本设置执行权限
        if platform.system() != 'Windows':
            os.chmod(sh_path, 0o755)
        
        log_info(f"训练脚本已生成: {bat_path}, {sh_path}")
        return {
            'bat_script': str(bat_path),
            'sh_script': str(sh_path),
            'command': script_cmd
        }
    
    def _run_cache_steps(self, task: TrainingTask, dataset_config_path: str, log_callback: Optional[Callable[[str], None]] = None) -> bool:
        """执行预处理缓存步骤"""
        config = task.config
        if config.training_type not in TRAINING_PRESETS:
            return True
            
        preset = TRAINING_PRESETS[config.training_type]
        cache_scripts = preset.get("cache_scripts", [])
        
        if not cache_scripts:
            return True
            
        musubi_dir = self.get_musubi_path()
        
        for script_name in cache_scripts:
            script_path = musubi_dir / script_name
            if not script_path.exists():
                log_error(f"缓存脚本不存在: {script_path}")
                return False
                
            log_info(f"执行预处理步骤: {script_name}")
            
            # 构建缓存命令 - 使用当前激活的Python解释器
            # 计算相对于musubi工作目录的相对路径
            try:
                config_path = Path(dataset_config_path).resolve()
                musubi_path = Path(musubi_dir).resolve()
                relative_config_path = os.path.relpath(config_path, musubi_path)
                # 确保使用正确的路径分隔符
                relative_config_path = relative_config_path.replace('\\', '/')
                log_info(f"TOML绝对路径: {config_path}")
                log_info(f"Musubi工作目录: {musubi_path}")
                log_info(f"使用相对路径: {relative_config_path}")
            except Exception as e:
                log_error(f"路径转换失败，使用绝对路径: {e}")
                relative_config_path = dataset_config_path
            
            cache_cmd = [sys.executable, str(script_path), "--dataset_config", relative_config_path]
            
            # 添加模型特定参数
            if "cache_latents" in script_name:
                if config.training_type == TrainingType.QWEN_IMAGE_LORA:
                    cache_cmd.extend(["--vae", config.qwen_config.vae_path])
            elif "cache_text_encoder" in script_name:
                if config.training_type == TrainingType.QWEN_IMAGE_LORA:
                    cache_cmd.extend([
                        "--text_encoder", config.qwen_config.text_encoder_path,
                        "--batch_size", "16"
                    ])
            
            # 执行缓存命令 - 使用实时输出
            try:
                # 设置环境变量，确保能找到musubi_tuner模块
                env = os.environ.copy()
                env['PYTHONPATH'] = str(musubi_dir / 'src') + os.pathsep + env.get('PYTHONPATH', '')
                # 设置UTF-8编码，避免Windows GBK编码问题
                env['PYTHONIOENCODING'] = 'utf-8'
                
                # 使用Popen进行实时输出监控
                cache_proc = subprocess.Popen(
                    cache_cmd,
                    cwd=str(musubi_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 实时读取输出，带超时控制
                import time
                start_time = time.time()
                timeout_seconds = 1800  # 30分钟超时
                
                while True:
                    output = cache_proc.stdout.readline()
                    if output == '' and cache_proc.poll() is not None:
                        break
                    if output:
                        line = output.strip()
                        log_info(f"[缓存] {line}")
                        if log_callback:
                            log_callback(f"[缓存] {line}")
                    
                    # 检查超时
                    if time.time() - start_time > timeout_seconds:
                        cache_proc.terminate()
                        raise subprocess.TimeoutExpired(cache_cmd, timeout_seconds)
                
                # 检查返回码
                return_code = cache_proc.poll()
                if return_code != 0:
                    error_msg = f"预处理失败，退出码: {return_code}"
                    log_error(error_msg)
                    if log_callback:
                        log_callback(f"[错误] {error_msg}")
                    return False
                    
                success_msg = f"预处理完成: {script_name}"
                log_success(success_msg)
                if log_callback:
                    log_callback(f"[完成] {success_msg}")
                
            except subprocess.TimeoutExpired:
                error_msg = f"预处理超时: {script_name}"
                log_error(error_msg)
                if log_callback:
                    log_callback(f"[超时] {error_msg}")
                return False
            except Exception as e:
                error_msg = f"预处理异常: {e}"
                log_error(error_msg)
                if log_callback:
                    log_callback(f"[异常] {error_msg}")
                return False
        
        return True
    
    def prepare_training(self, task: TrainingTask) -> Dict[str, Any]:
        """准备训练环境和文件"""
        try:
            # 验证配置
            self._validate_config(task.config)
            
            # 创建数据集配置
            dataset_config_path = self._create_dataset_config(task)
            
            # 获取训练目录
            training_dir = self._create_training_workspace(task)
            
            # 保存训练配置记录
            self._save_training_config(task, training_dir)
            
            # 生成训练脚本
            scripts_info = self._create_training_scripts(task, dataset_config_path, training_dir)
            
            return {
                'training_dir': training_dir,
                'dataset_config': dataset_config_path,
                'scripts_info': scripts_info,
                'log_file': training_dir / "logs" / "training.log"
            }
            
        except Exception as e:
            log_error(f"准备训练失败: {e}")
            raise TrainingError(f"准备训练失败: {e}")
    
    def _validate_config(self, config: TrainingConfig):
        """验证训练配置"""
        if config.training_type not in TRAINING_PRESETS:
            raise TrainingError(f"不支持的训练类型: {config.training_type}")
            
        preset = TRAINING_PRESETS[config.training_type]
        required_models = preset.get("required_models", [])
        
        # 验证模型路径
        if config.training_type == TrainingType.QWEN_IMAGE_LORA:
            qwen = config.qwen_config
            if not qwen.dit_path or not os.path.exists(qwen.dit_path):
                raise TrainingError("DiT模型路径无效")
            if not qwen.vae_path or not os.path.exists(qwen.vae_path):
                raise TrainingError("VAE模型路径无效")
            if not qwen.text_encoder_path or not os.path.exists(qwen.text_encoder_path):
                raise TrainingError("Text Encoder路径无效")
    
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

            training_info = self.prepare_training(task)
            
            # 执行预处理缓存步骤
            log_info("开始预处理缓存步骤...")
            if log_callback:
                log_callback("开始预处理缓存步骤...")
                
            cache_success = self._run_cache_steps(task, training_info['dataset_config'], log_callback)
            if not cache_success:
                task.state = TrainingState.FAILED
                task.error_message = "预处理缓存失败"
                return False
            
            # 开始训练
            task.state = TrainingState.RUNNING
            if progress_callback:
                progress_callback({"state": task.state.value})

            log_info(f"开始训练: {task.config.name}")
            if log_callback:
                log_callback(f"开始训练: {task.config.name}")

            # 启动训练进程
            cmd = training_info['scripts_info']['command']
            musubi_dir = self.get_musubi_path()
            
            # 设置环境变量，确保能找到musubi_tuner模块
            env = os.environ.copy()
            env['PYTHONPATH'] = str(musubi_dir / 'src') + os.pathsep + env.get('PYTHONPATH', '')
            # 设置UTF-8编码，避免Windows GBK编码问题
            env['PYTHONIOENCODING'] = 'utf-8'
            
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(musubi_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',  # 遇到无法解码的字符时替换为?
                bufsize=1,
                universal_newlines=True
            )

            # 实时读取输出并监控进度
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
            if not self._proc:
                return False

            while True:
                output = self._proc.stdout.readline()
                if output == '' and self._proc.poll() is not None:
                    break

                if output:
                    line = output.strip()
                    if log_callback:
                        log_callback(line)

                    # 解析训练进度
                    progress_info = self._parse_training_output(line)
                    if progress_info:
                        # 更新任务状态
                        for key, value in progress_info.items():
                            if hasattr(task, key):
                                setattr(task, key, value)

                        # 计算进度百分比
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
            return_code = self._proc.poll()
            if return_code == 0:
                task.state = TrainingState.COMPLETED
                log_success(f"训练完成: {task.config.name}")
                
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
        """解析训练输出，提取进度信息"""
        try:
            progress_info = {}

            # 解析步数和轮次
            epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                progress_info['current_epoch'] = int(epoch_match.group(1))
                progress_info['total_epochs'] = int(epoch_match.group(2))

            step_match = re.search(r'Step (\d+)/(\d+)', line)
            if step_match:
                progress_info['current_step'] = int(step_match.group(1))
                progress_info['total_steps'] = int(step_match.group(2))

            # 解析loss
            loss_match = re.search(r'loss:?\s*([\d.]+)', line, re.IGNORECASE)
            if loss_match:
                progress_info['loss'] = float(loss_match.group(1))

            # 解析学习率
            lr_match = re.search(r'lr:?\s*([\d.e-]+)', line, re.IGNORECASE)
            if lr_match:
                progress_info['learning_rate'] = float(lr_match.group(1))

            # 解析速度
            speed_match = re.search(r'([\d.]+)\s*it/s', line)
            if speed_match:
                progress_info['speed'] = float(speed_match.group(1))

            # 解析ETA
            eta_match = re.search(r'ETA:?\s*(\d{2}):(\d{2}):(\d{2})', line)
            if eta_match:
                hours, minutes, seconds = map(int, eta_match.groups())
                progress_info['eta_seconds'] = hours * 3600 + minutes * 60 + seconds

            return progress_info if progress_info else None

        except Exception as e:
            log_error(f"解析训练输出失败: {str(e)}")
            return None

    def cancel_training(self):
        """取消训练"""
        if self._proc and self._proc.poll() is None:
            try:
                log_info("正在取消训练...")
                if os.name == 'nt':  # Windows
                    self._proc.terminate()
                else:  # Unix/Linux
                    self._proc.send_signal(signal.SIGTERM)
                
                # 等待进程结束
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    
                log_info("训练已取消")
            except Exception as e:
                log_error(f"取消训练时出错: {e}")
            finally:
                self._proc = None

    def is_available(self) -> bool:
        """检查Musubi-Tuner是否可用"""
        try:
            musubi_dir = self.get_musubi_path()
            
            # 检查关键脚本是否存在
            for training_type in TRAINING_PRESETS:
                script_path = musubi_dir / TRAINING_PRESETS[training_type]["script_path"]
                if not script_path.exists():
                    return False
            
            return True
            
        except Exception:
            return False

    def build_job(self, task: TrainingTask) -> Job:
        """构建训练任务（兼容旧接口）"""
        return Job(
            id=self._id,
            name=task.config.name,
            run=lambda log_cb, progress_cb: self.run_training(task, progress_cb, log_cb),
            cancel=self.cancel_training
        )