"""
Training Manager - 训练任务管理器
"""

import uuid
import json
import threading
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime

from .models import TrainingConfig, TrainingTask, TrainingState, TrainingType
from .qwen_trainer import QwenImageTrainer
from ..common.events import EventBus, JobQueue, Job
from ...utils.logger import log_info, log_error, log_success
from ...utils.exceptions import TrainingError, TrainingNotFoundError
from ...config import get_config

# ✅ 从 trainers.types 引用，避免循环导入
from trainers.types import TrainingConfig as OldTrainingConfig, TrainingBackend
from trainers.musubi_qwen_image import MusubiQwenImageTrainer
from .trainers.musubi_trainer import MusubiTrainer

class TrainingManager:
    """训练任务管理器"""
    
    def __init__(self, bus: Optional[EventBus] = None, queue: Optional[JobQueue] = None):
        self.config = get_config()
        self.tasks: Dict[str, TrainingTask] = {}
        self.trainers = {
            TrainingType.QWEN_IMAGE_LORA: QwenImageTrainer()
        }
        
        # 事件总线和任务队列（来自旧版training_manager.py）
        self.bus = bus
        self.queue = queue
        
        # 初始化训练器实例
        self._trainers = {
            TrainingBackend.MUSUBI_QWEN_IMAGE: MusubiQwenImageTrainer(self.bus),
        }
        
        # 延迟初始化 Musubi 训练器（避免启动时的依赖检查）
        if self.bus and TrainingBackend.MUSUBI_HUNYUAN_VIDEO:
            try:
                musubi_trainer = MusubiTrainer(self.bus)
                self._trainers[TrainingBackend.MUSUBI_HUNYUAN_VIDEO] = musubi_trainer
            except Exception as e:
                log_error(f"Musubi 训练器初始化失败: {e}")
                # 不阻止应用启动，用户可以稍后配置
        # task_id -> (last_ts, last_step)
        self._speed_cache: Dict[str, Tuple[float, int]] = {}
        
        # 任务持久化目录
        self.tasks_dir = Path(self.config.storage.workspace_root) / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        
        # 事件回调
        self.callbacks: Dict[str, List[Callable]] = {
            'task_state': [],
            'task_progress': [],
            'task_log': []
        }
        
        # 加载现有任务
        self.load_tasks()
    
    def total_steps(self, cfg: OldTrainingConfig) -> int:
        """
        总步数只与样本数/重复/轮数/batch/梯度累计相关（按你的规则）
        """
        num = cfg.dataset_size * max(1, cfg.repeats)
        denom = max(1, cfg.batch_size * max(1, cfg.grad_accum))
        steps_per_epoch = math.ceil(num / denom)
        return steps_per_epoch * max(1, cfg.epochs)
    
    def run_training(self, cfg: OldTrainingConfig) -> str:
        """
        创建并提交训练任务，返回 job_id
        """
        total = self.total_steps(cfg)
        trainer = self._trainers[cfg.backend]
        job: Job = trainer.build_job(cfg, total_steps=total)

        def _on_prog(p: dict):
            """
            统一进度事件出口：优先用后端提供的 eta，
            缺失/不准时用本地回退估算（最近窗口 steps/sec）
            """
            now = time.time()
            step = int(p.get("step", 0))
            eta = p.get("eta_secs", None)

            if eta in (None, 0):
                last = self._speed_cache.get(job.id)
                if last:
                    t0, s0 = last
                    dt = max(1e-3, now - t0)
                    ds = max(0, step - s0)
                    sps = ds / dt
                    remain = max(0, total - step)
                    eta = int(remain / sps) if sps > 0 else None
                self._speed_cache[job.id] = (now, step)

            # 向总线抛统一进度
            payload = {
                "id": job.id,
                **p,
                "total_steps": total,
                "eta_secs": eta,
            }
            self.bus.emit("task_progress", payload)

        # 提交到队列
        jid = self.queue.submit(
            Job(
                id=job.id,
                name=cfg.name,
                run=lambda log_cb, progress_cb: trainer.run(
                    log_cb, _on_prog, cfg, total
                ),
                cancel=lambda: trainer.cancel(),
            )
        )
        return jid

    def create_task(self, config: TrainingConfig) -> str:
        """创建训练任务"""
        try:
            # 生成任务ID
            task_id = str(uuid.uuid4())
            config.task_id = task_id
            
            # 创建任务
            task = TrainingTask(task_id=task_id, config=config)
            self.tasks[task_id] = task
            
            # 保存任务
            self.save_task(task_id)
            
            # 通知任务创建
            self.emit('task_state', {
                'id': task_id,
                'state': task.state.value,
                'name': config.name
            })
            
            log_info(f"创建训练任务: {config.name} ({task_id})")
            return task_id
            
        except Exception as e:
            log_error(f"创建训练任务失败: {str(e)}")
            raise TrainingError(f"创建任务失败: {str(e)}")
    
    def start_task(self, task_id: str) -> bool:
        """启动训练任务"""
        try:
            if task_id not in self.tasks:
                raise TrainingNotFoundError(task_id)
            
            task = self.tasks[task_id]
            
            if task.state != TrainingState.PENDING:
                raise TrainingError(f"任务状态无效: {task.state.value}")
            
            # 获取对应的训练器
            trainer = self.trainers.get(task.config.training_type)
            if not trainer:
                raise TrainingError(f"不支持的训练类型: {task.config.training_type}")
            
            # 设置为准备状态
            task.state = TrainingState.PREPARING
            task.started_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 通知状态变更
            self.emit('task_state', {
                'id': task_id,
                'state': task.state.value
            })
            
            # 保存状态变更
            self.save_task(task_id)
            
            # 创建回调函数
            def progress_callback(progress_info: Dict[str, Any]):
                self._on_training_progress(task_id, progress_info)
            
            def log_callback(log_line: str):
                self._on_training_log(task_id, log_line)
            
            # 在新线程中启动训练
            def training_thread():
                try:
                    success = trainer.run_training(task, progress_callback, log_callback)
                    if success:
                        task.completed_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        log_success(f"训练任务完成: {task.config.name}")
                    self.save_task(task_id)
                except Exception as e:
                    task.state = TrainingState.FAILED
                    task.error_message = str(e)
                    log_error(f"训练任务异常: {str(e)}")
                    self.save_task(task_id)
            
            thread = threading.Thread(target=training_thread, daemon=True)
            thread.start()
            
            log_info(f"启动训练任务: {task.config.name}")
            return True
            
        except Exception as e:
            log_error(f"启动训练任务失败: {str(e)}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """取消训练任务"""
        try:
            if task_id not in self.tasks:
                raise TrainingNotFoundError(task_id)
            
            task = self.tasks[task_id]
            
            if task.state not in [TrainingState.PENDING, TrainingState.PREPARING, TrainingState.RUNNING]:
                return False
            
            # 获取训练器并取消
            trainer = self.trainers.get(task.config.training_type)
            if trainer:
                trainer.cancel_training()
            
            task.state = TrainingState.CANCELLED
            self.save_task(task_id)
            
            # 通知状态变更
            self.emit('task_state', {
                'id': task_id,
                'state': task.state.value
            })
            
            log_info(f"取消训练任务: {task.config.name}")
            return True
            
        except Exception as e:
            log_error(f"取消训练任务失败: {str(e)}")
            return False
    
    def cancel(self, job_id: str) -> bool:
        """取消训练任务（来自旧版training_manager.py）"""
        return self.queue.cancel(job_id)
    
    def delete_task(self, task_id: str) -> bool:
        """删除训练任务"""
        try:
            if task_id not in self.tasks:
                raise TrainingNotFoundError(task_id)
            
            task = self.tasks[task_id]
            
            # 如果任务正在运行，先取消
            if task.state in [TrainingState.PREPARING, TrainingState.RUNNING]:
                self.cancel_task(task_id)
            
            # 删除任务文件
            task_file = self.tasks_dir / f"{task_id}.json"
            if task_file.exists():
                task_file.unlink()
            
            # 从内存中删除
            del self.tasks[task_id]
            
            log_info(f"删除训练任务: {task.config.name}")
            return True
            
        except Exception as e:
            log_error(f"删除训练任务失败: {str(e)}")
            return False
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """获取训练任务"""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[TrainingTask]:
        """获取所有训练任务"""
        return list(self.tasks.values())
    
    def get_memory_estimate(self, config: TrainingConfig) -> Dict[str, Any]:
        """获取显存使用预估"""
        trainer = self.trainers.get(config.training_type)
        if trainer and hasattr(trainer, 'get_memory_usage_estimate'):
            return trainer.get_memory_usage_estimate(config)
        return {"estimated_vram_mb": 0, "estimated_vram_gb": 0.0}
    
    def save_task(self, task_id: str):
        """保存任务到文件"""
        try:
            if task_id not in self.tasks:
                return
            
            task = self.tasks[task_id]
            task_file = self.tasks_dir / f"{task_id}.json"
            
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            log_error(f"保存训练任务失败 {task_id}: {str(e)}")
    
    def load_tasks(self):
        """加载所有任务"""
        try:
            for task_file in self.tasks_dir.glob("*.json"):
                try:
                    with open(task_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    task = TrainingTask.from_dict(data)
                    self.tasks[task.task_id] = task
                    
                except Exception as e:
                    log_error(f"加载训练任务失败 {task_file}: {str(e)}")
            
            log_info(f"加载了 {len(self.tasks)} 个训练任务")
            
        except Exception as e:
            log_error(f"加载训练任务列表失败: {str(e)}")
    
    def register_callback(self, event: str, callback: Callable):
        """注册事件回调"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def unregister_callback(self, event: str, callback: Callable):
        """注销事件回调"""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def emit(self, event: str, data: Dict[str, Any]):
        """触发事件"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    log_error(f"事件回调异常 {event}: {str(e)}")
    
    def _on_training_progress(self, task_id: str, progress_info: Dict[str, Any]):
        """训练进度回调"""
        if task_id in self.tasks:
            # 保存任务状态
            self.save_task(task_id)
            
            # 发送进度事件
            progress_data = {'id': task_id, **progress_info}
            self.emit('task_progress', progress_data)
    
    def _on_training_log(self, task_id: str, log_line: str):
        """训练日志回调"""
        self.emit('task_log', {
            'id': task_id,
            'line': log_line
        })
    
    def get_training_presets(self) -> Dict[str, Dict[str, Any]]:
        """获取训练预设配置"""
        return self.config.training.memory_presets
    
    def create_qwen_task(self, 
                        name: str, 
                        dataset_id: str,
                        dit_path: str,
                        vae_path: str, 
                        text_encoder_path: str,
                        **kwargs) -> str:
        """创建Qwen-Image训练任务的便捷方法"""
        from .models import QwenImageConfig
        
        qwen_config = QwenImageConfig(
            dit_path=dit_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            **{k: v for k, v in kwargs.items() if hasattr(QwenImageConfig, k)}
        )
        
        training_config = TrainingConfig(
            task_id="",  # 将在create_task中设置
            name=name,
            training_type=TrainingType.QWEN_IMAGE_LORA,
            dataset_id=dataset_id,
            qwen_config=qwen_config,
            **{k: v for k, v in kwargs.items() if hasattr(TrainingConfig, k)}
        )
        
        return self.create_task(training_config)