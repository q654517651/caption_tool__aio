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
from ..common.events import EventBus, JobQueue, Job
from ...utils.logger import log_info, log_error, log_success
from ...utils.exceptions import TrainingError, TrainingNotFoundError
from ...config import get_config
from .trainers.musubi_trainer import MusubiTrainer


class TrainingManager:
    """训练任务管理器"""

    def __init__(self, bus: Optional[EventBus] = None, queue: Optional[JobQueue] = None):
        self.config = get_config()
        self.tasks: Dict[str, TrainingTask] = {}

        # 事件总线和任务队列（来自旧版training_manager.py）
        self.bus = bus
        self.queue = queue

        # 初始化统一的训练器
        try:
            self.musubi_trainer = MusubiTrainer(self.bus)
            log_info("Musubi 训练器初始化成功")
        except Exception as e:
            log_error(f"Musubi 训练器初始化失败: {e}")
            self.musubi_trainer = None

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

    def create_task(self, config: TrainingConfig) -> str:
        """创建训练任务"""
        try:
            task_id = str(uuid.uuid4())

            # 创建任务对象
            task = TrainingTask(
                id=task_id,
                name=config.name,
                config=config,
                state=TrainingState.PENDING,
                created_at=datetime.now(),
                progress=0.0,
                logs=[]
            )

            # 保存任务
            self.tasks[task_id] = task
            self.save_task(task)

            log_info(f"创建训练任务: {config.name} (ID: {task_id})")
            return task_id

        except Exception as e:
            log_error(f"创建训练任务失败: {e}")
            raise TrainingError(f"创建训练任务失败: {e}")

    def start_task(self, task_id: str) -> bool:
        """开始训练任务"""
        try:
            task = self.get_task(task_id)
            if not task:
                raise TrainingNotFoundError(f"任务不存在: {task_id}")

            if task.state != TrainingState.PENDING:
                log_error(f"任务状态不允许启动: {task.state}")
                return False

            # 更新任务状态
            task.state = TrainingState.RUNNING
            task.started_at = datetime.now()
            self.save_task(task)

            # 检查训练器可用性
            if not self.musubi_trainer:
                raise TrainingError("Musubi训练器未初始化")

            # 直接启动训练（简化逻辑）
            def run_training():
                try:
                    success = self.musubi_trainer.run_training(
                        task,
                        progress_callback=lambda data: self._on_progress(task_id, data),
                        log_callback=lambda line: self._on_log(task_id, line)
                    )
                    
                    if success:
                        task.state = TrainingState.COMPLETED
                        task.progress = 1.0
                        log_success(f"训练任务完成: {task.config.name}")
                    else:
                        task.state = TrainingState.FAILED
                        log_error(f"训练任务失败: {task.config.name}")
                        
                except Exception as e:
                    task.state = TrainingState.FAILED
                    task.error_message = str(e)
                    log_error(f"训练任务异常: {e}")
                finally:
                    task.completed_at = datetime.now()
                    self.save_task(task)
                    self._emit_event('task_state', {'task_id': task_id, 'state': task.state})

            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()

            log_info(f"开始训练任务: {task.name}")
            return True

        except Exception as e:
            log_error(f"启动训练任务失败: {e}")
            return False


    def cancel_task(self, task_id: str) -> bool:
        """取消训练任务"""
        try:
            task = self.get_task(task_id)
            if not task:
                return False

            if task.state != TrainingState.RUNNING:
                return False

            # 取消训练
            if self.musubi_trainer:
                self.musubi_trainer.cancel_training()
                
            task.state = TrainingState.CANCELLED
            task.completed_at = datetime.now()
            self.save_task(task)

            log_info(f"取消训练任务: {task.name}")
            self._emit_event('task_state', {'task_id': task_id, 'state': task.state})
            return True

        except Exception as e:
            log_error(f"取消训练任务失败: {e}")
            return False

    def delete_task(self, task_id: str) -> bool:
        """删除训练任务"""
        try:
            task = self.get_task(task_id)
            if not task:
                return False

            # 不能删除正在运行的任务
            if task.state == TrainingState.RUNNING:
                log_error("不能删除正在运行的任务")
                return False

            # 删除任务文件
            task_file = self.tasks_dir / f"{task_id}.json"
            if task_file.exists():
                task_file.unlink()

            # 从内存中删除
            del self.tasks[task_id]

            log_info(f"删除训练任务: {task.name}")
            return True

        except Exception as e:
            log_error(f"删除训练任务失败: {e}")
            return False

    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """获取训练任务"""
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[TrainingTask]:
        """列出所有训练任务"""
        return list(self.tasks.values())

    def save_task(self, task: TrainingTask) -> None:
        """保存训练任务到文件"""
        try:
            task_file = self.tasks_dir / f"{task.id}.json"
            task_data = {
                'id': task.id,
                'name': task.name,
                'config': {
                    'name': task.config.name,
                    'training_type': task.config.training_type.value,
                    'dataset_id': task.config.dataset_id,
                    'task_id': task.config.task_id,
                    'epochs': task.config.epochs,
                    'batch_size': task.config.batch_size,
                    'learning_rate': task.config.learning_rate,
                    'resolution': task.config.resolution,
                    'network_dim': task.config.network_dim,
                    'network_alpha': task.config.network_alpha,
                    'repeats': task.config.repeats,
                    'dataset_size': task.config.dataset_size,
                    'enable_bucket': task.config.enable_bucket,
                    'optimizer': task.config.optimizer,
                    'scheduler': task.config.scheduler,
                    'sample_prompt': task.config.sample_prompt,
                    'sample_every_n_steps': task.config.sample_every_n_steps,
                    'save_every_n_epochs': task.config.save_every_n_epochs,
                    'gpu_ids': task.config.gpu_ids,
                    'max_data_loader_n_workers': task.config.max_data_loader_n_workers,
                    'persistent_data_loader_workers': task.config.persistent_data_loader_workers,
                    'seed': task.config.seed,
                },
                'state': task.state.value,
                'progress': task.progress,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'logs': task.logs[-100:]  # 只保存最近100条日志
            }

            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            log_error(f"保存训练任务失败: {e}")

    def load_tasks(self) -> None:
        """从文件加载训练任务"""
        try:
            for task_file in self.tasks_dir.glob("*.json"):
                try:
                    with open(task_file, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)

                    # 重构配置对象
                    config_data = task_data['config']
                    config = TrainingConfig(
                        name=config_data['name'],
                        training_type=TrainingType(
                            config_data.get('training_type', config_data.get('type', 'qwen_image_lora'))),
                        dataset_id=config_data['dataset_id'],
                        task_id=config_data.get('task_id', ''),
                        epochs=config_data.get('epochs', 16),
                        batch_size=config_data.get('batch_size', 1),
                        learning_rate=config_data.get('learning_rate', 1e-4),
                        resolution=config_data.get('resolution', '1024,1024'),
                        network_dim=config_data.get('network_dim', 32),
                        network_alpha=config_data.get('network_alpha', 16),
                        repeats=config_data.get('repeats', 1),
                        dataset_size=config_data.get('dataset_size', 0),
                        enable_bucket=config_data.get('enable_bucket', True),
                        optimizer=config_data.get('optimizer', 'adamw'),
                        scheduler=config_data.get('scheduler', 'cosine'),
                        sample_prompt=config_data.get('sample_prompt', ''),
                        sample_every_n_steps=config_data.get('sample_every_n_steps', 200),
                        save_every_n_epochs=config_data.get('save_every_n_epochs', 1),
                        gpu_ids=config_data.get('gpu_ids', [0]),
                        max_data_loader_n_workers=config_data.get('max_data_loader_n_workers', 2),
                        persistent_data_loader_workers=config_data.get('persistent_data_loader_workers', True),
                        seed=config_data.get('seed', 42),
                    )

                    # 重构任务对象
                    task = TrainingTask(
                        id=task_data['id'],
                        name=task_data['name'],
                        config=config,
                        state=TrainingState(task_data['state']),
                        progress=task_data.get('progress', 0.0),
                        created_at=datetime.fromisoformat(task_data['created_at']),
                        started_at=datetime.fromisoformat(task_data['started_at']) if task_data.get(
                            'started_at') else None,
                        completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data.get(
                            'completed_at') else None,
                        logs=task_data.get('logs', [])
                    )

                    self.tasks[task.id] = task

                except Exception as e:
                    log_error(f"加载任务文件失败 {task_file}: {e}")
                    continue

            log_info(f"加载了 {len(self.tasks)} 个训练任务")

        except Exception as e:
            log_error(f"加载训练任务失败: {e}")

    def _on_progress(self, task_id: str, progress_info: Dict[str, Any]) -> None:
        """训练进度回调"""
        task = self.get_task(task_id)
        if task:
            # 更新任务进度信息
            if 'progress' in progress_info:
                task.progress = progress_info['progress']
            if 'step' in progress_info:
                task.current_step = progress_info['step']
            if 'total_steps' in progress_info:
                task.total_steps = progress_info['total_steps']
            if 'epoch' in progress_info:
                task.current_epoch = progress_info['epoch']
            if 'loss' in progress_info:
                task.loss = progress_info['loss']
            if 'lr' in progress_info:
                task.learning_rate = progress_info['lr']
            if 'speed' in progress_info:
                task.speed = progress_info['speed']
            if 'eta_seconds' in progress_info:
                task.eta_seconds = progress_info['eta_seconds']
                
            self.save_task(task)
            
            # 发送进度事件
            event_data = {'task_id': task_id}
            event_data.update(progress_info)
            self._emit_event('task_progress', event_data)

    def _on_log(self, task_id: str, message: str) -> None:
        """训练日志回调"""
        task = self.get_task(task_id)
        if task:
            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] {message}"
            task.logs.append(log_entry)

            # 限制内存中的日志数量
            if len(task.logs) > 1000:
                task.logs = task.logs[-1000:]

            # 保存到任务文件
            self.save_task(task)
            
            # 同时写入实时日志文件（追加模式）
            try:
                self._write_log_to_file(task_id, log_entry)
            except Exception as e:
                log_error(f"写入日志文件失败: {e}")
            
            self._emit_event('task_log', {'task_id': task_id, 'message': log_entry})
    
    def _write_log_to_file(self, task_id: str, log_entry: str) -> None:
        """将日志写入实时日志文件"""
        from pathlib import Path
        
        # 创建日志目录
        log_dir = Path(self.config.storage.workspace_root) / "trainings" / task_id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        log_file = log_dir / "training_realtime.log"
        
        # 追加写入日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
            f.flush()  # 强制刷新缓冲区

    def _emit_event(self, event: str, data: Dict[str, Any]) -> None:
        """发送事件"""
        callbacks = self.callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                log_error(f"事件回调失败: {e}")

    def add_callback(self, event: str, callback: Callable) -> None:
        """添加事件回调"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callable) -> None:
        """移除事件回调"""
        if event in self.callbacks:
            try:
                self.callbacks[event].remove(callback)
            except ValueError:
                pass
