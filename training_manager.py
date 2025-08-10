# training_manager.py
import math
import time
from typing import Dict, Tuple

from services.event_bus import EventBus
from services.job_queue import JobQueue, Job

# ✅ 从 trainers.types 引用，避免循环导入
from trainers.types import TrainingConfig, TrainingBackend
from trainers.musubi_qwen_image import MusubiQwenImageTrainer


class TrainingManager:
    """
    训练编排器：
    - 计算总步数
    - 选择具体 Trainer
    - 运行任务（交给 JobQueue）
    - 早期 ETA 回退估算
    """
    def __init__(self, bus: EventBus, queue: JobQueue):
        self.bus = bus
        self.queue = queue
        self._trainers = {
            TrainingBackend.MUSUBI_QWEN_IMAGE: MusubiQwenImageTrainer(self.bus),
        }
        # task_id -> (last_ts, last_step)
        self._speed_cache: Dict[str, Tuple[float, int]] = {}

    def total_steps(self, cfg: TrainingConfig) -> int:
        """
        总步数只与样本数/重复/轮数/batch/梯度累计相关（按你的规则）
        """
        num = cfg.dataset_size * max(1, cfg.repeats)
        denom = max(1, cfg.batch_size * max(1, cfg.grad_accum))
        steps_per_epoch = math.ceil(num / denom)
        return steps_per_epoch * max(1, cfg.epochs)

    def run_training(self, cfg: TrainingConfig) -> str:
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

    def cancel(self, job_id: str) -> bool:
        return self.queue.cancel(job_id)
