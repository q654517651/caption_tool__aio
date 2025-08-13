"""
Events and Job Queue - 事件总线和任务队列
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading

class EventBus:
    """
    事件总线，用于组件间解耦通信
    """
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def on(self, topic: str, handler: Callable[[Any], None]) -> None:
        """
        订阅事件
        
        Args:
            topic: 事件主题
            handler: 事件处理函数
        """
        self._subs[topic].append(handler)

    def emit(self, topic: str, payload: Any = None) -> None:
        """
        发布事件
        
        Args:
            topic: 事件主题
            payload: 事件负载数据
        """
        for h in list(self._subs.get(topic, [])):
            try:
                h(payload)
            except Exception:
                # 不让单个订阅者的异常拖垮总线
                pass

@dataclass
class Job:
    """
    任务定义
    """
    id: str
    name: str
    run: Callable[[Callable[[str], None], Callable[[dict], None]], int]
    # run(log_cb(line), progress_cb(dict)) -> returncode
    cancel: Callable[[], None]

class JobQueue:
    """
    任务队列，用于异步执行任务
    """
    def __init__(self, bus: EventBus, max_workers: int = 1) -> None:
        """
        初始化任务队列
        
        Args:
            bus: 事件总线实例
            max_workers: 最大并发工作线程数
        """
        self.bus = bus
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def submit(self, job: Job) -> str:
        """
        提交任务到队列
        
        Args:
            job: 任务对象
            
        Returns:
            str: 任务ID
        """
        with self._lock:
            self._jobs[job.id] = job
        self.bus.emit("task_state", {"id": job.id, "state": "QUEUED", "name": job.name})

        def _runner():
            self.bus.emit("task_state", {"id": job.id, "state": "RUNNING", "name": job.name})
            try:
                rc = job.run(
                    lambda line: self.bus.emit("task_log", {"id": job.id, "line": line}),
                    lambda prog: self.bus.emit("task_progress", {"id": job.id, **prog}),
                )
                self.bus.emit("task_state", {"id": job.id, "state": "COMPLETED" if rc == 0 else "FAILED", "rc": rc})
            except Exception as e:
                self.bus.emit("task_state", {"id": job.id, "state": "FAILED", "error": str(e)})

        self.pool.submit(_runner)
        return job.id

    def cancel(self, job_id: str) -> bool:
        """
        取消任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        try:
            job.cancel()
            self.bus.emit("task_state", {"id": job_id, "state": "CANCELED"})
            return True
        except Exception:
            return False