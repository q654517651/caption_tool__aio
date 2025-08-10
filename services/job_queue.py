# services/job_queue.py
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from .event_bus import EventBus

@dataclass
class Job:
    id: str
    name: str
    run: Callable[[Callable[[str], None], Callable[[dict], None]], int]
    # run(log_cb(line), progress_cb(dict)) -> returncode
    cancel: Callable[[], None]

class JobQueue:
    def __init__(self, bus: EventBus, max_workers: int = 1) -> None:
        self.bus = bus
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def submit(self, job: Job) -> str:
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
            except Exception:
                self.bus.emit("task_state", {"id": job.id, "state": "FAILED", "rc": -1})

        self.pool.submit(_runner)
        return job.id

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        try:
            job.cancel()
            self.bus.emit("task_state", {"id": job_id, "state": "CANCELED"})
            return True
        except Exception:
            return False
