#!/usr/bin/env python3

import threading
from datetime import datetime
from typing import List, Callable
import queue


class TerminalService:
    """终端服务，统一管理所有系统日志"""

    def __init__(self):
        self.logs = []
        self.max_logs = 1000  # 最大日志条数
        self.callbacks = []  # UI更新回调函数
        self.log_queue = queue.Queue()
        self.lock = threading.Lock()

        # 启动日志处理线程
        self.log_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.log_thread.start()

    def register_callback(self, callback: Callable[[str], None]):
        """注册UI更新回调函数"""
        with self.lock:
            self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[str], None]):
        """取消注册UI更新回调函数"""
        with self.lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    def clear_all_callbacks(self):
        """清除所有回调函数"""
        with self.lock:
            self.callbacks.clear()

    def _process_logs(self):
        """处理日志队列的后台线程"""
        while True:
            try:
                log_entry = self.log_queue.get(timeout=1)
                self._add_log_internal(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"日志处理错误: {e}")

    def _add_log_internal(self, log_entry: str):
        """内部添加日志方法"""
        with self.lock:
            # 添加时间戳
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_log = f"[{timestamp}] {log_entry}"

            self.logs.append(formatted_log)

            # 限制日志数量
            if len(self.logs) > self.max_logs:
                self.logs.pop(0)

            # 通知所有回调函数
            for callback in self.callbacks:
                try:
                    callback(formatted_log)
                except Exception as e:
                    print(f"回调函数执行错误: {e}")

    def log_info(self, message: str):
        """记录信息日志"""
        self.log_queue.put(f"[INFO] {message}")

    def log_error(self, message: str):
        """记录错误日志"""
        self.log_queue.put(f"[ERROR] {message}")

    def log_warning(self, message: str):
        """记录警告日志"""
        self.log_queue.put(f"[WARN] {message}")

    def log_success(self, message: str):
        """记录成功日志"""
        self.log_queue.put(f"[SUCCESS] {message}")

    def log_progress(self, current: int, total: int, message: str):
        """记录进度日志"""
        percentage = round(current / total * 100) if total > 0 else 0
        progress_bar = "█" * (percentage // 5) + "░" * (20 - percentage // 5)
        self.log_queue.put(f"[PROGRESS] [{progress_bar}] {percentage}% - {message}")

    def clear_logs(self):
        """清空日志"""
        with self.lock:
            self.logs.clear()
            for callback in self.callbacks:
                try:
                    callback("CLEAR_TERMINAL")
                except Exception as e:
                    print(f"回调函数执行错误: {e}")

    def get_all_logs(self) -> List[str]:
        """获取所有日志"""
        with self.lock:
            return self.logs.copy()

    def get_logs_text(self) -> str:
        """获取所有日志的文本形式"""
        with self.lock:
            return "\n".join(self.logs)


# 全局终端服务实例
terminal_service = TerminalService()


# 重写日志函数，统一使用终端服务
def log_info(message):
    """记录信息日志"""
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
    terminal_service.log_info(message)


def log_error(message):
    """记录错误日志"""
    print(f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
    terminal_service.log_error(message)


def log_warning(message):
    """记录警告日志"""
    print(f"[WARN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
    terminal_service.log_warning(message)


def log_success(message):
    """记录成功日志"""
    print(f"[SUCCESS] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
    terminal_service.log_success(message)


def log_progress(current: int, total: int, message: str):
    """记录进度日志"""
    terminal_service.log_progress(current, total, message)
