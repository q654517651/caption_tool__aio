"""
Terminal Service - 新架构的终端日志服务
"""

import flet as ft
from typing import List, Callable
from datetime import datetime


class TerminalService:
    """终端日志服务"""
    
    def __init__(self):
        self.logs: List[str] = []
        self.callbacks: List[Callable[[str], None]] = []
        self.max_logs = 1000  # 最大日志条数
    
    def log_info(self, message: str):
        """记录信息日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [INFO] {message}"
        self._add_log(log_entry)
    
    def log_error(self, message: str):
        """记录错误日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [ERROR] {message}"
        self._add_log(log_entry)
    
    def log_success(self, message: str):
        """记录成功日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [SUCCESS] {message}"
        self._add_log(log_entry)
    
    def log_warning(self, message: str):
        """记录警告日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [WARN] {message}"
        self._add_log(log_entry)
    
    def log_progress(self, message: str):
        """记录进度日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [PROGRESS] {message}"
        self._add_log(log_entry)
    
    def _add_log(self, log_entry: str):
        """添加日志条目"""
        self.logs.append(log_entry)
        
        # 限制日志数量
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # 通知所有回调
        for callback in self.callbacks:
            try:
                callback(log_entry)
            except Exception:
                # 忽略回调异常，避免影响日志记录
                pass
    
    def register_callback(self, callback: Callable[[str], None]):
        """注册日志回调"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[str], None]):
        """注销日志回调"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def clear_all_callbacks(self):
        """清除所有回调"""
        self.callbacks.clear()
    
    def clear_logs(self):
        """清空日志"""
        self.logs.clear()
        # 发送清空信号
        for callback in self.callbacks:
            try:
                callback("CLEAR_TERMINAL")
            except Exception:
                pass
    
    def get_logs_text(self) -> str:
        """获取所有日志文本"""
        return "\n".join(self.logs)


class TerminalView:
    """终端视图"""
    
    def __init__(self, page: ft.Page, terminal_service: TerminalService):
        self.page = page
        self.terminal_service = terminal_service
        
        # 终端显示区域
        self.terminal_display = ft.TextField(
            value="",
            multiline=True,
            min_lines=30,
            max_lines=50,
            read_only=True,
            bgcolor=ft.Colors.BLACK,
            color=ft.Colors.GREEN,
            text_style=ft.TextStyle(font_family="Courier New", size=12),
            expand=True
        )
        
        self.root_container = None
        self._build_ui()
        self._setup_callbacks()
    
    def _build_ui(self):
        """构建UI"""
        # 清空按钮
        clear_btn = ft.ElevatedButton(
            text="清空终端",
            icon=ft.Icons.CLEAR,
            on_click=lambda e: self.terminal_service.clear_logs()
        )
        
        # 测试按钮组
        test_buttons = ft.Row([
            ft.ElevatedButton(
                text="测试信息",
                on_click=lambda e: self.terminal_service.log_info("这是一条测试信息")
            ),
            ft.ElevatedButton(
                text="测试错误",
                on_click=lambda e: self.terminal_service.log_error("这是一条测试错误")
            ),
            ft.ElevatedButton(
                text="测试成功",
                on_click=lambda e: self.terminal_service.log_success("这是一条测试成功消息")
            )
        ])
        
        # 主容器
        self.root_container = ft.Column([
            ft.Container(
                content=ft.Text("💻 系统终端", size=24, weight=ft.FontWeight.BOLD),
                padding=ft.padding.all(20)
            ),
            ft.Container(
                content=ft.Row([
                    clear_btn,
                    ft.Text("  |  测试日志:"),
                    test_buttons
                ]),
                padding=ft.padding.symmetric(horizontal=20)
            ),
            ft.Container(
                content=ft.Container(
                    content=self.terminal_display,
                    expand=True,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=ft.border_radius.all(10),
                    padding=ft.padding.all(10)
                ),
                expand=True,
                padding=ft.padding.all(20)
            )
        ], expand=True)
    
    def _setup_callbacks(self):
        """设置日志回调"""
        # 加载现有日志
        existing_logs = self.terminal_service.get_logs_text()
        if existing_logs:
            self.terminal_display.value = existing_logs
        
        # 清除所有旧回调函数（关键步骤！）
        self.terminal_service.clear_all_callbacks()
        
        # 注册终端更新回调
        def terminal_callback(log_entry: str):
            if log_entry == "CLEAR_TERMINAL":
                self.terminal_display.value = ""
            else:
                # 添加颜色标记
                if "[ERROR]" in log_entry:
                    self.terminal_display.value += f"🔴 {log_entry}\n"
                elif "[SUCCESS]" in log_entry:
                    self.terminal_display.value += f"🟢 {log_entry}\n"
                elif "[WARN]" in log_entry:
                    self.terminal_display.value += f"🟡 {log_entry}\n"
                elif "[PROGRESS]" in log_entry:
                    self.terminal_display.value += f"🔵 {log_entry}\n"
                else:
                    self.terminal_display.value += f"⚪ {log_entry}\n"
            
            # 自动滚动到底部
            self.terminal_display.selection = ft.TextSelection(
                base_offset=len(self.terminal_display.value),
                extent_offset=len(self.terminal_display.value)
            )
            
            # 限制显示的行数，避免过多内容
            lines = self.terminal_display.value.split('\n')
            if len(lines) > 1000:
                self.terminal_display.value = '\n'.join(lines[-1000:])
            
            if self.page:
                self.page.update()
        
        self.terminal_service.register_callback(terminal_callback)
        
        # 记录终端视图已打开
        self.terminal_service.log_info("终端视图已打开")
    
    def build(self) -> ft.Container:
        """构建并返回根容器"""
        return self.root_container