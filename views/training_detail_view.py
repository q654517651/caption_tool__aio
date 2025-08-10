# views/training_detail_view.py
import flet as ft
from typing import Dict, Any, Callable, Optional



class TrainingDetailView(ft.Container):
    def __init__(
            self,
            task_id: str,
            cfg: Dict[str, Any],
            on_cancel: Callable[[], None],
            on_back: [Callable[[], None]] = None,
    ):

        super().__init__(padding=10)
        self.task_id = task_id
        self.cfg = cfg
        self.on_cancel = on_cancel

        self.on_back = on_back
        # 顶部状态条（左侧信息，右侧操作）
        self.title = ft.Text(f"任务：{cfg.get('name', '')}", size=18, weight=ft.FontWeight.BOLD)
        self.state = ft.Text("状态：RUNNING")
        self.step = ft.Text("Step: 0 / 0")
        self.eta = ft.Text("ETA: --")
        self.ips = ft.Text("IPS: --")
        self.actions = ft.Row([
            ft.OutlinedButton("停止", icon=ft.Icons.STOP, on_click=lambda e: self.on_cancel()),
        ])
        back_btn = (
            ft.IconButton(
                icon=ft.Icons.ARROW_BACK,
                tooltip="返回",
                on_click=lambda e: self.on_back() if self.on_back else None,
            )
            if self.on_back
            else None
        )
        header = ft.Row([
            back_btn,
            ft.Column([self.title, ft.Row(
                [self.state, ft.Text(" • "), self.step, ft.Text(" • "), self.eta, ft.Text(" • "), self.ips])]),
            ft.Container(expand=True),
            self.actions
        ])

        # Tab1: 参数
        params = []
        for k, v in cfg.items():
            params.append(ft.Row([ft.Text(k, width=180, color=ft.Colors.GREY), ft.Text(str(v))]))
        tab_params = ft.Container(content=ft.Column(params, scroll=ft.ScrollMode.AUTO), padding=10)

        # Tab2: 训练详情（左：日志；右：GPU / 产物占位）
        self.log = ft.Text("", selectable=True)
        log_view = ft.Container(content=ft.ListView(controls=[self.log], auto_scroll=True, expand=True),
                                bgcolor=ft.Colors.BLACK, padding=8)
        # 右侧占位（后续接入 GPU 监控、样张与 ckpt）
        right = ft.Column([
            ft.Text("GPU 监控（占位）", weight=ft.FontWeight.BOLD),
            ft.Text("温度 / 风扇 / 功耗 / 显存 / 利用率 ..."),
            ft.Divider(),
            ft.Text("产物（占位）", weight=ft.FontWeight.BOLD),
            ft.Text("样张缩略图 / ckpt 列表 / TensorBoard"),
        ])

        tab_detail = ft.Row([ft.Container(expand=2, content=log_view), ft.Container(width=320, content=right)],
                            expand=True)

        self.tabs = ft.Tabs(
            tabs=[
                ft.Tab(text="任务参数", content=tab_params),
                ft.Tab(text="训练详情", content=tab_detail),
            ],
            expand=True
        )

        self.content = ft.Column([header, self.tabs], expand=True)

    # === 供外部更新 ===
    def append_log(self, line: str):
        self.log.value += (line + "\n")
        self.update()

    def update_progress(self, step: int, total_steps: int, ips: float | None, eta_secs: int | None):
        self.step.value = f"Step: {step} / {total_steps}"
        if ips is not None:
            self.ips.value = f"IPS: {ips:.2f}"
        self.eta.value = f"ETA: {self._fmt_eta(eta_secs)}"
        self.update()

    def update_state(self, state: str):
        self.state.value = f"状态：{state}"
        self.update()

    def _fmt_eta(self, s: int | None) -> str:
        if not s or s <= 0: return "--"
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        if h: return f"{h}h{m}m"
        if m: return f"{m}m{sec}s"
        return f"{sec}s"
