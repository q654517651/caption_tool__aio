"""
Toast Service - 新架构的通知服务
"""

import asyncio
import threading
from typing import Optional
import flet as ft

_KIND_BG = {
    "info": ft.Colors.BLUE_600,
    "success": ft.Colors.GREEN_600,
    "warning": ft.Colors.AMBER_600,
    "error": ft.Colors.RED_600,
}
_KIND_FG = {"warning": ft.Colors.BLACK}


class ToastService:
    """
    PC 端右上角叠放 toast：
    - 进入：右→左滑入（easeOutBack，微弹性）+ 淡入
    - 退出：先淡出，再可见性塌缩（animate_size），其余项自然上移
    - 悬停暂停、最多 3 条、任意线程可调用
    """
    def __init__(self, page: ft.Page, max_on_screen: int = 3):
        self.page = page
        self.max_on_screen = max_on_screen
        self._lock = threading.Lock()

        # 右上角宿主（不要 expand，避免遮挡全页）
        self._host = ft.Column(
            spacing=8,
            controls=[],
            horizontal_alignment=ft.CrossAxisAlignment.END,
            alignment=ft.MainAxisAlignment.START,
        )
        # 贴右上的轻量 overlay 容器
        self._host_row = ft.Row(
            controls=[ft.Container(content=self._host, padding=10)],
            alignment=ft.MainAxisAlignment.END,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        if self._host_row not in self.page.overlay:
            self.page.overlay.append(self._host_row)
            self.page.update()

    def show(
        self,
        message: str,
        kind: str = "info",
        duration: int = 2000,
        action_text: Optional[str] = None,
        on_action=None,
    ) -> None:

        def _build_once():
            # 超量则移除最旧
            while len(self._host.controls) >= self.max_on_screen:
                self._host.controls.pop(0)

            fg = _KIND_FG.get(kind, ft.Colors.WHITE)
            bg = _KIND_BG.get(kind, ft.Colors.BLUE_600)

            text = ft.Text(
                message, color=fg, size=14, max_lines=3,
                overflow=ft.TextOverflow.ELLIPSIS
            )
            close_btn = ft.IconButton(icon=ft.Icons.CLOSE, icon_size=16,
                                      style=ft.ButtonStyle(color=fg))
            act_btn = (
                ft.TextButton(
                    text=action_text,
                    on_click=lambda e: on_action() if on_action else None,
                    style=ft.ButtonStyle(color=fg),
                ) if action_text else None
            )

            row_items = []
            if act_btn:
                row_items.append(act_btn)
            row_items.append(text)
            row_items.append(close_btn)

            card = ft.Container(
                content=ft.Row(row_items, alignment=ft.MainAxisAlignment.END, spacing=8),
                bgcolor=bg,
                padding=12,
                border_radius=8,
                # 初始：右侧偏移 + 透明
                opacity=0.0,
                offset=ft.Offset(0.35, 0),
                # 动画
                animate_opacity=ft.Animation(220, curve=ft.AnimationCurve.DECELERATE),
                animate_offset=ft.Animation(380, curve=getattr(ft.AnimationCurve, "EASE_OUT_BACK",
                                                               ft.AnimationCurve.EASE_OUT)),
                animate_size=ft.Animation(220, curve=ft.AnimationCurve.DECELERATE),

                visible=True,  # 退场时会切到 False 产生塌缩动画
            )

            # 悬停暂停
            paused = {"v": False}
            card.on_hover = lambda e: paused.update(v=(e.data == "true"))

            # 关闭行为：先淡出 + 塌缩，再移除
            async def _close_with_anim():
                # 淡出内容卡片
                card.opacity = 0.0
                self.page.update()
                await asyncio.sleep(0.22)  # 匹配 animate_opacity=220ms

                # 让外层 slot 做"高度塌缩"，其余项自然上移
                slot.height = 0
                slot.padding = ft.padding.all(0)
                self.page.update()
                await asyncio.sleep(0.24)  # 匹配 animate_size=220ms，略留余量

                # 移除 slot（而不是 card）
                if slot in self._host.controls:
                    self._host.controls.remove(slot)
                    self.page.update()

            # 关闭按钮点击：触发动画关闭
            close_btn.on_click = lambda e: self.page.run_task(_close_with_anim)

            # 外层 slot：专门用来做"塌缩（上移）动画"
            slot = ft.Container(
                content=card,
                padding=ft.padding.only(bottom=0),  # 可按需给每条增加下间距
                animate_size=ft.Animation(220, curve=ft.AnimationCurve.DECELERATE),
                # 注意：不要给 slot 设置 expand/width，保持自适应内容
            )

            # 放入宿主并渲染首帧
            self._host.controls.append(slot)
            self.page.update()

            async def _play_in_and_auto_close():
                # 让首帧真正渲染，再触发位移/透明度动画
                await asyncio.sleep(0.016)  # 一帧更稳
                card.opacity = 1.0
                card.offset = ft.Offset(0, 0)
                self.page.update()

                # 自动关闭（错误且 duration<0 则不自动关）
                if kind == "error" and duration < 0:
                    return
                remain = max(0, duration) / 1000.0
                while remain > 0:
                    await asyncio.sleep(0.1)
                    if not paused["v"]:
                        remain -= 0.1
                await _close_with_anim()  # 走下方关闭流程

            self.page.run_task(_play_in_and_auto_close)

        # 切回 UI 线程执行（需要 Awaitable）
        with self._lock:
            async def _ui_async():
                _build_once()
            self.page.run_task(_ui_async)