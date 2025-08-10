# views/dialogs/delete_confirm_dialog.py
import flet as ft
from typing import Callable, Optional

class DeleteConfirmDialog:
    """
    删除确认弹窗（危险样式）：
      dlg = DeleteConfirmDialog(page, name, on_confirm=lambda: delete_fn(id)).open()
    """
    def __init__(
        self,
        page: ft.Page,
        item_name: str,
        on_confirm: Optional[Callable[[], None]] = None,
        extra_hint: str = "此操作不可撤销。",
        title: str = "删除确认",
        confirm_text: str = "删除",
        cancel_text: str = "取消",
    ):
        self.page = page
        self.on_confirm = on_confirm

        content = ft.Column(
            spacing=8,
            tight=True,
            controls=[
                ft.Text(f"确定要删除「{item_name}」吗？"),
                ft.Text(extra_hint, size=12, color=ft.Colors.RED_400),
            ],
        )

        self._dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.WARNING_AMBER, color=ft.Colors.RED_400),
                    ft.Text(title),
                ],
                spacing=8,
            ),
            content=content,
            actions_alignment=ft.MainAxisAlignment.END,
        )

        # 危险按钮 + 次要按钮
        self._dialog.actions = [
            ft.TextButton(cancel_text, on_click=lambda e: self.close()),
            ft.FilledButton(
                confirm_text,
                icon=ft.Icons.DELETE_FOREVER,
                style=ft.ButtonStyle(
                    bgcolor=ft.Colors.RED_500,
                    color=ft.Colors.WHITE,
                    overlay_color=ft.Colors.RED_700,
                ),
                on_click=self._handle_confirm,
            ),
        ]

    def _handle_confirm(self, e):
        self.close()
        if callable(self.on_confirm):
            self.on_confirm()

    def open(self):
        self.page.open(self._dialog)

    def close(self):
        self.page.close(self._dialog)
        self.page.update()
