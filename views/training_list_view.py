# views/training_list_view.py
import flet as ft
from typing import Callable, Dict

class TrainingListView(ft.Column):
    def __init__(self, on_create_task: Callable[[], None], on_open_task: Callable[[str], None]):
        super().__init__(spacing=10)
        self.on_create_task = on_create_task
        self.on_open_task = on_open_task
        self.tasks: Dict[str, Dict] = {}  # id -> {name, state, progress, eta}
        self.header = ft.Row([
            ft.Text("训练任务", size=20, weight=ft.FontWeight.BOLD),
            ft.Container(expand=True),
            ft.FilledButton("创建训练任务", on_click=lambda e: self.on_create_task()),
        ])
        self.list = ft.Column(spacing=6, scroll=ft.ScrollMode.AUTO)
        self.controls = [self.header, self.list]

    def upsert_task(self, task_id: str, name: str, state: str, progress: float = 0.0, eta: str = ""):
        self.tasks[task_id] = {"name": name, "state": state, "progress": progress, "eta": eta}
        # 重绘列表（简单实现：全量重建；任务多了可做 diff）
        cards = []
        for tid, t in self.tasks.items():
            bar = ft.ProgressBar(value=t["progress"])
            row = ft.ListTile(
                title=ft.Text(t["name"]),
                subtitle=ft.Text(f"{t['state']}  •  ETA {t['eta'] or '--'}"),
                trailing=ft.Container(width=160, content=bar),
                on_click=lambda e, x=tid: self.on_open_task(x),
            )
            cards.append(ft.Card(content=row))
        self.list.controls = cards
        self.update()
