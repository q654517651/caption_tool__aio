import flet as ft
from typing import Callable

PATH_HOME = "/datasets"
PATH_TRAINING = "/training"
PATH_TASK = "/training/{id}"    # 详情
PATH_DATASET = "/datasets/{id}"    # 可选

INDEX_TO_PATH = {0: PATH_HOME, 1: PATH_TRAINING}

class NavigationService:
    def __init__(self, page: ft.Page, render_fn: Callable[[str], None]):
        self.page = page
        self.render_fn = render_fn
        self.history: list[str] = []

        # 监听路由变化
        self.page.on_route_change = self._on_route_change

    def go(self, path: str):
        self.page.go(path)

    def back(self):
        # 浏览器/桌面都走统一历史兜底
        if len(self.history) > 1:
            # 弹出当前，再回到前一个
            self.history.pop()
            self.page.go(self.history[-1])
        else:
            # 没有历史就回到首页
            self.page.go(PATH_HOME)

    def _on_route_change(self, e: ft.RouteChangeEvent):
        route = self.page.route or PATH_HOME
        if not self.history or self.history[-1] != route:
            self.history.append(route)
        # 让外部把 content_host.content 换掉
        self.render_fn(route)
