"""
Navigation Service - 导航服务
"""

import flet as ft
from typing import Callable

# 定义路径常量
PATH_HOME = "/datasets"
PATH_TRAINING = "/training"
PATH_TASK = "/training/{id}"    # 训练任务详情
PATH_DATASET = "/datasets/{id}"    # 数据集详情

# 索引到路径的映射
INDEX_TO_PATH = {0: PATH_HOME, 1: PATH_TRAINING}

class NavigationService:
    """
    导航服务类，用于管理应用内的页面导航
    """
    def __init__(self, page: ft.Page, render_fn: Callable[[str], None]):
        """
        初始化导航服务
        
        Args:
            page: Flet页面对象
            render_fn: 页面渲染函数
        """
        self.page = page
        self.render_fn = render_fn
        self.history: list[str] = []

        # 监听路由变化
        self.page.on_route_change = self._on_route_change

    def go(self, path: str):
        """
        导航到指定路径
        
        Args:
            path: 目标路径
        """
        self.page.go(path)

    def back(self):
        """
        返回上一页
        """
        # 浏览器/桌面都走统一历史兜底
        if len(self.history) > 1:
            # 弹出当前，再回到前一个
            self.history.pop()
            self.page.go(self.history[-1])
        else:
            # 没有历史就回到首页
            self.page.go(PATH_HOME)

    def _on_route_change(self, e: ft.RouteChangeEvent):
        """
        路由变化事件处理函数
        
        Args:
            e: 路由变化事件
        """
        route = self.page.route or PATH_HOME
        if not self.history or self.history[-1] != route:
            self.history.append(route)
        # 让外部把 content_host.content 换掉
        self.render_fn(route)