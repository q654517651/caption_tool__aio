"""
Datasets View - 新架构的数据集管理视图
"""

import flet as ft
from datetime import datetime
from typing import Callable, List, Optional

class DatasetsView:
    """数据集管理视图"""
    
    def __init__(self, 
                 page: ft.Page,
                 dataset_manager,
                 on_open_dataset: Callable[[str], None],
                 on_delete_dataset: Callable[[str], None],
                 toast_service):
        self.page = page
        self.dataset_manager = dataset_manager
        self.on_open_dataset = on_open_dataset
        self.on_delete_dataset = on_delete_dataset
        self.toast_service = toast_service
        
        # UI组件
        self.dataset_list = ft.ListView(
            expand=True,
            spacing=10,
            padding=20,
            auto_scroll=True
        )
        
        self.root_container = None
        self._build_ui()
    
    def _build_ui(self):
        """构建UI"""
        # 顶部工具栏
        toolbar = ft.Container(
            content=ft.Row([
                ft.ElevatedButton(
                    "创建新数据集",
                    icon=ft.Icons.ADD,
                    on_click=self._create_dataset
                ),
                ft.ElevatedButton(
                    "刷新列表",
                    icon=ft.Icons.REFRESH,
                    on_click=lambda e: self.refresh()
                ),
            ], alignment=ft.MainAxisAlignment.START),
            padding=ft.padding.symmetric(horizontal=20, vertical=10)
        )
        
        # 主容器
        self.root_container = ft.Column([
            ft.Container(
                content=ft.Text("📊 数据集管理", size=24, weight=ft.FontWeight.BOLD),
                padding=ft.padding.all(20)
            ),
            toolbar,
            ft.Container(
                content=self.dataset_list,
                expand=True,
                padding=ft.padding.all(20),
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=10,
                margin=ft.margin.symmetric(horizontal=20),
            )
        ], expand=True)
    
    def _create_dataset(self, e):
        """创建数据集"""
        # 创建输入对话框
        name_field = ft.TextField(
            label="数据集名称",
            value=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            width=300
        )
        description_field = ft.TextField(
            label="描述（可选）",
            multiline=True,
            width=300,
            height=100
        )
        
        def create(e):
            name = name_field.value.strip()
            description = description_field.value.strip()
            
            if not name:
                self.toast_service.show("请输入数据集名称", "warning")
                return
            
            success, message = self.dataset_manager.create_dataset(name, description)
            dialog.open = False
            self.page.update()
            
            if success:
                self.toast_service.show(f"数据集创建成功: {name}", "success")
                self.refresh()
            else:
                self.toast_service.show(f"创建失败: {message}", "error")
        
        def cancel(e):
            dialog.open = False
            self.page.update()
        
        dialog = ft.AlertDialog(
            title=ft.Text("创建新数据集"),
            content=ft.Container(
                content=ft.Column([
                    name_field,
                    description_field
                ], spacing=10),
                width=350,
                height=200
            ),
            actions=[
                ft.TextButton("取消", on_click=cancel),
                ft.ElevatedButton("创建", on_click=create)
            ]
        )
        
        self.page.open(dialog)
    
    def _create_dataset_item(self, dataset) -> ft.Card:
        """创建数据集列表项"""
        stats = dataset.get_stats()
        
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.ListTile(
                        leading=ft.Icon(ft.Icons.FOLDER, color=ft.Colors.BLUE),
                        title=ft.Text(dataset.name, weight=ft.FontWeight.BOLD),
                        subtitle=ft.Text(f"创建于: {dataset.created_time}"),
                        trailing=ft.PopupMenuButton(
                            icon=ft.Icons.MORE_VERT,
                            items=[
                                ft.PopupMenuItem(
                                    text="查看", 
                                    icon=ft.Icons.VISIBILITY,
                                    on_click=lambda e, d=dataset: self.on_open_dataset(d.dataset_id)
                                ),
                                ft.PopupMenuItem(
                                    text="删除", 
                                    icon=ft.Icons.DELETE,
                                    on_click=lambda e, d=dataset: self.on_delete_dataset(d.dataset_id)
                                ),
                            ],
                        ),
                    ),
                    ft.Container(
                        content=ft.Row([
                            ft.Text(f"图片数量: {stats['total']}"),
                            ft.Text(f"已标注: {stats['labeled']}"),
                            ft.Text(f"完成度: {stats['completion_rate']}%"),
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        padding=ft.padding.symmetric(horizontal=15, vertical=5),
                    ),
                    ft.Container(
                        content=ft.Row([
                            ft.FilledButton(
                                "查看内容",
                                icon=ft.Icons.VISIBILITY,
                                on_click=lambda e, d=dataset: self.on_open_dataset(d.dataset_id),
                            ),
                        ], alignment=ft.MainAxisAlignment.END),
                        padding=ft.padding.only(right=15, bottom=10),
                    ),
                ]),
                padding=ft.padding.all(5),
            ),
            elevation=2,
        )
    
    def refresh(self):
        """刷新数据集列表"""
        try:
            # 清空现有内容
            self.dataset_list.controls.clear()
            
            # 获取并显示数据集
            datasets = self.dataset_manager.list_datasets()
            
            if not datasets:
                self.dataset_list.controls.append(
                    ft.Text("没有数据集，请创建新数据集", italic=True, color=ft.Colors.GREY_600)
                )
            else:
                for dataset in datasets:
                    self.dataset_list.controls.append(self._create_dataset_item(dataset))
            
            if self.page:
                self.page.update()
                
        except Exception as e:
            self.toast_service.show(f"刷新失败: {str(e)}", "error")
    
    def build(self) -> ft.Container:
        """构建并返回根容器"""
        self.refresh()
        return self.root_container
