# views/datasets_view.py
import flet as ft

class DatasetsView:
    def __init__(self, page, dataset_manager, terminal_service,
                 on_open_dataset, on_delete_dataset):
        self.page = page
        self.dm = dataset_manager
        self.term = terminal_service
        self.on_open_dataset = on_open_dataset        # 回调：打开详情页
        self.on_delete_dataset = on_delete_dataset    # 回调：弹确认删除对话框

        self.list_view = ft.ListView(expand=1, spacing=10, padding=10, auto_scroll=True)
        self.root = None

    # 事件：创建数据集（沿用你原来的“立即创建”语义）
    def _create_dataset_immediately(self, e):
        from datetime import datetime
        name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ok, msg = self.dm.create_dataset(name, "")
        if ok:
            self.term.log_success(f"已创建数据集: {name}")
            self.refresh()
        else:
            self.term.log_error(msg)

    def build(self):
        # 顶部工具条
        toolbar = ft.Container(
            content=ft.Row(
                [
                    ft.ElevatedButton("创建新数据集", icon=ft.Icons.ADD,
                                      on_click=self._create_dataset_immediately),
                    ft.ElevatedButton("刷新列表", icon=ft.Icons.REFRESH,
                                      on_click=lambda e: self.refresh()),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            padding=ft.padding.symmetric(horizontal=20)
        )

        # 标题 + 工具条 + 列表
        self.root = ft.Column(
            [
                ft.Container(
                    content=ft.Text("📊 数据集管理", size=24, weight=ft.FontWeight.BOLD),
                    padding=ft.padding.all(20),
                ),
                toolbar,
                ft.Container(
                    content=self.list_view,
                    expand=True,
                    padding=ft.padding.all(20),
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=10,
                    margin=ft.margin.symmetric(horizontal=20),
                ),
            ],
            expand=True,
        )
        return self.root

    def refresh(self):
        self.list_view.controls.clear()
        datasets = self.dm.list_datasets()

        if not datasets:
            self.list_view.controls.append(
                ft.Text("没有数据集，请创建新数据集", italic=True, color=ft.Colors.GREY_600)
            )
        else:
            for ds in datasets:
                stats = ds.get_stats()
                card = ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.ListTile(
                                leading=ft.Icon(ft.Icons.FOLDER, color=ft.Colors.BLUE),
                                title=ft.Text(ds.name, weight=ft.FontWeight.BOLD),
                                subtitle=ft.Text(f"创建于: {ds.created_time}"),
                                trailing=ft.PopupMenuButton(
                                    icon=ft.Icons.MORE_VERT,
                                    items=[
                                        ft.PopupMenuItem(
                                            text="查看", icon=ft.Icons.VISIBILITY,
                                            on_click=lambda e, d=ds: self.on_open_dataset(d.dataset_id)
                                        ),
                                        ft.PopupMenuItem(
                                            text="删除", icon=ft.Icons.DELETE,
                                            on_click=lambda e, d=ds: self.on_delete_dataset(d.dataset_id)
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
                                        on_click=lambda e, d=ds: self.on_open_dataset(d.dataset_id),
                                    ),
                                ], alignment=ft.MainAxisAlignment.END),
                                padding=ft.padding.only(right=15, bottom=10),
                            ),
                        ]),
                        padding=ft.padding.all(5),
                    ),
                    elevation=2,
                )
                self.list_view.controls.append(card)

        self.page.update()
