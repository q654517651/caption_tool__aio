# views/dataset_detail_view.py
import flet as ft
from typing import Callable, Optional, Dict, Any


class DatasetDetailView(ft.Column):
    """
    数据集详情视图（Flet 新 API）
    约定：
      - on_render_images: (dataset_id: str, grid: ft.GridView) -> None
        回调内负责向 grid.controls 追加图片卡片并调用 page.update()。
    """
    def __init__(
        self,
        page: ft.Page,
        dataset_id: str,
        dataset_name: str,
        on_back: Callable[[], None],
        on_batch_translate: Optional[Callable[[str], None]] = None,
        on_batch_label: Optional[Callable[[str], None]] = None,
        on_import_files: Optional[Callable[[str], None]] = None,
        on_render_images: Optional[Callable[[str, ft.GridView], None]] = None,
        *,
        style: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(expand=True, spacing=0, horizontal_alignment=ft.CrossAxisAlignment.STRETCH,)
        self.page = page
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name

        self.on_back = on_back
        self.on_batch_translate = on_batch_translate
        self.on_batch_label = on_batch_label
        self.on_import_files = on_import_files
        self.on_render_images = on_render_images

        # 存储选中的图片文件名
        self.selected_images = set()
        self.all_images = []

        self._title_text = ft.Text(f"数据集: {self.dataset_name}", size=24, weight=ft.FontWeight.BOLD)

        self.image_grid = ft.GridView(
            runs_count=0, 
            max_extent=350, 
            child_aspect_ratio=0.75,
            spacing=15, 
            run_spacing=15, 
            padding=ft.padding.all(20),
            expand=True,
        )

        back_btn = ft.IconButton(icon=ft.Icons.ARROW_BACK, tooltip="返回数据集列表",
                                 on_click=lambda e: self.on_back() if self.on_back else None)

        self.select_all_btn = ft.IconButton(
            icon=ft.Icons.SELECT_ALL,
            tooltip="全选",
            on_click=self.select_all_images
        )
        
        self.clear_selection_btn = ft.IconButton(
            icon=ft.Icons.CLEAR_ALL,
            tooltip="清空选择",
            on_click=self.clear_selection
        )

        header = ft.Container(
            content=ft.Row([
                back_btn, 
                self._title_text,
                ft.Container(expand=True),
                self.clear_selection_btn,
                self.select_all_btn
            ], alignment=ft.MainAxisAlignment.START),
            padding=ft.padding.all(20),
        )

        batch_buttons = ft.Row(
            controls=[
                ft.ElevatedButton("批量翻译", icon=ft.Icons.TRANSLATE,
                                  on_click=lambda e: self.on_batch_translate and self.on_batch_translate(self.dataset_id)),
                ft.ElevatedButton("批量打标", icon=ft.Icons.AUTO_AWESOME,
                                  on_click=lambda e: self.on_batch_label and self.on_batch_label(self.dataset_id)),
                ft.ElevatedButton("导入文件", icon=ft.Icons.UPLOAD_FILE,
                                  on_click=lambda e: self.on_import_files and self.on_import_files(self.dataset_id)),
            ],
            spacing=10,
        )
        batch_bar = ft.Container(content=batch_buttons, padding=ft.padding.symmetric(horizontal=20))

        self.grid_wrap = ft.Container(
            content=self.image_grid,
            bgcolor=ft.Colors.GREY_50,
            border_radius=10,
            expand=True,
        )

        if style:
            for k, v in style.items():
                setattr(self, k, v)

        self.controls.extend([header, batch_bar, self.grid_wrap])

        self._mounted = False
        self._loading = False

    def set_dataset(self, dataset_id: str, dataset_name: str):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self._title_text.value = f"数据集: {dataset_name}"
        self.page.update()
        self.refresh_images()

    def refresh_images(self):
        """清空并让回调渲染图片和标签"""
        if self._loading:
            return
        self._loading = True

        self.image_grid.controls.clear()
        self.image_grid.controls.append(
            ft.Row([ft.ProgressRing()], alignment=ft.MainAxisAlignment.CENTER)
        )
        self.page.update()

        if not self.on_render_images:
            self.image_grid.controls.clear()
            self.image_grid.controls.append(
                ft.Container(content=ft.Text("on_render_images 未接入", size=12, color=ft.Colors.GREY_600), padding=20)
            )
            self.page.update()
            self._loading = False
            return

        def worker():
            try:
                # 让外部回调把 grid 填满（包括标签编辑框）
                self.image_grid.controls.clear()
                self.on_render_images(self.dataset_id, self.image_grid)
            except Exception as ex:
                self.image_grid.controls.clear()
                self.image_grid.controls.append(
                    ft.Container(content=ft.Text(f"加载失败：{ex}", size=12, color=ft.Colors.RED_400), padding=20)
                )
                self.page.update()
            finally:
                self._loading = False

        # 同步调用更直观；如需异步可改：self.page.run_thread(worker)
        worker()

    def did_mount(self):
        self._mounted = True
        self.refresh_images()

    def will_unmount(self):
        self._mounted = False

    def toggle_image_selection(self, filename: str):
        """切换图片选择状态"""
        if filename in self.selected_images:
            self.selected_images.remove(filename)
        else:
            self.selected_images.add(filename)
        self.update_selection_ui()

    def select_all_images(self, e):
        """选择所有图片"""
        self.selected_images = set(self.all_images)
        self.update_selection_ui()
        self.page.update()

    def clear_selection(self, e):
        """清空选择"""
        self.selected_images.clear()
        self.update_selection_ui()
        self.page.update()

    def update_selection_ui(self):
        """更新选择相关的UI状态"""
        # 遍历图像网格中的控件并更新选中状态
        if hasattr(self, 'image_grid') and self.image_grid and self.image_grid.controls:
            for card in self.image_grid.controls:
                if isinstance(card, ft.Container) and isinstance(card.content, ft.Column):
                    # 优先用 card.data
                    filename_from_card = getattr(card, "data", None)

                    if not filename_from_card:
                        # 兼容回退：从第二行文本里推断
                        column = card.content
                        if len(column.controls) >= 2:
                            filename_container = column.controls[1]
                            if isinstance(filename_container, ft.Container) and isinstance(filename_container.content,
                                                                                           ft.Text):
                                filename_text = filename_container.content.value
                                for fname in self.all_images:
                                    if (filename_text.endswith("...") and fname.startswith(
                                            filename_text[:-3])) or fname == filename_text:
                                        filename_from_card = fname
                                        break

                    # 更新样式
                    if filename_from_card and filename_from_card in self.selected_images:
                        card.bgcolor = ft.Colors.BLUE_50
                        card.border = ft.border.all(3, ft.Colors.BLUE)
                    else:
                        card.bgcolor = ft.Colors.WHITE
                        card.border = None

            self.page.update()

