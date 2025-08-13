"""
Dataset Detail View - 新架构的数据集详情视图
"""

import flet as ft
import os
from typing import Callable, Set
from pathlib import Path

class DatasetDetailView:
    """数据集详情视图"""
    
    def __init__(self,
                 page: ft.Page,
                 dataset_id: str,
                 dataset_name: str,
                 dataset_manager,
                 labeling_service,
                 on_back: Callable[[], None],
                 toast_service):
        self.page = page
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.dataset_manager = dataset_manager
        self.labeling_service = labeling_service
        self.on_back = on_back
        self.toast_service = toast_service
        
        # 选择状态
        self.selected_images: Set[str] = set()
        self.all_images = []
        
        # UI组件
        self.image_grid = ft.GridView(
            expand=True,
            runs_count=0,
            max_extent=300,
            child_aspect_ratio=0.8,
            spacing=10,
            run_spacing=10,
            padding=10
        )
        
        self.selection_info = ft.Text("未选择图片")
        self.root_container = None
        self._build_ui()
    
    def _build_ui(self):
        """构建UI"""
        # 顶部工具栏
        toolbar = ft.Container(
            content=ft.Row([
                ft.IconButton(
                    icon=ft.Icons.ARROW_BACK,
                    tooltip="返回",
                    on_click=lambda e: self.on_back()
                ),
                ft.Text(
                    f"数据集: {self.dataset_name}",
                    size=20,
                    weight=ft.FontWeight.BOLD
                ),
                ft.Container(expand=True),  # 占位符，推动右侧按钮
                ft.ElevatedButton(
                    "导入文件",
                    icon=ft.Icons.UPLOAD_FILE,
                    on_click=self._import_files
                ),
                ft.ElevatedButton(
                    "批量打标",
                    icon=ft.Icons.AUTO_AWESOME,
                    on_click=self._batch_label,
                    disabled=True  # 初始禁用，选择图片后启用
                ),
            ]),
            padding=ft.padding.all(20)
        )
        
        # 选择工具栏
        selection_toolbar = ft.Container(
            content=ft.Row([
                self.selection_info,
                ft.Container(expand=True),
                ft.TextButton(
                    "全选",
                    icon=ft.Icons.SELECT_ALL,
                    on_click=self._select_all
                ),
                ft.TextButton(
                    "清空选择",
                    icon=ft.Icons.DESELECT,
                    on_click=self._clear_selection
                )
            ]),
            padding=ft.padding.symmetric(horizontal=20, vertical=10),
            bgcolor=ft.Colors.GREY_100,
            border_radius=5
        )
        
        # 主容器
        self.root_container = ft.Column([
            toolbar,
            selection_toolbar,
            ft.Container(
                content=self.image_grid,
                expand=True,
                padding=ft.padding.all(10)
            )
        ], expand=True)
    
    def _import_files(self, e):
        """导入文件"""
        try:
            # 创建文件选择器
            file_picker = ft.FilePicker()
            
            def on_file_result(e):
                if e.files:
                    file_paths = [f.path for f in e.files]
                    success_count, message = self.dataset_manager.import_images_to_dataset(
                        self.dataset_id, file_paths
                    )
                    
                    if success_count > 0:
                        self.toast_service.show(message, "success")
                        self.refresh_images()
                    else:
                        self.toast_service.show(message, "error")
                
                # 移除文件选择器
                self.page.overlay.remove(file_picker)
                self.page.update()
            
            file_picker.on_result = on_file_result
            self.page.overlay.append(file_picker)
            self.page.update()
            
            # 打开文件选择对话框
            file_picker.pick_files(
                allow_multiple=True,
                allowed_extensions=["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "txt"]
            )
            
        except Exception as ex:
            self.toast_service.show(f"文件导入失败: {str(ex)}", "error")
    
    def _batch_label(self, e):
        """批量打标选中的图片"""
        if not self.selected_images:
            self.toast_service.show("请先选择要打标的图片", "warning")
            return
        
        try:
            # 获取数据集
            dataset = self.dataset_manager.get_dataset(self.dataset_id)
            if not dataset:
                self.toast_service.show("数据集不存在", "error")
                return
            
            # 准备图片路径
            image_paths = []
            for filename in self.selected_images:
                image_path = self.dataset_manager.get_dataset_image_path(
                    self.dataset_id, filename, "original"
                )
                if image_path:
                    image_paths.append(image_path)
            
            if not image_paths:
                self.toast_service.show("没有找到选中的图片", "error")
                return
            
            # 开始打标
            self.toast_service.show(f"开始打标 {len(image_paths)} 张图片...", "info")
            
            # 这里简化实现，实际应该是异步处理
            # TODO: 实现异步打标逻辑
            self.toast_service.show("打标功能开发中...", "info")
            
        except Exception as ex:
            self.toast_service.show(f"打标失败: {str(ex)}", "error")
    
    def _select_all(self, e):
        """全选图片"""
        self.selected_images = set(self.all_images)
        self._update_selection_ui()
        self.refresh_images()
    
    def _clear_selection(self, e):
        """清空选择"""
        self.selected_images.clear()
        self._update_selection_ui()
        self.refresh_images()
    
    def _update_selection_ui(self):
        """更新选择状态显示"""
        selected_count = len(self.selected_images)
        total_count = len(self.all_images)
        
        if selected_count == 0:
            self.selection_info.value = "未选择图片"
        else:
            self.selection_info.value = f"已选择 {selected_count}/{total_count} 张图片"
        
        # 更新批量打标按钮状态
        for control in self.root_container.controls[0].content.controls:
            if isinstance(control, ft.ElevatedButton) and "批量打标" in control.text:
                control.disabled = selected_count == 0
                break
        
        if self.page:
            self.page.update()
    
    def toggle_image_selection(self, filename: str):
        """切换图片选择状态"""
        if filename in self.selected_images:
            self.selected_images.remove(filename)
        else:
            self.selected_images.add(filename)
        
        self._update_selection_ui()
        self.refresh_images()
    
    def _create_image_card(self, filename: str, label: str) -> ft.Container:
        """创建图片卡片"""
        # 获取图片路径
        image_info = self.dataset_manager.resolve_image_src(
            self.dataset_id, filename, "original"
        )
        raw_path = image_info.get("src") if image_info else None

        # 规范化为可用的 URL
        def _to_image_uri(p: str | None) -> str | None:
            if not p:
                return None
            p = str(p)
            # 已是 URL/URI 的直接用
            if p.startswith(("http://", "https://", "file://", "data:")):
                # HTTP 资源做一次 cache-bust
                if p.startswith(("http://", "https://")):
                    try:
                        # 如果 image_info 里还带了本地文件真实路径，可用它取 mtime；否则仅附时间戳
                        local = image_info.get("local") if isinstance(image_info, dict) else None
                        ts = int(os.path.getmtime(local)) if local and os.path.exists(local) else int(
                            os.path.getmtime(p))
                    except Exception:
                        from time import time as _now
                        ts = int(_now())
                    sep = "&" if "?" in p else "?"
                    return f"{p}{sep}v={ts}"
                return p
            # 不是 URL，则视为本地文件路径：转 file:// URI（兼容 Windows 反斜杠）
            try:
                uri = Path(p).as_uri()  # -> file:///D:/...
            except Exception:
                return None
            return uri

        image_uri = _to_image_uri(raw_path)

        # 检查选择状态
        is_selected = filename in self.selected_images

        # 创建图片组件
        if image_uri:
            image_widget = ft.Image(
                src=image_uri,
                fit=ft.ImageFit.CONTAIN,
                error_content=ft.Container(
                    content=ft.Icon(ft.Icons.BROKEN_IMAGE, size=50, color=ft.Colors.GREY),
                    alignment=ft.alignment.center,
                    bgcolor=ft.Colors.GREY_100
                )
            )
        else:
            image_widget = ft.Container(
                content=ft.Icon(ft.Icons.BROKEN_IMAGE, size=50, color=ft.Colors.GREY),
                alignment=ft.alignment.center,
                bgcolor=ft.Colors.GREY_100,
                height=150
            )

        # 标签输入框
        label_field = ft.TextField(
            value=label,
            multiline=True,
            text_size=12,
            border=ft.InputBorder.NONE,
            filled=True,
            fill_color=ft.Colors.GREY_100,
            on_change=lambda e: self._update_label(filename, e.control.value),
            expand=True,
        )
        
        # 文件名显示
        display_name = filename[:25] + "..." if len(filename) > 25 else filename
        
        # 图片卡片
        card = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=image_widget,
                    height=150,
                    clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                    border_radius=5
                ),
                ft.Container(
                    content=ft.Text(
                        display_name,
                        size=12,
                        weight=ft.FontWeight.W_500,
                        max_lines=2,
                        overflow=ft.TextOverflow.ELLIPSIS
                    ),
                    padding=ft.padding.symmetric(horizontal=8, vertical=5)
                ),
                ft.Container(
                    content=label_field,
                    padding=ft.padding.symmetric(horizontal=8, vertical=5),
                    expand=True
                )
            ], spacing=0, expand=True),
            bgcolor=ft.Colors.BLUE_50 if is_selected else ft.Colors.WHITE,
            border_radius=8,
            border=ft.border.all(3, ft.Colors.BLUE) if is_selected else ft.border.all(1, ft.Colors.GREY_300),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=4,
                color=ft.Colors.BLACK12,
                offset=ft.Offset(0, 2)
            ),
            on_click=lambda e: self.toggle_image_selection(filename),
            data=filename,
            width=280,
            height=250
        )
        
        return card
    
    def _update_label(self, filename: str, label: str):
        """更新图片标签"""
        success = self.dataset_manager.update_dataset_label(
            self.dataset_id, filename, label
        )
        if not success:
            self.toast_service.show("标签保存失败", "error")
    
    def refresh_images(self):
        """刷新图片显示"""
        try:
            # 清空现有内容
            self.image_grid.controls.clear()
            
            # 获取数据集
            dataset = self.dataset_manager.get_dataset(self.dataset_id)
            if not dataset:
                self.image_grid.controls.append(
                    ft.Container(
                        content=ft.Text("数据集不存在", size=16),
                        alignment=ft.alignment.center,
                        expand=True
                    )
                )
                self.page.update()
                return
            
            # 更新所有图片列表（用于全选功能）
            self.all_images = list(dataset.images.keys())
            
            if not self.all_images:
                self.image_grid.controls.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Icon(ft.Icons.PHOTO_LIBRARY, size=64, color=ft.Colors.GREY),
                            ft.Text("暂无图片", size=18, color=ft.Colors.GREY),
                            ft.Text("点击上方「导入文件」添加图片", size=14, color=ft.Colors.GREY_600)
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        alignment=ft.alignment.center,
                        expand=True
                    )
                )
            else:
                # 添加图片卡片
                for filename, label in dataset.images.items():
                    card = self._create_image_card(filename, label)
                    self.image_grid.controls.append(card)
            
            # 更新选择状态
            self._update_selection_ui()
            
            if self.page:
                self.page.update()
                
        except Exception as e:
            self.toast_service.show(f"刷新图片失败: {str(e)}", "error")
    
    def build(self) -> ft.Container:
        """构建并返回根容器"""
        self.refresh_images()
        return self.root_container