#!/usr/bin/env python3
"""
TagTracker Flet Application - 新架构实现
"""

import flet as ft
import os
from pathlib import Path

# 导入新架构的核心模块
from ...config import get_config
from ...core.dataset import DatasetManager
from ...core.labeling import LabelingService  
from ...core.training import TrainingManager
from ...utils.logger import logger

# 导入新架构的UI组件
from .components import (
    ToastService, DeleteConfirmDialog,
    DatasetsView, DatasetDetailView,
    TrainingListView, TrainingDetailView,
    TerminalService, TerminalView
)

class TagTrackerApp:
    """TagTracker主应用 - 新架构实现"""
    
    def __init__(self, page: ft.Page):
        self.page = page
        self.config = get_config()
        
        # 配置页面
        self.setup_page()
        
        # 初始化核心服务
        self.terminal_service = TerminalService()
        self.dataset_manager = DatasetManager()
        self.labeling_service = LabelingService()
        self.training_manager = TrainingManager()
        
        # 初始化UI服务
        self.toast_service = ToastService(page)
        
        # 当前视图状态
        self.current_view = "datasets"
        self.current_dataset_id = None
        self.current_task_id = None
        
        # 注册日志回调
        logger.register_ui_callback(self._on_log_message)
        
        # 创建主要UI容器
        self.content_host = ft.Container(expand=True)
        self.nav_rail = self._create_nav_rail()
        
        # 设置页面布局
        main_layout = ft.Row([
            self.nav_rail,
            ft.VerticalDivider(width=1),
            self.content_host
        ], expand=True)
        
        self.page.add(main_layout)
        self.page.update()
        
        # 默认显示数据集视图
        self.show_datasets_view()
    
    def setup_page(self):
        """配置页面属性"""
        self.page.title = "TagTracker - 集成打标与训练的LoRA训练工具"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.window_width = self.config.ui.window_width
        self.page.window_height = self.config.ui.window_height
        self.page.window_min_width = 1200
        self.page.window_min_height = 800
        
        # 设置应用栏
        self.page.appbar = ft.AppBar(
            title=ft.Text("TagTracker"),
            center_title=False,
            bgcolor=ft.Colors.SURFACE,
        )
    
    def _create_nav_rail(self):
        """创建导航栏"""
        return ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=100,
            min_extended_width=200,
            leading=ft.FloatingActionButton(
                icon=ft.Icons.LABEL,
                text="TagTracker",
                width=150
            ),
            group_alignment=-0.9,
            destinations=[
                ft.NavigationRailDestination(
                    icon=ft.Icons.DATASET_OUTLINED,
                    selected_icon=ft.Icons.DATASET,
                    label="数据集管理"
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.MODEL_TRAINING_OUTLINED,
                    selected_icon=ft.Icons.MODEL_TRAINING,
                    label="模型训练"
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.TERMINAL_OUTLINED,
                    selected_icon=ft.Icons.TERMINAL,
                    label="终端"
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.SETTINGS_OUTLINED,
                    selected_icon=ft.Icons.SETTINGS,
                    label="设置"
                ),
            ],
            on_change=self._on_nav_change
        )
    
    def _on_nav_change(self, e):
        """导航切换"""
        selected_index = e.control.selected_index
        
        if selected_index == 0:
            self.show_datasets_view()
        elif selected_index == 1:
            self.show_training_view()
        elif selected_index == 2:
            self.show_terminal_view()
        elif selected_index == 3:
            self.show_settings_view()
    
    def show_datasets_view(self):
        """显示数据集管理视图"""
        self.current_view = "datasets"
        self.nav_rail.selected_index = 0
        
        datasets_view = DatasetsView(
            page=self.page,
            dataset_manager=self.dataset_manager,
            on_open_dataset=self.show_dataset_detail,
            on_delete_dataset=self.confirm_delete_dataset,
            toast_service=self.toast_service
        )
        
        self.content_host.content = datasets_view.build()
        self.page.update()
    
    def show_dataset_detail(self, dataset_id: str):
        """显示数据集详情视图"""
        try:
            # 获取数据集信息
            dataset = self.dataset_manager.get_dataset(dataset_id)
            if not dataset:
                self.toast_service.show("数据集不存在", "error")
                return
            
            self.current_view = "dataset_detail"
            self.current_dataset_id = dataset_id
            
            detail_view = DatasetDetailView(
                page=self.page,
                dataset_id=dataset_id,
                dataset_name=dataset.name,
                dataset_manager=self.dataset_manager,
                labeling_service=self.labeling_service,
                on_back=self.show_datasets_view,
                toast_service=self.toast_service
            )
            
            self.content_host.content = detail_view.build()
            self.page.update()
            
        except Exception as e:
            self.toast_service.show(f"打开数据集详情失败: {str(e)}", "error")
    
    def show_training_view(self):
        """显示训练视图"""
        try:
            self.current_view = "training"
            self.nav_rail.selected_index = 1
            
            training_view = TrainingListView(
                page=self.page,
                training_manager=self.training_manager,
                dataset_manager=self.dataset_manager,
                on_open_task=self.show_training_detail,
                toast_service=self.toast_service
            )
            
            self.content_host.content = training_view.build()
            self.page.update()
            
        except Exception as e:
            self.toast_service.show(f"显示训练视图失败: {str(e)}", "error")
    
    def show_training_detail(self, task_id: str):
        """显示训练详情视图"""
        try:
            task = self.training_manager.get_task(task_id)
            if not task:
                self.toast_service.show("训练任务不存在", "error")
                return
            
            self.current_view = "training_detail"
            self.current_task_id = task_id
            
            detail_view = TrainingDetailView(
                page=self.page,
                task_id=task_id,
                training_manager=self.training_manager,
                on_back=self.show_training_view,
                toast_service=self.toast_service
            )
            
            self.content_host.content = detail_view.build()
            self.page.update()
            
        except Exception as e:
            self.toast_service.show(f"显示训练详情失败: {str(e)}", "error")
    
    def show_terminal_view(self):
        """显示终端视图"""
        try:
            self.current_view = "terminal"
            self.nav_rail.selected_index = 2
            
            terminal_view = TerminalView(
                page=self.page,
                terminal_service=self.terminal_service
            )
            
            self.content_host.content = terminal_view.build()
            self.page.update()
            
        except Exception as e:
            self.toast_service.show(f"显示终端视图失败: {str(e)}", "error")
    
    def show_settings_view(self):
        """显示设置视图"""
        try:
            self.current_view = "settings"
            self.nav_rail.selected_index = 3
            
            config = get_config()
            
            # 模型路径设置字段
            musubi_dir_field = ft.TextField(
                label="Musubi-Tuner目录",
                value=config.model_paths.musubi_dir,
                expand=True
            )
            vae_path_field = ft.TextField(
                label="VAE模型路径",
                value=config.model_paths.vae_path,
                expand=True
            )
            clip_path_field = ft.TextField(
                label="CLIP模型路径",
                value=config.model_paths.clip_path,
                expand=True
            )
            t5_path_field = ft.TextField(
                label="T5模型路径",
                value=config.model_paths.t5_path,
                expand=True
            )
            unet_path_field = ft.TextField(
                label="UNet模型路径",
                value=config.model_paths.unet_path,
                expand=True
            )
            
            def save_settings(e):
                # TODO: 实现设置保存逻辑
                self.toast_service.show("设置保存功能开发中...", "info")
            
            settings_content = ft.Column([
                ft.Container(
                    content=ft.Text("⚙️ 设置", size=24, weight=ft.FontWeight.BOLD),
                    padding=ft.padding.all(20)
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("模型路径配置", size=18, weight=ft.FontWeight.BOLD),
                        musubi_dir_field,
                        vae_path_field,
                        clip_path_field,
                        t5_path_field,
                        unet_path_field,
                        ft.Container(height=20),
                        ft.ElevatedButton(
                            "保存设置",
                            icon=ft.Icons.SAVE,
                            on_click=save_settings,
                            style=ft.ButtonStyle(bgcolor=ft.Colors.PRIMARY, color=ft.Colors.WHITE)
                        )
                    ], spacing=15),
                    padding=ft.padding.all(20)
                )
            ], scroll=ft.ScrollMode.AUTO)
            
            self.content_host.content = settings_content
            self.page.update()
            
        except Exception as e:
            self.toast_service.show(f"显示设置视图失败: {str(e)}", "error")
    
    def confirm_delete_dataset(self, dataset_id: str):
        """确认删除数据集"""
        def on_confirm():
            success, message = self.dataset_manager.delete_dataset(dataset_id)
            if success:
                self.toast_service.show("数据集删除成功", "success")
                self.show_datasets_view()  # 刷新视图
            else:
                self.toast_service.show(f"删除失败: {message}", "error")
        
        # 获取数据集名称
        dataset = self.dataset_manager.get_dataset(dataset_id)
        dataset_name = dataset.name if dataset else "未知数据集"
        
        dialog = DeleteConfirmDialog(
            page=self.page,
            item_name=dataset_name,
            on_confirm=on_confirm,
            extra_hint="此操作将删除数据集及其所有图片和标签，不可撤销。"
        )
        dialog.open()
    
    # === 事件处理方法 ===
    
    def _handle_batch_translate(self, dataset_id: str):
        """处理批量翻译"""
        # TODO: 实现批量翻译功能
        self.toast_service.show("批量翻译功能开发中...", kind="info")
    
    def _handle_batch_label(self, dataset_id: str):
        """处理批量打标"""
        try:
            # 调用打标服务
            self.labeling_service.start_labeling(dataset_id)
            self.toast_service.show("开始批量打标...", kind="success")
        except Exception as e:
            self.toast_service.show(f"启动打标失败: {str(e)}", kind="error")
    
    def _handle_import_files(self, dataset_id: str):
        """处理文件导入"""
        try:
            # 创建文件选择器
            file_picker = ft.FilePicker()
            
            def on_file_result(e):
                if e.files:
                    file_paths = [f.path for f in e.files]
                    success_count, message = self.dataset_manager.import_images_to_dataset(
                        dataset_id, file_paths
                    )
                    
                    if success_count > 0:
                        self.toast_service.show(message, kind="success")
                        # 刷新当前视图
                        if hasattr(self, 'current_view') and hasattr(self.current_view, 'refresh_images'):
                            self.current_view.refresh_images()
                    else:
                        self.toast_service.show(message, kind="error")
                
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
            
        except Exception as e:
            self.toast_service.show(f"文件导入失败: {str(e)}", kind="error")
    
    def _handle_render_images(self, dataset_id: str, grid: ft.GridView):
        """渲染图片到网格中"""
        try:
            # 获取数据集
            dataset = None
            for ds in self.dataset_manager.list_datasets():
                if ds.dataset_id == dataset_id:
                    dataset = ds
                    break
            
            if not dataset:
                grid.controls.clear()
                grid.controls.append(
                    ft.Container(
                        content=ft.Text("找不到数据集", size=16),
                        padding=20
                    )
                )
                self.page.update()
                return
            
            # 获取图片文件名列表
            image_filenames = list(dataset.images.keys())
            
            if not image_filenames:
                grid.controls.clear()
                grid.controls.append(
                    ft.Container(
                        content=ft.Text("该数据集没有图片", size=16),
                        padding=20
                    )
                )
                self.page.update()
                return
            
            for i, img_filename in enumerate(image_filenames):
                try:
                    # 获取图片的实际路径 - 直接构建路径
                    dataset_path = self.dataset_manager.get_dataset_path(dataset_id)
                    
                    # 优先使用medium缩略图 (格式: filename.随机数.jpg)
                    medium_dir = dataset_path / "medium"
                    img_path = None
                    
                    # 查找medium目录中匹配的缩略图
                    if medium_dir.exists():
                        for medium_file in medium_dir.iterdir():
                            if medium_file.name.startswith(img_filename.split('.')[0]):
                                img_path = str(medium_file)
                                break
                    
                    if not img_path:
                        # 尝试原图 (在images目录)
                        original_path = dataset_path / "images" / img_filename
                        if original_path.exists():
                            img_path = str(original_path)
                        else:
                            continue  # 跳过找不到的图片
                    
                    # 获取标签（从数据集的images字典中）
                    label = dataset.images.get(img_filename, "")
                    has_label = bool(label.strip())
                    
                    card = ft.Container(
                        content=ft.Column([
                            ft.Image(
                                src=img_path,
                                width=200,
                                height=150,
                                fit=ft.ImageFit.CONTAIN,
                                error_content=ft.Text("加载失败", size=10)
                            ),
                            ft.Container(
                                content=ft.Text(
                                    img_filename if len(img_filename) <= 20 else img_filename[:17] + "...",
                                    size=12
                                ),
                                padding=5
                            ),
                            ft.Container(
                                content=ft.Text(
                                    "已标注" if has_label else "未标注",
                                    size=10,
                                    color=ft.Colors.GREEN if has_label else ft.Colors.RED
                                ),
                                padding=5
                            )
                        ]),
                        padding=10,
                        bgcolor=ft.Colors.WHITE,
                        border_radius=10,
                        data=img_filename  # 存储文件名用于选择逻辑
                    )
                    
                    grid.controls.append(card)
                    
                    # 每10张图片更新一次UI，提高响应性
                    if (i + 1) % 10 == 0:
                        self.page.update()
                    
                except Exception as img_e:
                    # 跳过有问题的图片
                    continue
            
            self.page.update()
            
        except Exception as e:
            self.terminal_service.log_error(f"渲染图片失败: {str(e)}")
            
            # 显示错误信息
            grid.controls.clear()
            grid.controls.append(
                ft.Container(
                    content=ft.Text(f"加载失败: {str(e)}", size=14, color=ft.Colors.RED),
                    padding=20
                )
            )
            self.page.update()
    
    def _handle_create_training_task(self):
        """处理创建训练任务"""
        try:
            # 检查是否有数据集
            datasets = self.dataset_manager.list_datasets()
            if not datasets:
                self.toast_service.show("请先创建数据集", kind="warning")
                return
            
            # 创建简单的训练任务配置对话框
            from ...core.training.models import TrainingConfig, TrainingType
            
            # 任务名称输入
            task_name_field = ft.TextField(
                label="任务名称",
                value=f"训练任务_{len(self.training_manager.list_tasks()) + 1}",
                width=300
            )
            
            # 数据集选择
            dataset_options = [ft.dropdown.Option(ds.dataset_id, ds.name) for ds in datasets]
            dataset_dropdown = ft.Dropdown(
                label="选择数据集",
                options=dataset_options,
                value=datasets[0].dataset_id,
                width=300
            )
            
            # 训练参数
            epochs_field = ft.TextField(label="训练轮数", value="16", width=150)
            batch_size_field = ft.TextField(label="批次大小", value="1", width=150)
            learning_rate_field = ft.TextField(label="学习率", value="1e-4", width=150)
            
            def create_task(e):
                try:
                    config = TrainingConfig(
                        task_id="",  # 将在manager中生成
                        name=task_name_field.value,
                        training_type=TrainingType.QWEN_IMAGE_LORA,
                        dataset_id=dataset_dropdown.value,
                        epochs=int(epochs_field.value),
                        batch_size=int(batch_size_field.value),
                        learning_rate=float(learning_rate_field.value)
                    )
                    
                    task_id = self.training_manager.create_task(config)
                    dialog.open = False
                    self.page.update()
                    
                    self.toast_service.show(f"训练任务创建成功: {task_name_field.value}", kind="success")
                    self.show_training_view()  # 刷新训练视图
                    
                except Exception as ex:
                    self.toast_service.show(f"创建失败: {str(ex)}", kind="error")
            
            def cancel(e):
                dialog.open = False
                self.page.update()
            
            dialog = ft.AlertDialog(
                title=ft.Text("创建训练任务"),
                content=ft.Container(
                    content=ft.Column([
                        task_name_field,
                        dataset_dropdown,
                        ft.Row([epochs_field, batch_size_field, learning_rate_field]),
                        ft.Text("注意：需要在设置中配置模型路径", size=12, color=ft.Colors.ORANGE)
                    ], spacing=10),
                    width=400,
                    height=250
                ),
                actions=[
                    ft.TextButton("取消", on_click=cancel),
                    ft.ElevatedButton("创建", on_click=create_task)
                ]
            )
            
            self.page.open(dialog)
            
        except Exception as e:
            self.toast_service.show(f"打开创建对话框失败: {str(e)}", kind="error")
    
    def _handle_cancel_task(self, task_id: str):
        """处理取消训练任务"""
        try:
            success = self.training_manager.cancel_task(task_id)
            if success:
                self.toast_service.show("训练任务已取消", kind="success")
                self.show_training_view()  # 返回训练列表
            else:
                self.toast_service.show("取消训练任务失败", kind="error")
        except Exception as e:
            self.toast_service.show(f"取消任务失败: {str(e)}", kind="error")
    
    def _save_settings(self, e):
        """保存设置"""
        # TODO: 实现设置保存
        self.toast_service.show("设置保存成功", kind="success")
    
    def _on_log_message(self, message: str, level):
        """日志消息回调"""
        # 通过terminal_service处理日志
        if hasattr(self, 'terminal_service'):
            if level == 'INFO':
                self.terminal_service.log_info(message)
            elif level == 'ERROR':
                self.terminal_service.log_error(message)
            elif level == 'SUCCESS':
                self.terminal_service.log_success(message)

def main():
    """应用入口"""
    def app_main(page: ft.Page):
        app = TagTrackerApp(page)
    
    # 获取workspace路径作为assets目录
    config = get_config()
    assets_dir = os.path.abspath(config.storage.workspace_root)
    
    ft.app(
        target=app_main, 
        assets_dir=assets_dir
        # 不设置view参数，默认为原生桌面窗口
    )

if __name__ == "__main__":
    main()