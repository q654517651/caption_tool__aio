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
    TerminalService, TerminalView, TrainingCreateView
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
        # self.page.update()

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
        self.page.window.center()

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
                    icon=ft.Icons.ADD_OUTLINED,
                    selected_icon=ft.Icons.ADD,
                    label="创建训练"
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
            self.show_create_training_view()
        elif selected_index == 3:
            self.show_terminal_view()
        elif selected_index == 4:
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

    def show_create_training_view(self):
        """显示创建训练视图"""
        try:
            self.current_view = "create_training"
            self.nav_rail.selected_index = 2

            
            create_training_view = TrainingCreateView(
                page=self.page,
                dataset_manager=self.dataset_manager,
                training_manager=self.training_manager,
                on_back=self.show_training_view,  # 返回训练列表页面
                toast_service=self.toast_service
            )

            self.content_host.content = create_training_view.build()
            self.page.update()

        except Exception as e:
            self.toast_service.show(f"显示创建训练视图失败: {str(e)}", "error")

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

            # Musubi状态显示
            musubi_status_text = ft.Text(
                "检查 Musubi-Tuner 状态...",
                color=ft.Colors.GREY
            )
            
            # 自动检查状态
            def auto_check_status():
                try:
                    from ...utils.musubi_helper import check_musubi_status
                    status = check_musubi_status()
                    if status["available"]:
                        musubi_status_text.value = f"✅ {status['status']}"
                        musubi_status_text.color = ft.Colors.GREEN
                    else:
                        musubi_status_text.value = f"❌ {status['status']}"
                        musubi_status_text.color = ft.Colors.ERROR
                except Exception as ex:
                    musubi_status_text.value = f"❌ 检查失败: {str(ex)}"
                    musubi_status_text.color = ft.Colors.ERROR
            
            auto_check_status()  # 立即执行检查
            
            # Musubi训练器目录
            musubi_dir_field = ft.TextField(
                label="Musubi训练器目录",
                value=config.model_paths.musubi_dir,
                expand=True
            )
            
            # Qwen-Image 模型路径
            qwen_dit_field = ft.TextField(
                label="DiT模型路径",
                value=config.model_paths.qwen_image.dit_path,
                expand=True
            )
            qwen_vae_field = ft.TextField(
                label="VAE模型路径",
                value=config.model_paths.qwen_image.vae_path,
                expand=True
            )
            qwen_text_encoder_field = ft.TextField(
                label="Text Encoder路径",
                value=config.model_paths.qwen_image.text_encoder_path,
                expand=True
            )
            
            # Flux 模型路径
            flux_dit_field = ft.TextField(
                label="DiT模型路径",
                value=config.model_paths.flux.dit_path,
                expand=True
            )
            flux_vae_field = ft.TextField(
                label="VAE模型路径",
                value=config.model_paths.flux.vae_path,
                expand=True
            )
            flux_text_encoder_field = ft.TextField(
                label="Text Encoder路径",
                value=config.model_paths.flux.text_encoder_path,
                expand=True
            )
            flux_clip_field = ft.TextField(
                label="CLIP模型路径",
                value=config.model_paths.flux.clip_path,
                expand=True
            )
            
            # Stable Diffusion 模型路径
            sd_unet_field = ft.TextField(
                label="UNet模型路径",
                value=config.model_paths.stable_diffusion.unet_path,
                expand=True
            )
            sd_vae_field = ft.TextField(
                label="VAE模型路径",
                value=config.model_paths.stable_diffusion.vae_path,
                expand=True
            )
            sd_text_encoder_field = ft.TextField(
                label="Text Encoder路径",
                value=config.model_paths.stable_diffusion.text_encoder_path,
                expand=True
            )
            sd_clip_field = ft.TextField(
                label="CLIP模型路径",
                value=config.model_paths.stable_diffusion.clip_path,
                expand=True
            )

            def save_settings(e):
                try:
                    from ...config import save_config, get_config
                    # 获取当前配置
                    current_config = get_config()
                    
                    # 更新模型路径配置
                    current_config.model_paths.musubi_dir = musubi_dir_field.value
                    
                    # Qwen-Image 路径
                    current_config.model_paths.qwen_image.dit_path = qwen_dit_field.value
                    current_config.model_paths.qwen_image.vae_path = qwen_vae_field.value
                    current_config.model_paths.qwen_image.text_encoder_path = qwen_text_encoder_field.value
                    
                    # Flux 路径
                    current_config.model_paths.flux.dit_path = flux_dit_field.value
                    current_config.model_paths.flux.vae_path = flux_vae_field.value
                    current_config.model_paths.flux.text_encoder_path = flux_text_encoder_field.value
                    current_config.model_paths.flux.clip_path = flux_clip_field.value
                    
                    # Stable Diffusion 路径
                    current_config.model_paths.stable_diffusion.unet_path = sd_unet_field.value
                    current_config.model_paths.stable_diffusion.vae_path = sd_vae_field.value
                    current_config.model_paths.stable_diffusion.text_encoder_path = sd_text_encoder_field.value
                    current_config.model_paths.stable_diffusion.clip_path = sd_clip_field.value
                    
                    # 保存配置
                    save_config(current_config)
                    self.toast_service.show("设置已保存", "success")
                except Exception as ex:
                    self.toast_service.show(f"保存设置失败: {str(ex)}", "error")
            
            def check_musubi_status(e):
                try:
                    from ...utils.musubi_helper import check_musubi_status
                    status = check_musubi_status()
                    if status["available"]:
                        musubi_status_text.value = f"✅ {status['status']}"
                        musubi_status_text.color = ft.Colors.GREEN
                        self.toast_service.show(f"✅ {status['status']}", "success")
                    else:
                        musubi_status_text.value = f"❌ {status['status']}"
                        musubi_status_text.color = ft.Colors.ERROR
                        msg = f"❌ {status['status']}\n" + "\n".join(status['missing_components'])
                        self.toast_service.show(msg, "error")
                    self.page.update()
                except Exception as ex:
                    musubi_status_text.value = f"❌ 检查失败: {str(ex)}"
                    musubi_status_text.color = ft.Colors.ERROR
                    self.page.update()
                    self.toast_service.show(f"检查失败: {str(ex)}", "error")
            
            def init_submodules(e):
                """初始化git子模块"""
                try:
                    import subprocess
                    self.toast_service.show("正在初始化 Musubi-Tuner 子模块...", "info")
                    
                    result = subprocess.run(
                        ["git", "submodule", "update", "--init", "--recursive"],
                        capture_output=True,
                        text=True,
                        cwd=Path(__file__).parent.parent.parent.parent.parent.parent
                    )
                    
                    if result.returncode == 0:
                        self.toast_service.show("✅ 子模块初始化成功", "success")
                        check_musubi_status(None)  # 重新检查状态
                    else:
                        self.toast_service.show(f"❌ 子模块初始化失败: {result.stderr}", "error")
                        
                except Exception as ex:
                    self.toast_service.show(f"初始化失败: {str(ex)}", "error")

            settings_content = ft.Column([
                ft.Container(
                    content=ft.Text("⚙️ 设置", size=24, weight=ft.FontWeight.BOLD),
                    padding=ft.padding.all(20)
                ),
                ft.Container(
                    content=ft.Column([
                        ft.Text("🔧 Musubi-Tuner 状态", size=18, weight=ft.FontWeight.BOLD),
                        ft.Container(
                            content=ft.Column([
                                musubi_status_text,
                                ft.Text("Musubi-Tuner 已内置，无需手动配置", size=12, color=ft.Colors.GREY),
                                ft.Row([
                                    ft.ElevatedButton(
                                        "检查状态",
                                        icon=ft.Icons.CHECK_CIRCLE,
                                        on_click=check_musubi_status
                                    ),
                                    ft.ElevatedButton(
                                        "初始化子模块",
                                        icon=ft.Icons.DOWNLOAD,
                                        on_click=init_submodules
                                    )
                                ])
                            ]),
                            padding=ft.padding.all(15),
                            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                            border_radius=8
                        ),
                        ft.Container(height=10),
                        ft.Text("📁 模型路径配置", size=18, weight=ft.FontWeight.BOLD),
                        
                        # Musubi训练器配置
                        ft.Container(height=10),
                        ft.Text("🔧 Musubi训练器", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_GREY_700),
                        musubi_dir_field,
                        
                        # Qwen-Image 配置
                        ft.Container(height=15),
                        ft.Text("🎯 Qwen-Image LoRA", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.DEEP_ORANGE_700),
                        qwen_dit_field,
                        qwen_vae_field,
                        qwen_text_encoder_field,
                        
                        # Flux 配置
                        ft.Container(height=15),
                        ft.Text("⚡ Flux LoRA", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.PURPLE_700),
                        flux_dit_field,
                        flux_vae_field,
                        flux_text_encoder_field,
                        flux_clip_field,
                        
                        # Stable Diffusion 配置
                        ft.Container(height=15),
                        ft.Text("🎨 Stable Diffusion LoRA", size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_700),
                        sd_unet_field,
                        sd_vae_field,
                        sd_text_encoder_field,
                        sd_clip_field,
                        ft.Container(height=20),
                        ft.Row([
                            ft.ElevatedButton(
                                "保存设置",
                                icon=ft.Icons.SAVE,
                                on_click=save_settings,
                                style=ft.ButtonStyle(bgcolor=ft.Colors.PRIMARY, color=ft.Colors.WHITE)
                            ),
                            ft.OutlinedButton(
                                "检查 Musubi 状态",
                                icon=ft.Icons.HEALTH_AND_SAFETY,
                                on_click=check_musubi_status
                            )
                        ])
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
        # page.window.center()

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
