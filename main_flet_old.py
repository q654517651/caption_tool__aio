#!/usr/bin/env python3

import flet as ft
import os
import threading
import time
import asyncio
from typing import cast, Optional

# 导入服务模块
from services.labeling_service import LabelingService
# from translation_service import TranslationService
from settings_manager import SettingsManager
from services.terminal_service import TerminalService
from services.dataset_manager import DatasetManager
from views.datasets_view import DatasetsView
from views.dialogs import DeleteConfirmDialog
from views.dataset_detail_view import DatasetDetailView
from views.dialogs import ToastService
from services.event_bus import EventBus
from services.job_queue import JobQueue
from training_manager import TrainingManager
from trainers.types import TrainingConfig, TrainingBackend, BACKEND_PRESETS
from views.training_list_view import TrainingListView
from views.training_detail_view import TrainingDetailView
from services.navigation import NavigationService


# 路由常量
PATH_HOME = "/"
PATH_DATASETS = "/datasets"
PATH_TRAINING = "/training"
PATH_TERMINAL = "/terminal"
PATH_SETTINGS = "/settings"


class ImageLabelingApp:
    """主应用类"""

    def __init__(self, page: ft.Page):
        # ========= 基础页面配置 =========
        self.page = page
        self.page.title = "图像打标系统"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.window_width = 1400
        self.page.window_height = 900

        # ========= UI 容器与页面状态 =========
        # 所有内容统一挂到 content_host；不要再使用 main_content
        self.content_host = ft.Container(expand=True)

        # 这些控件/引用在后续 view 构建时会被赋值，这里先置空占位
        self.nav_rail = None
        self._datasets_view = None
        self._dataset_detail_view = None
        self.current_dataset_id = None

        # 训练视图内使用到的输入控件（按你原来地占位保留）
        self.current_view = None
        self.terminal_display = None
        self.settings_status = None
        self.unet_path_input = None
        self.t5_path_input = None
        self.clip_path_input = None
        self.vae_path_input = None
        self.musubi_dir_input = None
        self.tensorboard_container = None
        self.training_output = None
        self.sample_prompts_input = None
        self.lora_name_input = None
        self.learning_rate_input = None
        self.max_epochs_input = None
        self.num_repeats_input = None
        self.batch_size_input = None
        self.resolution_input = None
        self.dataset_type_radio = None
        self.dataset_path_input = None
        self.training_list = None

        # ========= 服务模块 =========
        # 提示/Toast
        self.toast_service = ToastService(self.page)
        self.toast = lambda msg, kind="info", duration=2000: \
            self.toast_service.show(msg, kind=kind, duration=duration)

        # 路由导航（由 NavigationService 回调 _render_route 统一渲染）
        self.nav = NavigationService(self.page, render_fn=self._render_route)

        # 设置 / 数据集 / 打标服务
        self.settings_manager = SettingsManager()
        self.dataset_manager = DatasetManager()
        self.dataset_manager.platform_mode = "web" if getattr(self.page, "platform", None) == "web" else "pc"
        self.terminal_service = TerminalService()
        self.labeling_service = LabelingService(self.terminal_service)

        # ========= 训练调度（事件总线、队列、管理器）=========
        self.bus = EventBus()
        self.queue = JobQueue(self.bus, max_workers=1)  # 先单并发，后续可按 GPU 扩展
        self.training_manager = TrainingManager(self.bus, self.queue)
        self.current_training_detail = {}  # task_id -> TrainingDetailView

        # 打标刷新任务控制（若你在别处用到）
        self._labeling_refresh_stop = None
        self._labeling_refresh_task = None

        # ========= 文件选择器 & Overlay =========
        self.file_picker = ft.FilePicker()
        self.page.overlay.append(self.file_picker)

        # 初次更新（让 AppBar / Overlay 等生效）
        self.page.update()

        # ========= 启动 UI =========
        self.setup_ui()

    def setup_ui(self):
        """设置UI（由 NavigationService 驱动）"""
        # AppBar
        self.page.appbar = ft.AppBar(title=ft.Text("图像打标系统"), center_title=False)

        # 承载主内容的容器（如果外部还没创建就补上）
        if not hasattr(self, "content_host") or self.content_host is None:
            self.content_host = ft.Container(expand=True)

        # 侧栏切换 -> 路由跳转
        def _on_nav_change(e):
            idx = e.control.selected_index
            paths = [PATH_DATASETS, PATH_TRAINING, PATH_TERMINAL, PATH_SETTINGS]
            if idx < 0 or idx >= len(paths):
                idx = 0
            self.nav.go(paths[idx])

        # NavigationRail（保持你原来的样式和文案）
        self.nav_rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=100,
            min_extended_width=200,
            leading=ft.FloatingActionButton(
                icon=ft.Icons.LABEL,
                text="标注系统"
            ),
            group_alignment=-0.9,
            destinations=[
                ft.NavigationRailDestination(
                    icon=ft.Icons.SAVE_OUTLINED,
                    selected_icon=ft.Icons.SAVE,
                    label="数据管理"
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.ROCKET_LAUNCH_OUTLINED,
                    selected_icon=ft.Icons.ROCKET_LAUNCH,
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
                )
            ],
            on_change=_on_nav_change
        )

        # 页面主布局：侧栏 + 分割线 + 主内容容器
        self.page.controls.clear()
        self.page.add(
            ft.Row(
                controls=[self.nav_rail, ft.VerticalDivider(width=1), self.content_host],
                expand=True
            )
        )

        # 默认进入“数据管理”页（通过路由驱动 _render_route）
        self.nav.go(PATH_DATASETS)

    def _render_route(self, route: str):
        # 1) 侧栏选中态
        if route.startswith(PATH_TRAINING):
            self.nav_rail.selected_index = 1
        elif route.startswith(PATH_TERMINAL):
            self.nav_rail.selected_index = 2
        elif route.startswith(PATH_SETTINGS):
            self.nav_rail.selected_index = 3
        else:
            self.nav_rail.selected_index = 0

        # 2) 路由分发
        if route == PATH_TRAINING:
            # 训练任务列表
            # 复用已实例。如果第一次进入，这里构建一次
            if not getattr(self, "training_list", None):
                self.show_training_view()  # 内部会创建 self.training_list
            else:
                self.content_host.content = self.training_list
        elif route.startswith(f"{PATH_TRAINING}/"):  # 任务详情
            task_id = route.split("/", 2)[-1]
            # 统一用已有详情缓存；不存在就让 show_training_view 创建
            if not hasattr(self, "current_training_detail") or task_id not in self.current_training_detail:
                self.show_training_view()  # 会把新建的 detail 放到缓存
            if task_id in self.current_training_detail:
                self.content_host.content = self.current_training_detail[task_id]
        elif route.startswith(f"{PATH_DATASETS}/"):  # 数据集详情
            ds_id = route.split("/", 2)[-1]
            # 直接进入详情渲染
            self.view_dataset(ds_id)
        elif route == PATH_TERMINAL:
            self.show_terminal_view()
        elif route == PATH_SETTINGS:
            self.show_settings_view()
        else:
            # 默认首页：数据集列表
            self.show_datasets_view()

        # 3) 返回按钮显隐（AppBar 上）
        is_detail = route.startswith(f"{PATH_TRAINING}/") or route.startswith(f"{PATH_DATASETS}/")
        self.page.appbar.leading = (
            ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda e: self.nav.back())
            if is_detail else None
        )

        self.page.update()

    def update_dataset_label(self, dataset_id, filename, label):
        """更新数据集标签"""
        self.dataset_manager.update_dataset_label(dataset_id, filename, label)

    def confirm_delete_dataset(self, dataset_id: str):
        """确认删除数据集（委托给 DeleteConfirmDialog 组件）"""
        dataset = self.dataset_manager.get_dataset(dataset_id)
        if not dataset:
            self.toast("数据集不存在", kind="warning")
            return

        def _do_delete():
            try:
                success, msg = self.dataset_manager.delete_dataset(dataset_id)
            except Exception as ex:
                success, msg = False, f"删除异常：{ex}"

            if success:
                # 刷新视图
                if hasattr(self, "_datasets_view") and self._datasets_view:
                    self._datasets_view.refresh()
                self.toast(f"✅ 已删除：{dataset.name}", kind="success")
            else:
                self.toast(f"❌ 删除失败：{msg}", kind="error")

        DeleteConfirmDialog(
            page=self.page,
            item_name=dataset.name,
            on_confirm=_do_delete,
        ).open()

    def pick_files_for_dataset(self, dataset_id):
        """调用系统文件对话框并把选中文件导入到指定数据集"""
        self.page.update()  # 先刷新 UI，防止焦点被占用

        # 这里的 on_result 回调必须是单参 (e)
        # 把 dataset_id 闭包到外部作用域里
        self.file_picker.on_result = lambda e: self.on_files_selected(e, dataset_id)

        # 弹出文件选择对话框
        self.file_picker.pick_files(
            allow_multiple=True,
            allowed_extensions=["jpg", "jpeg", "png", "gif", "bmp", "txt"]
        )

    def on_files_selected(self, e, dataset_id):
        """文件选择器回调"""
        if e.files:
            file_paths = [f.path for f in e.files]
            count, message = self.dataset_manager.import_images_to_dataset(
                dataset_id, file_paths
            )

            if count > 0:
                self.terminal_service.log_success(message)
                # 如果当前在数据集详情页，刷新图片列表
                if self.current_view == "dataset_detail" and self.current_dataset_id == dataset_id:
                    if hasattr(self, "_dataset_detail_view") and self._dataset_detail_view:
                        self._dataset_detail_view.refresh_images()
            else:
                self.terminal_service.log_error(message)

    def view_dataset(self, dataset_id: str):
        ds = self.dataset_manager.get_dataset(dataset_id)
        if not ds:
            self.terminal_service.log_error("数据集不存在")
            self.toast("数据集不存在")
            return

        self.current_view = "dataset_detail"
        self.current_dataset_id = dataset_id
        self.terminal_service.log_info(f"正在加载数据集: {ds.name}")

        def _on_back():
            self.show_datasets_view()

        def _on_batch_translate(dsid: str):
            self.toast("TODO: 批量翻译接入")

        def _on_batch_label(dsid: str):
            # 实现批量打标功能
            self.batch_label_images(dsid)

        def _on_import_files(dsid: str):
            self.pick_files_for_dataset(dsid)

        def _on_render_images(dsid: str, grid: ft.GridView):
            """把原 load_dataset_images 的渲染逻辑搬过来，针对传入的 grid"""
            grid.controls.clear()
            dataset = self.dataset_manager.get_dataset(dsid)
            if not dataset:
                self.terminal_service.log_error("数据集不存在")
                self.page.update()
                return

            if not dataset.images:
                grid.controls.append(
                    ft.Container(
                        content=ft.Text("暂无图片，点击上方「导入文件」添加",
                                        size=18, color=ft.Colors.GREY_600, italic=True),
                        alignment=ft.alignment.center,
                        height=200
                    )
                )
                self.page.update()
                return

            # 更新数据集详情视图中的选中图片集合
            if hasattr(self, '_dataset_detail_view') and self._dataset_detail_view:
                # 为全选功能准备所有图片列表
                self._dataset_detail_view.all_images = list(dataset.images.keys())

            for filename, label in dataset.images.items():
                try:
                    info = self.dataset_manager.resolve_image_src(dsid, filename, kind="medium")
                    src = info.get("src")
                    if not src:
                        self.terminal_service.log_error(f"无法解析图片资源: {filename}")
                        continue

                    abs_path = info.get("abs")
                    if abs_path and not os.path.exists(abs_path):
                        self.terminal_service.log_error(f"文件不存在: {abs_path}")

                    image_widget = ft.Image(
                        src=src,
                        fit=ft.ImageFit.CONTAIN,
                        error_content=ft.Container(
                            content=ft.Icon(ft.Icons.BROKEN_IMAGE, size=50, color=ft.Colors.GREY),
                            alignment=ft.alignment.center,
                            bgcolor=ft.Colors.GREY_100
                        )
                    )

                    # 检查图片是否被选中
                    is_selected = (hasattr(self, '_dataset_detail_view') and
                                   self._dataset_detail_view and
                                   filename in self._dataset_detail_view.selected_images)

                    # 创建图片卡片容器
                    image_card = ft.Container(
                        content=ft.Column([
                            ft.Container(
                                content=image_widget,
                                height=200,
                                clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                            ),
                            ft.Container(
                                content=ft.Text(
                                    filename[:30] + "..." if len(filename) > 30 else filename,
                                    size=12, weight=ft.FontWeight.W_500
                                ),
                                padding=ft.padding.symmetric(horizontal=10, vertical=5)
                            ),
                            ft.Container(
                                content=ft.TextField(
                                    value=label,  # 这里使用从dataset.images中获取的标签
                                    multiline=True,
                                    text_size=12,
                                    border=ft.InputBorder.NONE,
                                    filled=True,
                                    fill_color=ft.Colors.GREY_100,
                                    on_change=lambda e, f=filename: self.update_dataset_label(
                                        dsid, f, e.control.value
                                    ),
                                    expand=True,
                                ),
                                padding=ft.padding.only(left=10, right=10, bottom=10),
                                expand=True,
                            )
                        ], spacing=0, expand=True),
                        bgcolor=ft.Colors.WHITE if not is_selected else ft.Colors.BLUE_50,
                        border_radius=8,
                        shadow=ft.BoxShadow(
                            spread_radius=1,
                            blur_radius=4,
                            color=ft.Colors.BLACK12,
                            offset=ft.Offset(0, 2)
                        ),
                        border=ft.border.all(3, ft.Colors.BLUE) if is_selected else None,
                        on_click=lambda e, f=filename: self.toggle_image_selection(dsid, f) if hasattr(self,
                                                                                                       '_dataset_detail_view') and self._dataset_detail_view else None,
                        data=filename,
                    )

                    grid.controls.append(image_card)

                except Exception as ex:
                    self.terminal_service.log_error(f"加载图片失败 {filename}: {ex}")
                    grid.controls.append(
                        ft.Container(
                            content=ft.Column([
                                ft.Container(
                                    content=ft.Icon(ft.Icons.BROKEN_IMAGE, size=50, color=ft.Colors.GREY),
                                    height=200,
                                    alignment=ft.alignment.center,
                                    bgcolor=ft.Colors.GREY_100
                                ),
                                ft.Container(
                                    content=ft.Text(f"加载失败: {filename}", size=12, weight=ft.FontWeight.W_500),
                                    padding=ft.padding.all(10)
                                )
                            ]),
                            bgcolor=ft.Colors.WHITE,
                            border_radius=8,
                            width=300
                        )
                    )

            self.page.update()

        self._dataset_detail_view = DatasetDetailView(
            page=self.page,
            dataset_id=ds.dataset_id,
            dataset_name=ds.name,
            on_back=_on_back,
            on_batch_translate=_on_batch_translate,
            on_batch_label=_on_batch_label,
            on_import_files=_on_import_files,
            on_render_images=_on_render_images,  # ← 这里换成"渲染 grid"的回调
        )

        # 添加全选和清空选择的方法
        def select_all_images(e):
            if hasattr(self, '_dataset_detail_view') and self._dataset_detail_view:
                self._dataset_detail_view.selected_images = set(self._dataset_detail_view.all_images)
                self._dataset_detail_view.refresh_images()

        def clear_selection(e):
            if hasattr(self, '_dataset_detail_view') and self._dataset_detail_view:
                self._dataset_detail_view.selected_images.clear()
                self._dataset_detail_view.refresh_images()

        # 将方法绑定到视图
        self._dataset_detail_view.select_all_images = select_all_images
        self._dataset_detail_view.clear_selection = clear_selection

        self.content_host.content = self._dataset_detail_view
        self.page.update()

    def toggle_image_selection(self, dataset_id: str, filename: str):
        """切换图片选择状态"""
        if hasattr(self, '_dataset_detail_view') and self._dataset_detail_view:
            self._dataset_detail_view.toggle_image_selection(filename)

    def batch_label_images(self, dataset_id: str):
        """批量打标选中的图片（UI 线程安全 + 实时刷新）"""
        if not hasattr(self, '_dataset_detail_view') or not self._dataset_detail_view:
            self.toast("视图未初始化")
            return

        selected_images = self._dataset_detail_view.selected_images
        if not selected_images:
            self.toast("请先选择要打标的图片")
            return

        dataset = self.dataset_manager.get_dataset(dataset_id)
        if not dataset:
            self.toast("数据集不存在")
            return

        # 选中图片的完整路径
        image_paths = []
        for filename in selected_images:
            image_path = self.dataset_manager.get_dataset_image_path(dataset_id, filename)
            if image_path:
                image_paths.append(image_path)
        if not image_paths:
            self.toast("没有找到选中的图片")
            return

        prompt = self.labeling_service.default_labeling_prompt()

        # 打开进度弹窗（UI 线程）
        progress_dialog = ft.AlertDialog(
            title=ft.Text("正在打标"),
            content=ft.Row([ft.ProgressRing(), ft.Text("  处理中...")], alignment=ft.MainAxisAlignment.CENTER),
        )
        self.page.dialog = progress_dialog
        progress_dialog.open = True
        self.page.update()

        # ==== 异步轮询刷新（运行在 UI 事件循环），不要在 run_task 里加括号！====
        if getattr(self, "_labeling_refresh_stop", None) is not None:
            self._labeling_refresh_stop.set()
        self._labeling_refresh_stop = threading.Event()

        async def live_refresh_loop():
            # 在 UI 事件循环里安全刷新
            try:
                while not self._labeling_refresh_stop.is_set():
                    self.update_label_texts(dataset.images)  # 内部会调用 self.page.update()
                    await asyncio.sleep(0.4)
            except Exception:
                # 忽略刷新异常，避免打断主流程
                pass

        # 保存任务句柄（可选），注意这里传函数对象，不是调用！
        self._labeling_refresh_task = self.page.run_task(live_refresh_loop)

        # ==== 后台打标线程 ====
        def run_labeling():
            print("开始打标...")
            try:
                success_count, message = self.labeling_service.label_images(
                    images=image_paths,
                    labels=dataset.images,
                    prompt=prompt,
                    model_type="LM_Studio",
                    delay=1.0,
                )

                # 持久化
                self.dataset_manager.save_dataset(dataset_id)

                # 停止刷新
                if self._labeling_refresh_stop:
                    self._labeling_refresh_stop.set()

                async def finish_ui():
                    # 保险再刷一次
                    self.update_label_texts(dataset.images)
                    # 关闭弹窗并提示
                    self.page.dialog = None
                    self.toast(f"打标完成: {message}")
                    self.page.update()

                # 这里也传函数，不要加括号！
                self.page.run_task(finish_ui)

            except Exception as e:
                if self._labeling_refresh_stop:
                    self._labeling_refresh_stop.set()

                async def error_ui():
                    self.page.dialog = None
                    self.toast(f"打标失败: {str(e)}")
                    self.page.update()
                    self.terminal_service.log_error(f"批量打标失败: {str(e)}")

                # 同样传函数，不要加括号！
                self.page.run_task(error_ui)

        threading.Thread(target=run_labeling, daemon=True).start()

    def update_label_texts(self, images_dict: dict) -> None:
        """就地刷新网格中每张卡片的标签文本；显式 cast 消除 'Control | None 没有 value' 报错"""
        view = getattr(self, "_dataset_detail_view", None)
        if view is None:
            return

        grid = getattr(view, "image_grid", None)
        if not isinstance(grid, ft.GridView):
            return

        controls = getattr(grid, "controls", None)
        if not isinstance(controls, list) or not controls:
            return

        def _resolve_filename(text_on_card: str) -> Optional[str]:
            # 把可能被 ... 截断的显示名还原成真实文件名
            if text_on_card in images_dict:
                return text_on_card
            if text_on_card.endswith("..."):
                prefix = text_on_card[:-3]
                for fname in images_dict.keys():
                    if fname.startswith(prefix):
                        return fname
            return None

        updated = 0

        for card in controls:
            if not isinstance(card, ft.Container):
                continue
            content = card.content
            if not isinstance(content, ft.Column):
                continue

            # 1) 优先用 card.data（创建卡片时建议 data=filename）
            card_filename = getattr(card, "data", None)

            # 2) 回退：从第二行的 Text 里读显示名并还原
            if not card_filename:
                col = cast(ft.Column, content)
                if len(col.controls) >= 2:
                    filename_container = col.controls[1]
                    if isinstance(filename_container, ft.Container) and isinstance(filename_container.content, ft.Text):
                        text_ctrl = cast(ft.Text, filename_container.content)  # 显式 cast 消除 'value' 报错
                        filename_text = (text_ctrl.value or "").strip()
                        card_filename = _resolve_filename(filename_text)

            if not card_filename or card_filename not in images_dict:
                continue

            # 3) 第三行应是 TextField，显式 cast 后再改 value
            col = cast(ft.Column, content)
            if len(col.controls) < 3:
                continue
            label_container = col.controls[2]
            if not (isinstance(label_container, ft.Container) and isinstance(label_container.content, ft.TextField)):
                continue
            tf = cast(ft.TextField, label_container.content)  # 显式 cast
            new_value = images_dict[card_filename]
            if tf.value != new_value:
                tf.value = new_value
                updated += 1

        if updated:
            self.page.update()
            try:
                self.terminal_service.log_info(f"已更新 {updated} 个标签文本")
            except Exception:
                pass

        if hasattr(view, "update_selection_ui"):
            try:
                view.update_selection_ui()
            except Exception:
                pass

    def show_datasets_view(self):
        """显示数据集视图（独立视图类）"""
        self.current_view = "datasets"

        # 实例化视图（把"查看/删除"的回调接回主类现有方法）
        self._datasets_view = DatasetsView(
            page=self.page,
            dataset_manager=self.dataset_manager,
            terminal_service=self.terminal_service,
            on_open_dataset=self.view_dataset,
            on_delete_dataset=self.confirm_delete_dataset,
        )

        # 挂到主容器并刷新
        self.content_host.content = self._datasets_view.build()
        self.page.update()

        # 填充数据
        self._datasets_view.refresh()

    def show_training_view(self):

        # 列表视图
        def create_task():
            import time
            # 1) 选数据集
            datasets = self.dataset_manager.list_datasets()
            if not datasets:
                self.toast("没有数据集，先去创建一个", "warning")
                return
            ds = datasets[0]
            ds_size = len(ds.images)

            # 2) 合并预设（可选）
            try:
                PRESETS = BACKEND_PRESETS
            except Exception:
                PRESETS = {}

            backend = TrainingBackend.MUSUBI_QWEN_IMAGE
            temp_id = f"QwenImage-{time.strftime('%H%M%S')}"  # 用 name 作为临时ID
            base_cfg = {
                "backend": backend,
                "name": temp_id,
                "dataset_id": ds.dataset_id,
                "dataset_size": ds_size,
                "repeats": 1,
                "epochs": 1,
                "batch_size": 2,
                "grad_accum": 1,
                "resolution": "1024,1024",
                "base_model": "",
                "gpu_index": 0,
                "sample_prompt": "a cute cat",
                "sample_every_n_steps": 200,
            }
            base_cfg.update(PRESETS.get(backend, {}))
            cfg = TrainingConfig(**base_cfg)

            # 3) 回调（返回/停止）
            def _on_back():
                self.content_host.content = self.training_list
                self.page.update()

            def _on_cancel():
                self.toast("已取消（未启动）", "warning")

            # 4) 先创建详情视图，再定义 _on_start，把当前 detail 闭包进去
            from views.training_detail_view import TrainingDetailView
            detail = TrainingDetailView(
                task_id=temp_id,
                cfg=cfg.__dict__,
                on_cancel=_on_cancel,
                on_back=_on_back,
                on_start=None,  # 先占位，下面再赋值
            )

            def _on_start(updated_cfg: dict, view=detail):
                """手动开始训练：使用当前详情视图实例，不从 dict 里找，避免 None"""
                real_jid = self.training_manager.run_training(TrainingConfig(**updated_cfg))

                # 列表：移除临时ID，插入真实ID
                if hasattr(self.training_list, "remove_task"):
                    try:
                        self.training_list.remove_task(view.task_id)
                    except Exception:
                        pass
                if hasattr(self.training_list, "upsert_task"):
                    self.training_list.upsert_task(
                        real_jid,
                        name=updated_cfg.get("name", real_jid),
                        state="RUNNING",
                        progress=0.0,
                        eta=""
                    )
