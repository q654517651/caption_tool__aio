#!/usr/bin/env python3

import flet as ft
import os
import time
from datetime import datetime
import base64
from PIL import Image
import io
import threading


# 导入服务模块
from labeling_service import LabelingService
# from translation_service import TranslationService
from settings_manager import SettingsManager
from training_manager import TrainingManager
from terminal_service import TerminalService
from dataset_manager import DatasetManager



class ImageLabelingApp:
    """主应用类"""

    def __init__(self, page: ft.Page):
        self.page = page

        self.is_web = getattr(self.page, "platform", None) == "web"
        print("[INFO] platform:", getattr(self.page, "platform", None), "is_web:", self.is_web)

        # 初始化服务模块
        self.terminal_service = TerminalService()
        # self.labeling_service = LabelingService(self.terminal_service)
        # self.translation_service = TranslationService()
        self.settings_manager = SettingsManager()
        self.training_manager = TrainingManager(self.settings_manager)
        self.dataset_manager = DatasetManager()

        self.image_cards = []
        self.current_view = "data"  # 当前视图

        # 在 MainApp 类中添加以下属性（在 __init__ 方法内）
        self.datasets = []  # 存储所有数据集
        self.current_dataset = None  # 当前选中的数据集
        self.dataset_view = None  # 数据集视图容器

        # 配置页面
        self.page.title = "图像打标系统"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.window_width = 1400
        self.page.window_height = 900

        self.file_picker = ft.FilePicker(on_result=self.on_files_selected)
        self.page.overlay.append(self.file_picker)
        self.page.update()

        self.desktop_use_file_uri = True
        # 创建UI
        self.setup_ui()

    def setup_ui(self):
        """设置UI"""
        # 创建侧边导航
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
            on_change=self.nav_change
        )

        # 创建主内容区域
        self.main_content = ft.Container(expand=True)

        # 添加到页面
        self.page.add(
            ft.Row([
                self.nav_rail,
                ft.VerticalDivider(width=1),
                self.main_content
            ], expand=True)
        )

        # 默认显示数据加载页面
        self.show_datasets_view()

    def nav_change(self, e):
        """导航变化事件"""
        selected = e.control.selected_index

        if selected == 0:
            self.show_datasets_view()
        elif selected == 1:
            self.show_training_view()
        elif selected == 2:
            self.show_terminal_view()
        elif selected == 3:
            self.show_settings_view()

    def load_datasets_list(self):
        """加载数据集列表"""
        self.datasets_list.controls.clear()

        # 获取所有数据集
        datasets = self.dataset_manager.list_datasets()

        if not datasets:
            self.datasets_list.controls.append(
                ft.Text("没有数据集，请创建新数据集", italic=True, color=ft.Colors.GREY_600)
            )
        else:
            for dataset in datasets:
                # 获取数据集统计信息
                stats = dataset.get_stats()

                # 创建数据集卡片
                dataset_card = ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.ListTile(
                                leading=ft.Icon(ft.Icons.FOLDER, color=ft.Colors.BLUE),
                                title=ft.Text(dataset.name, weight=ft.FontWeight.BOLD),
                                subtitle=ft.Text(f"创建于: {dataset.created_time}"),
                                trailing=ft.PopupMenuButton(
                                    icon=ft.Icons.MORE_VERT,
                                    items=[
                                        ft.PopupMenuItem(text="查看", icon=ft.Icons.VISIBILITY,
                                                         on_click=lambda e, d=dataset: self.view_dataset(d.dataset_id)),
                                        ft.PopupMenuItem(text="删除", icon=ft.Icons.DELETE,
                                                         on_click=lambda e, d=dataset: self.confirm_delete_dataset(
                                                             d.dataset_id))
                                    ]
                                )
                            ),
                            ft.Container(
                                content=ft.Row([
                                    ft.Text(f"图片数量: {stats['total']}"),
                                    ft.Text(f"已标注: {stats['labeled']}"),
                                    ft.Text(f"完成度: {stats['completion_rate']}%")
                                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                                padding=ft.padding.symmetric(horizontal=15, vertical=5)
                            ),
                            ft.Container(
                                content=ft.Row([
                                    ft.FilledButton(
                                        "查看内容",
                                        icon=ft.Icons.VISIBILITY,
                                        on_click=lambda e, d=dataset: self.view_dataset(d.dataset_id)
                                    ),
                                ], alignment=ft.MainAxisAlignment.END),
                                padding=ft.padding.only(right=15, bottom=10)
                            )
                        ]),
                        padding=ft.padding.all(5)
                    ),
                    elevation=2
                )

                self.datasets_list.controls.append(dataset_card)

        self.page.update()

    def create_dataset_immediately(self, e):
        """直接创建一个数据集（按时间戳命名），无需弹窗"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f"dataset_{timestamp}"
        success, message = self.dataset_manager.create_dataset(name, "")
        if success:
            self.terminal_service.log_success(f"已创建数据集: {name}")
            self.load_datasets_list()  # 立即刷新列表
        else:
            self.terminal_service.log_error(message)

    def show_create_dataset_dialog(self, e):
        """显示创建数据集对话框"""
        name_field = ft.TextField(
            label="数据集名称",
            autofocus=True,
            expand=True
        )

        description_field = ft.TextField(
            label="描述 (可选)",
            multiline=True,
            min_lines=2,
            max_lines=4,
            expand=True
        )

        def close_dialog():
            self.page.dialog.open = False
            self.page.update()

        def create_dataset():
            name = name_field.value.strip()
            description = description_field.value.strip()

            if not name:
                name_field.error_text = "请输入数据集名称"
                self.page.update()
                return

            success, message = self.dataset_manager.create_dataset(name, description)

            if success:
                close_dialog()
                self.load_datasets_list()
            else:
                name_field.error_text = message
                self.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("创建新数据集"),
            content=ft.Column([
                name_field,
                description_field
            ], tight=True, spacing=10, width=400),
            actions=[
                ft.TextButton("取消", on_click=lambda e: close_dialog()),
                ft.TextButton("创建", on_click=lambda e: create_dataset())
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def update_dataset_label(self, dataset_id, filename, label):
        """更新数据集标签"""
        self.dataset_manager.update_dataset_label(dataset_id, filename, label)

    def confirm_delete_dataset(self, dataset_id):
        """确认删除数据集"""
        dataset = self.dataset_manager.get_dataset(dataset_id)
        if not dataset:
            return

        # 1) 先生成对话框，占位按钮
        dialog = ft.AlertDialog(
            title=ft.Text("确认删除"),
            content=ft.Text(f"确定要删除数据集『{dataset.name}』吗？"),
            actions=[],  # 先空着，下面再填
            modal=True,  # 可选：点击背景不关闭
        )

        # 2) 共用的关闭逻辑，方便复用
        def close_dialog():
            self.page.close(dialog)  # 关键：把 dialog 传进来
            self.page.update()

        # 3) 删除按钮回调
        def delete_action(e):
            print("[DEBUG] delete_action fired for id:", dataset_id)
            try:
                success, msg = self.dataset_manager.delete_dataset(dataset_id)
            finally:
                close_dialog()  # 不管成功失败都先关弹窗

            if success:
                self.load_datasets_list()
                self.page.snack_bar = ft.SnackBar(ft.Text("✅ 删除成功"))
            else:
                self.page.snack_bar = ft.SnackBar(ft.Text("❌ 删除失败"))
            self.page.snack_bar.open = True
            self.page.update()

        # 4) 把按钮补进去（此时 delete_action 可以捕获到 dialog 变量）
        dialog.actions = [
            ft.TextButton("取消", on_click=lambda e: close_dialog()),
            ft.TextButton("删除", on_click=delete_action),
        ]

        # 5) 打开对话框
        self.page.open(dialog)




    # 保留这一份 ―― 放在类方法区靠前位置即可
    def pick_files_for_dataset(self, dataset_id):
        """调用系统文件对话框并把选中文件导入到指定数据集"""
        self.page.update()  # 先刷新 UI，防止焦点被占用
        self.file_picker.on_result = lambda e, d=dataset_id: self.on_files_selected(e, d)
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
                    self.load_dataset_images(dataset_id)
            else:
                self.terminal_service.log_error(message)

    def batch_label_dataset_images(self, dataset_id):
        """批量打标数据集图片"""
        dataset = self.dataset_manager.get_dataset(dataset_id)
        if not dataset or not dataset.images:
            self.terminal_service.log_error("数据集为空或不存在")
            return

        # 创建打标提示词输入对话框
        prompt_field = ft.TextField(
            label="打标提示词",
            value=LabelingService.default_labeling_prompt(),
            multiline=True,
            min_lines=3,
            max_lines=5,
            expand=True
        )

        model_choice = ft.RadioGroup(
            content=ft.Row([
                ft.Radio(value="LLM_Studio", label="LLM_Studio"),
                ft.Radio(value="GPT", label="GPT")
            ]),
            value="LLM_Studio"
        )

        def close_dialog():
            self.page.dialog.open = False
            self.page.update()

        def start_labeling():
            prompt = prompt_field.value
            model = model_choice.value

            if not prompt:
                prompt_field.error_text = "请输入提示词"
                self.page.update()
                return

            close_dialog()

            # 开始打标处理
            self.terminal_service.log_info(f"开始批量打标数据集: {dataset.name}")

            # 这里应该调用实际的打标服务
            # 为简化示例，这里只是模拟处理
            def labeling_task():
                unlabeled_images = {f: l for f, l in dataset.images.items() if not l.strip()}
                total = len(unlabeled_images)

                if total == 0:
                    self.terminal_service.log_info("所有图片已有标签")
                    return

                for i, (filename, _) in enumerate(unlabeled_images.items()):
                    # 模拟打标过程
                    self.terminal_service.log_progress(f"正在打标 {filename} ({i + 1}/{total})")
                    time.sleep(0.5)  # 模拟处理时间

                    # 生成模拟标签
                    new_label = f"AI生成的标签 - {filename}"
                    self.dataset_manager.update_dataset_label(dataset_id, filename, new_label)

                self.terminal_service.log_success(f"完成批量打标，处理了 {total} 个文件")

                # 如果当前在数据集详情页，刷新图片列表
                if self.current_view == "dataset_detail" and self.current_dataset_id == dataset_id:
                    self.page.add_action(lambda: self.load_dataset_images(dataset_id))

            # 在后台线程中执行
            threading.Thread(target=labeling_task, daemon=True).start()

        dialog = ft.AlertDialog(
            title=ft.Text("批量AI打标"),
            content=ft.Column([
                ft.Text("为未标注的图片生成标签"),
                prompt_field,
                ft.Text("选择模型:"),
                model_choice
            ], tight=True, spacing=10, width=400),
            actions=[
                ft.TextButton("取消", on_click=lambda e: close_dialog()),
                ft.TextButton("开始打标", on_click=lambda e: start_labeling())
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def batch_translate_dataset_labels(self, dataset_id):
        """批量翻译数据集标签"""
        dataset = self.dataset_manager.get_dataset(dataset_id)
        if not dataset:
            return

        # 创建翻译设置对话框
        source_lang = ft.Dropdown(
            label="源语言",
            options=[
                ft.dropdown.Option("auto", "自动检测"),
                ft.dropdown.Option("en", "英语"),
                ft.dropdown.Option("zh", "中文"),
                ft.dropdown.Option("ja", "日语")
            ],
            value="auto"
        )

        target_lang = ft.Dropdown(
            label="目标语言",
            options=[
                ft.dropdown.Option("en", "英语"),
                ft.dropdown.Option("zh", "中文"),
                ft.dropdown.Option("ja", "日语")
            ],
            value="zh"
        )

        def close_dialog():
            self.page.dialog.open = False
            self.page.update()

        def start_translation():
            from_lang = source_lang.value
            to_lang = target_lang.value

            if from_lang == to_lang and from_lang != "auto":
                self.terminal_service.log_warning("源语言和目标语言相同")
                return

            close_dialog()

            # 开始翻译处理
            self.terminal_service.log_info(f"开始批量翻译数据集: {dataset.name}")

            # 这里应该调用实际的翻译服务
            # 为简化示例，这里只是模拟处理
            def translation_task():
                labeled_images = {f: l for f, l in dataset.images.items() if l.strip()}
                total = len(labeled_images)

                if total == 0:
                    self.terminal_service.log_info("没有可翻译的标签")
                    return

                for i, (filename, label) in enumerate(labeled_images.items()):
                    # 模拟翻译过程
                    self.terminal_service.log_progress(f"正在翻译 {filename} ({i + 1}/{total})")
                    time.sleep(0.5)  # 模拟处理时间

                    # 生成模拟翻译
                    translated = f"翻译后的标签: {label}"
                    self.dataset_manager.update_dataset_label(dataset_id, filename, translated)

                self.terminal_service.log_success(f"完成批量翻译，处理了 {total} 个文件")

                # 如果当前在数据集详情页，刷新图片列表
                if self.current_view == "dataset_detail" and self.current_dataset_id == dataset_id:
                    self.page.add_action(lambda: self.load_dataset_images(dataset_id))

            # 在后台线程中执行
            threading.Thread(target=translation_task, daemon=True).start()

        dialog = ft.AlertDialog(
            title=ft.Text("批量翻译标签"),
            content=ft.Column([
                ft.Text("翻译所有标签"),
                source_lang,
                target_lang
            ], tight=True, spacing=10, width=400),
            actions=[
                ft.TextButton("取消", on_click=lambda e: close_dialog()),
                ft.TextButton("开始翻译", on_click=lambda e: start_translation())
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def view_dataset(self, dataset_id):
        """查看数据集详情"""
        dataset = self.dataset_manager.get_dataset(dataset_id)
        if not dataset:
            self.terminal_service.log_error("数据集不存在")
            return

        self.current_view = "dataset_detail"
        self.current_dataset_id = dataset_id

        self.terminal_service.log_info(f"正在加载数据集: {dataset.name}")

        # 返回和操作按钮
        back_btn = ft.IconButton(
            icon=ft.Icons.ARROW_BACK,
            tooltip="返回数据集列表",
            on_click=lambda e: self.show_datasets_view()
        )

        batch_buttons = ft.Row([
            ft.ElevatedButton("批量翻译", icon=ft.Icons.TRANSLATE,
                              on_click=lambda e: self.batch_translate_dataset_labels(dataset_id)),
            ft.ElevatedButton("批量打标", icon=ft.Icons.AUTO_AWESOME,
                              on_click=lambda e: self.batch_label_dataset_images(dataset_id)),
            ft.ElevatedButton("导入文件", icon=ft.Icons.UPLOAD_FILE,
                              on_click=lambda e, d=dataset_id: self.pick_files_for_dataset(d))
        ])

        # 使用 GridView 替代 ListView
        self.image_grid = ft.GridView(
            expand=1,
            runs_count=0,  # 自动计算列数
            max_extent=350,  # 每个项目的最大宽度
            child_aspect_ratio=0.75,  # 宽高比
            spacing=15,
            run_spacing=15,
            padding=ft.padding.all(20)
        )

        # 创建内容区域
        content_area = ft.Column([
            ft.Container(
                content=ft.Row([
                    back_btn,
                    ft.Text(f"数据集: {dataset.name}", size=24, weight=ft.FontWeight.BOLD)
                ]),
                padding=ft.padding.all(20)
            ),
            ft.Container(content=batch_buttons, padding=ft.padding.symmetric(horizontal=20)),
            ft.Container(
                content=self.image_grid,
                expand=True,
                bgcolor=ft.Colors.GREY_50,
                border_radius=10
            )
        ], expand=True)

        # 直接渲染内容（去掉拖拽区域）
        self.main_content.content = content_area

        self.page.update()

        # 延迟加载图片，确保UI先渲染
        def delayed_load():
            self.load_dataset_images(dataset_id)

        # 使用page.run_task来在下一个事件循环中加载图片
        import threading
        threading.Timer(0.1, delayed_load).start()

    def load_dataset_images(self, dataset_id):
        """加载数据集图片（网格显示）
        - PC：使用绝对路径（不加 file://）
        - Web：使用相对 assets_dir 的 URL（datasets/<id>/...）
        - 列表用中清晰度；必要时回退原图
        """
        import os

        self.image_grid.controls.clear()

        dataset = self.dataset_manager.get_dataset(dataset_id)
        if not dataset:
            self.terminal_service.log_error("数据集不存在")
            self.page.update()
            return

        if not dataset.images:
            self.image_grid.controls.append(
                ft.Container(
                    content=ft.Text("暂无图片，点击上方「导入文件」添加",
                                    size=18, color=ft.Colors.GREY_600, italic=True),
                    alignment=ft.alignment.center,
                    height=200
                )
            )
            self.page.update()
            return

        # 平台判定
        is_web = getattr(self.page, "platform", None) == "web"

        # 统一取图：Web -> url_for(相对路径)；PC -> 绝对路径（不加 file://）
        def get_image_src(filename: str, kind: str = "medium") -> str | None:
            try:
                if is_web:
                    # 例如: datasets/<id>/medium/xxx.jpg 或回退 datasets/<id>/images/<file>
                    rel = self.dataset_manager.url_for(dataset_id, filename, kind=kind)
                    if not rel:
                        return None
                    # （可选）存在性校验
                    abs_check = os.path.join(self.dataset_manager.workspace_root, rel.replace("/", os.sep))
                    if not os.path.exists(abs_check):
                        self.terminal_service.log_error(f"Web路径不存在: {abs_check}")
                    return rel
                else:
                    # 绝对路径（PC 最稳）
                    if kind == "medium":
                        p = self.dataset_manager.ensure_medium(dataset_id, filename)
                    else:
                        p = os.path.join(self.dataset_manager.get_images_dir(dataset_id), filename)
                    p = os.path.abspath(p)
                    if not os.path.exists(p):
                        self.terminal_service.log_error(f"PC文件不存在: {p}")
                        return None
                    return p
            except Exception as ex:
                self.terminal_service.log_error(f"获取图片路径失败 {filename}: {ex}")
                return None

        # 渲染网格
        for filename, label in dataset.images.items():
            try:
                src_medium = get_image_src(filename, "medium")
                if not src_medium:
                    continue

                image_widget = ft.Image(
                    src=src_medium,  # PC: 绝对路径；Web: 相对 assets_dir 的路径
                    fit=ft.ImageFit.COVER,
                    error_content=ft.Container(
                        content=ft.Icon(ft.Icons.BROKEN_IMAGE, size=50, color=ft.Colors.GREY),
                        alignment=ft.alignment.center,
                        bgcolor=ft.Colors.GREY_100
                    )
                )

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
                                value=label,
                                multiline=True,
                                min_lines=2,
                                max_lines=3,
                                text_size=12,
                                border=ft.InputBorder.NONE,
                                filled=True,
                                fill_color=ft.Colors.GREY_100,
                                on_change=lambda e, f=filename: self.update_dataset_label(
                                    dataset_id, f, e.control.value
                                ),
                            ),
                            padding=ft.padding.only(left=10, right=10, bottom=10)
                        )
                    ], spacing=0),
                    bgcolor=ft.Colors.WHITE,
                    border_radius=8,
                    shadow=ft.BoxShadow(
                        spread_radius=1,
                        blur_radius=4,
                        color=ft.Colors.BLACK12,
                        offset=ft.Offset(0, 2)
                    )
                )

                self.image_grid.controls.append(image_card)

            except Exception as ex:
                self.terminal_service.log_error(f"加载图片失败 {filename}: {ex}")
                self.image_grid.controls.append(
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

    def show_datasets_view(self):
        """显示数据集视图（无拖拽，统一用 FilePicker）"""
        self.current_view = "datasets"

        # 数据集列表
        self.datasets_list = ft.ListView(
            expand=1,
            spacing=10,
            padding=10,
            auto_scroll=True
        )

        # 新建数据集
        create_dataset_btn = ft.ElevatedButton(
            text="创建新数据集",
            icon=ft.Icons.ADD,
            on_click=self.create_dataset_immediately
        )

        # 刷新
        refresh_btn = ft.ElevatedButton(
            text="刷新列表",
            icon=ft.Icons.REFRESH,
            on_click=lambda e: self.load_datasets_list()
        )

        # 顶部工具条（只保留按钮）
        toolbar = ft.Container(
            content=ft.Row(
                [create_dataset_btn, refresh_btn],
                alignment=ft.MainAxisAlignment.START,
            ),
            padding=ft.padding.symmetric(horizontal=20)
        )

        # 数据集列表容器
        datasets_box = ft.Container(
            content=self.datasets_list,
            expand=True,
            padding=ft.padding.all(20),
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=10,
            margin=ft.margin.symmetric(horizontal=20),
        )

        # 组装
        datasets_view = ft.Column(
            [
                ft.Container(
                    content=ft.Text("📊 数据集管理", size=24, weight=ft.FontWeight.BOLD),
                    padding=ft.padding.all(20),
                ),
                toolbar,
                datasets_box,
                # 🧹 已去掉拖拽提示区与 DragTarget
            ],
            expand=True,
        )

        self.main_content.content = datasets_view
        self.page.update()

        # 加载数据集列表
        self.load_datasets_list()

    def show_training_view(self):
        """显示模型训练视图"""
        self.current_view = "training"

        # 训练配置
        self.dataset_path_input = ft.TextField(
            label="数据集路径",
            hint_text="选择包含datasets文件夹的根目录",
            expand=True
        )

        self.dataset_type_radio = ft.RadioGroup(
            content=ft.Row([
                ft.Radio(value="image", label="图片"),
                ft.Radio(value="video", label="视频")
            ]),
            value="image"
        )

        self.resolution_input = ft.TextField(
            label="分辨率 (宽,高)",
            value=self.settings_manager.get("default_resolution"),
            width=150
        )

        self.batch_size_input = ft.TextField(
            label="批次大小",
            value=self.settings_manager.get("default_batch_size"),
            width=100
        )

        self.num_repeats_input = ft.TextField(
            label="重复次数",
            value="10",
            width=100
        )

        self.max_epochs_input = ft.TextField(
            label="最大轮数",
            value=self.settings_manager.get("default_epochs"),
            width=100
        )

        self.learning_rate_input = ft.TextField(
            label="学习率",
            value=self.settings_manager.get("default_lr"),
            width=150
        )

        self.lora_name_input = ft.TextField(
            label="模型名称",
            value="my_lora",
            width=200
        )

        self.sample_prompts_input = ft.TextField(
            label="采样提示词",
            value="a beautiful landscape",
            multiline=True,
            min_lines=3,
            max_lines=5
        )

        # 控制按钮
        start_training_btn = ft.ElevatedButton(
            text="🚀 开始训练",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self.start_training
        )

        stop_training_btn = ft.ElevatedButton(
            text="⏹️ 停止训练",
            icon=ft.Icons.STOP,
            on_click=self.stop_training
        )

        generate_script_btn = ft.ElevatedButton(
            text="📝 生成脚本",
            icon=ft.Icons.CODE,
            on_click=self.generate_training_script
        )

        # 输出显示
        self.training_output = ft.TextField(
            label="训练输出",
            multiline=True,
            min_lines=15,
            max_lines=20,
            read_only=True,
            expand=True
        )

        # TensorBoard
        self.tensorboard_container = ft.Container(
            content=ft.Text("TensorBoard将在训练开始后显示"),
            height=400,
            border=ft.border.all(1, ft.Colors.GREY_300),
            border_radius=ft.border_radius.all(10)
        )

        # 组装训练视图
        training_view = ft.Column([
            ft.Container(
                content=ft.Text("🚀 模型训练", size=24, weight=ft.FontWeight.BOLD),
                padding=ft.padding.all(20)
            ),
            ft.Container(
                content=ft.Column([
                    # 数据集配置
                    ft.Text("数据集配置", size=16, weight=ft.FontWeight.BOLD),
                    self.dataset_path_input,
                    ft.Row([
                        ft.Text("数据集类型:", weight=ft.FontWeight.BOLD),
                        self.dataset_type_radio
                    ]),

                    # 训练参数
                    ft.Text("训练参数", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        self.resolution_input,
                        self.batch_size_input,
                        self.num_repeats_input,
                        self.max_epochs_input
                    ]),
                    ft.Row([
                        self.learning_rate_input,
                        self.lora_name_input
                    ]),
                    self.sample_prompts_input,

                    # 控制按钮
                    ft.Row([
                        start_training_btn,
                        stop_training_btn,
                        generate_script_btn
                    ])
                ]),
                padding=ft.padding.all(20)
            ),
            # 输出区域
            ft.Container(
                content=ft.Row([
                    ft.Container(
                        content=self.training_output,
                        expand=2
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("TensorBoard", weight=ft.FontWeight.BOLD),
                            self.tensorboard_container
                        ]),
                        expand=1
                    )
                ], expand=True),
                expand=True,
                padding=ft.padding.symmetric(horizontal=20, vertical=10)
            )
        ], scroll=ft.ScrollMode.AUTO, expand=True)

        self.main_content.content = training_view
        self.page.update()

    def show_settings_view(self):
        """显示设置视图"""
        self.current_view = "settings"

        # 路径设置
        self.musubi_dir_input = ft.TextField(
            label="Musubi-Tuner目录",
            value=self.settings_manager.get("musubi_dir"),
            expand=True
        )

        self.vae_path_input = ft.TextField(
            label="VAE模型路径",
            value=self.settings_manager.get("vae_path"),
            expand=True
        )

        self.clip_path_input = ft.TextField(
            label="CLIP模型路径",
            value=self.settings_manager.get("clip_path"),
            expand=True
        )

        self.t5_path_input = ft.TextField(
            label="T5模型路径",
            value=self.settings_manager.get("t5_path"),
            expand=True
        )

        self.unet_path_input = ft.TextField(
            label="UNet模型路径",
            value=self.settings_manager.get("unet_path"),
            expand=True
        )

        # 保存按钮
        save_settings_btn = ft.ElevatedButton(
            text="💾 保存设置",
            icon=ft.Icons.SAVE,
            on_click=self.save_settings
        )

        self.settings_status = ft.Text("", color=ft.Colors.GREEN)

        # 组装设置视图
        settings_view = ft.Column([
            ft.Container(
                content=ft.Text("⚙️ 设置", size=24, weight=ft.FontWeight.BOLD),
                padding=ft.padding.all(20)
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("模型路径配置", size=16, weight=ft.FontWeight.BOLD),
                    self.musubi_dir_input,
                    self.vae_path_input,
                    self.clip_path_input,
                    self.t5_path_input,
                    self.unet_path_input,
                    save_settings_btn,
                    self.settings_status
                ]),
                padding=ft.padding.all(20)
            )
        ], scroll=ft.ScrollMode.AUTO)

        self.main_content.content = settings_view
        self.page.update()

    def show_terminal_view(self):
        """显示终端视图"""
        self.current_view = "terminal"

        # 创建终端显示区域
        self.terminal_display = ft.TextField(
            value="",
            multiline=True,
            min_lines=30,
            max_lines=50,
            read_only=True,
            bgcolor=ft.Colors.BLACK,
            color=ft.Colors.GREEN,
            text_style=ft.TextStyle(font_family="Courier New", size=12),
            expand=True
        )

        # 加载现有日志
        existing_logs = self.terminal_service.get_logs_text()
        if existing_logs:
            self.terminal_display.value = existing_logs

        # 清除所有旧回调函数（关键步骤！）
        self.terminal_service.clear_all_callbacks()

        # 注册终端更新回调
        def terminal_callback(log_entry: str):
            if log_entry == "CLEAR_TERMINAL":
                self.terminal_display.value = ""
            else:
                # 添加颜色标记
                if "[ERROR]" in log_entry:
                    self.terminal_display.value += f"🔴 {log_entry}\n"
                elif "[SUCCESS]" in log_entry:
                    self.terminal_display.value += f"🟢 {log_entry}\n"
                elif "[WARN]" in log_entry:
                    self.terminal_display.value += f"🟡 {log_entry}\n"
                elif "[PROGRESS]" in log_entry:
                    self.terminal_display.value += f"🔵 {log_entry}\n"
                else:
                    self.terminal_display.value += f"⚪ {log_entry}\n"

            # 自动滚动到底部
            self.terminal_display.selection = ft.TextSelection(
                base_offset=len(self.terminal_display.value),
                extent_offset=len(self.terminal_display.value)
            )

            # 限制显示的行数，避免过多内容
            lines = self.terminal_display.value.split('\n')
            if len(lines) > 1000:
                self.terminal_display.value = '\n'.join(lines[-1000:])

            self.page.update()

        self.terminal_service.register_callback(terminal_callback)

        # 清空按钮
        clear_btn = ft.ElevatedButton(
            text="清空终端",
            icon=ft.Icons.CLEAR,
            on_click=lambda e: self.terminal_service.clear_logs()
        )

        # 测试按钮组
        test_buttons = ft.Row([
            ft.ElevatedButton(
                text="测试信息",
                on_click=lambda e: self.terminal_service.log_info("这是一条测试信息")
            ),
            ft.ElevatedButton(
                text="测试错误",
                on_click=lambda e: self.terminal_service.log_error("这是一条测试错误")
            ),
            ft.ElevatedButton(
                text="测试成功",
                on_click=lambda e: self.terminal_service.log_success("这是一条测试成功消息")
            )
        ])

        # 组装终端视图
        terminal_view = ft.Column([
            ft.Container(
                content=ft.Text("💻 系统终端", size=24, weight=ft.FontWeight.BOLD),
                padding=ft.padding.all(20)
            ),
            ft.Container(
                content=ft.Row([
                    clear_btn,
                    ft.Text("  |  测试日志:"),
                    test_buttons
                ]),
                padding=ft.padding.symmetric(horizontal=20)
            ),
            ft.Container(
                content=ft.Container(
                    content=self.terminal_display,
                    expand=True,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=ft.border_radius.all(10),
                    padding=ft.padding.all(10)
                ),
                expand=True,
                padding=ft.padding.all(20)
            )
        ], expand=True)

        self.main_content.content = terminal_view
        self.page.update()

        self.terminal_service.log_info("终端视图已打开")


    # 训练相关方法
    def start_training(self, e):
        """开始训练"""
        from training_manager import TrainingConfig

        config = TrainingConfig(
            dataset_path=self.dataset_path_input.value,
            dataset_type=self.dataset_type_radio.value,
            resolution=self.resolution_input.value,
            batch_size=self.batch_size_input.value,
            num_repeats=self.num_repeats_input.value,
            max_epochs=self.max_epochs_input.value,
            learning_rate=self.learning_rate_input.value,
            lora_name=self.lora_name_input.value,
            sample_prompts=self.sample_prompts_input.value
        )

        def progress_callback(message):
            self.page.add_action(lambda: (
                setattr(self.training_output, "value", self.training_output.value + message + "\n"),
                self.page.update()
            ))

        def training_task():
            tb_port = self.training_manager.run_training(config, progress_callback)
            if tb_port > 0:
                self.page.add_action(lambda: (
                    setattr(self.tensorboard_container, "content",
                            ft.Text(f"TensorBoard: http://localhost:{tb_port}")),
                    self.page.update()
                ))

        threading.Thread(target=training_task, daemon=True).start()

    def stop_training(self, e):
        """停止训练"""
        result = self.training_manager.stop_training()
        self.training_output.value += f"\n{result}\n"
        self.tensorboard_container.content = ft.Text("TensorBoard已停止")
        self.page.update()

    def generate_training_script(self, e):
        """生成训练脚本"""
        from training_manager import TrainingConfig

        try:
            config = TrainingConfig(
                dataset_path=self.dataset_path_input.value,
                dataset_type=self.dataset_type_radio.value,
                resolution=self.resolution_input.value,
                batch_size=self.batch_size_input.value,
                num_repeats=self.num_repeats_input.value,
                max_epochs=self.max_epochs_input.value,
                learning_rate=self.learning_rate_input.value,
                lora_name=self.lora_name_input.value,
                sample_prompts=self.sample_prompts_input.value
            )

            script_path = self.training_manager.generate_training_script(config)
            self.training_output.value += f"\n脚本已生成: {script_path}\n"
            self.page.update()

        except Exception as ex:
            self.training_output.value += f"\n生成脚本失败: {str(ex)}\n"
            self.page.update()

    def save_settings(self, e):
        """保存设置"""
        self.settings_manager.set("musubi_dir", self.musubi_dir_input.value)
        self.settings_manager.set("vae_path", self.vae_path_input.value)
        self.settings_manager.set("clip_path", self.clip_path_input.value)
        self.settings_manager.set("t5_path", self.t5_path_input.value)
        self.settings_manager.set("unet_path", self.unet_path_input.value)

        self.settings_status.value = "✅ 设置已保存"
        self.page.update()



def main(page: ft.Page):
    """主函数"""
    app = ImageLabelingApp(page)


if __name__ == "__main__":
    ft.app(target=main, assets_dir=os.path.abspath("./workspace"))

# view=ft.WEB_BROWSER