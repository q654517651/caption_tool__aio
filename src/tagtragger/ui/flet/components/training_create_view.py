"""
Training Create View - 新建训练任务视图
"""

import flet as ft
from typing import Callable, List, Dict, Any
from ....core.training.models import TrainingConfig, TrainingType, QwenImageConfig
from ....core.dataset.models import Dataset


class TrainingCreateView:
    """新建训练任务视图"""

    def __init__(self,
                 page: ft.Page,
                 dataset_manager,
                 training_manager,
                 on_back: Callable[[], None],
                 toast_service):
        self.page = page
        self.dataset_manager = dataset_manager
        self.training_manager = training_manager
        self.on_back = on_back
        self.toast_service = toast_service

        # UI组件
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
                ft.Text("创建训练任务", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True),
                ft.ElevatedButton(
                    "创建训练",
                    icon=ft.Icons.CHECK,
                    on_click=self._create_training,
                    style=ft.ButtonStyle(
                        color=ft.Colors.WHITE,
                        bgcolor=ft.Colors.BLUE
                    )
                ),
            ]),
            padding=ft.padding.all(20)
        )

        # 任务基本信息
        self.task_name_field = ft.TextField(
            label="任务名称",
            hint_text="输入训练任务的名称",
            value=f"Qwen-Image训练_{len(self.training_manager.list_tasks()) + 1}",
            width=400
        )

        # 数据集选择
        datasets = self.dataset_manager.list_datasets()
        dataset_options = [ft.dropdown.Option(ds.dataset_id, ds.name) for ds in datasets]
        
        self.dataset_dropdown = ft.Dropdown(
            label="选择数据集",
            hint_text="请选择用于训练的数据集",
            options=dataset_options,
            value=datasets[0].dataset_id if datasets else None,
            width=400
        )

        # 训练类型选择
        self.training_type_dropdown = ft.Dropdown(
            label="训练类型",
            options=[
                ft.dropdown.Option(TrainingType.QWEN_IMAGE_LORA.value, "Qwen-Image LoRA"),
            ],
            value=TrainingType.QWEN_IMAGE_LORA.value,
            width=400,
            on_change=self._on_training_type_change
        )

        basic_info_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("基础信息", size=18, weight=ft.FontWeight.BOLD),
                    self.task_name_field,
                    self.dataset_dropdown,
                    self.training_type_dropdown
                ], spacing=15),
                padding=20
            )
        )

        # Qwen-Image训练参数
        self._build_qwen_image_params()

        # 主容器
        self.root_container = ft.Column([
            toolbar,
            ft.Divider(),
            ft.Container(
                content=ft.Column([
                    basic_info_card,
                    self.qwen_params_card
                ], spacing=20),
                padding=ft.padding.symmetric(horizontal=20),
                expand=True
            )
        ], expand=True, scroll=ft.ScrollMode.AUTO)

    def _build_qwen_image_params(self):
        """构建Qwen-Image训练参数UI"""
        # 基础训练参数
        self.epochs_field = ft.TextField(
            label="训练轮数",
            value="16",
            width=200,
            input_filter=ft.NumbersOnlyInputFilter()
        )

        self.batch_size_field = ft.TextField(
            label="批次大小",
            value="1",
            width=200,
            input_filter=ft.NumbersOnlyInputFilter()
        )

        self.learning_rate_field = ft.TextField(
            label="学习率",
            value="1e-4",
            width=200
        )

        self.network_dim_field = ft.TextField(
            label="网络维度",
            value="32",
            width=200,
            input_filter=ft.NumbersOnlyInputFilter()
        )

        self.network_alpha_field = ft.TextField(
            label="网络Alpha",
            value="16",
            width=200,
            input_filter=ft.NumbersOnlyInputFilter()
        )

        self.resolution_field = ft.TextField(
            label="分辨率 (宽,高)",
            value="1024,1024",
            width=200
        )

        self.repeats_field = ft.TextField(
            label="重复次数",
            value="1",
            width=200,
            input_filter=ft.NumbersOnlyInputFilter()
        )

        # 高级参数开关
        self.advanced_switch = ft.Switch(
            label="显示高级参数",
            on_change=self._toggle_advanced_params
        )

        # 高级参数 - 默认隐藏
        self.mixed_precision_dropdown = ft.Dropdown(
            label="混合精度",
            options=[
                ft.dropdown.Option("bf16", "BF16"),
                ft.dropdown.Option("fp16", "FP16"),
                ft.dropdown.Option("no", "无")
            ],
            value="bf16",
            width=200
        )

        self.timestep_sampling_dropdown = ft.Dropdown(
            label="时间步采样",
            options=[
                ft.dropdown.Option("shift", "Shift"),
                ft.dropdown.Option("qwen_shift", "Qwen Shift"),
                ft.dropdown.Option("uniform", "Uniform")
            ],
            value="shift",
            width=200
        )

        self.weighting_scheme_dropdown = ft.Dropdown(
            label="权重方案",
            options=[
                ft.dropdown.Option("none", "None"),
                ft.dropdown.Option("logit_normal", "Logit Normal"),
                ft.dropdown.Option("mode", "Mode"),
                ft.dropdown.Option("cosmap", "Cosmap")
            ],
            value="none",
            width=200
        )

        self.discrete_flow_shift_field = ft.TextField(
            label="离散流偏移值",
            value="3.0",
            width=200
        )

        self.optimizer_type_dropdown = ft.Dropdown(
            label="优化器类型",
            options=[
                ft.dropdown.Option("adamw8bit", "AdamW 8bit"),
                ft.dropdown.Option("adamw", "AdamW"),
                ft.dropdown.Option("adafactor", "Adafactor"),
                ft.dropdown.Option("lion", "Lion")
            ],
            value="adamw8bit",
            width=200
        )

        self.gradient_checkpointing_switch = ft.Switch(
            label="梯度检查点",
            value=True
        )

        self.fp8_options_switches = ft.Column([
            ft.Switch(label="FP8 Base", value=False),
            ft.Switch(label="FP8 Scaled", value=False),
            ft.Switch(label="FP8 VL (文本编码器)", value=False)
        ])

        self.attention_type_dropdown = ft.Dropdown(
            label="注意力机制",
            options=[
                ft.dropdown.Option("sdpa", "SDPA"),
                ft.dropdown.Option("xformers", "XFormers"),
                ft.dropdown.Option("flash_attn", "Flash Attention")
            ],
            value="sdpa",
            width=200
        )

        self.split_attn_switch = ft.Switch(
            label="分割注意力",
            value=False
        )

        self.blocks_to_swap_field = ft.TextField(
            label="交换块数量",
            value="0",
            width=200,
            input_filter=ft.NumbersOnlyInputFilter()
        )

        # 数据加载参数
        self.max_data_loader_n_workers_field = ft.TextField(
            label="数据加载线程数",
            value="2",
            width=200,
            input_filter=ft.NumbersOnlyInputFilter()
        )

        self.persistent_data_loader_workers_switch = ft.Switch(
            label="持久化数据加载器",
            value=True
        )

        self.seed_field = ft.TextField(
            label="随机种子",
            value="42",
            width=200,
            input_filter=ft.NumbersOnlyInputFilter()
        )

        self.advanced_params_container = ft.Container(
            content=ft.Column([
                ft.Text("高级参数", size=16, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Row([self.mixed_precision_dropdown, self.timestep_sampling_dropdown]),
                ft.Row([self.weighting_scheme_dropdown, self.discrete_flow_shift_field]),
                ft.Row([self.optimizer_type_dropdown, self.network_alpha_field]),
                self.gradient_checkpointing_switch,
                ft.Text("FP8 选项", size=14),
                self.fp8_options_switches,
                ft.Row([self.attention_type_dropdown, self.split_attn_switch]),
                ft.Row([self.blocks_to_swap_field, self.resolution_field]),
                ft.Row([self.repeats_field, self.seed_field]),
                ft.Row([self.max_data_loader_n_workers_field, self.persistent_data_loader_workers_switch])
            ], spacing=15),
            visible=False
        )

        self.qwen_params_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Qwen-Image 训练参数", size=18, weight=ft.FontWeight.BOLD),
                    ft.Divider(),
                    ft.Row([self.epochs_field, self.batch_size_field]),
                    ft.Row([self.learning_rate_field, self.network_dim_field]),
                    ft.Divider(),
                    self.advanced_switch,
                    self.advanced_params_container
                ], spacing=15),
                padding=20
            )
        )

    def _toggle_advanced_params(self, e):
        """切换高级参数显示/隐藏"""
        self.advanced_params_container.visible = self.advanced_switch.value
        self.page.update()

    def _on_training_type_change(self, e):
        """训练类型改变时的回调"""
        # 目前只有一种类型，但为将来扩展保留
        pass

    def _create_training(self, e):
        """创建训练任务"""
        try:
            # 验证必填字段
            task_name = self.task_name_field.value.strip()
            dataset_id = self.dataset_dropdown.value
            training_type = self.training_type_dropdown.value

            if not task_name:
                self.toast_service.show("请输入任务名称", "warning")
                return

            if not dataset_id:
                self.toast_service.show("请选择数据集", "warning")
                return

            # 获取训练参数
            config = self._get_training_config(task_name, dataset_id, training_type)
            
            # 创建训练任务
            task_id = self.training_manager.create_task(config)
            
            self.toast_service.show(f"训练任务创建成功: {task_name}", "success")
            
            # 返回训练列表页面
            self.on_back()
            
        except Exception as ex:
            self.toast_service.show(f"创建失败: {str(ex)}", "error")

    def _get_training_config(self, task_name: str, dataset_id: str, training_type: str) -> TrainingConfig:
        """获取训练配置"""
        # 基础参数
        epochs = int(self.epochs_field.value) if self.epochs_field.value else 16
        batch_size = int(self.batch_size_field.value) if self.batch_size_field.value else 1
        learning_rate = float(self.learning_rate_field.value) if self.learning_rate_field.value else 1e-4
        network_dim = int(self.network_dim_field.value) if self.network_dim_field.value else 32
        network_alpha = int(self.network_alpha_field.value) if self.network_alpha_field.value else 16
        resolution = self.resolution_field.value if self.resolution_field.value else "1024,1024"
        repeats = int(self.repeats_field.value) if self.repeats_field.value else 1
        discrete_flow_shift = float(self.discrete_flow_shift_field.value) if self.discrete_flow_shift_field.value else 3.0
        blocks_to_swap = int(self.blocks_to_swap_field.value) if self.blocks_to_swap_field.value else 0
        max_data_loader_n_workers = int(self.max_data_loader_n_workers_field.value) if self.max_data_loader_n_workers_field.value else 2
        seed = int(self.seed_field.value) if self.seed_field.value else 42

        # 创建Qwen-Image特定配置
        qwen_config = QwenImageConfig(
            mixed_precision=self.mixed_precision_dropdown.value,
            timestep_sampling=self.timestep_sampling_dropdown.value,
            weighting_scheme=self.weighting_scheme_dropdown.value,
            discrete_flow_shift=discrete_flow_shift,
            optimizer_type=self.optimizer_type_dropdown.value,
            gradient_checkpointing=self.gradient_checkpointing_switch.value,
            fp8_base=self._get_switch_value(self.fp8_options_switches.controls[0]),
            fp8_scaled=self._get_switch_value(self.fp8_options_switches.controls[1]),
            fp8_vl=self._get_switch_value(self.fp8_options_switches.controls[2]),
            attention_type=self.attention_type_dropdown.value,
            split_attn=self.split_attn_switch.value,
            blocks_to_swap=blocks_to_swap
        )

        # 创建完整配置
        config = TrainingConfig(
            task_id="",  # 会在创建时自动生成
            name=task_name,
            training_type=TrainingType(training_type),
            dataset_id=dataset_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            resolution=resolution,
            network_dim=network_dim,
            network_alpha=network_alpha,
            repeats=repeats,
            max_data_loader_n_workers=max_data_loader_n_workers,
            persistent_data_loader_workers=self.persistent_data_loader_workers_switch.value,
            seed=seed,
            qwen_config=qwen_config
        )

        return config

    def _get_switch_value(self, switch: ft.Switch) -> bool:
        """获取开关控件的值"""
        return switch.value if switch else False

    def build(self) -> ft.Container:
        """构建并返回根容器"""
        return self.root_container