"""
Training View - 新架构的训练管理视图
"""

import flet as ft
from typing import Callable, List, Dict, Any
from ....core.training.models import TrainingConfig, TrainingType, TrainingState

class TrainingListView:
    """训练任务列表视图"""
    
    def __init__(self,
                 page: ft.Page,
                 training_manager,
                 dataset_manager,
                 on_open_task: Callable[[str], None],
                 toast_service):
        self.page = page
        self.training_manager = training_manager
        self.dataset_manager = dataset_manager
        self.on_open_task = on_open_task
        self.toast_service = toast_service
        
        # UI组件
        self.task_list = ft.Column(
            expand=True,
            spacing=6,
            scroll=ft.ScrollMode.AUTO
        )
        
        self.root_container = None
        self._build_ui()
    
    def _build_ui(self):
        """构建UI"""
        # 顶部工具栏
        toolbar = ft.Container(
            content=ft.Row([
                ft.ElevatedButton(
                    "创建训练任务",
                    icon=ft.Icons.ADD,
                    on_click=self._create_task
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
                content=ft.Text("🚀 模型训练", size=24, weight=ft.FontWeight.BOLD),
                padding=ft.padding.all(20)
            ),
            toolbar,
            ft.Container(
                content=self.task_list,
                expand=True,
                padding=ft.padding.all(20),
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=10,
                margin=ft.margin.symmetric(horizontal=20),
            )
        ], expand=True)
    
    def _create_task(self, e):
        """创建训练任务"""
        try:
            # 检查数据集
            datasets = self.dataset_manager.list_datasets()
            if not datasets:
                self.toast_service.show("请先创建数据集", "warning")
                return
            
            # 创建配置对话框
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
            
            # 训练类型选择
            training_type_dropdown = ft.Dropdown(
                label="训练类型",
                options=[
                    ft.dropdown.Option(TrainingType.QWEN_IMAGE_LORA.value, "Qwen-Image LoRA"),
                    # 可以添加更多训练类型
                ],
                value=TrainingType.QWEN_IMAGE_LORA.value,
                width=300
            )
            
            def create(e):
                name = task_name_field.value.strip()
                dataset_id = dataset_dropdown.value
                training_type = training_type_dropdown.value
                
                if not name:
                    self.toast_service.show("请输入任务名称", "warning")
                    return
                
                if not dataset_id:
                    self.toast_service.show("请选择数据集", "warning")
                    return
                
                dialog.open = False
                self.page.update()
                
                # 获取全局配置的模型路径
                from ....config import get_config
                from ....core.training.models import QwenImageConfig
                
                app_config = get_config()
                
                # 根据训练类型检查对应的模型路径
                if training_type == TrainingType.QWEN_IMAGE_LORA.value:
                    qwen_paths = app_config.model_paths.qwen_image
                    missing_paths = []
                    if not qwen_paths.dit_path:
                        missing_paths.append("DiT模型路径")
                    if not qwen_paths.vae_path:
                        missing_paths.append("VAE模型路径") 
                    if not qwen_paths.text_encoder_path:
                        missing_paths.append("Text Encoder路径")
                    
                    if missing_paths:
                        self.toast_service.show(f"请先在设置中配置Qwen-Image模型路径: {', '.join(missing_paths)}", "warning")
                        return
                    
                    # 创建Qwen配置，使用对应的模型路径
                    qwen_config = QwenImageConfig(
                        dit_path=qwen_paths.dit_path,
                        vae_path=qwen_paths.vae_path,
                        text_encoder_path=qwen_paths.text_encoder_path
                    )
                else:
                    # 其他训练类型的配置
                    qwen_config = QwenImageConfig()  # 默认空配置
                
                # 创建训练配置
                config = TrainingConfig(
                    task_id="",  # 会在创建时自动生成
                    name=name,
                    training_type=TrainingType(training_type),
                    dataset_id=dataset_id,
                    qwen_config=qwen_config
                )
                
                try:
                    task_id = self.training_manager.create_task(config)
                    self.toast_service.show(f"训练任务创建成功: {name}", "success")
                    self.refresh()
                except Exception as ex:
                    self.toast_service.show(f"创建失败: {str(ex)}", "error")
            
            def cancel(e):
                dialog.open = False
                self.page.update()
            
            dialog = ft.AlertDialog(
                title=ft.Text("创建训练任务"),
                content=ft.Container(
                    content=ft.Column([
                        task_name_field,
                        dataset_dropdown,
                        training_type_dropdown
                    ], spacing=10),
                    width=350,
                    height=250
                ),
                actions=[
                    ft.TextButton("取消", on_click=cancel),
                    ft.ElevatedButton("创建", on_click=create)
                ]
            )
            
            self.page.open(dialog)
            
        except Exception as e:
            self.toast_service.show(f"创建任务失败: {str(e)}", "error")
    
    def _create_task_item(self, task) -> ft.Card:
        """创建任务列表项"""
        # 获取任务状态文本
        state_text = {
            TrainingState.PENDING: "待开始",
            TrainingState.PREPARING: "准备中",
            TrainingState.RUNNING: "训练中",
            TrainingState.COMPLETED: "已完成",
            TrainingState.FAILED: "失败",
            TrainingState.CANCELLED: "已取消"
        }.get(task.state, "未知")
        
        # 计算进度（如果可用）
        progress_value = 0.0
        if hasattr(task, 'progress_info') and task.progress_info:
            progress_value = task.progress_info.get('progress', 0.0)
        
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.ListTile(
                        title=ft.Text(task.config.name, weight=ft.FontWeight.BOLD),
                        subtitle=ft.Text(f"状态: {state_text}"),
                        trailing=ft.PopupMenuButton(
                            icon=ft.Icons.MORE_VERT,
                            items=[
                                ft.PopupMenuItem(
                                    text="查看详情",
                                    icon=ft.Icons.VISIBILITY,
                                    on_click=lambda e, tid=task.task_id: self.on_open_task(tid)
                                ),
                                ft.PopupMenuItem(
                                    text="删除任务",
                                    icon=ft.Icons.DELETE,
                                    on_click=lambda e, tid=task.task_id: self._delete_task(tid)
                                ),
                            ],
                        ),
                        on_click=lambda e, tid=task.task_id: self.on_open_task(tid)
                    ),
                    ft.Container(
                        content=ft.Row([
                            ft.ProgressBar(value=progress_value, width=200),
                            ft.Text(f"{progress_value*100:.1f}%"),
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        padding=ft.padding.symmetric(horizontal=15, vertical=5),
                    ),
                    ft.Container(
                        content=ft.Row([
                            ft.Text(f"类型: {task.config.training_type.value}"),
                            ft.Text(f"创建时间: {task.created_time}"),
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        padding=ft.padding.symmetric(horizontal=15, vertical=5),
                    ),
                ]),
                padding=ft.padding.all(5),
            ),
            elevation=2,
        )
    
    def refresh(self):
        """刷新任务列表"""
        try:
            # 清空现有内容
            self.task_list.controls.clear()
            
            # 获取并显示任务
            tasks = self.training_manager.list_tasks()
            
            if not tasks:
                self.task_list.controls.append(
                    ft.Text("暂无训练任务，请创建训练任务", italic=True, color=ft.Colors.GREY_600)
                )
            else:
                # 按创建时间倒序排列
                sorted_tasks = sorted(tasks, key=lambda x: x.created_time, reverse=True)
                for task in sorted_tasks:
                    self.task_list.controls.append(self._create_task_item(task))
            
            if self.page:
                self.page.update()
                
        except Exception as e:
            self.toast_service.show(f"刷新失败: {str(e)}", "error")
    
    def _delete_task(self, task_id: str):
        """删除任务"""
        def confirm_delete(e):
            try:
                success = self.training_manager.delete_task(task_id)
                if success:
                    self.toast_service.show("任务删除成功", "success")
                    self.refresh()
                else:
                    self.toast_service.show("任务删除失败", "error")
            except Exception as ex:
                self.toast_service.show(f"删除失败: {str(ex)}", "error")
            confirm_dialog.open = False
            self.page.update()
        
        def cancel_delete(e):
            confirm_dialog.open = False
            self.page.update()
        
        confirm_dialog = ft.AlertDialog(
            title=ft.Text("确认删除"),
            content=ft.Text("确定要删除这个训练任务吗？此操作不可恢复。"),
            actions=[
                ft.TextButton("取消", on_click=cancel_delete),
                ft.ElevatedButton("删除", on_click=confirm_delete, bgcolor=ft.Colors.RED, color=ft.Colors.WHITE)
            ]
        )
        
        self.page.open(confirm_dialog)
    
    def build(self) -> ft.Container:
        """构建并返回根容器"""
        self.refresh()
        return self.root_container
    
    def update_task_progress(self, task_id: str, progress_info: Dict[str, Any]):
        """更新任务进度（来自事件回调）"""
        # 这个方法会在事件回调中被调用，用于实时更新进度
        # 由于我们现在的实现是刷新整个列表，所以这里直接调用refresh()
        # 在更复杂的实现中，可以精确更新特定任务项
        self.refresh()

class TrainingDetailView:
    """训练任务详情视图"""
    
    def __init__(self,
                 page: ft.Page,
                 task_id: str,
                 training_manager,
                 on_back: Callable[[], None],
                 toast_service):
        self.page = page
        self.task_id = task_id
        self.training_manager = training_manager
        self.on_back = on_back
        self.toast_service = toast_service
        
        # UI组件
        self.log_display = ft.TextField(
            value="",
            multiline=True,
            min_lines=20,
            max_lines=30,
            read_only=True,
            bgcolor=ft.Colors.BLACK,
            color=ft.Colors.GREEN,
            text_style=ft.TextStyle(font_family="Courier New", size=12),
            expand=True
        )
        
        self.progress_bar = ft.ProgressBar(value=0.0, width=400)
        self.status_text = ft.Text("状态: 未知")
        self.progress_text = ft.Text("进度: 0%")
        self.step_text = ft.Text("步骤: 0/0")
        self.eta_text = ft.Text("预计时间: --")
        
        self.root_container = None
        self._build_ui()
    
    def _build_ui(self):
        """构建UI"""
        # 获取任务信息
        task = self.training_manager.get_task(self.task_id)
        task_name = task.config.name if task else "未知任务"
        
        # 训练控制按钮（根据状态动态显示）
        self.start_button = ft.ElevatedButton(
            "开始训练",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._start_training,
            style=ft.ButtonStyle(color=ft.Colors.GREEN)
        )
        
        self.stop_button = ft.ElevatedButton(
            "停止训练",
            icon=ft.Icons.STOP,
            on_click=self._stop_training,
            style=ft.ButtonStyle(color=ft.Colors.RED)
        )
        
        self.control_buttons = ft.Row([
            self.start_button,
            self.stop_button
        ], spacing=10)

        # 顶部工具栏
        toolbar = ft.Container(
            content=ft.Row([
                ft.IconButton(
                    icon=ft.Icons.ARROW_BACK,
                    tooltip="返回",
                    on_click=lambda e: self.on_back()
                ),
                ft.Text(
                    f"训练详情: {task_name}",
                    size=20,
                    weight=ft.FontWeight.BOLD
                ),
                ft.Container(expand=True),
                self.control_buttons,
            ]),
            padding=ft.padding.all(20)
        )
        
        # 状态信息区域
        status_area = ft.Container(
            content=ft.Column([
                ft.Row([self.status_text, self.progress_text]),
                self.progress_bar,
                ft.Row([self.step_text, self.eta_text])
            ], spacing=10),
            padding=ft.padding.all(20),
            bgcolor=ft.Colors.GREY_100,
            border_radius=5
        )
        
        # 日志区域
        log_area = ft.Container(
            content=ft.Column([
                ft.Text("训练日志", size=16, weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=self.log_display,
                    expand=True,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=5,
                    padding=5
                )
            ]),
            expand=True,
            padding=ft.padding.all(20)
        )
        
        # 主容器
        self.root_container = ft.Column([
            toolbar,
            status_area,
            log_area
        ], expand=True)
    
    def _start_training(self, e):
        """开始训练"""
        success = self.training_manager.start_task(self.task_id)
        if success:
            self.toast_service.show("训练已开始", "success")
            self._update_button_state()
        else:
            self.toast_service.show("启动训练失败", "error")
    
    def _stop_training(self, e):
        """停止训练"""
        success = self.training_manager.cancel_task(self.task_id)
        if success:
            self.toast_service.show("训练已停止", "warning")
            self._update_button_state()
        else:
            self.toast_service.show("停止训练失败", "error")
    
    def _update_button_state(self):
        """根据任务状态更新按钮显示"""
        task = self.training_manager.get_task(self.task_id)
        if not task:
            return
        
        # 根据状态决定按钮可见性
        if task.state == TrainingState.PENDING:
            self.start_button.visible = True
            self.stop_button.visible = False
        elif task.state in [TrainingState.PREPARING, TrainingState.RUNNING]:
            self.start_button.visible = False
            self.stop_button.visible = True
        else:  # COMPLETED, FAILED, CANCELLED
            self.start_button.visible = False
            self.stop_button.visible = False
        
        if self.page:
            self.page.update()
    
    def update_progress(self, progress: float, current_step: int, total_steps: int, eta_seconds: int = None):
        """更新进度信息"""
        self.progress_bar.value = progress
        self.progress_text.value = f"进度: {progress:.1%}"
        self.step_text.value = f"步骤: {current_step}/{total_steps}"
        
        if eta_seconds:
            hours = eta_seconds // 3600
            minutes = (eta_seconds % 3600) // 60
            self.eta_text.value = f"预计时间: {hours:02d}:{minutes:02d}"
        
        if self.page:
            self.page.update()
    
    def update_status(self, status: str):
        """更新状态"""
        self.status_text.value = f"状态: {status}"
        self._update_button_state()
        if self.page:
            self.page.update()
    
    def append_log(self, log_line: str):
        """添加日志行"""
        current_logs = self.log_display.value
        new_logs = current_logs + log_line + "\n"
        
        # 限制日志行数，避免过多内容
        lines = new_logs.split('\n')
        if len(lines) > 1000:
            new_logs = '\n'.join(lines[-1000:])
        
        self.log_display.value = new_logs
        
        # 自动滚动到底部
        self.log_display.selection = ft.TextSelection(
            base_offset=len(new_logs),
            extent_offset=len(new_logs)
        )
        
        if self.page:
            self.page.update()
    
    def build(self) -> ft.Container:
        """构建并返回根容器"""
        # 加载任务状态
        task = self.training_manager.get_task(self.task_id)
        if task:
            self.update_status(task.state.value)
            self.update_progress(task.progress, task.current_step, task.total_steps, task.eta_seconds)
            self._update_button_state()
        
        return self.root_container