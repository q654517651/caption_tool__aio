# views/training_detail_view.py
import flet as ft
from typing import Dict, Any, Callable, Optional, List, Tuple


class TrainingDetailView(ft.Container):
    def __init__(
            self,
            task_id: str,
            cfg: Dict[str, Any],
            on_cancel: Callable[[], None],
            on_back: Optional[Callable[[], None]] = None,
            on_start: Optional[Callable[[Dict[str, Any]], None]] = None,  # ← 新增：手动开始回调
    ):

        super().__init__(padding=10)
        self.task_id = task_id
        self.cfg = cfg
        self.on_cancel = on_cancel
        self.on_back = on_back
        self.on_start = on_start

        # 顶部状态条（左侧信息，右侧操作）
        self.title = ft.Text(f"任务：{cfg.get('name', '')}", size=18, weight=ft.FontWeight.BOLD)
        self.state = ft.Text("状态：RUNNING")
        self.step = ft.Text("Step: 0 / 0")
        self.eta = ft.Text("ETA: --")
        self.ips = ft.Text("IPS: --")
        self.actions = ft.Row([
            ft.OutlinedButton("停止", icon=ft.Icons.STOP, on_click=lambda e: self.on_cancel()),
        ])
        back_btn = (
            ft.IconButton(
                icon=ft.Icons.ARROW_BACK,
                tooltip="返回",
                on_click=lambda e: self.on_back() if self.on_back else None,
            )
            if self.on_back
            else None
        )
        header = ft.Row([
            back_btn,
            ft.Column([self.title, ft.Row(
                [self.state, ft.Text(" • "), self.step, ft.Text(" • "), self.eta, ft.Text(" • "), self.ips])]),
            ft.Container(expand=True),
            self.actions
        ])

        # === Tab1: 可编辑参数（分组 + 折叠 + 描述） ===


        # 1) 参数规格元数据：分组、展示名、描述、类型
        #    只写你现有 TrainingConfig 里常用的键；不存在的会自动忽略
        #    group 取值：数据集设置 | 学习率与调度器 | 过程中采样 | 保存相关
        PARAM_SPECS: Dict[str, Dict[str, Any]] = {
            # 数据集设置
            "dataset_id": {"label": "数据集ID", "desc": "用于选择训练的数据集唯一标识。", "group": "数据集设置",
                           "type": "str", "readonly": True},
            "dataset_size": {"label": "样本数量", "desc": "数据集中图片/样本数量（自动统计）。", "group": "数据集设置",
                             "type": "int", "readonly": True},
            "repeats": {"label": "重复次数", "desc": "每张样本在若干 epoch 内被重复的次数。", "group": "数据集设置",
                        "type": "int"},
            "epochs": {"label": "总Epoch", "desc": "完整遍历数据集的轮次。", "group": "数据集设置", "type": "int"},
            "batch_size": {"label": "批大小", "desc": "每步更新时的样本批量（显存相关）。", "group": "数据集设置",
                           "type": "int"},
            "grad_accum": {"label": "梯度累积", "desc": "用于小显存模拟大batch的累积步数。", "group": "数据集设置",
                           "type": "int"},
            "resolution": {"label": "训练分辨率", "desc": "宽,高（如 1024,1024）。", "group": "数据集设置", "type": "str"},
            "base_model": {"label": "基座模型路径", "desc": "可选：指定基座/底模权重路径。", "group": "数据集设置",
                           "type": "str"},
            "clip_path": {"label": "CLIP 路径", "desc": "可选：覆盖默认 CLIP 路径。", "group": "数据集设置",
                          "type": "str"},
            "vae_path": {"label": "VAE 路径", "desc": "可选：覆盖默认 VAE 路径。", "group": "数据集设置", "type": "str"},
            "gpu_index": {"label": "GPU 序号", "desc": "选择用于训练的 GPU 索引。", "group": "数据集设置",
                          "type": "int"},

            # 学习率与调度器
            "optimizer": {"label": "优化器", "desc": "如 adamw8bit / adamw / lion 等。", "group": "学习率与调度器",
                          "type": "str"},
            "lr": {"label": "学习率", "desc": "初始学习率（如 1e-4）。", "group": "学习率与调度器", "type": "float"},
            "scheduler": {"label": "调度器", "desc": "如 cosine / linear / constant 等。", "group": "学习率与调度器",
                          "type": "str"},
            "warmup_ratio": {"label": "Warmup 比例", "desc": "学习率预热阶段比例（0~1）。", "group": "学习率与调度器",
                             "type": "float"},
            "weight_decay": {"label": "权重衰减", "desc": "L2 正则强度（0 表示关闭）。", "group": "学习率与调度器",
                             "type": "float"},
            "precision": {"label": "精度", "desc": "训练精度（fp16 / bf16 / fp32）。", "group": "学习率与调度器",
                          "type": "str"},

            # 过程中采样
            "sample_prompt": {"label": "采样提示词", "desc": "用于中途/阶段性生成样张的提示词。", "group": "过程中采样",
                              "type": "str"},
            "sample_every_n_steps": {"label": "采样间隔步数", "desc": "每隔多少步生成一次样张。", "group": "过程中采样",
                                     "type": "int"},
            "sample_resolution": {"label": "采样分辨率", "desc": "样张生成的分辨率。", "group": "过程中采样",
                                  "type": "str"},
            "sample_num_images": {"label": "样张数量", "desc": "每次采样生成的样张数量。", "group": "过程中采样",
                                  "type": "int"},

            # 保存相关（目前你的 TrainingConfig 里没有“保存间隔/输出目录”等字段，可先预留）
            # "save_every_n_steps": {"label": "保存间隔步数", "desc": "每隔多少步保存一次权重。", "group": "保存相关", "type": "int"},
            # "output_dir":         {"label": "输出目录", "desc": "保存权重与日志的目录。", "group": "保存相关", "type": "str"},
        }

        GROUP_ORDER: List[str] = ["数据集设置", "学习率与调度器", "过程中采样", "保存相关"]

        # 2) 工厂：根据 spec + 当前 cfg 生成控件（带 helper 描述）
        self.param_inputs: Dict[str, ft.Control] = {}

        def _make_field(key: str, value: Any, spec: Dict[str, Any]) -> ft.Control:
            label = spec.get("label", key)
            desc = spec.get("desc", "")
            typ = spec.get("type", "str")
            readonly = spec.get("readonly", False)

            if typ == "bool":
                ctrl = ft.Switch(label=label, value=bool(value), tooltip=desc, disabled=readonly)
            else:
                # TextField 支持 helper_text，展示描述文案
                ctrl = ft.TextField(
                    label=label,
                    value=str(value if value is not None else ""),
                    helper_text=desc,
                    dense=True,
                    text_align=ft.TextAlign.LEFT,
                    read_only=readonly,
                )
            self.param_inputs[key] = ctrl
            return ctrl

        # 3) 将 cfg 拆分到各组（只有当前 cfg 中存在的键才渲染）
        tiles: List[ft.Control] = []
        for group in GROUP_ORDER:
            rows: List[ft.Control] = []
            for key, val in self.cfg.items():
                spec = PARAM_SPECS.get(key)
                if not spec or spec.get("group") != group:
                    continue
                rows.append(_make_field(key, val, spec))
            if not rows:
                continue  # 跳过空组
            # 每组为一个可折叠面板
            tiles.append(
                ft.ExpansionTile(
                    title=ft.Text(group),
                    initially_expanded=True if group in ("数据集设置", "学习率与调度器") else False,
                    controls=[ft.Column(rows, tight=True, spacing=8)]
                )
            )

        # 4) 参数收集（按 spec 做类型回填）
        def _collect_cfg() -> Dict[str, Any]:
            new_cfg: Dict[str, Any] = {}
            for key, orig in self.cfg.items():
                spec = PARAM_SPECS.get(key, {})
                typ = spec.get("type", "str")
                c = self.param_inputs.get(key)
                if c is None:
                    continue
                val = getattr(c, "value", None)
                try:
                    if typ == "int":
                        new_cfg[key] = int(val)
                    elif typ == "float":
                        new_cfg[key] = float(val)
                    elif typ == "bool":
                        new_cfg[key] = bool(val)
                    else:
                        new_cfg[key] = str(val) if val is not None else ""
                except Exception:
                    # 失败则回退原值
                    new_cfg[key] = orig
            return new_cfg

        def _on_start_click(e):
            if self.on_start:
                self.on_start(_collect_cfg())
                self.state.value = "状态：RUNNING"
                self.update()

        # 操作区：开始/停止
        actions_row = ft.Row([
            ft.FilledButton("开始训练", icon=ft.Icons.PLAY_ARROW, on_click=_on_start_click),
            ft.OutlinedButton("停止", icon=ft.Icons.STOP, on_click=lambda e: self.on_cancel()),
        ])

        tab_params = ft.Container(
            content=ft.Column([ft.Column(tiles, spacing=8), ft.Divider(), actions_row], expand=True),
            padding=10
        )

        # === Tab2: 训练详情（左：日志；右：GPU / 产物占位） ===
        self.log = ft.Text("", selectable=True)
        log_view = ft.Container(
            content=ft.ListView(controls=[self.log], auto_scroll=True, expand=True),
            bgcolor=ft.Colors.BLACK,
            padding=8
        )
        right = ft.Column([
            ft.Text("GPU 监控（占位）", weight=ft.FontWeight.BOLD),
            ft.Text("温度 / 风扇 / 功耗 / 显存 / 利用率 ..."),
            ft.Divider(),
            ft.Text("产物（占位）", weight=ft.FontWeight.BOLD),
            ft.Text("样张缩略图 / ckpt 列表 / TensorBoard"),
        ])
        tab_detail = ft.Row(
            [ft.Container(expand=2, content=log_view), ft.Container(width=320, content=right)],
            expand=True
        )

        # === Tabs 总装配 ===
        self.tabs = ft.Tabs(
            tabs=[
                ft.Tab(text="任务参数", content=tab_params),
                ft.Tab(text="训练详情", content=tab_detail),
            ],
            expand=True
        )

        # === 页面内容 ===
        self.content = ft.Column([header, self.tabs], expand=True)



    # === 供外部更新 ===
    def append_log(self, line: str):
        self.log.value += (line + "\n")
        self.update()

    def update_progress(self, step: int, total_steps: int, ips: float | None, eta_secs: int | None):
        self.step.value = f"Step: {step} / {total_steps}"
        if ips is not None:
            self.ips.value = f"IPS: {ips:.2f}"
        self.eta.value = f"ETA: {self._fmt_eta(eta_secs)}"
        self.update()

    def update_state(self, state: str):
        self.state.value = f"状态：{state}"
        self.update()

    def _fmt_eta(self, s: int | None) -> str:
        if not s or s <= 0: return "--"
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        if h: return f"{h}h{m}m"
        if m: return f"{m}m{sec}s"
        return f"{sec}s"
