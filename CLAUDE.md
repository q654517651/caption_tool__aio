# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
需要使用中文来回答用户的全部问题
！！！非常重要：要先把代码实时方案和我讨论并且确认无误后再开始实施


## Project Overview

TagTracker是一个基于Python的集成打标与LoRA训练工具，支持图像数据集管理、AI自动打标、标签处理和模型训练。系统采用现代化架构设计，使用Flet桌面界面。

## 重构后的新架构

### 目录结构 (清理后)
```
tagtragger/
├── main.py                    # 统一入口点
├── src/tagtragger/           # 核心代码
│   ├── config/               # 配置管理
│   │   ├── settings.py       # 应用配置
│   │   └── constants.py      # 常量定义
│   ├── core/                 # 核心业务逻辑
│   │   ├── common/           # 公共组件
│   │   │   └── events.py     # 事件系统
│   │   ├── dataset/          # 数据集管理
│   │   │   ├── manager.py    # 数据集管理器
│   │   │   ├── models.py     # 数据模型
│   │   │   └── image_processor.py # 图像处理
│   │   ├── training/         # 训练管理
│   │   │   ├── manager.py    # 训练管理器
│   │   │   ├── models.py     # 训练模型
│   │   │   ├── qwen_trainer.py # Qwen训练器
│   │   │   └── trainers/     # 训练器集合
│   │   └── labeling/         # 打标服务
│   │       ├── service.py    # 打标服务
│   │       └── ai_client.py  # AI客户端
│   ├── ui/                   # 用户界面
│   │   └── flet/             # Flet桌面界面
│   │       ├── app.py        # 主应用
│   │       └── components/   # UI组件
│   ├── storage/              # 数据持久化
│   │   └── database.py       # 数据库管理
│   └── utils/                # 工具模块
│       ├── exceptions.py     # 自定义异常
│       ├── logger.py         # 统一日志系统
│       ├── validators.py     # 数据验证
│       └── musubi_helper.py  # Musubi工具
├── third_party/              # 第三方模块
│   └── musubi-tuner/         # 训练后端
└── workspace/                # 数据工作区
    ├── datasets/             # 数据集存储
    ├── models/               # 模型输出
    └── tasks/                # 任务记录
```

### app功能结构
```
首页/
 ├──数据集加载
 │   └────数据集详情
 │         ├────数据预览
 │         └────标签编辑
 ├──创建训练
 ├──训练任务
 │   └────任务详情
 └──设置
```

## 新架构设计

### 核心组件
- **统一入口** (`main.py`): 智能启动器，支持新架构和回退机制
- **核心业务层** (`src/tagtragger/core/`): 模块化的业务逻辑实现
- **用户界面层** (`src/tagtragger/ui/flet/`): 现代化Flet桌面应用
- **配置管理** (`src/tagtragger/config/`): 统一的配置系统
- **存储层** (`src/tagtragger/storage/`): 混合存储架构 (SQLite + 文件系统)

### 核心服务模块
- **DatasetManager** (`core/dataset/manager.py`): 数据集CRUD操作、图像导入、工作区管理
- **LabelingService** (`core/labeling/service.py`): AI自动打标，支持多种模型 (GPT, LM Studio)
- **TrainingManager** (`core/training/manager.py`): 训练任务管理、进度跟踪
- **EventBus + JobQueue** (`core/common/events.py`): 异步任务执行和事件通信
- **统一日志系统** (`utils/logger.py`): 集中化日志记录和状态报告

### UI组件架构
- **主应用** (`ui/flet/app.py`): 应用主框架和路由管理  
- **数据集视图** (`ui/flet/components/datasets_view.py`): 数据集列表和操作
- **数据集详情** (`ui/flet/components/dataset_detail_view.py`): 图像预览和标签编辑
- **训练视图** (`ui/flet/components/training_view.py`): 训练任务创建和监控
- **公共服务** (`ui/flet/components/`): ToastService、TerminalService等UI工具

### Data Storage
- **Workspace Structure**: `workspace/datasets/{dataset_id}/` containing images, medium thumbnails, and configs
- **Configuration**: JSON-based dataset configs and settings management
- **Labels**: Text files alongside images (same name with `.txt` extension)

## Common Development Commands

### Running the Application
```bash
# 运行Flet桌面界面(推荐)
python main.py --interface flet

# 直接运行桌面界面
python main.py

# 启用调试模式
python main.py --debug

# 指定配置文件
python main.py --config /path/to/config.json

# 安装依赖
pip install -r requirements.txt
```

### 配置管理
```bash
# 配置文件位置
# Windows: %APPDATA%/TagTracker/config.json  
# Linux/Mac: ~/.tagtracker/config.json

# 支持环境变量
export TAGTRACKER_CONFIG=/path/to/config.json
export TAGTRACKER_DEBUG=1
```

### Training Configuration
Set environment variables for training:
```bash
# For Musubi-Tuner integration
export MUSUBI_QWEN_TRAIN="/path/to/musubi/train_script.py"
```

### Testing
The codebase does not include formal tests. Manual testing should focus on:
- Dataset import/export workflows
- AI labeling with different models
- Training job execution and monitoring
- UI navigation and state management

## Development Guidelines

### Adding New Training Backends
1. Create trainer class in `trainers/` implementing the required interface
2. Update `TrainingBackend` enum in `trainers/types.py` 
3. Add backend preset configuration to `BACKEND_PRESETS`
4. Register with `TrainingManager` in `training_manager.py`

### Service Integration
- Services communicate via `EventBus` for loose coupling
- Use `TerminalService` for consistent logging across modules
- Follow the async job pattern for long-running operations

### UI Components (Flet)
- Views are in `views/` and should be self-contained
- Use callback pattern for parent-child communication
- Navigation is handled centrally via `NavigationService`
- Toast notifications via `ToastService` for user feedback

### Data Handling
- Images are automatically resized to medium thumbnails (max 1280px)
- Labels are stored as UTF-8 text files alongside images
- Dataset metadata in JSON format under `workspace/datasets/configs/`

### Error Handling
- Use centralized logging through `TerminalService`
- UI errors should show toast notifications
- Training errors are captured and displayed in training detail views

### 新架构开发指南

#### 异常处理
```python
from tagtragger.utils.exceptions import DatasetError, ValidationError
from tagtragger.utils.logger import log_error

try:
    # 业务逻辑
    result = some_operation()
except ValidationError as e:
    log_error(f"验证失败: {e.message}", e)
    raise
except DatasetError as e:
    log_error(f"数据集操作失败: {e.message}", e)
    return False, e.message
```

#### 配置使用
```python
from tagtragger.config import get_config, update_config

config = get_config()
model_path = config.model_paths.vae_path
batch_size = config.training.default_batch_size

# 更新配置
update_config(training={'default_batch_size': 4})
```

#### 日志系统
```python
from tagtracker.utils.logger import log_info, log_error, log_success

log_info("操作开始")
log_success("操作成功完成")
log_error("操作失败", exception=e)
```

#### 数据验证
```python
from tagtragger.utils.validators import validate_image_file, validate_resolution

try:
    validate_image_file("/path/to/image.jpg")
    validate_resolution("1024,1024")
except ValidationError as e:
    print(f"验证失败: {e.message}")
```

### 迁移完成状态

✅ **已完成的迁移工作**：

1. **核心模块迁移完成**:
   - ✅ 数据集管理 (`core/dataset/manager.py`) - 完整功能实现
   - ✅ 打标服务 (`core/labeling/service.py`) - AI客户端和批量处理
   - ✅ 训练管理 (`core/training/manager.py`) - 支持新旧两套训练系统
   - ✅ 事件系统 (`core/common/events.py`) - EventBus和JobQueue
   - ✅ 公共服务迁移完成
   
2. **配置和工具系统**:
   - ✅ 统一配置管理 (`config/settings.py`)
   - ✅ 结构化异常处理 (`utils/exceptions.py`)
   - ✅ 统一日志系统 (`utils/logger.py`)
   - ✅ 数据验证系统 (`utils/validators.py`)
   
3. **UI架构完成**:
   - ✅ Flet应用主框架 (`ui/flet/app.py`)
   - ✅ 数据集视图 (`ui/flet/components/datasets_view.py`)
   - ✅ 数据集详情 (`ui/flet/components/dataset_detail_view.py`)
   - ✅ 训练视图 (`ui/flet/components/training_view.py`)
   - ✅ UI服务组件 (toast、dialog、terminal等)

4. **训练系统**:
   - ✅ Qwen-Image LoRA训练支持
   - ✅ 新旧训练器共存 (支持向后兼容)
   - ✅ 混合存储架构 (SQLite + 文件系统)

5. **应用入口**:
   - ✅ 统一启动入口 (`main.py`)
   - ✅ 自动回退机制 (新版本失败时自动使用旧版本)

✅ **架构清理完成** (2025-08-13):
- ✅ 已删除 `services_removed/` 目录 - 旧服务模块已完全迁移
- ✅ 已删除 `views_removed/` 目录 - 旧视图模块已完全迁移  
- ✅ 已删除 `services/` 目录 - 重复的服务文件
- ✅ 已删除 `settings_manager.py` - 被新配置系统取代
- ✅ 已删除 `src/tagtragger/ui/flet/services/` - 未使用的UI服务
- ✅ 已删除 `terminal_service.py` - 未使用的终端服务
- ✅ 清理了所有 `.pyc` 文件和 `__pycache__` 目录

### Flet UI开发注意事项

#### API版本兼容性 (重要！)
当前使用Flet 0.28.3版本，请务必使用正确的API格式：

**关键修复点**：
- ❌ `ft.colors.*` → ✅ `ft.Colors.*` (大写C)
- ❌ `ft.Colors.SURFACE_VARIANT` → ✅ `ft.Colors.SURFACE` (在0.28.3中不存在SURFACE_VARIANT)
- ❌ 在`__init__`中调用`self.update()` → ✅ 在控件添加到页面后再初始化视图
- ❌ `view=ft.WEB_BROWSER` → ✅ 不设置view参数(默认桌面模式)

```python
# ✅ 正确的API使用方式
import flet as ft

# 颜色API - 使用大写的 Colors
container = ft.Container(bgcolor=ft.Colors.PRIMARY)
text = ft.Text("Hello", color=ft.Colors.GREY)

# 图标API - 使用大写的 Icons  
icon = ft.Icon(ft.Icons.FAVORITE, color=ft.Colors.RED)
button = ft.ElevatedButton("按钮", icon=ft.Icons.ADD)

# ❌ 错误的API使用方式 (会导致AttributeError)
# container = ft.Container(bgcolor=ft.colors.PRIMARY)  # ❌ 错误
# icon = ft.Icon(ft.icons.FAVORITE)  # ❌ 错误
```

#### 常用Flet组件API
```python
# 主题颜色 (Flet 0.28.3)
ft.Colors.PRIMARY, ft.Colors.SECONDARY, ft.Colors.ERROR
ft.Colors.SURFACE, ft.Colors.ON_SURFACE, ft.Colors.SURFACE_CONTAINER_HIGHEST
ft.Colors.GREY, ft.Colors.WHITE, ft.Colors.BLACK

# 常用图标
ft.Icons.ADD, ft.Icons.DELETE, ft.Icons.EDIT, ft.Icons.SAVE
ft.Icons.FAVORITE, ft.Icons.STAR, ft.Icons.HOME
ft.Icons.ADD_CIRCLE_OUTLINE, ft.Icons.ERROR
```

### 项目当前状态总结

🎉 **重构基本完成** - TagTracker已成功从单体架构迁移到模块化新架构！

#### 新架构优势
1. **模块解耦**: 核心业务逻辑完全独立于UI层
2. **类型安全**: 完整的数据模型和异常处理体系
3. **可扩展性**: 新的训练器和功能模块易于添加
4. **向后兼容**: 保留旧版本功能，平滑过渡
5. **统一管理**: 配置、日志、错误处理全面统一

#### 添加新功能的标准流程
1. 在`core/`下创建新的功能目录
2. 实现业务逻辑类，继承相应的基类  
3. 添加相应的异常类到`utils/exceptions.py`
4. 在UI层(`ui/flet/components/`)添加相应的视图组件
5. 更新配置文件(`config/settings.py`)支持新功能的配置项
6. 在主应用(`ui/flet/app.py`)中注册新服务

## 训练系统架构深度分析

### 当前训练系统组成

#### 1. 训练管理层级
```
应用层 (ui/flet/app.py)
  ├── 事件回调注册 (训练日志、进度、状态)
  ├── 训练视图管理 (TrainingDetailView)
  └── 用户交互处理

训练管理层 (core/training/manager.py)
  ├── 任务生命周期管理 (创建、启动、取消、删除)
  ├── 任务持久化 (JSON文件存储)
  ├── 事件回调系统 (task_log, task_progress, task_state)
  └── 训练器调度

训练器层 (core/training/qwen_trainer.py)
  ├── 配置验证和数据集准备
  ├── TOML配置文件生成
  ├── Musubi-Tuner集成
  └── 进程管理和日志解析
```

#### 2. 训练数据流
```
UI创建训练 → TrainingConfig → TrainingTask → 
数据集配置(TOML) → latents缓存 → accelerate训练 → 
实时日志 → 进度解析 → UI更新
```

#### 3. 核心配置类
- `TrainingConfig` (models.py): 包含所有训练参数
- `QwenImageConfig` (models.py): Qwen-Image特定配置
- `TrainingTask` (models.py): 任务运行时状态


### musubi 训练方式
生成dataset.toml文件
以下是配置规则
# resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale should be set in either general or datasets
# otherwise, the default values will be used for each item

# general configurations
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/path/to/image_dir"
cache_directory = "/path/to/cache_directory"
num_repeats = 1 # optional, default is 1. Number of times to repeat the dataset. Useful to balance the multiple datasets with different sizes.

# other datasets can be added here. each dataset can have different configurations

musubi训练之前要先缓存文本latent和Text Encoder
第1步
python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model
第2步
python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 16
第3步-开启训练
accelerate launch ^
    --num_cpu_threads_per_process 1 ^
    --mixed_precision bf16 ^
    src/musubi_tuner/qwen_image_train_network.py ^
    --dit D:\AI\Qwen-model\qwen_image_bf16.safetensors ^
    --vae D:\AI\Qwen-model\vae.safetensors ^
    --text_encoder D:\AI\Qwen-model\qwen_2.5_vl_7b.safetensors ^
    --dataset_config D:\AI\train\QWEN_IMAGE\event-banner-3d-v021\dataset.toml ^
    --sdpa --mixed_precision bf16 ^
    --timestep_sampling shift ^
    --weighting_scheme none --discrete_flow_shift 3.0 ^
    --optimizer_type adamw --learning_rate 1e-4 --gradient_checkpointing ^
    --max_data_loader_n_workers 2 --persistent_data_loader_workers ^
    --network_module musubi_tuner.networks.lora_qwen_image ^
    --network_dim 32 ^
    --network_alpha 16 ^
    --max_train_epochs 8 --save_every_n_epochs 2 --seed 42 ^
    --output_dir D:\AI\train\QWEN_IMAGE\event-banner-3d-v021\output --output_name event-banner-3d-qwen ^
    --fp8_base --fp8_scaled ^
    --blocks_to_swap 16 ^
    --sample_prompts D:\AI\train\QWEN_IMAGE\event-banner-3d-v021\sample_prompts.txt ^
    --sample_every_n_epochs 1  --sample_at_first ^
    --logging_dir=logs ^

以下是一些配置说明
Uses qwen_image_train_network.py.
Requires specifying --dit, --vae, and --text_encoder.
The LoRA network for Qwen-Image (networks.lora_qwen_image) is automatically selected.
--mixed_precision bf16 is recommended for Qwen-Image training.
Memory saving options like --fp8_base and --fp8_scaled (for DiT), and --fp8_vl (for Text Encoder) are available.
--gradient_checkpointing is available for memory savings.

--fp8_vl is recommended for GPUs with less than 16GB of VRAM.
--sdpa uses PyTorch's scaled dot product attention. Other options like --xformers and --flash_attn are available. flash3 cannot be used currently.
If you specify --split_attn, the attention computation will be split, slightly reducing memory usage. Please specify --split_attn if you are using anything other than --sdpa.
--timestep_sampling allows you to choose the sampling method for the timesteps. shift with --discrete_flow_shift is the default. qwen_shift is also available. qwen_shift is a same method during inference. It uses the dynamic shift value based on the resolution of each image (typically around 2.2 for 1328x1328 images).
--discrete_flow_shift is set quite low for Qwen-Image during inference (as described), so a lower value than other models may be preferable.
The appropriate settings for each parameter are unknown. Feedback is welcome.

其中--dit，--vae，--text_encoder，--dataset_config，--output_dir，--sample_prompts这几个都需要动态的替换路径
“src/musubi_tuner/qwen_image_train_network.py”这个路径要替换成我们项目中实际的程序路径，让他能够找到就行

必须要说明的是musubi是一款通用的训练器，他的训练流程与逻辑不同模型之间都是公用的，只不过在数据集配置以及参数方面略有不同

### 训练的开启逻辑
用户点击开始训练按钮，musubi输出的信息要同步到训练详情的终端里，同时要保留一份train.bat或者sh以及dataset toml到相关目录下
缓存潜空间和文本编码器的内容也要再详情终端显示


[2025-08-14T01:35:44.222542] [缓存] INFO:musubi_tuner.qwen_image.qwen_image_utils:Setting Qwen2.5-VL to dtype: torch.bfloat16
[2025-08-14T01:35:44.261867] [缓存] INFO:musubi_tuner.qwen_image.qwen_image_utils:Loading tokenizer from Qwen/Qwen-Image
[2025-08-14T01:35:45.308710] [缓存] INFO:__main__:Encoding with Qwen2.5-VL
[2025-08-14T01:35:45.309713] [缓存] INFO:musubi_tuner.cache_text_encoder_outputs:Encoding dataset [0]
[2025-08-14T01:35:45.311714] [缓存] 
[2025-08-14T01:35:45.966023] [缓存] 0it [00:00, ?it/s]
[2025-08-14T01:35:46.008988] [缓存] 1it [00:00,  1.52it/s]
[2025-08-14T01:35:46.011420] [缓存] 2it [00:00,  2.86it/s]
[2025-08-14T01:35:46.710288] [缓存] [0m
[2025-08-14T01:35:46.712800] [完成] 预处理完成: src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py
[2025-08-14T01:35:46.714802] 开始训练: Qwen-Image训练_1
[2025-08-14T01:35:49.111628] E:\Program\programlearn\tagtragger\.venv\Scripts\python.exe: No module named accelerate.__main__; 'accelerate' is a package and cannot be directly executed
[2025-08-14T01:35:49.412851] [0m