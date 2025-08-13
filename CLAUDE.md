# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
需要使用中文来回答用户的全部问题，提出方案后需要和用户确认再开始实施

## Project Overview

TagTracker是一个基于Python的集成打标与LoRA训练工具，支持图像数据集管理、AI自动打标、标签处理和模型训练。系统采用现代化架构设计，使用Flet桌面界面。

## 重构后的新架构

### 目录结构
```
tagtragger/
├── main.py                    # 统一入口点
├── src/tagtragger/           # 核心代码
│   ├── config/               # 配置管理
│   │   ├── settings.py       # 应用配置
│   │   └── constants.py      # 常量定义
│   ├── core/                 # 核心业务逻辑
│   │   ├── dataset/          # 数据集管理
│   │   ├── training/         # 训练管理
│   │   └── labeling/         # 打标服务
│   ├── ui/                   # 用户界面
│   │   └── flet/             # Flet桌面界面
│   ├── storage/              # 数据持久化
│   └── utils/                # 工具模块
│       ├── exceptions.py     # 自定义异常
│       ├── logger.py         # 统一日志系统
│       └── validators.py     # 数据验证
├── services/                 # 原有服务(待迁移)
├── views/                    # 原有视图(待迁移)
└── workspace/                # 数据工作区
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

⚠️ **过渡期保留**:
- 旧的`services/`目录 - 作为兼容层保留 
- 旧的`views/`目录 - 部分组件仍在使用
- 旧版本文件 (`main_flet_old.py`) - 作为回退选项

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

### 后续优化计划 (可选)

#### 阶段一：代码清理 (优先级：低)
- 移除不再使用的`services/`和部分`views/`文件
- 清理`main_flet_old.py`和`router.py`等过时文件  
- 统一导入路径，减少兼容层依赖

#### 阶段二：功能增强 (按需)
- 完善UI组件的高级交互功能
- 添加更多训练器支持
- 实现更丰富的数据集管理功能

#### 阶段三：性能优化 (按需)
- 数据库查询优化
- 大数据集处理性能提升
- UI响应速度优化

**注意**: 当前系统已经完全可用，上述优化为可选项目，可根据实际需求进行。
