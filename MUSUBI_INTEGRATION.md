# Musubi-Tuner 集成说明

TagTracker 现已完全内置 Musubi-Tuner 训练器，实现真正的**无感使用体验**！

## 🚀 一键安装

### 快速开始

```bash
# 1. 克隆项目（包含子模块）
git clone --recurse-submodules https://github.com/your-repo/tagtragger.git
cd tagtragger

# 2. 一键安装
python setup_musubi.py

# 3. 启动应用
python main.py
```

### 手动安装（如果一键安装失败）

```bash
# 1. 初始化子模块
git submodule update --init --recursive

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python main.py
```

## 🎯 集成特性

### ✅ 完全内置
- **无需手动下载** - Musubi-Tuner 作为 Git 子模块集成
- **无需路径配置** - 自动使用内置的训练器
- **统一依赖管理** - 一个 requirements.txt 解决所有依赖
- **开箱即用** - 安装完成即可使用

### 🔧 支持的训练模式
- **HunyuanVideo LoRA** - 视频生成模型训练
- **Qwen-Image LoRA** - 图像生成模型训练
- **FLUX.1 Kontext** - 多模态模型训练
- **Wan2.1/2.2** - 先进的视频模型训练

### 🎛️ 用户体验
- **智能状态检查** - 自动检测安装状态
- **一键修复** - 子模块未初始化时可一键修复
- **实时进度监控** - 训练过程可视化
- **错误诊断** - 友好的错误提示和解决方案

## 📋 使用流程

### 1. 检查状态
在设置页面中查看 Musubi-Tuner 状态：
- ✅ 绿色：一切正常，可以开始训练
- ❌ 红色：需要修复，点击"初始化子模块"

### 2. 创建训练任务
1. 准备数据集（图片 + 标签文件）
2. 在训练页面选择 "Musubi HunyuanVideo" 或 "Musubi Qwen-Image"
3. 配置训练参数
4. 开始训练

### 3. 监控训练
- 实时查看训练日志
- 跟踪训练进度
- 支持取消和恢复

## 🛠️ 技术实现

### Git 子模块集成
```
tagtragger/
├── third_party/
│   └── musubi-tuner/           # Git 子模块
│       ├── src/musubi_tuner/   # 训练脚本
│       └── pyproject.toml      # 依赖配置
├── src/tagtragger/
│   └── core/training/trainers/
│       └── musubi_trainer.py   # 集成训练器
└── requirements.txt            # 统一依赖
```

### 依赖管理策略
- **统一版本** - 避免版本冲突
- **平台适配** - 自动选择合适的 PyTorch 版本
- **可选依赖** - 开发工具标记为可选

### 进程隔离调用
- 通过 `subprocess` 调用训练脚本
- 实时解析输出获取进度
- 支持优雅的取消和清理

## 📦 分发优势

### 对用户
- **零配置** - 下载即用
- **无依赖冲突** - 预先测试的版本组合
- **完整功能** - 包含所有训练能力

### 对开发者
- **易于维护** - 统一的代码库
- **版本控制** - 子模块固定特定版本
- **自动化测试** - CI/CD 可以测试完整集成

## 🔧 故障排除

### 常见问题

**Q: 显示"子模块未初始化"？**
A: 点击设置页面的"初始化子模块"按钮，或运行：
```bash
git submodule update --init --recursive
```

**Q: PyTorch 版本不兼容？**
A: 重新运行安装脚本：
```bash
python setup_musubi.py
```

**Q: 训练脚本找不到？**
A: 确保子模块正确初始化，检查 `third_party/musubi-tuner/` 目录是否存在。

**Q: accelerate 命令不可用？**
A: 安装 accelerate：
```bash
pip install accelerate
```

### 手动修复
如果自动修复失败，可以手动操作：
```bash
# 重新克隆项目
git clone --recurse-submodules https://github.com/your-repo/tagtragger.git

# 或强制更新子模块
git submodule update --init --recursive --force
```

## 🎉 成功案例

采用这种集成方式，TagTracker 用户可以：
- **5分钟内** 完成完整安装
- **零学习成本** 开始使用 Musubi 训练
- **无需关心** 底层技术细节
- **专注于** 数据集和训练效果

这正是 **kohya_ss** 项目成功的原因 - 将复杂的技术栈包装成用户友好的工具！