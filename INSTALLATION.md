# TagTracker 安装指南

TagTracker 现已升级到现代化的 `pyproject.toml` 依赖管理系统！

## 🚀 快速安装

### 方法一：一键安装脚本（推荐）
```bash
# 克隆项目（包含 Musubi-Tuner）
git clone --recurse-submodules https://github.com/your-repo/tagtragger.git
cd tagtragger

# 一键安装
python setup_musubi.py
```

### 方法二：使用现代工具 UV（推荐高级用户）
```bash
# 安装 UV 包管理器
pip install uv

# 克隆项目
git clone --recurse-submodules https://github.com/your-repo/tagtragger.git
cd tagtragger

# 初始化子模块
git submodule update --init --recursive

# 安装依赖（自动选择 CUDA 版本）
uv pip install -e .[cu124]     # CUDA 12.4
# 或
uv pip install -e .[cu128]     # CUDA 12.8
# 或
uv pip install -e .[cpu]       # CPU 版本
```

### 方法三：传统 pip 安装
```bash
# 克隆项目
git clone --recurse-submodules https://github.com/your-repo/tagtragger.git
cd tagtragger

# 初始化子模块
git submodule update --init --recursive

# 安装依赖
pip install -e .[cu124]        # CUDA 12.4 版本
```

## 📦 可选依赖组

TagTracker 支持灵活的依赖安装：

### 🎮 基础使用
```bash
pip install -e .               # 仅核心依赖
```

### 🔥 GPU 加速训练
```bash
pip install -e .[cu124]        # CUDA 12.4（推荐）
pip install -e .[cu128]        # CUDA 12.8（最新）
pip install -e .[cpu]          # CPU 版本
```

### 🛠️ 开发环境
```bash
pip install -e .[dev]          # 完整开发工具
pip install -e .[debug]        # 仅调试工具
```

### 🚀 组合安装
```bash
pip install -e .[cu124,dev]    # GPU + 开发工具
pip install -e .[cu124,debug]  # GPU + 调试工具
```

## 🔧 依赖管理对比

### pyproject.toml 的优势

| 特性 | pyproject.toml | requirements.txt |
|------|----------------|------------------|
| **标准化** | ✅ Python 官方标准 | ❌ 非正式约定 |
| **项目元数据** | ✅ 包含完整信息 | ❌ 仅依赖列表 |
| **可选依赖** | ✅ 灵活的组合安装 | ❌ 需要多个文件 |
| **CUDA 版本管理** | ✅ 自动冲突检测 | ❌ 手动处理 |
| **工具配置** | ✅ 统一配置文件 | ❌ 分散配置 |

### 实际使用体验

**现代 pyproject.toml 方式**：
```bash
pip install -e .[cu124]                  # 一条命令，自动处理
pip install -e .[cu124,dev]              # 灵活组合
# 自动冲突检测和解决
```

## 🎯 平台特定安装

### Windows
```bash
# 自动检测并安装 CUDA 12.4 版本
python setup_musubi.py

# 或手动指定
pip install -e .[cu124]
```

### Linux
```bash
# 自动检测并安装 CUDA 12.4 版本
python setup_musubi.py

# 或手动指定
pip install -e .[cu124]
```

### macOS
```bash
# 自动安装 CPU 版本
python setup_musubi.py

# 或手动指定
pip install -e .[cpu]
```

## 🔍 验证安装

安装完成后，验证是否正确：

```bash
# 检查核心功能
python -c "import tagtragger; print('✅ TagTracker 安装成功')"

# 检查训练器
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "import accelerate; print('✅ Accelerate 可用')"

# 启动应用
python main.py
```

## 🛠️ 开发者安装

如果你要参与开发：

```bash
# 完整开发环境
pip install -e .[cu124,dev]

# 包含代码格式化、类型检查、测试工具
black .                    # 代码格式化
isort .                    # 导入排序
mypy .                     # 类型检查
pytest                     # 运行测试
```

## 📋 故障排除

### 常见问题

**Q: `error: Microsoft Visual C++ 14.0 is required`？**
A: 安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**Q: PyTorch CUDA 版本不匹配？**
A: 重新安装指定版本：
```bash
pip uninstall torch torchvision
pip install -e .[cu124]  # 或 cu128
```

**Q: UV 安装失败？**
A: 回退到 pip：
```bash
pip install -e .[cu124]
```

### 完全重新安装

如果遇到严重问题：
```bash
# 清理环境
pip uninstall tagtragger torch torchvision -y

# 重新安装
python setup_musubi.py
```

## 🎉 成功！

安装完成后，你将拥有：
- ✅ **完整的 TagTracker** - 数据集管理 + AI 打标
- ✅ **内置 Musubi-Tuner** - 先进的 LoRA 训练器
- ✅ **现代化依赖管理** - 灵活的安装选项
- ✅ **跨平台支持** - Windows/Linux/macOS

立即开始你的 AI 训练之旅吧！ 🚀