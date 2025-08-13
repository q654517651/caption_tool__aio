"""
Musubi-Tuner 集成助手工具
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional, List

from ..config import get_config, update_config
from .logger import log_info, log_error, log_success, log_warning


class MusubiHelper:
    """Musubi-Tuner 集成助手"""
    
    @staticmethod
    def check_installation() -> Tuple[bool, str, List[str]]:
        """
        检查 Musubi-Tuner 安装状态
        
        Returns:
            (是否安装, 状态信息, 缺失组件列表)
        """
        missing_components = []
        
        # 检查内置的 Musubi-Tuner 子模块
        project_root = Path(__file__).parent.parent.parent.parent
        musubi_dir = project_root / "third_party" / "musubi-tuner"
        
        if not musubi_dir.exists():
            missing_components.append("Musubi-Tuner 子模块未初始化，请运行: git submodule update --init --recursive")
        else:
            # 检查关键文件
            required_files = [
                "src/musubi_tuner/hv_train_network.py",
                "src/musubi_tuner/qwen_image_train_network.py",
                "pyproject.toml"
            ]
            
            for file_path in required_files:
                if not (musubi_dir / file_path).exists():
                    missing_components.append(f"缺少文件: {file_path}")
        
        # 检查 accelerate 命令
        if not shutil.which("accelerate"):
            missing_components.append("accelerate 命令不可用 (需要安装: pip install accelerate)")
        
        # 检查 Python 版本
        import sys
        if sys.version_info < (3, 10):
            missing_components.append(f"Python 版本过低 ({sys.version}) - Musubi 需要 Python 3.10+")
        
        # 检查 PyTorch
        try:
            import torch
            if torch.__version__ < "2.5.0":
                missing_components.append(f"PyTorch 版本过低 ({torch.__version__}) - Musubi 需要 2.5.0+")
        except ImportError:
            missing_components.append("PyTorch 未安装")
        
        is_available = len(missing_components) == 0
        status = "Musubi-Tuner 可用" if is_available else f"发现 {len(missing_components)} 个问题"
        
        return is_available, status, missing_components
    
    @staticmethod
    def setup_musubi_path(musubi_path: str) -> Tuple[bool, str]:
        """
        设置 Musubi-Tuner 路径
        
        Args:
            musubi_path: Musubi-Tuner 安装路径
            
        Returns:
            (成功状态, 状态消息)
        """
        try:
            musubi_dir = Path(musubi_path).resolve()
            
            # 验证路径
            if not musubi_dir.exists():
                return False, f"路径不存在: {musubi_dir}"
            
            # 验证是否为 Musubi 项目
            if not (musubi_dir / "src" / "musubi_tuner").exists():
                return False, "不是有效的 Musubi-Tuner 项目目录"
            
            # 更新配置
            update_config(model_paths={'musubi_dir': str(musubi_dir)})
            
            log_success(f"Musubi-Tuner 路径已设置: {musubi_dir}")
            return True, "路径设置成功"
            
        except Exception as e:
            log_error(f"设置 Musubi 路径失败: {e}")
            return False, f"设置失败: {e}"
    
    @staticmethod
    def get_installation_guide() -> str:
        """获取安装指南"""
        return """
# Musubi-Tuner 安装指南

## 1. 克隆项目
```bash
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
```

## 2. 创建虚拟环境 (推荐)
```bash
python -m venv musubi_env
# Windows
musubi_env\\Scripts\\activate
# Linux/Mac  
source musubi_env/bin/activate
```

## 3. 安装依赖
```bash
pip install -e .
pip install accelerate
```

## 4. 在 TagTracker 中配置路径
在设置页面中将 "Musubi-Tuner目录" 设置为克隆的项目路径。

## 5. 验证安装
使用 TagTracker 的检查功能验证安装是否成功。
"""
    
    @staticmethod
    def test_training_command() -> Tuple[bool, str]:
        """
        测试训练命令是否可用
        
        Returns:
            (测试成功, 测试结果信息)
        """
        try:
            config = get_config()
            musubi_dir = Path(config.model_paths.musubi_dir)
            
            if not musubi_dir.exists():
                return False, "Musubi-Tuner 目录未配置"
            
            script_path = musubi_dir / "src/musubi_tuner/hv_train_network.py"
            if not script_path.exists():
                return False, f"训练脚本不存在: {script_path}"
            
            # 测试 accelerate 命令
            result = subprocess.run(
                ["accelerate", "launch", str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True, "训练命令测试成功"
            else:
                return False, f"训练命令测试失败: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "命令执行超时"
        except Exception as e:
            return False, f"测试失败: {e}"
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """获取可用的训练后端"""
        available = []
        
        is_available, _, _ = MusubiHelper.check_installation()
        if is_available:
            available.extend([
                "musubi.hunyuan_video",
                "musubi.qwen_image"
            ])
        
        return available
    
    @staticmethod
    def auto_detect_musubi() -> Optional[str]:
        """自动检测 Musubi-Tuner 安装路径"""
        # 常见的安装位置
        search_paths = [
            Path.cwd() / "musubi-tuner",
            Path.home() / "musubi-tuner", 
            Path("/opt/musubi-tuner"),
            Path("C:/musubi-tuner"),
        ]
        
        # 搜索环境变量
        if "MUSUBI_DIR" in os.environ:
            search_paths.insert(0, Path(os.environ["MUSUBI_DIR"]))
        
        for path in search_paths:
            if path.exists() and (path / "src" / "musubi_tuner").exists():
                log_info(f"自动检测到 Musubi-Tuner: {path}")
                return str(path)
        
        return None


# 便捷函数
def check_musubi_status() -> dict:
    """检查 Musubi 状态并返回详细信息"""
    is_available, status, missing = MusubiHelper.check_installation()
    
    return {
        "available": is_available,
        "status": status,
        "missing_components": missing,
        "installation_guide": MusubiHelper.get_installation_guide() if not is_available else None,
        "auto_detected_path": MusubiHelper.auto_detect_musubi()
    }