#!/usr/bin/env python3
"""
TagTracker + Musubi-Tuner 一键安装脚本
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, check=True, shell=False):
    """运行命令并处理输出"""
    print(f"执行: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            shell=shell,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        if e.stderr:
            print(f"错误输出: {e.stderr}")
        if check:
            raise
        return e


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version < (3, 10):
        print(f"[ERROR] Python版本过低: {version.major}.{version.minor}")
        print("Musubi-Tuner需要Python 3.10+")
        return False
    elif version >= (3, 13):
        print(f"[WARNING] Python版本过高: {version.major}.{version.minor}")
        print("Musubi-Tuner最高支持Python 3.12")
        return False
    else:
        print(f"[OK] Python版本合适: {version.major}.{version.minor}")
        return True


def check_git():
    """检查Git是否可用"""
    try:
        result = run_command(["git", "--version"], check=False)
        if result.returncode == 0:
            print("[OK] Git可用")
            return True
        else:
            print("[ERROR] Git不可用")
            return False
    except FileNotFoundError:
        print("[ERROR] Git未安装")
        return False


def init_submodules():
    """初始化Git子模块"""
    print("\n[INFO] 初始化Musubi-Tuner子模块...")
    
    # 检查子模块是否已存在
    submodule_path = Path("third_party/musubi-tuner")
    if submodule_path.exists() and any(submodule_path.iterdir()):
        print("[OK] 子模块已存在")
        return True
    
    # 初始化子模块
    try:
        run_command(["git", "submodule", "update", "--init", "--recursive"])
        print("[OK] 子模块初始化成功")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] 子模块初始化失败")
        return False


def check_uv():
    """检查UV包管理器是否可用"""
    try:
        result = run_command(["uv", "--version"], check=False)
        if result.returncode == 0:
            print("[OK] UV包管理器可用")
            return True
        else:
            print("[INFO] UV包管理器不可用，将使用pip")
            return False
    except FileNotFoundError:
        print("[INFO] UV包管理器未安装，将使用pip")
        return False

def install_dependencies():
    """安装依赖包"""
    print("\n[INFO] 安装依赖包...")
    
    # 检查配置文件
    pyproject_file = Path("pyproject.toml")
    
    if not pyproject_file.exists():
        print("[ERROR] 未找到 pyproject.toml 配置文件")
        return False
    
    # 优先使用 pyproject.toml + uv
    use_uv = check_uv()
    
    # 根据平台选择CUDA版本
    system = platform.system().lower()
    if system in ["windows", "linux"]:
        cuda_extra = "cu124"  # 默认使用 CUDA 12.4
    else:
        cuda_extra = "cpu"    # macOS 使用 CPU 版本
    
    try:
        if use_uv:
            print(f"使用 UV 安装依赖（{cuda_extra} 版本）...")
            # UV 可以自动处理 pyproject.toml
            cmd = ["uv", "pip", "install", "-e", f".[{cuda_extra}]"]
            run_command(cmd)
        else:
            print("使用 pip 安装依赖...")
            # 从 pyproject.toml 安装
            cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{cuda_extra}]"]
            run_command(cmd)
        
        print("[OK] 依赖安装成功")
        return True
        
    except subprocess.CalledProcessError:
        print("[ERROR] 依赖安装失败")
        return False


def verify_installation():
    """验证安装是否成功"""
    print("\n🔍 验证安装...")
    
    try:
        # 检查关键包
        packages = ["torch", "accelerate", "diffusers", "transformers", "flet"]
        for pkg in packages:
            try:
                __import__(pkg)
                print(f"[OK] {pkg}")
            except ImportError:
                print(f"[ERROR] {pkg} 未安装")
                return False
        
        # 检查Musubi脚本
        script_path = Path("third_party/musubi-tuner/src/musubi_tuner/hv_train_network.py")
        if script_path.exists():
            print("[OK] Musubi训练脚本")
        else:
            print("[ERROR] Musubi训练脚本缺失")
            return False
        
        print("[OK] 安装验证成功")
        return True
        
    except Exception as e:
        print(f"[ERROR] 验证失败: {e}")
        return False


def main():
    """主安装流程"""
    print("*** TagTracker + Musubi-Tuner 一键安装 ***")
    print("=" * 50)
    
    # 检查环境
    if not check_python_version():
        sys.exit(1)
    
    if not check_git():
        print("请先安装Git: https://git-scm.com/")
        sys.exit(1)
    
    # 执行安装
    success = True
    
    if not init_submodules():
        success = False
    
    if success and not install_dependencies():
        success = False
    
    if success and not verify_installation():
        success = False
    
    # 显示结果
    print("\n" + "=" * 50)
    if success:
        print("*** 安装完成! ***")
        print("\n使用方法:")
        print("  python main.py")
        print("\n或者:")
        print("  python main.py --debug")
    else:
        print("安装失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()