#!/usr/bin/env python3
"""
TagTracker - 集成打标与训练的LoRA训练工具

统一入口文件
"""

import sys
import os
import argparse
from pathlib import Path

# 添加src路径到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def run_flet_app():
    """运行Flet桌面应用"""
    print("启动TagTracker新架构...")
    try:
        from tagtragger.ui.flet.app import main as flet_main
        flet_main()
    except Exception as e:
        print(f"新架构启动失败，回退到旧版本: {e}")
        print("使用稳定版本启动...")
        import flet as ft
        from main_flet_old import main as old_main
        ft.app(target=old_main, assets_dir=os.path.abspath("./workspace"))


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='TagTracker - 集成打标与训练的LoRA训练工具')
    parser.add_argument(
        '--config', '-c',
        help='指定配置文件路径'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='启用调试模式'
    )

    args = parser.parse_args()

    # 设置调试模式
    if args.debug:
        os.environ['TAGTRACKER_DEBUG'] = '1'

    # 设置配置文件路径
    if args.config:
        os.environ['TAGTRACKER_CONFIG'] = args.config

    # 启动Flet桌面界面
    print("启动TagTracker桌面界面...")
    run_flet_app()


if __name__ == "__main__":
    main()
