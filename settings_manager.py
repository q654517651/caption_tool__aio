import json
import os
from pathlib import Path
from typing import Dict, Any


class SettingsManager:
    def __init__(self, config_file="config.json"):
        self.config_file = Path(config_file)
        self.default_settings = {
            "musubi_dir": "",
            "kohya_dir": "",
            "vae_path": "",
            "clip_path": "",
            "t5_path": "",
            "unet_path": "",
            "default_resolution": "960,544",
            "default_batch_size": "1",
            "default_lr": "2e-4",
            "default_epochs": "16",
            "tensorboard_port": 6006
        }
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """加载设置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                # 合并默认设置（确保新增设置项存在）
                for key, value in self.default_settings.items():
                    if key not in settings:
                        settings[key] = value
                return settings
            except Exception as e:
                print(f"加载设置失败: {e}")
                return self.default_settings.copy()
        return self.default_settings.copy()

    def save_settings(self) -> bool:
        """保存设置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存设置失败: {e}")
            return False

    def get(self, key: str, default=None):
        """获取设置值"""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any):
        """设置值并自动保存"""
        self.settings[key] = value
        self.save_settings()
