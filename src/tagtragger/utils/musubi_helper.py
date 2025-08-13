"""
Musubi-Tuner 工具函数
使用内嵌的git子模块，无需额外配置
"""

import subprocess
from pathlib import Path
from typing import Dict, Any
from ..utils.logger import log_info, log_error, log_success


def get_musubi_path() -> str:
    """获取内嵌的musubi-tuner路径"""
    project_root = Path(__file__).parent.parent.parent.parent
    return str(project_root / "third_party" / "musubi-tuner")


def check_musubi_status() -> Dict[str, Any]:
    """检查内嵌musubi-tuner状态"""
    try:
        musubi_dir = Path(get_musubi_path())
        
        if not musubi_dir.exists():
            return {
                "available": False,
                "status": "Git子模块未初始化，请运行: git submodule update --init --recursive"
            }
        
        # 检查关键训练脚本
        required_scripts = [
            "src/musubi_tuner/qwen_image_train_network.py",
            "src/musubi_tuner/qwen_image_cache_latents.py",
            "src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py"
        ]
        
        missing_scripts = []
        for script in required_scripts:
            script_path = musubi_dir / script
            if not script_path.exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            return {
                "available": False,
                "status": f"训练脚本缺失: {', '.join(missing_scripts)}"
            }
        
        # 检查Python环境和accelerate命令
        try:
            result = subprocess.run(
                ["accelerate", "--help"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode != 0:
                return {
                    "available": False,
                    "status": "accelerate命令不可用，请安装: pip install accelerate"
                }
        except FileNotFoundError:
            return {
                "available": False,
                "status": "accelerate命令未找到，请安装: pip install accelerate"
            }
        except subprocess.TimeoutExpired:
            return {
                "available": False,
                "status": "accelerate命令检查超时"
            }
        
        # 检查musubi_tuner模块是否可导入
        import sys
        musubi_src = musubi_dir / 'src'
        old_path = sys.path.copy()
        try:
            if str(musubi_src) not in sys.path:
                sys.path.insert(0, str(musubi_src))
            
            import musubi_tuner
            # 检查关键模块
            from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_QWEN_IMAGE
            
        except ImportError as e:
            sys.path = old_path
            error_msg = str(e)
            
            # 提取缺失的依赖名
            if "No module named" in error_msg:
                if "'" in error_msg:
                    missing_dep = error_msg.split("'")[1]
                else:
                    missing_dep = error_msg.replace("No module named ", "").strip()
            else:
                missing_dep = "未知依赖"
            
            # 常见依赖的特定安装建议
            install_suggestions = {
                "cv2": "opencv-python==4.10.0.84",
                "torch": "torch torchvision",
                "transformers": "transformers==4.54.1",
                "diffusers": "diffusers==0.32.1",
                "accelerate": "accelerate==1.6.0",
                "numpy": "numpy",
                "safetensors": "safetensors==0.4.5",
                "musubi_tuner": "完整musubi-tuner依赖"
            }
            
            suggested_package = install_suggestions.get(missing_dep, missing_dep)
            
            return {
                "available": False,
                "status": f"Musubi依赖缺失: {missing_dep}\n\n🔧 推荐解决方案:\n1. cd third_party/musubi-tuner\n2. pip install -e .\n\n📦 或手动安装关键包:\npip install opencv-python torch accelerate transformers diffusers safetensors\n\n⚠️ 当前缺失: {suggested_package}"
            }
        except Exception as e:
            sys.path = old_path
            return {
                "available": False,
                "status": f"Musubi模块检查失败: {e}"
            }
        finally:
            sys.path = old_path
        
        return {
            "available": True,
            "status": "Musubi-Tuner 就绪，所有组件正常"
        }
        
    except Exception as e:
        log_error(f"检查Musubi状态失败: {e}")
        return {
            "available": False,
            "status": f"检查失败: {str(e)}"
        }


def get_available_training_backends() -> Dict[str, bool]:
    """获取可用的训练后端"""
    musubi_dir = Path(get_musubi_path())
    
    backends = {
        "qwen_image": False,
        "flux": False,
        "sd": False
    }
    
    if not musubi_dir.exists():
        return backends
    
    # 检查各个模型的训练脚本
    script_mapping = {
        "qwen_image": "src/musubi_tuner/qwen_image_train_network.py",
        "flux": "src/musubi_tuner/flux_train_network.py",
        "sd": "src/musubi_tuner/sd_train_network.py"
    }
    
    for backend, script_path in script_mapping.items():
        if (musubi_dir / script_path).exists():
            backends[backend] = True
    
    return backends


def validate_musubi_installation() -> bool:
    """验证Musubi-Tuner安装完整性"""
    status = check_musubi_status()
    if not status["available"]:
        log_error(f"Musubi-Tuner不可用: {status['status']}")
        return False
    
    log_success("Musubi-Tuner验证通过")
    return True


def get_training_script_path(training_type: str) -> str:
    """获取训练脚本路径"""
    musubi_dir = Path(get_musubi_path())
    
    script_mapping = {
        "qwen_image": "src/musubi_tuner/qwen_image_train_network.py",
        "flux": "src/musubi_tuner/flux_train_network.py", 
        "sd": "src/musubi_tuner/sd_train_network.py"
    }
    
    if training_type not in script_mapping:
        raise ValueError(f"不支持的训练类型: {training_type}")
    
    script_path = musubi_dir / script_mapping[training_type]
    if not script_path.exists():
        raise FileNotFoundError(f"训练脚本不存在: {script_path}")
    
    return str(script_path)