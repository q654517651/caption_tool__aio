"""
Dataset Manager - 数据集管理器
"""

import os
import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .models import Dataset
from .image_processor import ImageProcessor
from ...utils.logger import log_info, log_error, log_success
from ...utils.exceptions import (
    DatasetError, DatasetNotFoundError, DatasetCreateError,
    ValidationError
)
from ...utils.validators import validate_dataset_name, validate_directory
from ...config import get_config

class DatasetManager:
    """数据集管理器"""
    
    def __init__(self):
        self.config = get_config()
        self.workspace_root = Path(self.config.storage.workspace_root).resolve()
        self.datasets_dir = self.workspace_root / self.config.storage.datasets_dir
        self.config_dir = self.datasets_dir / "configs"
        
        # 确保目录存在
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 图像处理器
        self.image_processor = ImageProcessor()
        
        # 内存中的数据集缓存
        self.datasets: Dict[str, Dataset] = {}
        
        # 加载现有数据集
        self.load_all_datasets()

    def create_dataset(self, name: str, description: str = "", dataset_type: str = "image") -> Tuple[bool, str]:
        """创建新数据集"""
        try:
            # 验证名称
            validate_dataset_name(name)
            
            # 检查名称是否已存在
            if any(ds.name == name for ds in self.datasets.values()):
                raise DatasetCreateError("数据集名称已存在")

            dataset_id = str(uuid.uuid4())
            dataset = Dataset(
                dataset_id=dataset_id,
                name=name,
                description=description,
                dataset_type=dataset_type
            )

            # 创建目录结构
            dataset_path = self.get_dataset_path(dataset_id)
            original_path = dataset_path / "original"
            cache_path = dataset_path / "cache"
            cache_path.mkdir(parents=True, exist_ok=True)
            original_path.mkdir(parents=True, exist_ok=True)
            (cache_path / "previews").mkdir(parents=True, exist_ok=True)
            (cache_path / "training").mkdir(parents=True, exist_ok=True)

            # 保存到内存和文件
            self.datasets[dataset_id] = dataset
            self.save_dataset_config(dataset_id)

            log_success(f"创建数据集成功: {name} ({dataset_id})")
            return True, f"数据集 '{name}' 创建成功"

        except ValidationError as e:
            log_error(f"数据集名称验证失败: {e.message}")
            return False, e.message
        except DatasetCreateError as e:
            log_error(f"创建数据集失败: {e.message}")
            return False, e.message
        except Exception as e:
            log_error(f"创建数据集异常: {str(e)}", e)
            return False, f"创建失败: {str(e)}"

    def delete_dataset(self, dataset_id: str) -> Tuple[bool, str]:
        """删除数据集"""
        try:
            if dataset_id not in self.datasets:
                raise DatasetNotFoundError(dataset_id)

            dataset = self.datasets[dataset_id]
            dataset_name = dataset.name

            # 删除配置文件
            config_file = self.config_dir / f"{dataset_id}.json"
            if config_file.exists():
                config_file.unlink()

            # 删除数据目录
            dataset_path = self.get_dataset_path(dataset_id)
            if dataset_path.exists():
                shutil.rmtree(dataset_path)

            # 从内存中删除
            del self.datasets[dataset_id]

            log_success(f"删除数据集成功: {dataset_name}")
            return True, f"数据集 '{dataset_name}' 删除成功"

        except DatasetNotFoundError as e:
            log_error(f"数据集不存在: {dataset_id}")
            return False, e.message
        except Exception as e:
            log_error(f"删除数据集异常: {str(e)}", e)
            return False, f"删除失败: {str(e)}"

    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """获取数据集"""
        return self.datasets.get(dataset_id)

    def list_datasets(self) -> List[Dataset]:
        """获取所有数据集列表"""
        return list(self.datasets.values())

    def search_datasets(self, keyword: str) -> List[Dataset]:
        """搜索数据集"""
        if not keyword:
            return self.list_datasets()

        keyword = keyword.lower()
        results = []

        for dataset in self.datasets.values():
            if (keyword in dataset.name.lower() or 
                keyword in dataset.description.lower() or 
                keyword in ' '.join(dataset.tags).lower()):
                results.append(dataset)

        return results

    def batch_update_labels(self, dataset_id: str, labels_dict: Dict[str, str]) -> Tuple[int, str]:
        """批量更新标签"""
        try:
            dataset = self.get_dataset(dataset_id)
            if not dataset:
                raise DatasetNotFoundError(dataset_id)

            success_count = 0

            for filename, label in labels_dict.items():
                if dataset.update_label(filename, label):
                    success_count += 1

            if success_count > 0:
                self.save_dataset_config(dataset_id)
                log_success(f"批量更新了 {success_count} 个标签")

            return success_count, f"成功更新 {success_count} 个标签"

        except DatasetNotFoundError as e:
            return 0, e.message
        except Exception as e:
            log_error(f"批量更新标签失败: {str(e)}", e)
            return 0, f"更新失败: {str(e)}"

    def export_dataset(self, dataset_id: str, export_path: str, format_type: str = "folder") -> Tuple[bool, str]:
        """导出数据集"""
        try:
            dataset = self.get_dataset(dataset_id)
            if not dataset:
                raise DatasetNotFoundError(dataset_id)

            export_path = Path(export_path)

            if format_type == "folder":
                # 导出为文件夹格式 (图片+txt)
                dataset_export_dir = export_path / f"{dataset.name}_{dataset.dataset_id[:8]}"
                dataset_export_dir.mkdir(parents=True, exist_ok=True)

                dataset_images_dir = self.get_dataset_path(dataset_id) / "original"

                for filename, label in dataset.images.items():
                    # 复制图片
                    src_image = dataset_images_dir / filename
                    if src_image.exists():
                        shutil.copy2(src_image, dataset_export_dir / filename)

                        # 保存标签
                        txt_filename = Path(filename).stem + '.txt'
                        txt_path = dataset_export_dir / txt_filename
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(label)

                return True, f"数据集已导出到: {dataset_export_dir}"

            elif format_type == "json":
                # 导出为JSON格式
                json_file = export_path / f"{dataset.name}_{dataset.dataset_id[:8]}.json"

                export_data = {
                    'dataset_info': dataset.to_dict(),
                    'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

                return True, f"数据集已导出到: {json_file}"

            else:
                return False, "不支持的导出格式"

        except DatasetNotFoundError as e:
            return False, e.message
        except Exception as e:
            log_error(f"导出数据集失败: {str(e)}", e)
            return False, f"导出失败: {str(e)}"

    def update_dataset_label(self, dataset_id: str, filename: str, label: str) -> bool:
        """更新数据集中图片的标签"""
        try:
            dataset = self.get_dataset(dataset_id)
            if not dataset:
                raise DatasetNotFoundError(dataset_id)

            # 更新数据集中的标签
            if dataset.update_label(filename, label):
                # 同时更新对应的txt文件
                self._save_label_file(dataset_id, filename, label)
                self.save_dataset_config(dataset_id)
                return True
            return False

        except Exception as e:
            log_error(f"更新标签失败: {str(e)}", e)
            return False

    def import_images_to_dataset(self, dataset_id: str, image_paths: List[str]) -> Tuple[int, str]:
        """导入图片到数据集"""
        try:
            dataset = self.get_dataset(dataset_id)
            if not dataset:
                raise DatasetNotFoundError(dataset_id)

            success_count = 0
            errors = []

            for image_path in image_paths:
                try:
                    # 处理单个图片
                    if self._import_single_image(dataset, image_path):
                        success_count += 1
                except Exception as e:
                    errors.append(f"{os.path.basename(image_path)}: {str(e)}")

            # 保存数据集配置
            self.save_dataset_config(dataset_id)

            message = f"成功导入 {success_count}/{len(image_paths)} 张图片"
            if errors:
                message += f"，失败：{', '.join(errors[:3])}"
                if len(errors) > 3:
                    message += f" 等{len(errors)}个"

            if success_count > 0:
                log_success(message)
            else:
                log_error(message)

            return success_count, message

        except DatasetNotFoundError as e:
            return 0, e.message
        except Exception as e:
            log_error(f"批量导入图片失败: {str(e)}", e)
            return 0, f"导入失败: {str(e)}"

    def get_dataset_path(self, dataset_id: str) -> Path:
        """获取数据集目录路径"""
        return self.datasets_dir / dataset_id

    def get_dataset_image_path(self, dataset_id: str, filename: str, image_type: str = "original") -> Optional[str]:
        """获取数据集中图片的路径"""
        dataset_path = self.get_dataset_path(dataset_id)
        
        if image_type == "original":
            image_path = dataset_path / "original" / filename
        elif image_type == "preview":
            # 生成预览图
            image_path = self._get_or_create_preview(dataset_id, filename)
        elif image_type == "training":
            # 生成训练图
            image_path = self._get_or_create_training_image(dataset_id, filename)
        else:
            return None

        return str(image_path) if image_path and image_path.exists() else None

    def resolve_image_src(self, dataset_id: str, filename: str, kind: str = "preview") -> Dict[str, Any]:
        """解析图片资源路径（兼容旧接口）"""
        image_path = self.get_dataset_image_path(dataset_id, filename, kind)
        if image_path:
            return {
                "src": f"file://{image_path}",
                "abs": image_path
            }
        return {"src": "", "abs": ""}

    def save_dataset_config(self, dataset_id: str) -> bool:
        """保存数据集配置到文件"""
        try:
            if dataset_id not in self.datasets:
                return False

            dataset = self.datasets[dataset_id]
            config_file = self.config_dir / f"{dataset_id}.json"

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            log_error(f"保存数据集配置失败: {str(e)}", e)
            return False

    def load_all_datasets(self):
        """加载所有数据集配置"""
        try:
            self.datasets.clear()

            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    dataset = Dataset.from_dict(data)
                    self.datasets[dataset.dataset_id] = dataset
                    
                    # 加载对应的标签文件
                    self._load_label_files(dataset)

                except Exception as e:
                    log_error(f"加载数据集配置失败 {config_file}: {str(e)}")

            log_info(f"加载了 {len(self.datasets)} 个数据集")

        except Exception as e:
            log_error(f"加载数据集列表失败: {str(e)}", e)

    def _import_single_image(self, dataset: Dataset, image_path: str) -> bool:
        """导入单个图片"""
        try:
            # 验证图片文件
            from ...utils.validators import validate_image_file
            validate_image_file(image_path)

            filename = os.path.basename(image_path)
            original_dir = self.get_dataset_path(dataset.dataset_id) / "original"
            
            # 复制到原图目录
            dest_path = original_dir / filename
            shutil.copy2(image_path, dest_path)
            
            # 加载现有标签（如果存在txt文件）
            label = self._load_label_from_txt(image_path)
            
            # 添加到数据集
            dataset.add_image(filename, label)
            
            # 保存标签文件
            if label:
                self._save_label_file(dataset.dataset_id, filename, label)

            return True

        except Exception as e:
            log_error(f"导入图片失败 {image_path}: {str(e)}")
            return False

    def _load_label_files(self, dataset: Dataset):
        """加载数据集的所有标签文件"""
        original_dir = self.get_dataset_path(dataset.dataset_id) / "original"
        if not original_dir.exists():
            return

        for image_file in original_dir.iterdir():
            if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}:
                label_file = image_file.with_suffix('.txt')
                if label_file.exists():
                    try:
                        with open(label_file, 'r', encoding='utf-8') as f:
                            label = f.read().strip()
                        dataset.images[image_file.name] = label
                    except Exception as e:
                        log_error(f"读取标签文件失败 {label_file}: {str(e)}")

    def _load_label_from_txt(self, image_path: str) -> str:
        """从txt文件加载标签"""
        txt_path = os.path.splitext(image_path)[0] + '.txt'
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                log_error(f"读取标签文件失败 {txt_path}: {str(e)}")
        return ""

    def _save_label_file(self, dataset_id: str, filename: str, label: str):
        """保存标签到txt文件"""
        try:
            original_dir = self.get_dataset_path(dataset_id) / "original"
            image_path = original_dir / filename
            label_path = image_path.with_suffix('.txt')
            
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(label)
        except Exception as e:
            log_error(f"保存标签文件失败 {filename}: {str(e)}")

    def _get_or_create_preview(self, dataset_id: str, filename: str) -> Optional[Path]:
        """获取或创建预览图"""
        preview_dir = self.get_dataset_path(dataset_id) / "cache" / "previews"
        preview_path = preview_dir / filename
        
        if preview_path.exists():
            return preview_path
            
        # 生成预览图
        original_path = self.get_dataset_path(dataset_id) / "original" / filename
        if original_path.exists():
            if self.image_processor.create_preview(
                str(original_path), 
                str(preview_path), 
                self.config.storage.preview_max_side
            ):
                return preview_path
        
        return None

    def _get_or_create_training_image(self, dataset_id: str, filename: str, resolution: str = "1024,1024") -> Optional[Path]:
        """获取或创建训练分辨率图片"""
        training_dir = self.get_dataset_path(dataset_id) / "cache" / "training" / resolution.replace(',', 'x')
        training_dir.mkdir(parents=True, exist_ok=True)
        training_path = training_dir / filename
        
        if training_path.exists():
            return training_path
            
        # 生成训练分辨率图片
        original_path = self.get_dataset_path(dataset_id) / "original" / filename
        if original_path.exists():
            width, height = map(int, resolution.split(','))
            if self.image_processor.create_training_image(
                str(original_path), 
                str(training_path), 
                width, height
            ):
                return training_path
        
        return None