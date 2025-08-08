#!/usr/bin/env python3

import os
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
from datetime import datetime
import uuid
from terminal_service import log_info, log_error, log_success, log_progress
from PIL import Image
import threading

class Dataset:
    """数据集类"""

    def __init__(self, dataset_id: str, name: str, description: str = "", created_time: str = None):
        self.dataset_id = dataset_id
        self.name = name
        self.description = description
        self.created_time = created_time or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.modified_time = self.created_time
        self.images = {}  # {image_filename: label_text}
        self.tags = []  # 数据集标签

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'dataset_id': self.dataset_id,
            'name': self.name,
            'description': self.description,
            'created_time': self.created_time,
            'modified_time': self.modified_time,
            'images': self.images,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建"""
        dataset = cls(
            dataset_id=data['dataset_id'],
            name=data['name'],
            description=data.get('description', ''),
            created_time=data.get('created_time')
        )
        dataset.modified_time = data.get('modified_time', dataset.created_time)
        dataset.images = data.get('images', {})
        dataset.tags = data.get('tags', [])
        return dataset

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = len(self.images)
        labeled = len([label for label in self.images.values() if label.strip()])
        return {
            'total': total,
            'labeled': labeled,
            'unlabeled': total - labeled,
            'completion_rate': round(labeled / total * 100) if total > 0 else 0
        }

    def add_image(self, image_path: str, label: str = "") -> bool:
        """添加图片到数据集"""
        try:
            filename = os.path.basename(image_path)
            self.images[filename] = label
            self.modified_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return True
        except Exception as e:
            log_error(f"添加图片失败: {e}")
            return False

    def update_label(self, filename: str, label: str):
        """更新标签"""
        if filename in self.images:
            self.images[filename] = label
            self.modified_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return True
        return False

    def remove_image(self, filename: str) -> bool:
        """移除图片"""
        if filename in self.images:
            del self.images[filename]
            self.modified_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return True
        return False


class DatasetManager:
    """数据集管理器"""
    def __init__(self, datasets_dir: str = "datasets"):
        # 统一工作根（可从设置注入）
        self.workspace_root = os.path.abspath("./workspace")
        self.medium_max_side = 1280
        self.web_strategy = "assets"
        self.assets_base_url = "assets"
        self.static_base_url = "/files"

        # 数据集根：workspace/datasets
        self.datasets_dir = Path(self.workspace_root) / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # 配置目录：workspace/datasets/configs
        self.config_dir = self.datasets_dir / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.datasets: Dict[str, Dataset] = {}
        self.load_datasets()


    def create_dataset(self, name: str, description: str = "") -> Tuple[bool, str]:
        """创建新数据集"""
        try:
            # 检查名称是否已存在
            if any(ds.name == name for ds in self.datasets.values()):
                return False, "数据集名称已存在"

            dataset_id = str(uuid.uuid4())
            dataset = Dataset(dataset_id, name, description)

            # 创建标准目录
            Path(self.get_images_dir(dataset_id)).mkdir(parents=True, exist_ok=True)
            Path(self.get_medium_dir(dataset_id)).mkdir(parents=True, exist_ok=True)

            self.datasets[dataset_id] = dataset
            self.save_dataset(dataset_id)

            log_success(f"创建数据集成功: {name}")
            return True, f"数据集 '{name}' 创建成功"

        except Exception as e:
            log_error(f"创建数据集失败: {e}")
            return False, f"创建失败: {str(e)}"

    def delete_dataset(self, dataset_id: str) -> Tuple[bool, str]:
        """删除数据集"""
        try:
            if dataset_id not in self.datasets:
                return False, "数据集不存在"

            dataset_name = self.datasets[dataset_id].name

            # 删除配置文件
            config_file = self.config_dir / f"{dataset_id}.json"
            if config_file.exists():
                config_file.unlink()

            # 删除图片目录
            dataset_dir = Path(self.get_dataset_path(dataset_id))
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)

            # 从内存中删除
            del self.datasets[dataset_id]

            log_success(f"删除数据集成功: {dataset_name}")
            return True, f"数据集 '{dataset_name}' 删除成功"

        except Exception as e:
            log_error(f"删除数据集失败: {e}")
            return False, f"删除失败: {str(e)}"

    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """获取数据集"""
        return self.datasets.get(dataset_id)

    def list_datasets(self) -> List[Dataset]:
        """获取所有数据集列表"""
        return list(self.datasets.values())

    def save_dataset(self, dataset_id: str) -> bool:
        """保存数据集配置"""
        try:
            if dataset_id not in self.datasets:
                return False

            dataset = self.datasets[dataset_id]
            config_file = self.config_dir / f"{dataset_id}.json"

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            log_error(f"保存数据集失败: {e}")
            return False

    def load_datasets(self):
        """加载所有数据集"""
        try:
            self.datasets.clear()

            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    dataset = Dataset.from_dict(data)

                    # —— 新增：把历史里带路径的 key 统一成文件名 —— #
                    fixed = {}
                    changed = False
                    for k, v in dataset.images.items():
                        fn = os.path.basename(k)
                        if fn != k:
                            changed = True
                        fixed[fn] = v
                    dataset.images = fixed
                    if changed:
                        # 立刻落盘，避免下次再修
                        self.save_dataset(dataset.dataset_id)

                    self.datasets[dataset.dataset_id] = dataset

                except Exception as e:
                    log_error(f"加载数据集配置失败 {config_file}: {e}")

            log_info(f"加载了 {len(self.datasets)} 个数据集")

        except Exception as e:
            log_error(f"加载数据集失败: {e}")

    def import_images_to_dataset(self, dataset_id: str, image_files: List[str]) -> Tuple[int, str]:
        """导入图片到数据集"""
        try:
            if dataset_id not in self.datasets:
                return 0, "数据集不存在"

            dataset = self.datasets[dataset_id]
            dataset_images_dir = Path(self.get_images_dir(dataset_id))

            success_count = 0
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

            for image_path in image_files:
                try:
                    if not os.path.exists(image_path):
                        continue

                    file_ext = Path(image_path).suffix.lower()
                    if file_ext not in supported_formats:
                        continue

                    filename = os.path.basename(image_path)
                    dest_path = dataset_images_dir / filename

                    # 如果文件名重复，添加时间戳
                    if dest_path.exists():
                        name_part = Path(filename).stem
                        ext_part = Path(filename).suffix
                        timestamp = int(time.time())
                        filename = f"{name_part}_{timestamp}{ext_part}"
                        dest_path = dataset_images_dir / filename

                    # 复制图片文件
                    shutil.copy2(image_path, dest_path)

                    # 检查是否有对应的标签文件
                    label_text = ""
                    txt_path = os.path.splitext(image_path)[0] + '.txt'
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                label_text = f.read().strip()
                        except Exception:
                            pass

                    # 添加到数据集
                    dataset.add_image(filename, label_text)
                    success_count += 1

                except Exception as e:
                    log_error(f"导入图片失败 {image_path}: {e}")
                    continue

            if success_count > 0:
                self.save_dataset(dataset_id)
                log_success(f"成功导入 {success_count} 张图片到数据集")

            return success_count, f"成功导入 {success_count} 张图片"

        except Exception as e:
            log_error(f"导入图片失败: {e}")
            return 0, f"导入失败: {str(e)}"

    def export_dataset(self, dataset_id: str, export_path: str, format_type: str = "folder") -> Tuple[bool, str]:
        """导出数据集"""
        try:
            if dataset_id not in self.datasets:
                return False, "数据集不存在"

            dataset = self.datasets[dataset_id]
            export_path = Path(export_path)

            if format_type == "folder":
                # 导出为文件夹格式 (图片+txt)
                dataset_export_dir = export_path / f"{dataset.name}_{dataset_id[:8]}"
                dataset_export_dir.mkdir(parents=True, exist_ok=True)

                dataset_images_dir = Path(self.get_images_dir(dataset_id))

                for filename, label in dataset.images.items():
                    # 复制图片
                    src_image = dataset_images_dir / filename
                    if src_image.exists():
                        shutil.copy2(src_image, dataset_export_dir / filename)

                        # 保存标签
                        txt_filename = os.path.splitext(filename)[0] + '.txt'
                        txt_path = dataset_export_dir / txt_filename
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(label)

                return True, f"数据集已导出到: {dataset_export_dir}"

            elif format_type == "json":
                # 导出为JSON格式
                json_file = export_path / f"{dataset.name}_{dataset_id[:8]}.json"

                export_data = {
                    'dataset_info': dataset.to_dict(),
                    'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

                return True, f"数据集已导出到: {json_file}"

            else:
                return False, "不支持的导出格式"

        except Exception as e:
            log_error(f"导出数据集失败: {e}")
            return False, f"导出失败: {str(e)}"

    def get_dataset_image_path(self, dataset_id: str, filename: str) -> Optional[str]:
        """获取数据集中图片的完整路径"""
        dataset_images_dir = Path(self.get_images_dir(dataset_id))
        image_path = dataset_images_dir / filename
        return str(image_path) if image_path.exists() else None

    def update_dataset_label(self, dataset_id: str, filename: str, label: str) -> bool:
        """更新数据集中的标签"""
        if dataset_id in self.datasets:
            result = self.datasets[dataset_id].update_label(filename, label)
            if result:
                self.save_dataset(dataset_id)
            return result
        return False

    def batch_update_labels(self, dataset_id: str, labels_dict: Dict[str, str]) -> Tuple[int, str]:
        """批量更新标签"""
        try:
            if dataset_id not in self.datasets:
                return 0, "数据集不存在"

            dataset = self.datasets[dataset_id]
            success_count = 0

            for filename, label in labels_dict.items():
                if dataset.update_label(filename, label):
                    success_count += 1

            if success_count > 0:
                self.save_dataset(dataset_id)
                log_success(f"批量更新了 {success_count} 个标签")

            return success_count, f"成功更新 {success_count} 个标签"

        except Exception as e:
            log_error(f"批量更新标签失败: {e}")
            return 0, f"更新失败: {str(e)}"

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

    # ---- 路径基础 ----
    def get_dataset_path(self, dataset_id: str) -> str:
        return os.path.join(self.workspace_root, "datasets", dataset_id)

    def get_images_dir(self, dataset_id: str) -> str:
        return os.path.join(self.get_dataset_path(dataset_id), "images")

    def get_medium_dir(self, dataset_id: str) -> str:
        return os.path.join(self.get_dataset_path(dataset_id), "medium")

    def list_images(self, dataset_id: str) -> list[str]:
        """仅返回 images/ 下的文件名（不含路径）"""
        d = self.get_images_dir(dataset_id)
        if not os.path.isdir(d):
            return []
        return sorted([f for f in os.listdir(d)
                       if os.path.isfile(os.path.join(d, f))])

    def _need_medium(self, src_path: str) -> bool:
        try:
            with Image.open(src_path) as im:
                w, h = im.size
            return max(w, h) > int(self.medium_max_side)
        except Exception:
            return False  # 读不到就当不需要，UI会回退原图

    def ensure_medium(self, dataset_id: str, filename: str) -> str:
        """
        返回中清晰度图的绝对路径：
        - 若原图尺寸 <= 阈值：直接返回原图路径（不生成 medium）
        - 若 > 阈值：在 medium/ 下生成 JPEG（最长边=阈值），并返回其路径
        """
        src = os.path.join(self.get_images_dir(dataset_id), filename)
        if not os.path.exists(src):
            return src  # 让上层自己处理不存在

        if not self._need_medium(src):
            return src  # 小图直接用原图作为中图

        os.makedirs(self.get_medium_dir(dataset_id), exist_ok=True)
        name, _ = os.path.splitext(filename)
        # 用 mtime 做版本号，源图更新会换文件名
        mtime = int(os.path.getmtime(src))
        medium_path = os.path.join(self.get_medium_dir(dataset_id), f"{name}.{mtime}.jpg")

        # 清理旧版本
        prefix = f"{name}."
        for f in os.listdir(self.get_medium_dir(dataset_id)):
            if f.startswith(prefix) and f != os.path.basename(medium_path):
                try:
                    os.remove(os.path.join(self.get_medium_dir(dataset_id), f))
                except:
                    pass

        # 生成（若不存在）
        if not os.path.exists(medium_path):
            with Image.open(src) as im:
                im.thumbnail((self.medium_max_side, self.medium_max_side), Image.Resampling.LANCZOS)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(medium_path, "JPEG", quality=80, optimize=True, progressive=True)

        return medium_path

    def path_for(self, dataset_id: str, filename: str, kind: Literal["images", "medium"]) -> str:
        """
        返回本地文件的 file:// URI，Flet 桌面端对 URI 更稳定。
        kind: "images" 原图；"medium" 中清晰度（内部会自动按阈值生成或回退原图）
        """
        if kind == "images":
            p = os.path.join(self.get_images_dir(dataset_id), filename)
        else:
            p = self.ensure_medium(dataset_id, filename)  # 小图会直接回退原图路径
        # 统一规范化为 file:// URI
        return Path(p).resolve().as_uri()

    # dataset_manager.py 中确认这个方法
    def url_for(self, dataset_id: str, filename: str, kind: Literal["images", "medium"]) -> Optional[str]:
        """Web端专用：返回相对于 assets_dir 的路径"""
        if kind == "images":
            abs_path = os.path.join(self.get_images_dir(dataset_id), filename)
        else:
            abs_path = self.ensure_medium(dataset_id, filename)

        if not os.path.exists(abs_path):
            return None

        # 关键：确保返回格式是 "datasets/abc123/images/photo.jpg"
        abs_path_norm = os.path.normpath(abs_path).replace("\\", "/")
        ws_root = os.path.normpath(self.workspace_root).replace("\\", "/")

        if abs_path_norm.startswith(ws_root):
            rel = abs_path_norm[len(ws_root):].lstrip("/")
            return rel  # 🔥 不要加 "assets/" 前缀
        return None





