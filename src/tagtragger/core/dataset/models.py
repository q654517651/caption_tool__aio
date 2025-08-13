"""
Dataset data models
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from ...utils.logger import log_error
from ...config.constants import DATASET_TYPES

@dataclass
class Dataset:
    """数据集模型"""
    dataset_id: str
    name: str
    dataset_type: str = "image"  # image, video, image_control
    description: str = ""
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    images: Dict[str, str] = field(default_factory=dict)  # {filename: label}
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.modified_time is None:
            self.modified_time = self.created_time

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'dataset_id': self.dataset_id,
            'name': self.name,
            'dataset_type': self.dataset_type,
            'description': self.description,
            'created_time': self.created_time,
            'modified_time': self.modified_time,
            'images': self.images,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Dataset':
        """从字典创建"""
        return cls(
            dataset_id=data['dataset_id'],
            name=data['name'],
            dataset_type=data.get('dataset_type', 'image'),
            description=data.get('description', ''),
            created_time=data.get('created_time'),
            modified_time=data.get('modified_time'),
            images=data.get('images', {}),
            tags=data.get('tags', [])
        )

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

    def add_image(self, filename: str, label: str = "") -> bool:
        """添加图片到数据集"""
        try:
            self.images[filename] = label
            self._update_modified_time()
            return True
        except Exception as e:
            log_error(f"添加图片失败: {e}")
            return False

    def update_label(self, filename: str, label: str) -> bool:
        """更新标签"""
        if filename in self.images:
            self.images[filename] = label
            self._update_modified_time()
            return True
        return False

    def remove_image(self, filename: str) -> bool:
        """移除图片"""
        if filename in self.images:
            del self.images[filename]
            self._update_modified_time()
            return True
        return False

    def get_label(self, filename: str) -> str:
        """获取图片标签"""
        return self.images.get(filename, "")

    def has_image(self, filename: str) -> bool:
        """检查是否包含图片"""
        return filename in self.images

    def get_image_count(self) -> int:
        """获取图片数量"""
        return len(self.images)

    def get_labeled_count(self) -> int:
        """获取已标注图片数量"""
        return len([label for label in self.images.values() if label.strip()])

    def get_unlabeled_images(self) -> List[str]:
        """获取未标注的图片列表"""
        return [filename for filename, label in self.images.items() if not label.strip()]

    def get_labeled_images(self) -> Dict[str, str]:
        """获取已标注的图片字典"""
        return {filename: label for filename, label in self.images.items() if label.strip()}

    def _update_modified_time(self):
        """更新修改时间"""
        self.modified_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def validate_type(self) -> bool:
        """验证数据集类型"""
        return self.dataset_type in DATASET_TYPES