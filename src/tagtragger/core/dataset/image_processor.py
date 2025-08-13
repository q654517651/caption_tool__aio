"""
Image Processor - 图像处理工具
"""

from PIL import Image, ImageOps
import os
from typing import Tuple, Optional
from ...utils.logger import log_error, log_info
from ...utils.exceptions import ImageProcessingError, ImageNotFoundError

class ImageProcessor:
    """图像处理器"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    def create_preview(self, source_path: str, dest_path: str, max_side: int = 512) -> bool:
        """创建预览图"""
        try:
            if not os.path.exists(source_path):
                raise ImageNotFoundError(source_path)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            with Image.open(source_path) as img:
                # 转换为RGB模式（如果需要）
                if img.mode in ('RGBA', 'LA', 'P'):
                    # 为透明图片添加白色背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 创建缩略图
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                
                # 保存为JPEG格式
                img.save(dest_path, 'JPEG', quality=85, optimize=True)
                
            return True
            
        except Exception as e:
            log_error(f"创建预览图失败 {source_path}: {str(e)}")
            return False
    
    def create_training_image(self, source_path: str, dest_path: str, width: int, height: int) -> bool:
        """创建训练分辨率图片"""
        try:
            if not os.path.exists(source_path):
                raise ImageNotFoundError(source_path)
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            with Image.open(source_path) as img:
                # 转换为RGB模式
                if img.mode != 'RGB':
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = background
                    else:
                        img = img.convert('RGB')
                
                # 调整大小，保持比例
                resized_img = self._resize_with_padding(img, width, height)
                
                # 保存
                resized_img.save(dest_path, 'JPEG', quality=95)
                
            return True
            
        except Exception as e:
            log_error(f"创建训练图片失败 {source_path}: {str(e)}")
            return False
    
    def _resize_with_padding(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """调整图片大小并添加填充，保持比例"""
        original_width, original_height = img.size
        
        # 计算缩放比例
        scale = min(target_width / original_width, target_height / original_height)
        
        # 缩放图片
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建目标大小的背景（白色）
        background = Image.new('RGB', (target_width, target_height), (255, 255, 255))
        
        # 居中粘贴
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2
        background.paste(img, (x, y))
        
        return background
    
    def get_image_info(self, image_path: str) -> Optional[dict]:
        """获取图片信息"""
        try:
            if not os.path.exists(image_path):
                return None
                
            with Image.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_bytes': os.path.getsize(image_path)
                }
        except Exception as e:
            log_error(f"获取图片信息失败 {image_path}: {str(e)}")
            return None
    
    def validate_image(self, image_path: str) -> bool:
        """验证图片文件"""
        try:
            if not os.path.exists(image_path):
                return False
                
            # 检查扩展名
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.supported_formats:
                return False
            
            # 尝试打开图片
            with Image.open(image_path) as img:
                img.verify()  # 验证图片完整性
                
            return True
            
        except Exception:
            return False
    
    def clean_cache(self, cache_dir: str) -> bool:
        """清理缓存目录"""
        try:
            import shutil
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                log_info(f"清理缓存目录: {cache_dir}")
                return True
            return False
        except Exception as e:
            log_error(f"清理缓存失败 {cache_dir}: {str(e)}")
            return False