#!/usr/bin/env python3

import os
import time
import threading
from typing import List, Tuple, Callable, Optional
from pathlib import Path
from datetime import datetime
from services.chat_service import AIChatTool


# from terminal_service import log_info, log_error, log_progress


class LabelingService:
    """AI打标服务"""

    def __init__(self, terminal_service):
        self.ai_chat_tool = AIChatTool()
        self.terminal_service = terminal_service

    @staticmethod
    def default_labeling_prompt() -> str:
        """默认的打标提示词"""
        return """你是一名图像理解专家，请根据以下图片内容，生成自然流畅、具体清晰的图像描述。要求如下：
1. 使用简洁准确的中文句子，使用逗号进行连接；
2. 避免使用"图中"、"这是一张图片"等冗余措辞；
3. 语言风格自然、具象，不使用抽象形容词或主观感受；
4. 描述的内容不要重复
5. 将描述结构划分为以下模块，并标明模块标题；

【描述的参考示例】
【主体与外貌】
一位银白短发的男性角色，神情坚毅，目光直视前方，
【服饰与道具】
佩戴带有蓝色装饰的银色战盔，头部饰有披风状发饰。身穿银白色装甲，肩部设有高位护甲，胸口嵌有蓝色发光核心。左手持一柄龙首造型的能量长枪，右手自然下垂，手背覆盖护甲。
【动作与姿态】
左腿踏出，右腿蹬地跃起，左手持枪向右上方突刺，身体前倾，披风和发饰随动作后扬，整体动作紧凑有力。
【环境与场景】
背景为夜晚城市废墟，天空中悬挂满月，角色身后浮现出蓝白色狼影，下方有敌方机械残骸。
【氛围与光效】
蓝白色能量线条环绕角色，长枪带出一道光弧，背景月光与冷色特效交织，营造出动势与压迫感。
【镜头视角信息】
[仰视，俯视，平视]三选一

【输出格式】
请按以下模块生成描述：

【主体与外貌】
【服饰与道具】
【动作与姿态】
【环境与场景】
【氛围与光效】
【镜头视角信息】

开始生成
"""

    def label_images(self,
                     images: List[str],
                     labels: dict,
                     prompt: str,
                     model_type: str,
                     delay: float,
                     progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, str]:
        """
        AI打标图片

        Args:
            images: 图片路径列表
            labels: 现有标签字典
            prompt: 打标提示词
            model_type: 模型类型
            delay: 调用间隔
            progress_callback: 进度回调函数

        Returns:
            (success_count, message)
        """
        try:
            # 只标注未标注的图片
            unlabeled_images = [img for img in images if not labels.get(os.path.basename(img), "").strip()]

            if not unlabeled_images:
                self.terminal_service.log_info("所有图片都已标注")
                return 0, "所有图片都已标注"

            self.terminal_service.log_info(f"开始AI打标，共 {len(unlabeled_images)} 张未标注图片")
            success_count = 0

            # 逐个处理图片
            for i, img_path in enumerate(unlabeled_images):
                try:
                    img_name = os.path.basename(img_path)
                    self.terminal_service.log_info(f"正在处理图片: {img_name}")

                    # 调用AI进行标注
                    label_text = self.ai_chat_tool.call_chatai(
                        model_type=model_type,
                        prompt=prompt,
                        image_path=img_path
                    )

                    if label_text and not label_text.startswith("错误"):
                        labels[img_name] = label_text
                        success_count += 1

                        # 保存到文件
                        txt_path = os.path.splitext(img_path)[0] + '.txt'
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(label_text)

                        self.terminal_service.log_info(f"✓ 成功标注: {img_name}")
                    else:
                        self.terminal_service.log_error(f"✗ 标注失败: {img_name} - {label_text}")

                    # 记录进度
                    self.terminal_service.log_progress(i + 1, len(unlabeled_images), f"处理图片: {img_name}")

                    # 更新UI进度
                    if progress_callback:
                        progress_callback(i + 1, len(unlabeled_images), img_name)

                    # 延迟避免API限制
                    if i < len(unlabeled_images) - 1:
                        self.terminal_service.log_info(f"等待 {delay} 秒后继续...")
                        time.sleep(delay)

                except Exception as e:
                    self.terminal_service.log_error(f"处理图片失败 {img_path}: {e}")
                    continue

            result_msg = f"AI标注完成，成功标注 {success_count}/{len(unlabeled_images)} 张图片"
            if success_count > 0:
                self.terminal_service.log_info(result_msg)
            else:
                self.terminal_service.log_error("AI标注未成功处理任何图片")

            return success_count, result_msg

        except Exception as e:
            error_msg = f"AI标注失败: {str(e)}"
            self.terminal_service.log_error(error_msg)
            return 0, error_msg
