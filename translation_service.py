#!/usr/bin/env python3

import os
import time
import threading
from typing import Dict, Tuple, Callable, Optional
from datetime import datetime
from chat_tool import AIChatTool
from terminal_service import log_info, log_error, log_progress, log_success


class TranslationService:
    """翻译服务"""

    def __init__(self):
        self.ai_chat_tool = AIChatTool()
        self.translation_preview = {}
        self.translation_ready = False

    @staticmethod
    def default_translation_prompt() -> str:
        """默认的翻译提示词"""
        return """【FLUX LoRA 图像打标专用翻译 Prompt】
将下方中文描述翻译为英文，严格遵守以下硬性规则：

1.准确传达原意，不得加入任何主观润色或感情色彩修饰。
2.全句仅使用主动语态，每句动作锚点必须前置，静态锚点必须具备可视化实体描述（如发型、道具、环境元素）。
3.每个视觉锚点必须拆解成一句独立短句，禁止在同一句出现多个动作、道具、服饰或背景信息。
4.句子顺序固定为：主体外貌 → 动作姿态 → 服饰道具 → 场景背景 → 光效氛围，严禁顺序颠倒。
5.句子之间仅使用英文逗号, 连接，不允许句子内部使用逗号。
6.禁止使用"and / but / or"等连词，禁止使用被动语态，禁止任何修饰性从句。
7.输出格式为：一整行英文逗号串，最后以英文句号. 结尾。
8.仅输出英文翻译，不要输出任何标签、换行或解释说明。

以下是一个翻译的参考示例
原文：
一位粉色长发的女性角色正漂浮在一个梦幻的深蓝色星空背景中，
身着一袭无袖深蓝色连衣裙，
裙摆呈现出亮黄色与粉色的渐变，
她的右手握着一支蓝色细长画笔，
画笔的末端流转着渐变的光彩，
她轻盈悬空，
左手优雅地伸出，
触碰一只白色海豚的嘴部，
双腿自然弯曲上扬，
她周围漂浮着抽象的圆环与多彩的线条，
远处有几条如锦鲤般的彩色元素在星空中游动，
整个场景充满着彩虹般的渐变光效，
营造出一种梦幻唯美的氛围。

翻译
A female character has long pink hair,
She floats lightly in the air,
She wears a sleeveless dark blue dress,
The dress hem displays a gradient of bright yellow and pink,
Her right hand holds a slender blue brush,
The brush tip shimmers with a gradient of colors,
Her left hand reaches out to touch the mouth of a white dolphin,
Her legs bend naturally upwards,
Abstract rings float around her,
Colorful lines swirl through the scene,
Several koi-like colorful elements swim in the distant starry background,
The background shows a dark blue starry sky,
Rainbow gradient light effects flow across the scene,
Soft glowing colors create a dreamy and beautiful atmosphere.
"""

    def translate_labels_preview(self,
                                 labels: Dict[str, str],
                                 prompt: str,
                                 model_type: str,
                                 progress_callback: Optional[Callable[[int, int, str], None]] = None) -> str:
        """
        翻译标签预览

        Args:
            labels: 标签字典
            prompt: 翻译提示词
            model_type: 模型类型
            progress_callback: 进度回调函数

        Returns:
            结果消息
        """
        try:
            # 收集需要翻译的标签
            labels_to_translate = {}
            for img_path, label_text in labels.items():
                if label_text and label_text.strip():
                    labels_to_translate[img_path] = label_text

            if not labels_to_translate:
                self.translation_ready = False
                log_info("没有标签需要翻译")
                return "没有标签需要翻译"

            log_info(f"开始翻译标签，共 {len(labels_to_translate)} 个标签")
            self.translation_preview = {}
            success_count = 0

            # 逐个翻译
            for i, (img_path, original_label) in enumerate(labels_to_translate.items()):
                try:
                    img_name = os.path.basename(img_path)
                    log_info(f"正在翻译: {img_name}")

                    # 调用翻译
                    translated = self.ai_chat_tool.call_chatai(
                        model_type=model_type,
                        prompt=prompt,
                        content=original_label
                    )

                    if translated and not translated.startswith("错误"):
                        self.translation_preview[img_path] = {
                            'original': original_label,
                            'translated': translated
                        }
                        success_count += 1
                        log_success(f"✓ 翻译完成: {img_name}")
                    else:
                        log_error(f"✗ 翻译失败: {img_name} - {translated}")

                    # 记录进度
                    log_progress(i + 1, len(labels_to_translate), f"翻译标签: {img_name}")

                    # 更新UI进度
                    if progress_callback:
                        progress_callback(i + 1, len(labels_to_translate), img_name)

                    # 短暂延迟
                    time.sleep(0.5)

                except Exception as e:
                    log_error(f"翻译失败 {img_path}: {e}")
                    continue

            # 生成对比结果
            if self.translation_preview:
                self.translation_ready = True
                result_msg = f"翻译完成，成功翻译 {success_count}/{len(labels_to_translate)} 个标签"
                log_success(result_msg)
                return result_msg
            else:
                self.translation_ready = False
                log_error("翻译失败，没有成功的翻译结果")
                return "翻译失败，没有成功的翻译结果"

        except Exception as e:
            error_msg = f"翻译失败: {str(e)}"
            log_error(error_msg)
            self.translation_ready = False
            return error_msg

    def save_translations(self, labels: Dict[str, str]) -> str:
        """保存翻译结果"""
        try:
            if not self.translation_preview:
                log_error("没有翻译结果可保存")
                return "❌ 没有翻译结果可保存"

            log_info("开始保存翻译结果...")
            success_count = 0

            for img_path, translation_data in self.translation_preview.items():
                try:
                    # 直接覆盖原始txt文件
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(translation_data['translated'])

                    # 更新内存中的标签
                    labels[img_path] = translation_data['translated']
                    success_count += 1

                    img_name = os.path.basename(img_path)
                    log_info(f"✓ 保存翻译: {img_name}")

                except Exception as e:
                    log_error(f"保存翻译失败 {img_path}: {e}")

            # 清空预览数据
            self.translation_preview = {}
            self.translation_ready = False

            if success_count > 0:
                result_msg = f"✅ 翻译保存完成，成功保存 {success_count} 个翻译结果"
                log_success(result_msg)
                return result_msg
            else:
                log_error("翻译保存失败")
                return "❌ 翻译保存失败"

        except Exception as e:
            error_msg = f"保存翻译失败: {str(e)}"
            log_error(error_msg)
            return f"❌ {error_msg}"

    def cancel_translations(self) -> str:
        """取消翻译"""
        self.translation_preview = {}
        self.translation_ready = False
        log_info("已取消翻译操作")
        return "✅ 已取消翻译操作"

    def get_translation_preview(self) -> Dict:
        """获取翻译预览结果"""
        return self.translation_preview

    def is_translation_ready(self) -> bool:
        """检查翻译是否准备就绪"""
        return self.translation_ready
