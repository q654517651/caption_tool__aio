#!/usr/bin/env python3

import gradio as gr
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
import base64
from PIL import Image
import io
from chat_tool import AIChatTool


# 简单的日志记录
def log_info(message):
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def log_error(message):
    print(f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


class SimpleImageLabelingSystem:
    def __init__(self):
        # 核心数据
        self.current_folder = ""
        # 保存扫描到的图片
        self.images = []
        self.labels = {}

        # 归一化相关
        self.normalization_analysis = None
        self.normalization_preview = {}

        # AI打标模块
        # self.ai_labeler = AILabeler()
        self.ai_chat_tool = AIChatTool()
        self.tag_normalizer = TagNormalizer(self.ai_chat_tool)
        # self.translator = Translator()
        self.dataset_manager = DatasetManager()

        # 配置
        self.config = {
            'labeling_prompt': self._default_labeling_prompt(),
            'translation_prompt': self._default_translation_prompt(),
            'model_type': 'GPT',
            'delay_between_calls': 2.0
        }

        # 支持的图像格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    def _default_labeling_prompt(self) -> str:
        """默认的打标提示词"""
        return """你是一名图像理解专家，请根据以下图片内容，生成自然流畅、具体清晰的图像描述。要求如下：
                1. 使用简洁准确的中文句子；
                2. 避免使用"图中"、"这是一张图片"等冗余措辞；
                3. 语言风格自然、具象，不使用抽象形容词或主观感受；
                4. 不要描述画面中的风格；
                5. 描述的结构为[主体] + [外观/服装] + [动作/姿势] + [背景/环境] + [氛围/灯光（可选）]；
                现在请描述这张图片的内容："""

    @staticmethod
    def _default_translation_prompt() -> str:
        """默认的翻译提示词"""
        return """请将以下中文描述翻译成英文，保持原意不变，要求：
                1. 翻译要准确、自然、流畅
                2. 保持原文的描述顺序和重点
                3. 使用适合AI绘画的描述风格
                4. 直接输出翻译结果，不要加任何额外说明
                
                中文描述："""

    def scan_images(self, folder_path: str) -> Tuple[List[str], str]:
        """扫描文件夹中的图片"""
        try:
            if not os.path.exists(folder_path):
                return [], "文件夹不存在"

            self.current_folder = folder_path
            self.images = []

            for file in os.listdir(folder_path):
                if Path(file).suffix.lower() in self.supported_formats:
                    img_path = os.path.join(folder_path, file)
                    self.images.append(img_path)

            self.images.sort()
            self.load_existing_labels()

            message = f"找到 {len(self.images)} 张图片"
            log_info(message)

            return self.images, message

        except Exception as e:
            error_msg = f"扫描文件夹出错: {str(e)}"
            log_error(error_msg)
            return [], error_msg

    def load_existing_labels(self):
        """加载已有的标签文件"""
        self.labels = {}

        for img_path in self.images:
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            label_text = ""

            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        label_text = f.read().strip()
                except Exception as e:
                    log_error(f"读取标签文件失败 {txt_path}: {e}")

            self.labels[img_path] = label_text


    # def create_image_gallery_html(self) -> str:
    #     """创建图片展示HTML"""
    #     if not self.images:
    #         return "<p>没有找到图片</p>"
    #
    #     html_parts = ["<div style='display: flex; flex-wrap: wrap; gap: 10px; padding: 10px;'>"]
    #
    #     for i, img_path in enumerate(self.images):
    #         img_name = os.path.basename(img_path)
    #         current_label = self.labels.get(img_path, "")
    #
    #         # 读取图片并转换为base64
    #         try:
    #             with Image.open(img_path) as img:
    #                 # 创建缩略图
    #                 img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    #                 buffer = io.BytesIO()
    #                 img.save(buffer, format='JPEG', quality=85)
    #                 img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    #                 img_src = f"data:image/jpeg;base64,{img_base64}"
    #         except:
    #             img_src = ""
    #
    #         # 创建图片卡片
    #         img_container = f"""
    #         <div style='
    #             width: 300px;
    #             border: 1px solid #ddd;
    #             padding: 10px;
    #             border-radius: 5px;
    #             background: white;
    #         '>
    #             <div style='text-align: center; margin-bottom: 10px;'>
    #                 <img src='{img_src}' style='max-width: 100%; max-height: 200px;' alt='{img_name}'/>
    #             </div>
    #             <div style='font-weight: bold; margin-bottom: 5px; word-break: break-all;'>
    #                 {img_name}
    #             </div>
    #             <div>
    #                 <textarea id='label_{i}'
    #                           style='width: 100%; height: 80px; resize: vertical;'
    #                           placeholder='标签内容...'
    #                           readonly>{current_label}</textarea>
    #             </div>
    #             <div style='text-align: right; margin-top: 5px; color: {"green" if current_label else "orange"}'>
    #                 {'✓ 已标注' if current_label else '○ 未标注'}
    #             </div>
    #         </div>
    #         """
    #         html_parts.append(img_container)
    #
    #     html_parts.append("</div>")
    #
    #     # 统计信息
    #     labeled_count = len([l for l in self.labels.values() if l.strip()])
    #     stats_html = f"""
    #     <div style='background: #f0f0f0; padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
    #         <h3>数据集统计</h3>
    #         <p>总数: {len(self.images)} | 已标注: {labeled_count} | 未标注: {len(self.images) - labeled_count}</p>
    #     </div>
    #     """
    #
    #     return stats_html + ''.join(html_parts)


    def create_image_gallery_html(self) -> str:
        """创建图片展示HTML"""
        if not self.images:
            return "<p>没有找到图片</p>"

        # 响应式网格布局容器
        html_parts = ["""
        <div style='
            width: 100%;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            padding: 15px;
            box-sizing: border-box;
        '>
        """]

        for i, img_path in enumerate(self.images):
            img_name = os.path.basename(img_path)
            current_label = self.labels.get(img_path, "")

            # 初始化默认值
            img_src = ""

            # 读取图片并转换为base64
            try:
                with Image.open(img_path) as img:
                    # 创建缩略图
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    img_src = f"data:image/jpeg;base64,{img_base64}"
            except (OSError, IOError, ValueError) as e:
                log_error(f"读取图片失败 {img_path}: {e}")
                img_src = ""

            # 创建响应式图片卡片
            img_container = f"""
            <div style='
                border: 1px solid #ddd;
                padding: 15px;
                border-radius: 8px;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                min-height: 400px;
            '>
                <div style='
                    text-align: center; 
                    margin-bottom: 15px;
                    flex: 0 0 auto;
                '>
                    {f"<img src='{img_src}' style='width: 100%; max-height: 250px; object-fit: contain; border-radius: 4px;' alt='{img_name}'/>" if img_src else f"<div style='height: 250px; display: flex; align-items: center; justify-content: center; background: #f5f5f5; color: #999; border-radius: 4px; flex-direction: column;'><div>图片加载失败</div><div style='font-size: 12px; margin-top: 5px;'>{img_name}</div></div>"}
                </div>
                <div style='
                    font-weight: bold; 
                    margin-bottom: 10px; 
                    word-break: break-all;
                    font-size: 14px;
                    color: #333;
                    flex: 0 0 auto;
                '>
                    {img_name}
                </div>
                <div style='flex: 1 1 auto; display: flex; flex-direction: column;'>
                    <textarea id='label_{i}' 
                              style='
                                  width: 100%; 
                                  height: 100px; 
                                  resize: vertical; 
                                  border: 1px solid #ccc;
                                  border-radius: 4px;
                                  padding: 8px;
                                  font-family: inherit;
                                  font-size: 13px;
                                  box-sizing: border-box;
                                  flex: 1;
                              ' 
                              placeholder='标签内容...'
                              readonly>{current_label}</textarea>
                </div>
                <div style='
                    text-align: right; 
                    margin-top: 10px; 
                    color: {"green" if current_label else "orange"};
                    font-size: 12px;
                    font-weight: bold;
                    flex: 0 0 auto;
                '>
                    {'✓ 已标注' if current_label else '○ 未标注'}
                </div>
            </div>
            """
            html_parts.append(img_container)

        html_parts.append("</div>")

        # 统计信息
        labeled_count = len([l for l in self.labels.values() if l.strip()])
        stats_html = f"""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            width: 100%;
            box-sizing: border-box;
        '>
            <h3 style='margin: 0 0 15px 0; font-size: 24px; font-weight: 600;'>📊 数据集统计</h3>
            <div style='
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
                gap: 15px;
                text-align: center;
            '>
                <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 28px; font-weight: bold; margin-bottom: 5px;'>{len(self.images)}</div>
                    <div style='font-size: 14px; opacity: 0.9;'>总图片数</div>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 28px; font-weight: bold; margin-bottom: 5px; color: #4ade80;'>{labeled_count}</div>
                    <div style='font-size: 14px; opacity: 0.9;'>已标注</div>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 28px; font-weight: bold; margin-bottom: 5px; color: #f97316;'>{len(self.images) - labeled_count}</div>
                    <div style='font-size: 14px; opacity: 0.9;'>未标注</div>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;'>
                    <div style='font-size: 28px; font-weight: bold; margin-bottom: 5px; color: #06b6d4;'>{round(labeled_count / len(self.images) * 100) if self.images else 0}%</div>
                    <div style='font-size: 14px; opacity: 0.9;'>完成度</div>
                </div>
            </div>
        </div>
        """

        return stats_html + ''.join(html_parts)


    def start_ai_labeling(self, prompt: str, model_type: str, delay: float) -> str:
        """开始AI打标"""
        try:
            # 只标注未标注的图片
            unlabeled_images = [img for img in self.images if not self.labels.get(img, "").strip()]

            if not unlabeled_images:
                return "所有图片都已标注"

            success_count = 0

            # 逐个处理图片
            for i, img_path in enumerate(unlabeled_images):
                try:
                    # 调用AI进行标注
                    label_text = self.ai_chat_tool.call_chatai(model_type=model_type, prompt=prompt,
                                                               image_path=img_path)
                    # if model_type == "本地LLM Studio":
                    #     label_text = self.ai_chat_tool.call_chatai(model_type=ModelType.LOCAL, prompt=prompt,
                    #                                                image_path=img_path)
                    # else:  # GPT
                    #     label_text = self.ai_labeler.call_gpt(img_path, prompt)

                    if label_text and not label_text.startswith("错误"):
                        self.labels[img_path] = label_text
                        success_count += 1

                        # 保存到文件
                        txt_path = os.path.splitext(img_path)[0] + '.txt'
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(label_text)

                    log_info(f"完成 {i + 1}/{len(unlabeled_images)}: {os.path.basename(img_path)}")

                    # 延迟避免API限制
                    if i < len(unlabeled_images) - 1:
                        time.sleep(delay)

                except Exception as e:
                    log_error(f"处理图片失败 {img_path}: {e}")
                    continue

            return f"AI标注完成，成功标注 {success_count}/{len(unlabeled_images)} 张图片"

        except Exception as e:
            error_msg = f"AI标注失败: {str(e)}"
            log_error(error_msg)
            return error_msg


    def analyze_normalization(self, model_type: str) -> Tuple[str, str]:
        """分析归一化规则，返回规则描述和对比表格"""
        try:
            # 收集所有标签
            all_labels = {}
            for img_path, label_text in self.labels.items():
                if label_text and label_text.strip():
                    all_labels[os.path.basename(img_path)] = label_text

            if not all_labels:
                return "没有标签需要归一化", ""

            # 调用TagNormalizer进行分析
            analysis_result = self.tag_normalizer.analyze_normalization(all_labels, model_type)


            if isinstance(analysis_result, dict) and 'normalized_labels' in analysis_result:
                # 生成规则描述和对比表格
                rules_html = self.tag_normalizer.generate_rules_display()
                comparison_html = self.tag_normalizer.generate_comparison_table(self.labels)
                return rules_html, comparison_html
            else:
                error_msg = f"分析失败: {analysis_result.get('error', '未知错误')}"
                return error_msg, ""

        except Exception as e:
            error_msg = f"归一化分析失败: {str(e)}"
            log_error(error_msg)
            return error_msg, ""


    def apply_normalization(self) -> str:
        """应用归一化修改"""
        try:
            # 调用TagNormalizer应用归一化
            new_labels, changes_count = self.tag_normalizer.apply_normalization(self.labels, self.images)

            if changes_count == 0:
                return "没有需要修改的标签"

            # 更新标签并保存到文件
            for img_path in self.images:
                img_name = os.path.basename(img_path)
                if img_name in new_labels and self.labels[img_path] != new_labels[img_path]:
                    self.labels[img_path] = new_labels[img_path]

                    # 保存到文件
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(new_labels[img_path])

                    log_info(f"归一化修改: {img_name}")

                # 清除分析结果
                self.tag_normalizer.clear_analysis()

            return f"✅ 归一化完成！成功修改了 {changes_count} 个标签"

        except Exception as e:
            error_msg = f"应用归一化失败: {str(e)}"
            log_error(error_msg)
            return error_msg

    def cancel_normalization(self) -> str:
        """取消归一化修改"""
        self.normalization_analysis = None
        return "❌ 已取消归一化操作"

    def translate_labels(self, prompt: str, model_type: str) -> str:
        """翻译标签"""
        try:
            # 收集需要翻译的标签
            labels_to_translate = {}
            for img_path, label_text in self.labels.items():
                if label_text and label_text.strip():
                    labels_to_translate[img_path] = label_text

            if not labels_to_translate:
                return "没有标签需要翻译"

            success_count = 0

            # 逐个翻译
            for img_path, original_label in labels_to_translate.items():
                try:
                    # 调用翻译
                    translated = self.ai_chat_tool.call_chatai(model_type=model_type, prompt=prompt,
                                                               content=original_label)

                    if translated and not translated.startswith("错误"):
                        # 保存翻译结果到新文件
                        base_path = os.path.splitext(img_path)[0]
                        translated_file = f"{base_path}_translated.txt"
                        with open(translated_file, 'w', encoding='utf-8') as f:
                            f.write(translated)
                        success_count += 1

                        log_info(f"翻译完成: {os.path.basename(img_path)}")

                    # 短暂延迟
                    time.sleep(0.5)

                except Exception as e:
                    log_error(f"翻译失败 {img_path}: {e}")
                    continue

            return f"翻译完成，成功翻译 {success_count}/{len(labels_to_translate)} 个标签"

        except Exception as e:
            error_msg = f"翻译失败: {str(e)}"
            log_error(error_msg)
            return error_msg

    def save_dataset(self, format_type: str) -> str:
        """保存数据集"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if format_type == "json":
                # 保存为JSON格式
                output_file = os.path.join(self.current_folder, f"dataset_{timestamp}.json")
                dataset = {
                    'images': [],
                    'created': timestamp
                }

                for img_path, label in self.labels.items():
                    dataset['images'].append({
                        'filename': os.path.basename(img_path),
                        'path': img_path,
                        'label': label
                    })

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)

                return f"数据集已保存到: {output_file}"

            elif format_type == "csv":
                # 保存为CSV格式
                output_file = os.path.join(self.current_folder, f"dataset_{timestamp}.csv")
                df = pd.DataFrame([
                    {'filename': os.path.basename(path), 'label': label}
                    for path, label in self.labels.items()
                ])
                df.to_csv(output_file, index=False, encoding='utf-8')

                return f"数据集已保存到: {output_file}"

            else:
                return "不支持的格式"

        except Exception as e:
            return f"保存失败: {str(e)}"


# 标签归一化模块
class TagNormalizer:
    def __init__(self, ai_chat_tool):
        self.ai_chat_tool = ai_chat_tool
        self.analysis_result = None
        self.batch_size = 25  # 每批处理的标签数量

    def analyze_normalization(self, labels_dict: Dict[str, str], model_type: str) -> dict:
        """分批分析需要归一化的标签"""
        try:
            if not labels_dict:
                return {"error": "没有标签需要归一化"}

            # 分批处理
            label_items = list(labels_dict.items())
            batches = [label_items[i:i + self.batch_size]
                       for i in range(0, len(label_items), self.batch_size)]

            log_info(f"将 {len(label_items)} 个标签分成 {len(batches)} 批处理")

            all_suggestions = []
            all_normalized = {}

            # 分批分析
            for batch_idx, batch in enumerate(batches, 1):
                log_info(f"处理第 {batch_idx}/{len(batches)} 批")

                batch_result = self._analyze_batch(dict(batch), model_type)
                if "error" not in batch_result:
                    all_suggestions.extend(batch_result.get("suggestions", []))
                    all_normalized.update(batch_result.get("normalized_labels", {}))

                # 批次间延迟
                if batch_idx < len(batches):
                    time.sleep(1)

            # 合并结果
            self.analysis_result = {
                "suggestions": all_suggestions,
                "normalized_labels": all_normalized
            }

            return self.analysis_result

        except Exception as e:
            log_error(f"归一化分析失败: {e}")
            return {"error": str(e)}

    def _analyze_batch(self, batch_labels: Dict[str, str], model_type: str) -> dict:
        """分析单批标签"""
        prompt = f"""请分析以下 {len(batch_labels)} 个图像标签，找出需要归一化的内容。
                请识别相似表达、格式问题等需要统一的地方。
                
                标签列表："""

        for img_name, label in batch_labels.items():
            prompt += f"\n{img_name}: {label}"

        prompt += """请返回JSON格式：
                {
                    "suggestions": [{"原始": "xxx", "建议": "yyy", "原因": "zzz"}],
                    "normalized_labels": {"图片名": "归一化后的标签"}
                }
                只返回JSON，不要其他文字。"""

        try:
            result = self.ai_chat_tool.call_chatai(model_type=model_type, prompt=prompt)
            return self._parse_json_response(result) or {"suggestions": [], "normalized_labels": {}}
        except Exception as e:
            log_error(f"批次分析失败: {e}")
            return {"suggestions": [], "normalized_labels": {}}

    def generate_rules_display(self) -> str:
        """生成归一化规则HTML显示"""
        if not self.analysis_result or 'suggestions' not in self.analysis_result:
            return "<p>没有找到归一化建议</p>"

        suggestions = self.analysis_result.get('suggestions', [])
        normalized_count = len(self.analysis_result.get('normalized_labels', {}))

        html = f"""
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <h3 style='color: #2c3e50; margin-bottom: 15px;'>🔍 归一化规则分析</h3>
        """

        if suggestions:
            for i, suggestion in enumerate(suggestions[:10], 1):  # 只显示前10条
                html += f"""
                <div style='background: white; padding: 12px; margin: 8px 0; border-left: 4px solid #3498db; border-radius: 4px;'>
                    <h4 style='color: #2980b9; margin: 0 0 8px 0;'>规则 {i}</h4>
                    <p style='margin: 5px 0;'><strong>原始:</strong> {suggestion.get('原始', 'N/A')}</p>
                    <p style='margin: 5px 0;'><strong>建议:</strong> {suggestion.get('建议', 'N/A')}</p>
                    <p style='margin: 5px 0; color: #7f8c8d;'><strong>原因:</strong> {suggestion.get('原因', 'N/A')}</p>
                </div>
                """

            if len(suggestions) > 10:
                html += f"<p style='text-align: center; color: #7f8c8d;'>... 还有 {len(suggestions) - 10} 条规则未显示</p>"
        else:
            html += "<p>没有找到需要归一化的规则</p>"

        html += f"""
        <div style='background: #e8f5e8; padding: 10px; border-radius: 5px; margin-top: 15px;'>
            <h4 style='margin: 0 0 10px 0; color: #27ae60;'>📊 统计信息</h4>
            <p style='margin: 5px 0;'>发现规则数: {len(suggestions)}</p>
            <p style='margin: 5px 0;'>需修改标签: {normalized_count}</p>
        </div></div>
        """

        return html

    def generate_comparison_table(self, original_labels: Dict[str, str]) -> str:
        """生成修改前后对比表格"""
        if not self.analysis_result or 'normalized_labels' not in self.analysis_result:
            return "<p>没有对比数据</p>"

        normalized_labels = self.analysis_result['normalized_labels']

        html = """
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <h3 style='color: #2c3e50; margin-bottom: 15px;'>📋 标签修改对比</h3>
            <div style='overflow-x: auto;'>
                <table style='width: 100%; border-collapse: collapse; background: white; border-radius: 5px; overflow: hidden;'>
                    <thead>
                        <tr style='background: #34495e; color: white;'>
                            <th style='padding: 12px; text-align: left; width: 200px;'>图片名称</th>
                            <th style='padding: 12px; text-align: left;'>修改前</th>
                            <th style='padding: 12px; text-align: left;'>修改后</th>
                            <th style='padding: 12px; text-align: center; width: 100px;'>状态</th>
                        </tr>
                    </thead><tbody>
        """

        for img_path, original_label in original_labels.items():
            if not original_label or not original_label.strip():
                continue

            img_name = os.path.basename(img_path)
            normalized_label = normalized_labels.get(img_name, original_label)
            has_changes = original_label != normalized_label

            status_color = "#e74c3c" if has_changes else "#27ae60"
            status_text = "需修改" if has_changes else "无变化"
            row_bg = "#fff5f5" if has_changes else "#f0fff0"

            html += f"""
            <tr style='background: {row_bg}; border-bottom: 1px solid #ecf0f1;'>
                <td style='padding: 12px; font-weight: bold; word-break: break-word;'>{img_name}</td>
                <td style='padding: 12px; max-width: 300px; word-wrap: break-word;'>{original_label}</td>
                <td style='padding: 12px; max-width: 300px; word-wrap: break-word;'>{normalized_label}</td>
                <td style='padding: 12px; text-align: center;'>
                    <span style='color: {status_color}; font-weight: bold;'>{status_text}</span>
                </td>
            </tr>
            """

        html += "</tbody></table></div></div>"
        return html

    def apply_normalization(self, labels_dict: Dict[str, str], images: List[str]) -> Tuple[Dict[str, str], int]:
        """应用归一化，返回新标签字典和修改数量"""
        if not self.analysis_result or 'normalized_labels' not in self.analysis_result:
            return labels_dict, 0

        normalized_labels = self.analysis_result['normalized_labels']
        new_labels = labels_dict.copy()
        changes_count = 0

        for img_path in images:
            img_name = os.path.basename(img_path)
            if img_name in normalized_labels:
                new_label = normalized_labels[img_name]
                if new_labels[img_path] != new_label:
                    new_labels[img_path] = new_label
                    changes_count += 1

        return new_labels, changes_count

    def clear_analysis(self):
        """清除分析结果"""
        self.analysis_result = None

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """精简的JSON解析方法"""
        try:
            # 直接尝试解析整个响应
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # 如果失败，尝试提取JSON部分
                json_start = response.find('{')
                json_end = response.rfind('}')

                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = response[json_start:json_end + 1]
                    parsed = json.loads(json_str)

                    # 简单验证必要字段
                    if "suggestions" in parsed or "normalized_labels" in parsed:
                        return parsed

            except json.JSONDecodeError:
                pass

            # 解析失败时返回空结构
            log_error(f"JSON解析失败，原始响应: {response[:200]}...")
            return {"suggestions": [], "normalized_labels": {}}


# 简化的数据管理模块
class DatasetManager:
    pass


def create_gradio_interface():
    """创建Gradio界面"""
    system = SimpleImageLabelingSystem()

    with gr.Blocks(title="简化版图像打标系统", theme=gr.themes.Soft(), fill_width=True) as interface:
        gr.Markdown("# 🏷️ 简化版图像打标系统")

        with gr.Tab("📁 数据加载"):
            with gr.Row():
                folder_input = gr.Textbox(
                    label="图片文件夹路径",
                    placeholder="输入包含图片的文件夹路径..."
                )
                scan_btn = gr.Button("扫描图片", variant="primary")

            scan_status = gr.Textbox(label="扫描状态")
            gallery_display = gr.HTML(label="图片与标签")
            refresh_btn = gr.Button("刷新显示")

            def scan_and_display(folder_path):
                images, message = system.scan_images(folder_path)
                gallery_html = system.create_image_gallery_html()
                return message, gallery_html

            scan_btn.click(
                fn=scan_and_display,
                inputs=[folder_input],
                outputs=[scan_status, gallery_display]
            )

            refresh_btn.click(
                fn=lambda: system.create_image_gallery_html(),
                outputs=[gallery_display]
            )

        with gr.Tab("🤖 AI打标"):
            prompt_input = gr.Textbox(
                label="打标提示词",
                value=system.config['labeling_prompt'],
                lines=8
            )

            with gr.Row():
                model_choice = gr.Radio(
                    choices=["LLM Studio", "GPT"],
                    label="选择模型",
                    value="LLM Studio"
                )
                delay_slider = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.5,
                    label="调用间隔(秒)"
                )

            start_labeling_btn = gr.Button("开始AI打标", variant="primary")
            labeling_status = gr.Textbox(label="打标状态")

            start_labeling_btn.click(
                fn=lambda p, m, d: system.start_ai_labeling(prompt=p, model_type=m, delay=d),
                inputs=[prompt_input, model_choice, delay_slider],
                outputs=[labeling_status]
            )

        with gr.Tab("🔄 标签归一化"):
            gr.Markdown("### 步骤1: 分析归一化规则")

            with gr.Row():
                normalize_model = gr.Radio(
                    choices=["本地LLM Studio", "GPT"],
                    label="选择模型",
                    value="GPT"
                )
                analyze_btn = gr.Button("分析归一化规则", variant="primary")

            # 显示归一化规则
            rules_display = gr.HTML(label="归一化规则", visible=False)

            gr.Markdown("### 步骤2: 预览修改对比")
            # 显示修改前后对比
            comparison_display = gr.HTML(label="标签修改对比", visible=False)

            gr.Markdown("### 步骤3: 确认并应用修改")
            with gr.Row():
                apply_btn = gr.Button("✅ 确认并应用修改", variant="primary", visible=False)
                cancel_btn = gr.Button("❌ 取消修改", variant="secondary", visible=False)

            normalization_status = gr.Textbox(label="操作状态", lines=3)

            def analyze_normalization_rules(model):
                system.config['model_type'] = model
                rules_html, comparison_html = system.analyze_normalization(model)

                # 判断是否有有效的分析结果
                has_results = bool(
                    system.normalization_analysis and 'normalized_labels' in system.normalization_analysis)

                return (
                    rules_html,
                    comparison_html,
                    gr.update(visible=has_results),  # rules_display
                    gr.update(visible=has_results),  # comparison_display
                    gr.update(visible=has_results),  # apply_btn
                    gr.update(visible=has_results),  # cancel_btn
                    "✅ 分析完成，请查看规则和对比结果" if has_results else "❌ 分析失败或没有需要归一化的内容"
                )

            def apply_normalization_changes():
                result = system.apply_normalization()
                return (
                    result,
                    gr.update(visible=False),  # rules_display
                    gr.update(visible=False),  # comparison_display
                    gr.update(visible=False),  # apply_btn
                    gr.update(visible=False),  # cancel_btn
                    system.create_image_gallery_html()  # 更新图片显示
                )

            def cancel_normalization_changes():
                result = system.cancel_normalization()
                return (
                    result,
                    gr.update(visible=False),  # rules_display
                    gr.update(visible=False),  # comparison_display
                    gr.update(visible=False),  # apply_btn
                    gr.update(visible=False),  # cancel_btn
                )

            analyze_btn.click(
                fn=analyze_normalization_rules,
                inputs=[normalize_model],
                outputs=[
                    rules_display,
                    comparison_display,
                    rules_display,
                    comparison_display,
                    apply_btn,
                    cancel_btn,
                    normalization_status
                ]
            )

            apply_btn.click(
                fn=apply_normalization_changes,
                outputs=[
                    normalization_status,
                    rules_display,
                    comparison_display,
                    apply_btn,
                    cancel_btn,
                    gallery_display  # 更新主页面的图片显示
                ]
            )

            cancel_btn.click(
                fn=cancel_normalization_changes,
                outputs=[
                    normalization_status,
                    rules_display,
                    comparison_display,
                    apply_btn,
                    cancel_btn
                ]
            )

        with gr.Tab("🌐 标签翻译"):
            translation_prompt = gr.Textbox(
                label="翻译提示词",
                value=system.config['translation_prompt'],
                lines=5
            )

            with gr.Row():
                trans_model = gr.Radio(
                    choices=["LLM_Studio", "GPT"],
                    label="选择模型",
                    value="GPT"
                )
                # target_lang = gr.Textbox(
                #     label="目标语言",
                #     value="英文"
                # )

            translate_btn = gr.Button("开始翻译", variant="primary")
            translation_status = gr.Textbox(label="翻译状态")

            translate_btn.click(
                fn=system.translate_labels,
                inputs=[translation_prompt, trans_model],
                outputs=[translation_status]
            )

        with gr.Tab("💾 数据管理"):
            export_format = gr.Radio(
                choices=["json", "csv"],
                label="保存格式",
                value="json"
            )

            save_btn = gr.Button("保存数据集", variant="primary")
            save_status = gr.Textbox(label="保存状态")

            save_btn.click(
                fn=system.save_dataset,
                inputs=[export_format],
                outputs=[save_status]
            )

    return interface


def main():
    """主函数"""
    interface = create_gradio_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
