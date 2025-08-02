#!/usr/bin/env python3

import gradio as gr
import os
import json
import time
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from datetime import datetime
import base64
from PIL import Image
import io
from chat_tool import AIChatTool
from tag_normalizer_tool import TagNormalizer


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
        self.ai_chat_tool = AIChatTool()
        self.tag_normalizer = TagNormalizer(self.ai_chat_tool)

        # 翻译相关 - 新增这部分
        self.translation_preview = {}
        self.translation_ready = False

        # 配置
        self.config = {
            'labeling_prompt': self._default_labeling_prompt(),
            'translation_prompt': self._default_translation_prompt(),
            'model_type': 'GPT',
            'delay_between_calls': 2.0
        }

        # 支持的图像格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    @staticmethod
    def _default_labeling_prompt() -> str:
        """默认的打标提示词"""
        return """你是一名图像理解专家，请根据以下图片内容，生成自然流畅、具体清晰的图像描述。要求如下：
1. 使用简洁准确的中文句子，句与句之间用逗号衔接；
2. 避免使用“图中”、“这是一张图片”等冗余措辞；
3. 语言风格自然、具象，不使用抽象形容词或主观感受；
4. 将描述结构划分为以下模块，并标明模块标题；
5. 如果角色是面对镜头，注意左右手的描述不要弄反
6. 不同模块之间的描述不要重复

【描述的参考示例】
【主体与外貌】
一位银白短发的男性角色，神情坚毅，目光直视前方，佩戴带有蓝色装饰的银色战盔，头部饰有披风状发饰。
【服饰与道具】
身穿银白色装甲，肩部设有高位护甲，胸口嵌有蓝色发光核心。左手持一柄龙首造型的能量长枪，右手自然下垂，手背覆盖护甲。
【动作与姿态】
左腿踏出，右腿蹬地跃起，左手持枪向右上方突刺，身体前倾，披风和发饰随动作后扬，整体动作紧凑有力。
【环境与场景】
背景为夜晚城市废墟，天空中悬挂满月，角色身后浮现出蓝白色狼影，下方有敌方机械残骸。
【氛围与光效】
蓝白色能量线条环绕角色，长枪带出一道光弧，背景月光与冷色特效交织，营造出动势与压迫感。
【镜头视角信息】
仰视

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

    def translate_labels_preview(self, prompt: str, model_type: str) -> str:
        """翻译标签预览"""
        try:
            # 收集需要翻译的标签
            labels_to_translate = {}
            for img_path, label_text in self.labels.items():
                if label_text and label_text.strip():
                    labels_to_translate[img_path] = label_text

            if not labels_to_translate:
                self.translation_ready = False
                return "<p>没有标签需要翻译</p>"

            self.translation_preview = {}
            success_count = 0

            # 逐个翻译
            for img_path, original_label in labels_to_translate.items():
                try:
                    # 调用翻译
                    translated = self.ai_chat_tool.call_chatai(model_type=model_type, prompt=prompt,
                                                               content=original_label)

                    if translated and not translated.startswith("错误"):
                        self.translation_preview[img_path] = {
                            'original': original_label,
                            'translated': translated
                        }
                        success_count += 1

                        log_info(f"翻译完成: {os.path.basename(img_path)}")

                    # 短暂延迟
                    time.sleep(0.5)

                except Exception as e:
                    log_error(f"翻译失败 {img_path}: {e}")
                    continue

            # 生成对比HTML
            if self.translation_preview:
                self.translation_ready = True
                return self.create_translation_comparison_html()
            else:
                self.translation_ready = False
                return "<p>翻译失败，没有成功的翻译结果</p>"

        except Exception as e:
            error_msg = f"翻译失败: {str(e)}"
            log_error(error_msg)
            self.translation_ready = False
            return f"<p>{error_msg}</p>"

    def create_translation_comparison_html(self) -> str:
        """创建翻译对比HTML"""
        if not self.translation_preview:
            return "<p>没有翻译预览数据</p>"

        html_parts = ["""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        '>
            <h3 style='margin: 0 0 10px 0; font-size: 24px; font-weight: 600;'>🌐 翻译对比预览</h3>
            <p style='margin: 0; font-size: 16px; opacity: 0.9;'>共 {count} 个标签待翻译</p>
        </div>
        <div style='
            display: grid;
            gap: 15px;
            padding: 0;
        '>
        """.format(count=len(self.translation_preview))]

        for img_path, translation_data in self.translation_preview.items():
            img_name = os.path.basename(img_path)
            original = translation_data['original']
            translated = translation_data['translated']

            item_html = f"""
            <div style='
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                overflow: hidden;
                background: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            '>
                <div style='
                    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
                    color: white;
                    padding: 15px;
                    font-weight: bold;
                    font-size: 16px;
                '>
                    📄 {img_name}
                </div>
                <div style='padding: 20px;'>
                    <div style='margin-bottom: 20px;'>
                        <div style='
                            font-weight: bold;
                            color: #374151;
                            margin-bottom: 8px;
                            font-size: 14px;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;
                        '>
                            🇨🇳 原文
                        </div>
                        <div style='
                            background: #fef3c7;
                            border: 1px solid #fbbf24;
                            padding: 15px;
                            border-radius: 8px;
                            font-size: 14px;
                            line-height: 1.6;
                            color: #92400e;
                        '>
                            {original}
                        </div>
                    </div>
                    <div>
                        <div style='
                            font-weight: bold;
                            color: #374151;
                            margin-bottom: 8px;
                            font-size: 14px;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;
                        '>
                            🇺🇸 译文
                        </div>
                        <div style='
                            background: #dcfce7;
                            border: 1px solid #22c55e;
                            padding: 15px;
                            border-radius: 8px;
                            font-size: 14px;
                            line-height: 1.6;
                            color: #166534;
                        '>
                            {translated}
                        </div>
                    </div>
                </div>
            </div>
            """
            html_parts.append(item_html)

        html_parts.append("</div>")
        return ''.join(html_parts)

    def save_translations(self) -> str:
        """保存翻译结果"""
        try:
            if not self.translation_preview:
                return "❌ 没有翻译结果可保存"

            success_count = 0
            for img_path, translation_data in self.translation_preview.items():
                try:
                    # 直接覆盖原始txt文件
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(translation_data['translated'])

                    # 更新内存中的标签
                    self.labels[img_path] = translation_data['translated']
                    success_count += 1

                except Exception as e:
                    log_error(f"保存翻译失败 {img_path}: {e}")

            # 清空预览数据
            self.translation_preview = {}
            self.translation_ready = False

            if success_count > 0:
                return f"✅ 翻译保存完成，成功保存 {success_count} 个翻译结果"
            else:
                return "❌ 翻译保存失败"

        except Exception as e:
            error_msg = f"保存翻译失败: {str(e)}"
            log_error(error_msg)
            return f"❌ {error_msg}"

    def cancel_translations(self) -> str:
        """取消翻译"""
        self.translation_preview = {}
        self.translation_ready = False
        return "✅ 已取消翻译操作"

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
                    choices=["LLM_Studio", "GPT"],
                    label="选择模型",
                    value="LLM_Studio"
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

        # 简化后的Gradio界面 - 直接调用TagNormalizer

        with gr.Tab("🔄 标签归一化"):
            gr.Markdown("### 📋 三步归一化流程")

            # 模型选择
            normalize_model = gr.Radio(
                choices=["LLM_Studio", "GPT"],
                label="选择AI模型",
                value="GPT"
            )

            # 第一步：分析规则
            gr.Markdown("#### 步骤1️⃣: 分析归一化规则")
            step1_btn = gr.Button("🔍 分析归一化规则", variant="primary")
            rules_display = gr.HTML(label="归一化规则", visible=False)

            # 第二步：应用规则
            gr.Markdown("#### 步骤2️⃣: 应用规则生成对比")
            step2_btn = gr.Button("🔄 应用规则处理标签", variant="secondary", visible=False)
            comparison_display = gr.HTML(label="标签修改对比", visible=False)

            # 第三步：保存更改
            gr.Markdown("#### 步骤3️⃣: 确认并保存")
            with gr.Row():
                step3_save_btn = gr.Button("✅ 保存更改", variant="primary", visible=False)
                step3_cancel_btn = gr.Button("❌ 取消更改", variant="secondary", visible=False)

            # 状态显示
            normalization_status = gr.Textbox(label="操作状态", lines=2)

            def step1_analyze_rules(model):
                """第一步：直接分析规则"""
                try:
                    # 收集标签数据
                    all_labels = {}
                    for img_path, label_text in system.labels.items():
                        if label_text and label_text.strip():
                            all_labels[os.path.basename(img_path)] = label_text

                    if not all_labels:
                        return "<p>没有标签需要归一化</p>", gr.update(visible=False), gr.update(
                            visible=False), "❌ 没有可用标签"

                    # 直接调用TagNormalizer
                    rules_html = system.tag_normalizer.step1_analyze_rules(all_labels, model)
                    has_rules = system.tag_normalizer.has_rules()

                    return (
                        rules_html,
                        gr.update(visible=has_rules),  # rules_display
                        gr.update(visible=has_rules),  # step2_btn
                        "✅ 规则分析完成，请查看规则内容" if has_rules else "❌ 未找到归一化规则"
                    )
                except Exception as e:
                    log_error(f"规则分析失败: {e}")
                    return f"<p>规则分析失败: {str(e)}</p>", gr.update(visible=False), gr.update(
                        visible=False), f"❌ 分析失败: {str(e)}"

            def step2_apply_rules(model):
                """第二步：直接应用规则"""
                try:
                    # 直接调用TagNormalizer
                    comparison_html = system.tag_normalizer.step2_apply_rules(model)
                    has_changes = system.tag_normalizer.has_changes()

                    return (
                        comparison_html,
                        gr.update(visible=has_changes),  # comparison_display
                        gr.update(visible=has_changes),  # step3_save_btn
                        gr.update(visible=has_changes),  # step3_cancel_btn
                        "✅ 规则应用完成，请查看对比结果" if has_changes else "❌ 规则应用失败"
                    )
                except Exception as e:
                    log_error(f"应用规则失败: {e}")
                    return f"<p>应用规则失败: {str(e)}</p>", gr.update(visible=False), gr.update(
                        visible=False), gr.update(visible=False), f"❌ 应用失败: {str(e)}"

            def step3_save_changes():
                """第三步：直接保存更改"""
                try:
                    # 直接调用TagNormalizer保存
                    result = system.tag_normalizer.step3_save_changes(system.images)

                    # 如果保存成功，重新加载标签
                    if result.startswith("✅"):
                        system.load_existing_labels()

                    return (
                        result,
                        gr.update(visible=False),  # rules_display
                        gr.update(visible=False),  # step2_btn
                        gr.update(visible=False),  # comparison_display
                        gr.update(visible=False),  # step3_save_btn
                        gr.update(visible=False),  # step3_cancel_btn
                        system.create_image_gallery_html()  # 刷新图片显示
                    )
                except Exception as e:
                    log_error(f"保存更改失败: {e}")
                    return f"❌ 保存失败: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            def step3_cancel_changes():
                """取消更改"""
                try:
                    result = system.tag_normalizer.cancel_changes()

                    return (
                        result,
                        gr.update(visible=False),  # rules_display
                        gr.update(visible=False),  # step2_btn
                        gr.update(visible=False),  # comparison_display
                        gr.update(visible=False),  # step3_save_btn
                        gr.update(visible=False),  # step3_cancel_btn
                    )
                except Exception as e:
                    log_error(f"取消操作失败: {e}")
                    return f"❌ 取消失败: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            # 绑定事件
            step1_btn.click(
                fn=step1_analyze_rules,
                inputs=[normalize_model],
                outputs=[rules_display, rules_display, step2_btn, normalization_status]
            )

            step2_btn.click(
                fn=step2_apply_rules,
                inputs=[normalize_model],
                outputs=[comparison_display, comparison_display, step3_save_btn, step3_cancel_btn, normalization_status]
            )

            step3_save_btn.click(
                fn=step3_save_changes,
                outputs=[normalization_status, rules_display, step2_btn, comparison_display, step3_save_btn,
                         step3_cancel_btn, gallery_display]
            )

            step3_cancel_btn.click(
                fn=step3_cancel_changes,
                outputs=[normalization_status, rules_display, step2_btn, comparison_display, step3_save_btn,
                         step3_cancel_btn]
            )

        with gr.Tab("🌐 标签翻译"):
            gr.Markdown("### 📋 三步翻译流程")

            # 模型选择和提示词
            translation_prompt = gr.Textbox(
                label="翻译提示词",
                value=system.config['translation_prompt'],
                lines=5
            )

            translate_model = gr.Radio(
                choices=["LLM_Studio", "GPT"],
                label="选择AI模型",
                value="GPT"
            )

            # 第一步：开始翻译
            gr.Markdown("#### 步骤1️⃣: 开始翻译并生成对比")
            translate_step1_btn = gr.Button("🌐 开始翻译", variant="primary")
            translation_comparison_display = gr.HTML(label="翻译对比", visible=False)

            # 第二步：确认保存
            gr.Markdown("#### 步骤2️⃣: 确认并保存")
            with gr.Row():
                translate_save_btn = gr.Button("✅ 保存翻译", variant="primary", visible=False)
                translate_cancel_btn = gr.Button("❌ 取消翻译", variant="secondary", visible=False)

            # 状态显示
            translation_status = gr.Textbox(label="翻译状态", lines=2)

            def start_translation(prompt, model):
                """开始翻译"""
                try:
                    comparison_html = system.translate_labels_preview(prompt, model)
                    has_results = system.translation_ready

                    return (
                        comparison_html,
                        gr.update(visible=has_results),  # translation_comparison_display
                        gr.update(visible=has_results),  # translate_save_btn
                        gr.update(visible=has_results),  # translate_cancel_btn
                        "✅ 翻译完成，请查看对比结果" if has_results else "❌ 翻译失败或没有可翻译内容"
                    )
                except Exception as e:
                    log_error(f"翻译失败: {e}")
                    return (
                        f"<p>翻译失败: {str(e)}</p>",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        f"❌ 翻译失败: {str(e)}"
                    )

            def save_translation_results():
                """保存翻译结果"""
                try:
                    result = system.save_translations()

                    return (
                        result,
                        gr.update(visible=False),  # translation_comparison_display
                        gr.update(visible=False),  # translate_save_btn
                        gr.update(visible=False),  # translate_cancel_btn
                        system.create_image_gallery_html()  # 刷新图片显示
                    )
                except Exception as e:
                    log_error(f"保存翻译失败: {e}")
                    return f"❌ 保存失败: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update()

            def cancel_translation_results():
                """取消翻译"""
                try:
                    result = system.cancel_translations()

                    return (
                        result,
                        gr.update(visible=False),  # translation_comparison_display
                        gr.update(visible=False),  # translate_save_btn
                        gr.update(visible=False),  # translate_cancel_btn
                    )
                except Exception as e:
                    log_error(f"取消翻译失败: {e}")
                    return f"❌ 取消失败: {str(e)}", gr.update(), gr.update(), gr.update()

            # 绑定事件
            translate_step1_btn.click(
                fn=start_translation,
                inputs=[translation_prompt, translate_model],
                outputs=[translation_comparison_display, translation_comparison_display, translate_save_btn,
                         translate_cancel_btn, translation_status]
            )

            translate_save_btn.click(
                fn=save_translation_results,
                outputs=[translation_status, translation_comparison_display, translate_save_btn, translate_cancel_btn,
                         gallery_display]
            )

            translate_cancel_btn.click(
                fn=cancel_translation_results,
                outputs=[translation_status, translation_comparison_display, translate_save_btn, translate_cancel_btn]
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
