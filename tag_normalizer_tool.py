import os
import time
from typing import Dict, List
from datetime import datetime


# 简单的日志记录
def log_info(message):
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def log_error(message):
    print(f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


# 简化的标签归一化模块

class TagNormalizer:
    def __init__(self, ai_chat_tool):
        self.ai_chat_tool = ai_chat_tool
        self.batch_size = 20

        # 三步流程的数据
        self.rules = []  # 第一步：规则列表
        self.new_labels = {}  # 第二步：新标签（未保存）
        self.original_labels = {}  # 原始标签备份

    def step1_analyze_rules(self, labels_dict: Dict[str, str], model_type: str) -> str:
        """第一步：分析归一化规则"""
        try:
            self.original_labels = labels_dict.copy()

            # 分批分析规则
            label_items = list(labels_dict.items())
            batches = [label_items[i:i + self.batch_size]
                       for i in range(0, len(label_items), self.batch_size)]

            log_info(f"分析规则：{len(label_items)}个标签分{len(batches)}批处理")

            all_rules = []

            for batch_idx, batch in enumerate(batches, 1):
                log_info(f"分析第{batch_idx}/{len(batches)}批规则")

                prompt = self._build_rule_analysis_prompt(dict(batch))
                result = self.ai_chat_tool.call_chatai(model_type=model_type, prompt=prompt)

                batch_rules = self._extract_rules_from_response(result)
                all_rules.extend(batch_rules)

                if batch_idx < len(batches):
                    time.sleep(1)

            # 合并和去重规则
            self.rules = self._merge_rules(all_rules)

            return self._generate_rules_html()

        except Exception as e:
            log_error(f"规则分析失败: {e}")
            return f"<p>规则分析失败: {str(e)}</p>"

    def step2_apply_rules(self, model_type: str) -> str:
        """第二步：应用规则生成新标签"""
        try:
            if not self.rules:
                return "<p>没有可用的规则，请先分析规则</p>"

            self.new_labels = {}
            rules_prompt = self._build_rules_prompt()

            log_info(f"应用规则：处理{len(self.original_labels)}个标签")

            # 逐个处理标签（确保精确控制）
            for img_name, original_label in self.original_labels.items():
                try:
                    prompt = f"{rules_prompt}\n\n原标签：{original_label}\n\n请根据规则修改标签，直接返回修改后的标签："

                    new_label = self.ai_chat_tool.call_chatai(model_type=model_type, prompt=prompt)
                    self.new_labels[img_name] = new_label.strip()

                    time.sleep(0.5)  # 控制调用频率

                except Exception as e:
                    log_error(f"处理标签失败 {img_name}: {e}")
                    self.new_labels[img_name] = original_label  # 保持原标签

            return self._generate_comparison_html()

        except Exception as e:
            log_error(f"应用规则失败: {e}")
            return f"<p>应用规则失败: {str(e)}</p>"

    def step3_save_changes(self, images: List[str]) -> str:
        """第三步：保存更改到文件"""
        try:
            if not self.new_labels:
                return "没有可保存的更改"

            saved_count = 0

            for img_path in images:
                img_name = os.path.basename(img_path)

                if img_name in self.new_labels:
                    txt_path = os.path.splitext(img_path)[0] + '.txt'

                    try:
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(self.new_labels[img_name])
                        saved_count += 1
                    except Exception as e:
                        log_error(f"保存文件失败 {txt_path}: {e}")

            # 清理状态
            self.rules = []
            self.new_labels = {}
            self.original_labels = {}

            return f"✅ 成功保存 {saved_count} 个标签文件"

        except Exception as e:
            log_error(f"保存失败: {e}")
            return f"保存失败: {str(e)}"

    def cancel_changes(self) -> str:
        """取消所有更改"""
        self.rules = []
        self.new_labels = {}
        self.original_labels = {}
        return "❌ 已取消所有更改"

    def _build_rule_analysis_prompt(self, batch_labels: Dict[str, str]) -> str:
        """构建规则分析提示词"""
        prompt = f"""分析以下{len(batch_labels)}个图像标签，找出需要归一化的规则。

标签列表："""

        for img_name, label in batch_labels.items():
            prompt += f"\n{img_name}: {label}"

        prompt += """

请找出可以统一的表达方式、格式问题等，返回3-5条具体的归一化规则。
每条规则格式：规则描述|修改原因
例如：将"一位女性"统一为"女性"|简化表达
只返回规则列表，每行一条规则。"""

        return prompt

    def _build_rules_prompt(self) -> str:
        """构建应用规则的提示词"""
        prompt = "请根据以下归一化规则修改标签：\n\n"

        for i, rule in enumerate(self.rules, 1):
            prompt += f"{i}. {rule}\n"

        prompt += "\n请严格按照规则修改提示词，并转换为流畅的自然语言。"
        return prompt

    def _extract_rules_from_response(self, response: str) -> List[str]:
        """从AI响应中提取规则"""
        rules = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line and '|' in line and not line.startswith('规则'):
                rules.append(line)

        return rules

    def _merge_rules(self, all_rules: List[str]) -> List[str]:
        """合并和去重规则"""
        # 简单去重，实际可以更智能
        unique_rules = list(set(all_rules))
        return unique_rules[:10]  # 最多保留10条规则

    def _generate_rules_html(self) -> str:
        """生成规则展示HTML"""
        if not self.rules:
            return "<p>没有找到归一化规则</p>"

        html = """
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
            <h3 style='color: #2c3e50; margin-bottom: 15px;'>🔍 归一化规则</h3>
        """

        for i, rule in enumerate(self.rules, 1):
            parts = rule.split('|')
            rule_desc = parts[0] if len(parts) > 0 else rule
            rule_reason = parts[1] if len(parts) > 1 else ""

            html += f"""
            <div style='background: white; padding: 10px; margin: 8px 0; border-left: 4px solid #3498db; border-radius: 4px;'>
                <p style='margin: 0; font-weight: bold;'>规则 {i}: {rule_desc}</p>
                {f"<p style='margin: 5px 0 0 0; color: #666; font-size: 14px;'>原因: {rule_reason}</p>" if rule_reason else ""}
            </div>
            """

        html += f"""
            <div style='background: #e8f5e8; padding: 10px; border-radius: 5px; margin-top: 15px;'>
                <p style='margin: 0; color: #27ae60;'>📊 共发现 {len(self.rules)} 条归一化规则</p>
            </div>
        </div>
        """

        return html

    def _generate_comparison_html(self) -> str:
        """生成对比展示HTML"""
        if not self.new_labels:
            return "<p>没有对比数据</p>"

        html = """
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
            <h3 style='color: #2c3e50; margin-bottom: 15px;'>📋 标签修改对比</h3>
            <div style='overflow-x: auto;'>
                <table style='width: 100%; border-collapse: collapse; background: white; border-radius: 5px;'>
                    <thead>
                        <tr style='background: #34495e; color: white;'>
                            <th style='padding: 12px; text-align: left;'>图片名称</th>
                            <th style='padding: 12px; text-align: left;'>修改前</th>
                            <th style='padding: 12px; text-align: left;'>修改后</th>
                            <th style='padding: 12px; text-align: center;'>状态</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        changed_count = 0

        for img_name, original in self.original_labels.items():
            new_label = self.new_labels.get(img_name, original)
            has_change = original != new_label

            if has_change:
                changed_count += 1

            status_color = "#e74c3c" if has_change else "#27ae60"
            status_text = "已修改" if has_change else "无变化"
            row_bg = "#fff5f5" if has_change else "#f0fff0"

            html += f"""
            <tr style='background: {row_bg}; border-bottom: 1px solid #ecf0f1;'>
                <td style='padding: 12px; font-weight: bold;'>{img_name}</td>
                <td style='padding: 12px; max-width: 300px; word-wrap: break-word;'>{original}</td>
                <td style='padding: 12px; max-width: 300px; word-wrap: break-word;'>{new_label}</td>
                <td style='padding: 12px; text-align: center;'>
                    <span style='color: {status_color}; font-weight: bold;'>{status_text}</span>
                </td>
            </tr>
            """

        html += f"""
                    </tbody>
                </table>
            </div>
            <div style='background: #e8f5e8; padding: 10px; border-radius: 5px; margin-top: 15px;'>
                <p style='margin: 0; color: #27ae60;'>📊 共修改 {changed_count} 个标签</p>
            </div>
        </div>
        """

        return html

    def has_rules(self) -> bool:
        """是否有规则"""
        return len(self.rules) > 0

    def has_changes(self) -> bool:
        """是否有待保存的更改"""
        return len(self.new_labels) > 0
