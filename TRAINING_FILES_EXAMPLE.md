# 训练文件生成示例

TagTracker 现在会为每个训练任务生成完整的训练文件，既可以通过界面执行，也可以手动运行。

## 📁 生成的文件结构

每个训练任务会在 `workspace/trainings/{task_id}/` 目录下生成以下文件：

```
workspace/trainings/a1b2c3d4-e5f6-7890-abcd-ef1234567890/
├── datasets.toml          # Musubi 数据集配置文件
├── train.bat             # Windows 训练脚本
├── train.sh              # Linux/Mac 训练脚本  
├── config.json           # 完整的训练参数记录
├── cache/                # 训练缓存目录
└── logs/                 # 训练日志目录
    └── training.log      # 训练输出日志
```

## 📄 文件内容示例

### datasets.toml
```toml
[general]
resolution = [1024, 1024]
batch_size = 2
enable_bucket = true
bucket_no_upscale = false
bucket_resolution_steps = 64
min_bucket_reso = 256
max_bucket_reso = 1536

[[datasets]]
image_directory = "E:/Program/workspace/datasets/f118d5d7/original"
cache_directory = "E:/Program/workspace/trainings/a1b2c3d4/cache"
caption_extension = ".txt"
num_repeats = 1
```

### train.bat (Windows)
```batch
@echo off
echo ===== TagTracker Musubi 训练脚本 =====
echo 任务名称: Qwen-Image-LoRA-Test
echo 开始时间: %date% %time%
echo =====================================

cd /d "E:\Program\programlearn\tagtragger"

accelerate launch "third_party/musubi-tuner/src/musubi_tuner/qwen_image_train_network.py" ^
--dataset_config "workspace/trainings/a1b2c3d4/datasets.toml" ^
--output_dir "workspace/models/Qwen-Image-LoRA-Test" ^
--output_name "Qwen-Image-LoRA-Test" ^
--network_module networks.lora ^
--network_dim 32 ^
--network_alpha 16 ^
--max_train_epochs 16 ^
--learning_rate 1e-4 ^
--optimizer_type adamw8bit ^
--lr_scheduler cosine ^
--mixed_precision bf16 ^
--save_every_n_epochs 1 ^
--seed 42 ^
2>&1 | tee "workspace/trainings/a1b2c3d4/logs/training.log"

echo =====================================
echo 训练完成时间: %date% %time%
echo =====================================
pause
```

### train.sh (Linux/Mac)
```bash
#!/bin/bash
echo "===== TagTracker Musubi 训练脚本 ====="
echo "任务名称: Qwen-Image-LoRA-Test"
echo "开始时间: $(date)"
echo "====================================="

cd "E:/Program/programlearn/tagtragger"

accelerate launch "third_party/musubi-tuner/src/musubi_tuner/qwen_image_train_network.py" \
--dataset_config "workspace/trainings/a1b2c3d4/datasets.toml" \
--output_dir "workspace/models/Qwen-Image-LoRA-Test" \
--output_name "Qwen-Image-LoRA-Test" \
--network_module networks.lora \
--network_dim 32 \
--network_alpha 16 \
--max_train_epochs 16 \
--learning_rate 1e-4 \
--optimizer_type adamw8bit \
--lr_scheduler cosine \
--mixed_precision bf16 \
--save_every_n_epochs 1 \
--seed 42 \
2>&1 | tee "workspace/trainings/a1b2c3d4/logs/training.log"

echo "====================================="
echo "训练完成时间: $(date)"
echo "====================================="
```

### config.json
```json
{
  "backend": "musubi.qwen_image",
  "name": "Qwen-Image-LoRA-Test",
  "dataset_id": "f118d5d7-8385-4a14-aebb-a2cccfb6aeae",
  "dataset_size": 20,
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "repeats": 1,
  "epochs": 16,
  "batch_size": 2,
  "grad_accum": 1,
  "resolution": "1024,1024",
  "base_model": "",
  "optimizer": "adamw8bit",
  "lr": 1e-4,
  "scheduler": "cosine",
  "warmup_ratio": 0.0,
  "weight_decay": 0.0,
  "precision": "bf16",
  "sample_prompt": "",
  "sample_every_n_steps": 200,
  "gpu_index": 0,
  "created_at": "2024-08-12 15:30:45",
  "platform": "Windows"
}
```

## 🎯 使用方式

### 1. 通过 TagTracker 界面
- 正常创建训练任务
- 文件会自动生成并保存
- 训练通过界面管理和监控

### 2. 手动执行训练脚本
```bash
# Windows
cd /d "E:\Program\programlearn\tagtragger"
workspace\trainings\a1b2c3d4\train.bat

# Linux/Mac
cd "/path/to/tagtragger"
bash workspace/trainings/a1b2c3d4/train.sh
```

### 3. 自定义修改
- 编辑 `datasets.toml` 调整数据集配置
- 编辑 `train.bat/train.sh` 修改训练参数
- 重新执行脚本

## 💡 优势特点

### ✅ 完整的参数记录
- 所有训练参数都保存在文件中
- 方便后续复现和调试
- 可以作为训练历史的备份

### ✅ 独立的缓存管理
- 每个训练任务有独立的缓存目录
- 避免不同训练之间的缓存冲突
- 便于清理和管理存储空间

### ✅ 平台兼容性
- 同时生成 Windows 和 Linux/Mac 脚本
- 自动检测平台选择合适的执行方式
- 确保跨平台的一致性

### ✅ 调试友好
- 生成的脚本人类可读
- 可以手动修改和重新执行
- 日志文件统一管理

### ✅ 数据库 + 文件双重记录
- 数据库记录训练状态和进度
- 文件系统保存完整的训练配置
- 两种方式互为备份

这种设计既满足了程序化的自动执行需求，又提供了用户友好的手动操作能力！