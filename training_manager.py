import os
import subprocess
import time
import socket
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable, Optional
import psutil


@dataclass
class TrainingConfig:
    dataset_path: str
    dataset_type: str  # "video" or "image"
    resolution: str
    batch_size: str
    num_repeats: str
    max_epochs: str
    learning_rate: str
    lora_name: str
    sample_prompts: str


class TrainingManager:
    def __init__(self, settings_manager):
        self.settings = settings_manager
        self.running_processes = []
        self.tensorboard_process = None

    def create_toml_config(self, config: TrainingConfig) -> str:
        """创建TOML配置文件"""
        dataset_path = Path(config.dataset_path)
        cache_dir = dataset_path / "cache"
        datasets_dir = dataset_path / "datasets"

        # 创建必要目录
        cache_dir.mkdir(exist_ok=True)
        datasets_dir.mkdir(exist_ok=True)

        # 解析分辨率
        try:
            w, h = [x.strip() for x in config.resolution.split(",")]
        except ValueError:
            w, h = "960", "544"

        if config.dataset_type == "image":
            toml_content = f"""# Auto-generated config
[general]
resolution = [{w}, {h}]
caption_extension = ".txt"
batch_size = {config.batch_size}
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{datasets_dir.as_posix()}"
cache_directory = "{cache_dir.as_posix()}"
num_repeats = {config.num_repeats}
"""
        else:  # video
            toml_content = f"""# Auto-generated config
[general]
resolution = [{w}, {h}]
caption_extension = ".txt"
batch_size = {config.batch_size}
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "{datasets_dir.as_posix()}"
cache_directory = "{cache_dir.as_posix()}"
frame_extraction = "full"
max_frames = 50
num_repeats = {config.num_repeats}
"""

        toml_path = dataset_path / "dataset.toml"
        toml_path.write_text(toml_content, encoding='utf-8')
        return str(toml_path)

    def create_sample_file(self, config: TrainingConfig) -> str:
        """创建采样文件"""
        dataset_path = Path(config.dataset_path)
        output_dir = dataset_path / "outputs"
        output_dir.mkdir(exist_ok=True)

        sample_path = output_dir / "sample.txt"
        content = config.sample_prompts.strip() or "a beautiful landscape"
        sample_path.write_text(content, encoding='utf-8')
        return str(sample_path)

    def generate_training_script(self, config: TrainingConfig) -> str:
        """生成训练脚本"""
        dataset_path = Path(config.dataset_path)
        output_dir = dataset_path / "outputs"
        output_dir.mkdir(exist_ok=True)

        toml_path = self.create_toml_config(config)
        sample_path = self.create_sample_file(config)

        musubi_dir = self.settings.get("musubi_dir")
        if not musubi_dir:
            raise ValueError("请先在设置中配置Musubi目录")

        musubi_path = Path(musubi_dir)
        venv_python = musubi_path / "venv" / "Scripts" / "python.exe"
        accelerate_exe = musubi_path / "venv" / "Scripts" / "accelerate.exe"

        # 构建命令
        commands = []

        # 缓存潜在向量
        cache_cmd = f'"{venv_python}" wan_cache_latents.py --dataset_config "{toml_path}" --vae "{self.settings.get("vae_path")}" --clip "{self.settings.get("clip_path")}"'
        commands.append(cache_cmd)

        # 缓存文本编码器输出
        text_cmd = f'"{venv_python}" wan_cache_text_encoder_outputs.py --dataset_config "{toml_path}" --t5 "{self.settings.get("t5_path")}" --batch_size 16'
        commands.append(text_cmd)

        # 训练命令
        log_dir = output_dir / "logs"
        train_cmd = f'''"{accelerate_exe}" launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py --task i2v-14B --dit "{self.settings.get("unet_path")}" --dataset_config "{toml_path}" --sdpa --mixed_precision bf16 --fp8_base --optimizer_type adamw8bit --learning_rate {config.learning_rate} --gradient_checkpointing --lr_scheduler constant --max_data_loader_n_workers 2 --persistent_data_loader_workers --network_module networks.lora_wan --network_dim 32 --network_alpha 1 --timestep_sampling shift --discrete_flow_shift 3.0 --max_train_epochs {config.max_epochs} --save_every_n_epochs 1 --seed 42 --output_dir "{output_dir}" --output_name "{config.lora_name}" --blocks_to_swap 34 --logging_dir="{log_dir}" --vae "{self.settings.get("vae_path")}" --t5 "{self.settings.get("t5_path")}" --sample_prompts "{sample_path}" --sample_every_n_steps 500'''
        commands.append(train_cmd)

        # 创建批处理文件
        if os.name == 'nt':  # Windows
            bat_content = '\r\n\r\n'.join(commands) + '\r\n'
            script_path = dataset_path / f"{config.lora_name}_train.bat"
            script_path.write_text(bat_content, encoding='utf-8')
        else:  # Linux/Mac
            sh_content = '\n\n'.join(commands) + '\n'
            script_path = dataset_path / f"{config.lora_name}_train.sh"
            script_path.write_text(sh_content, encoding='utf-8')
            os.chmod(script_path, 0o755)

        return str(script_path)

    def start_tensorboard(self, log_dir: str) -> int:
        """启动TensorBoard"""
        port = self._find_free_port()
        musubi_dir = self.settings.get("musubi_dir")
        if not musubi_dir:
            return 0

        musubi_path = Path(musubi_dir)
        tb_exe = musubi_path / "venv" / "Scripts" / "tensorboard.exe"

        if not tb_exe.exists():
            return 0

        try:
            self.tensorboard_process = subprocess.Popen([
                str(tb_exe), "serve", "--logdir", log_dir,
                "--port", str(port), "--bind_all"
            ])
            time.sleep(2)
            return port
        except Exception as e:
            print(f"启动TensorBoard失败: {e}")
            return 0

    def _find_free_port(self, start_port=6006) -> int:
        """查找可用端口"""
        with socket.socket() as s:
            try:
                s.bind(("", start_port))
                return start_port
            except OSError:
                s.bind(("", 0))
                return s.getsockname()[1]

    def run_training(self, config: TrainingConfig, progress_callback: Optional[Callable] = None):
        """运行训练"""
        try:
            script_path = self.generate_training_script(config)
            musubi_dir = self.settings.get("musubi_dir")

            # 启动TensorBoard
            log_dir = Path(config.dataset_path) / "outputs" / "logs"
            tb_port = self.start_tensorboard(str(log_dir))

            if progress_callback:
                progress_callback(f"TensorBoard启动在端口: {tb_port}")
                progress_callback(f"开始训练，脚本: {script_path}")

            # 运行训练脚本
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(
                    script_path,
                    cwd=musubi_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # Linux/Mac
                process = subprocess.Popen(
                    ["bash", script_path],
                    cwd=musubi_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True
                )

            self.running_processes.append(process)

            # 实时输出
            for line in process.stdout:
                if progress_callback:
                    progress_callback(line.rstrip())

            process.wait()
            if progress_callback:
                progress_callback(f"训练完成，退出码: {process.returncode}")

            return tb_port

        except Exception as e:
            if progress_callback:
                progress_callback(f"训练失败: {str(e)}")
            return 0

    def stop_training(self):
        """停止所有训练进程"""
        stopped = 0
        for process in self.running_processes:
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    if child.is_running():
                        child.kill()
                if parent.is_running():
                    parent.kill()
                stopped += 1
            except psutil.NoSuchProcess:
                pass

        self.running_processes.clear()

        # 停止TensorBoard
        if self.tensorboard_process:
            try:
                self.tensorboard_process.kill()
            except:
                pass
            self.tensorboard_process = None

        return f"已停止 {stopped} 个训练进程"