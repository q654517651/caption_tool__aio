# trainers/musubi_qwen_image.py
import os, sys, time, math, shlex, subprocess, signal, platform, uuid, re
from typing import Callable, Optional
from ....core.common.events import EventBus, Job
from .types import TrainingConfig


class MusubiQwenImageTrainer:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self._proc: Optional[subprocess.Popen] = None
        self._id = uuid.uuid4().hex

    # —— 你确定具体命令后，把这里改成真实的 Musubi CLI —— #
    def _compose_cmd(self, cfg: TrainingConfig, total_steps: int) -> list[str]:
        # 示例：python -u tools/train_qwen_image_lora.py --dataset {..} ...
        # 下面暂时用一个不存在的文件名做检测，当你替换成真实脚本即可变为“真训练”
        script = os.environ.get("MUSUBI_QWEN_TRAIN", "")
        if not script:
            return []  # 返回空表示 dry-run
        args = [
            sys.executable, "-u", script,
            "--dataset-id", cfg.dataset_id,
            "--base-model", cfg.base_model,
            "--resolution", cfg.resolution,
            "--repeats", str(cfg.repeats),
            "--epochs", str(cfg.epochs),
            "--batch-size", str(cfg.batch_size),
            "--grad-accum", str(cfg.grad_accum),
            "--optimizer", cfg.optimizer,
            "--lr", str(cfg.lr),
            "--scheduler", cfg.scheduler,
            "--precision", cfg.precision,
            "--gpu", str(cfg.gpu_index),
        ]
        if cfg.sample_prompt:
            args += ["--sample-prompt", cfg.sample_prompt]
        if cfg.sample_every_n_steps:
            args += ["--sample-every-n-steps", str(cfg.sample_every_n_steps)]
        return args

    def build_job(self, cfg: TrainingConfig, total_steps: int) -> Job:
        # 仅返回一个占位 Job 对象；真正执行在 run()
        return Job(
            id=self._id,
            name=cfg.name,
            run=lambda log_cb, progress_cb: self.run(log_cb, progress_cb, cfg, total_steps),
            cancel=self.cancel
        )

    def run(self, log_cb: Callable[[str], None], progress_cb: Callable[[dict], None],
            cfg: TrainingConfig, total_steps: int) -> int:
        cmd = self._compose_cmd(cfg, total_steps)
        if not cmd:
            # —— Dry-run：没有配置 MUSUBI_QWEN_TRAIN 时，模拟训练 —— #
            log_cb("[DRYRUN] Musubi Qwen Image training simulation started")
            start = time.time()
            total_epochs = max(1, cfg.epochs)
            steps_per_epoch = math.ceil(
                (cfg.dataset_size * max(1, cfg.repeats)) /
                max(1, cfg.batch_size * max(1, cfg.grad_accum))
            )
            for step in range(1, total_steps + 1):
                epoch = 1 + (step - 1) // steps_per_epoch
                # 模拟日志
                if step % 10 == 0:
                    elapsed = time.time() - start
                    sps = step / max(1e-3, elapsed)
                    eta = int((total_steps - step) / max(sps, 1e-6))
                    log_cb(f"[SIM] step {step}/{total_steps} epoch {epoch}/{total_epochs} loss=0.{step%100:02d} lr={cfg.lr} ETA {eta}s")
                    progress_cb({"step": step, "total_steps": total_steps,
                                 "epoch": epoch, "total_epochs": total_epochs,
                                 "ips": round(sps * cfg.batch_size, 2),
                                 "eta_secs": eta})
                time.sleep(0.05)  # 模拟计算
            log_cb("[DRYRUN] Training finished.")
            return 0

        # —— 真训练：启动子进程 —— #
        env = os.environ.copy()
        # 单卡选择
        env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_index)

        creationflags = 0
        preexec_fn = None
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # 便于 Ctrl-Break
        else:
            preexec_fn = os.setsid  # 独立进程组

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=preexec_fn,
            creationflags=creationflags,
        )

        # 正则解析 Musubi 日志中的关键信息（示例：自行按真实日志调整）
        re_step = re.compile(r"step\s+(\d+)\s*/\s*(\d+)")
        re_epoch = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)")
        re_eta = re.compile(r"ETA\s+(\d+)s")
        re_ips = re.compile(r"ips\s*=\s*([\d\.]+)")

        cur_step = 0
        cur_epoch = 1
        total_epochs = max(1, cfg.epochs)

        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            line = raw.rstrip()
            log_cb(line)

            m = re_step.search(line)
            if m:
                cur_step = int(m.group(1))
                # total_steps 以我们计算为准，日志里的总步数不一定可信
            m2 = re_epoch.search(line)
            if m2:
                cur_epoch = int(m2.group(1))
            eta = None
            m3 = re_eta.search(line)
            if m3:
                eta = int(m3.group(1))
            ips = None
            m4 = re_ips.search(line)
            if m4:
                try:
                    ips = float(m4.group(1))
                except:
                    pass

            # 只要抓到 step，就推一次结构化进度
            if cur_step:
                progress_cb({
                    "step": cur_step,
                    "total_steps": total_steps,
                    "epoch": cur_epoch,
                    "total_epochs": total_epochs,
                    "ips": ips,
                    "eta_secs": eta,  # 让 manager 决定是否用回退算法
                })

        return self._proc.wait()

    def cancel(self) -> None:
        p = self._proc
        if not p:
            return
        try:
            if platform.system() == "Windows":
                # 发送 Ctrl-Break
                os.kill(p.pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                time.sleep(1)
                if p.poll() is None:
                    p.terminate()
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass