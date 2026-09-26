"""01 在同一个 Conda 环境内并行启动三个独立的学习率实验。"""

import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
RUN_ROOT = REPO_ROOT / "runs" / "resnet18"
LOG_DIR = RUN_ROOT / "parallel_lr_logs"
TRAIN_SCRIPT = PROJECT_DIR / "src" / "train.py"

# 每组只改变学习率；训练入口仍负责按自己的 experiment_id 保存指标与 checkpoint。
CONFIGS = (
    PROJECT_DIR / "configs" / "config09_lr_0.004_20ep.json",
    PROJECT_DIR / "configs" / "config10_lr_0.005_20ep.json",
    PROJECT_DIR / "configs" / "config11_lr_0.006_20ep.json",
)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:surrogateescape"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "matplotlib")

    running = []
    try:
        for config_path in CONFIGS:
            # 各进程使用本 Conda 环境的同一个 Python，不再并发调用 conda run。
            log_path = LOG_DIR / f"{config_path.stem}.log"
            if log_path.exists():
                raise FileExistsError(f"拒绝覆盖旧运行日志：{log_path}")
            log_file = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-u", str(TRAIN_SCRIPT), "--config", str(config_path)],
                cwd=PROJECT_DIR,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            running.append((config_path.name, process, log_file, log_path))
            print(f"started pid={process.pid} config={config_path.name} log={log_path}", flush=True)
            time.sleep(2)  # 错开预训练权重读取，训练阶段仍会并行。

        # 逐个报告完成状态；某组失败时保留其余组的独立实验结果。
        pending = list(running)
        failures = []
        while pending:
            for item in pending[:]:
                config_name, process, log_file, log_path = item
                code = process.poll()
                if code is None:
                    continue
                log_file.close()
                pending.remove(item)
                print(f"finished config={config_name} exit={code} log={log_path}", flush=True)
                if code != 0:
                    failures.append(config_name)
            if pending:
                time.sleep(5)

        if failures:
            raise SystemExit(f"failed experiments: {', '.join(failures)}")
    except KeyboardInterrupt:
        # 手动中断监督进程时，同时停止它启动的训练进程。
        for _, process, _, _ in running:
            if process.poll() is None:
                process.terminate()
        for _, process, _, _ in running:
            process.wait()
        raise
    finally:
        for _, _, log_file, _ in running:
            if not log_file.closed:
                log_file.close()


if __name__ == "__main__":
    main()
