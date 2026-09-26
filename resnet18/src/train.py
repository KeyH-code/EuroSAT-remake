"""配置驱动的 ResNet-18 微调主流程：CLI、续训校验、逐轮记录与 checkpoint。

训练循环在 ``loop.py``，评价与图表在 ``metrics.py`` / ``visualization.py``；
本模块只负责把这些编排成一次实验。
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from data import make_train_val_loaders, DEVICE, CONFIG_PATH, REPO_ROOT
from transforms import make_transforms
from torch import nn
from optimizer import make_optimizer

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))  # 让正式入口能导入 Agent 支持模块。
from src.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from src.loop import train_one_epoch  # noqa: E402
from src.metrics import append_epoch_record, evaluate_classifier  # noqa: E402
from src.visualization import save_experiment_visualizations  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "runs" / "resnet18"


def main():
    parser = argparse.ArgumentParser(description="EuroSAT ResNet-18 微调训练入口")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH / "config01.json")
    parser.add_argument("--resume", action="store_true", help="从该实验的 last.pt 继续")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_id = config.get("experiment_id", config_path.stem)
    run_dir = (OUTPUT_ROOT / experiment_id).resolve()
    if not run_dir.is_relative_to(OUTPUT_ROOT.resolve()) or run_dir == OUTPUT_ROOT.resolve():
        raise ValueError("experiment_id 必须是独立微调缓存目录下的子目录名")

    # 同一个 seed 固定初始化与 DataLoader 采样起点；续训时改由 checkpoint 恢复状态。
    seed = config["loader"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 当前学习者模型保持直接定义；延迟导入，缺配置时不会请求预训练权重。
    from model import model, TRAINABLE_SCOPE

    model.to(DEVICE)
    train_transform,val_transform = make_transforms(config)
    train_loader,val_loader = make_train_val_loaders(
        train_transform,
        val_transform,
        config["loader"]["batch_size"],
        config["loader"]["seed"]
    )
    optimizer = make_optimizer(model, config)
    # 默认 65536 对解冻后的 layer4 过大，首个 step 会溢出跳步；1024 实测全程 0 跳步。
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    criterion = nn.CrossEntropyLoss()

    config_snapshot = run_dir / "config.json"
    if args.resume:
        if not config_snapshot.is_file():
            raise FileNotFoundError("续训目录缺少 config.json")
        previous_config = json.loads(config_snapshot.read_text(encoding="utf-8"))
        # 可训练范围属于数据/模型契约，必须在恢复优化器状态之前先核对：从「只训分类头」
        # 的实验续训到「解冻 layer4」的实现，会凭空训练一批从未被优化过的权重。
        checkpoint = torch.load(
            run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False
        )
        recorded_scope = checkpoint.get("config", {}).get("trainable_scope")
        if recorded_scope is None:
            raise ValueError(
                "checkpoint 未记录可训练范围（trainable_scope），无法确认与当前实现一致，拒绝续训"
            )
        if recorded_scope != TRAINABLE_SCOPE:
            raise ValueError(
                f"可训练范围不一致：checkpoint={recorded_scope!r} 当前={TRAINABLE_SCOPE!r}；"
                "跨范围续训会训练一批从未被优化过的权重，请新建实验目录"
            )
        # 续训只允许增加目标总轮数；数据、模型输入和优化器参数必须保持一致。
        previous_settings = {key: value for key, value in previous_config.items() if key != "epoch"}
        current_settings = {key: value for key, value in config.items() if key != "epoch"}
        if previous_settings != current_settings or config["epoch"] < previous_config["epoch"]:
            raise ValueError("续训只能增加 epoch；其余配置必须与原实验一致")
        checkpoint = load_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model=model, optimizer=optimizer, scaler=scaler, map_location="cpu",
            train_loader=train_loader, restore_rng=True,
        )
        start_epoch = checkpoint["epoch"] + 1
        best_epoch = checkpoint["best_epoch"]
        best_val_accuracy = checkpoint["best_val_accuracy"]
        if config["epoch"] > previous_config["epoch"]:
            # 模型和随机状态成功恢复后，再把当前总轮数写入运行目录。
            config_snapshot.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)  # 拒绝覆盖旧实验。
        config_snapshot.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        start_epoch, best_epoch, best_val_accuracy = 1, 0, -1.0

    for epoch in range(start_epoch, config["epoch"] + 1):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        train_data = train_one_epoch(model, train_loader, optimizer, scaler)
        # 训练函数由学习者提供这两个按样本累计的数值；Agent 不替写更新步骤。
        train_loss = float(train_data["train_loss_ave"])
        train_accuracy = float(train_data["train_accuracy"])
        val_result = evaluate_classifier(model, val_loader, criterion, DEVICE)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        is_best = val_result["accuracy"] > best_val_accuracy
        if is_best:
            best_epoch, best_val_accuracy = epoch, val_result["accuracy"]

        append_epoch_record(
            run_dir, epoch, train_loss, train_accuracy, val_result,
            extras={
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": seconds,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
                "is_best": is_best,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
            },
        )
        checkpoint_dir = run_dir / "checkpoints"
        save_checkpoint(
            checkpoint_dir / "last.pt", model=model, optimizer=optimizer,
            scaler=scaler, epoch=epoch, best_epoch=best_epoch,
            best_val_accuracy=best_val_accuracy, config=config, train_loader=train_loader,
        )
        if is_best:
            save_checkpoint(
                checkpoint_dir / "best.pt", model=model, optimizer=optimizer,
                scaler=scaler, epoch=epoch, best_epoch=best_epoch,
                best_val_accuracy=best_val_accuracy, config=config, train_loader=train_loader,
            )
        save_experiment_visualizations(run_dir)
        print(f"epoch={epoch} train_acc={train_accuracy:.4f} val_acc={val_result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
