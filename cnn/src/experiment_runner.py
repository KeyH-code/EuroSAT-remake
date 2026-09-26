"""配置驱动的 EuroSAT 正式实验 runner。

训练与评价算法直接复用学习者已经完成的 train_fit_loop；本文件只负责
配置校验、连续指标、checkpoint、最终评价和实验目录边界。
"""

import hashlib
import json
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiment_visualization import save_experiment_visualizations
from train_fit_loop import (
    CLASS_NAMES,
    IDX_TO_CLASS,
    MANIFEST,
    NUM_CLASSES,
    EurosatCNN,
    evaluate,
    fix_seed,
    make_loader,
    train_one_epoch,
)
from transforms import TRAIN_MEAN, TRAIN_STD, build_eval_transform, build_train_transform
from eurosat_paths import PROJECT_ROOT


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _append_jsonl_rows(path, values):
    """一次追加一个 epoch 的逐 step 记录，避免频繁开关文件。"""
    with Path(path).open("a", encoding="utf-8", newline="\n") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _per_class_rows(cm):
    """沿用既有混淆矩阵口径，返回 10 类可序列化指标。"""
    cm_np = cm.cpu().numpy()
    true_count = cm_np.sum(axis=1)
    pred_count = cm_np.sum(axis=0)
    correct = np.diag(cm_np)
    recall = np.divide(
        correct,
        true_count,
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=true_count > 0,
    )
    precision = np.divide(
        correct,
        pred_count,
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=pred_count > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=(precision + recall) > 0,
    )
    return [
        {
            "class_idx": index,
            "class_name": IDX_TO_CLASS[index],
            "true_count": int(true_count[index]),
            "pred_count": int(pred_count[index]),
            "correct": int(correct[index]),
            "recall": float(recall[index]),
            "precision": float(precision[index]),
            "f1": float(f1[index]),
        }
        for index in range(NUM_CLASSES)
    ]


def _validate_config(config_path, config, resume):
    """训练前拒绝口径漂移和意外覆盖。"""
    experiment_dir = config_path.parent
    training = config["training"]
    data = config["data"]

    configured_manifest = Path(data["manifest_path"])
    if not configured_manifest.is_absolute():
        configured_manifest = PROJECT_ROOT / configured_manifest
    # 只接受本仓库清单；其他位置的清单一律拒绝。
    if configured_manifest.resolve() != MANIFEST.resolve():
        raise ValueError("config manifest_path 不是当前正式 my_split.csv")
    actual_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if data["manifest_sha256"] != actual_hash:
        raise ValueError("config manifest_sha256 与当前文件不一致")
    frame = pd.read_csv(MANIFEST)
    actual_counts = frame["split"].value_counts().to_dict()
    expected_counts = {
        "train": data["train_samples"],
        "val": data["val_samples"],
        "test": data["test_samples"],
    }
    if actual_counts != expected_counts:
        raise ValueError(f"split 数量漂移：{actual_counts}")
    if config["model"] != {"name": "EurosatCNN", "num_classes": NUM_CLASSES}:
        raise ValueError("正式基线模型配置与 EurosatCNN 契约不一致")
    if training["precision"] != "fp32":
        raise ValueError("本实验只允许真正的 FP32")
    if training["augmentation"] != "none":
        raise ValueError("本实验不允许训练增强")
    # 优化器的专属参数留在配置中；旧 AdamW 配置仍使用 PyTorch 默认 betas。
    optimizer_name = training["optimizer"]
    if optimizer_name == "AdamW":
        if "momentum" in training:
            raise ValueError("AdamW 不接受 SGD 的 momentum 参数")
        if "betas" in training:
            beta1, beta2 = training["betas"]
            if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
                raise ValueError("AdamW betas 必须在 [0, 1) 内")
    elif optimizer_name == "SGD":
        if "betas" in training:
            raise ValueError("SGD 不接受 AdamW 的 betas 参数")
        if not 0 <= training.get("momentum", -1) < 1:
            raise ValueError("SGD momentum 必须在 [0, 1) 内")
    else:
        raise ValueError(f"不支持的 optimizer：{optimizer_name}")
    if training["weight_decay"] < 0:
        raise ValueError("weight_decay 不能为负")
    scheduler_config = training["scheduler"]
    if scheduler_config is not None:
        scheduler_name = scheduler_config.get("name")
        if scheduler_name == "linear_warmup_then_constant":
            expected_keys = {
                "name",
                "start_learning_rate",
                "target_learning_rate",
                "warmup_epochs",
                "step_unit",
            }
        elif scheduler_name == "linear_warmup_then_step_up":
            expected_keys = {
                "name",
                "start_learning_rate",
                "warmup_target_learning_rate",
                "target_learning_rate",
                "warmup_epochs",
                "step_up_epoch",
                "step_unit",
            }
        else:
            raise ValueError(f"不支持的 scheduler：{scheduler_name}")
        if set(scheduler_config) != expected_keys:
            raise ValueError("warmup scheduler 配置字段不完整或包含未知字段")
        if scheduler_config["step_unit"] != "optimizer_step":
            raise ValueError("warmup 必须按 optimizer step 推进")
        if scheduler_config["warmup_epochs"] < 1:
            raise ValueError("warmup_epochs 必须至少为 1")
        if scheduler_config["target_learning_rate"] != training["learning_rate"]:
            raise ValueError("target_learning_rate 必须等于 optimizer 基准学习率")
        if not 0 < scheduler_config["start_learning_rate"] <= training["learning_rate"]:
            raise ValueError("warmup 起始学习率必须在 (0, target] 内")
        if scheduler_name == "linear_warmup_then_step_up":
            warmup_target = scheduler_config["warmup_target_learning_rate"]
            if not scheduler_config["start_learning_rate"] <= warmup_target < training["learning_rate"]:
                raise ValueError("阶跃策略要求 start <= warmup_target < target")
            step_up_epoch = scheduler_config["step_up_epoch"]
            if not scheduler_config["warmup_epochs"] < step_up_epoch <= training["epochs"]:
                raise ValueError("step_up_epoch 必须位于 warmup 之后且不超过总 epoch")
    if not torch.cuda.is_available():
        raise RuntimeError("正式训练要求 CUDA")

    output_files = [
        experiment_dir / config["outputs"]["metrics"],
        experiment_dir / config["outputs"]["per_class_metrics"],
        experiment_dir / config["outputs"]["checkpoints"] / "last.pt",
        experiment_dir / config["outputs"]["checkpoints"] / "best.pt",
    ]
    if not resume and any(path.exists() for path in output_files):
        existing = [str(path) for path in output_files if path.exists()]
        raise FileExistsError(f"新实验拒绝覆盖已有训练产物：{existing}")
    return frame, actual_hash


def _validate_resume_config(config_path, config, checkpoint):
    """恢复前核对实验口径；只允许显式延长总轮数及带来源的目录复制。

    续训只在本仓库产物之间进行：checkpoint 必须自带 config 与 config_sha256。
    """
    for field in ("config", "config_sha256", "config_path"):
        if field not in checkpoint:
            raise ValueError(f"checkpoint 缺少 {field}，不是本仓库训练的产物，不支持续训")
    saved_hash = checkpoint["config_sha256"]
    current_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if current_hash == saved_hash:
        original = config
    else:
        # 新检查点自带当时的配置，原配置文件后来被修改也可审计差异。
        original = checkpoint["config"]

    if config["training"]["epochs"] <= checkpoint["epoch"]:
        raise ValueError("目标 epochs 必须大于 checkpoint 已完成的 epoch")

    original_core = deepcopy(original)
    current_core = deepcopy(config)
    original_epochs = original_core["training"].pop("epochs")
    current_epochs = current_core["training"].pop("epochs")
    if current_epochs < original_epochs:
        raise ValueError("恢复训练只能保持或延长总轮数，不能缩短")

    if original_core != current_core:
        # 独立续训实验可改 ID，但必须准确记录来源并仅改总轮数。
        lineage = current_core.pop("lineage", None)
        source_config_path = Path(checkpoint["config_path"])
        is_recorded_copy = (
            isinstance(lineage, dict)
            and lineage.get("source_config_sha256") == saved_hash
            and lineage.get("source_epoch") == checkpoint["epoch"]
            and Path(lineage.get("source_directory", "")).resolve() == source_config_path.parent.resolve()
        )
        if is_recorded_copy:
            current_core["experiment_id"] = original_core["experiment_id"]
        if not is_recorded_copy or original_core != current_core:
            raise ValueError("恢复配置与 checkpoint 不一致：除有来源的目录复制和 epochs 延长外不允许修改")

def _build_lr_scheduler(optimizer, scheduler_config, steps_per_epoch):
    """按实验配置创建逐 optimizer-step scheduler；None 表示固定学习率。"""
    if scheduler_config is None:
        return None

    warmup_steps = scheduler_config["warmup_epochs"] * steps_per_epoch
    target_lr = scheduler_config["target_learning_rate"]
    start_factor = scheduler_config["start_learning_rate"] / target_lr
    if scheduler_config["name"] == "linear_warmup_then_constant":
        # LinearLR 到达 total_iters 后保持 end_factor=1，不再改变学习率。
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

    warmup_target_factor = (
        scheduler_config["warmup_target_learning_rate"] / target_lr
    )
    step_up_after_steps = (
        scheduler_config["step_up_epoch"] - 1
    ) * steps_per_epoch

    def learning_rate_factor(completed_steps):
        """先线性升到中间值；完成指定 epoch 后为下一轮设置最终值。"""
        if completed_steps <= warmup_steps:
            progress = completed_steps / warmup_steps
            return start_factor + (
                warmup_target_factor - start_factor
            ) * progress
        if completed_steps < step_up_after_steps:
            return warmup_target_factor
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_factor,
    )


def run_experiment(config_path, resume=False):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    experiment_dir = config_path.parent
    frame, manifest_sha256 = _validate_config(config_path, config, resume)
    training = config["training"]
    outputs = config["outputs"]

    checkpoints_dir = experiment_dir / outputs["checkpoints"]
    confusion_dir = experiment_dir / outputs["confusion_matrices"]
    final_dir = experiment_dir / outputs["final_evaluation"]
    for directory in (checkpoints_dir, confusion_dir, final_dir):
        directory.mkdir(parents=True, exist_ok=True)
    last_path = checkpoints_dir / "last.pt"
    best_path = checkpoints_dir / "best.pt"
    metrics_path = experiment_dir / outputs["metrics"]
    per_class_path = experiment_dir / outputs["per_class_metrics"]
    sessions_path = experiment_dir / outputs["sessions"]

    fix_seed(training["seed"])
    device = torch.device("cuda")
    train_loader = make_loader(
        "train",
        build_train_transform(),
        training["batch_size"],
        shuffle=True,
        seed=training["seed"],
    )
    val_loader = make_loader(
        "val",
        build_eval_transform(),
        training["batch_size"],
        shuffle=False,
    )
    model = EurosatCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # 只在优化器对象的构造处分支；训练、评价和记录仍复用同一套主流程。
    if training["optimizer"] == "AdamW":
        adamw_options = {}
        if "betas" in training:
            adamw_options["betas"] = tuple(training["betas"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
            **adamw_options,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training["learning_rate"],
            momentum=training["momentum"],
            weight_decay=training["weight_decay"],
        )
    scheduler_config = training["scheduler"]
    lr_scheduler = _build_lr_scheduler(
        optimizer,
        scheduler_config,
        len(train_loader),
    )
    scaler = None  # 本实验是纯 FP32；train_one_epoch 因此不会进入 autocast。
    start_epoch = 1
    best_epoch = 0
    best_val_acc = float("-inf")

    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        # 先核对实验口径，再把旧状态装入当前模型、优化器和调度器。
        _validate_resume_config(config_path, config, checkpoint)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_epoch = checkpoint["best_epoch"]
        best_val_acc = checkpoint["best_val_acc"]
        random.setstate(checkpoint["rng_state"]["pythonRandom"])
        np.random.set_state(checkpoint["rng_state"]["numpyRandom"])
        torch.set_rng_state(checkpoint["rng_state"]["torchRandom"])
        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cudaRandom"])
        train_loader.generator.set_state(checkpoint["rng_state"]["loaderRandom"])

    session_id = datetime.now().astimezone().strftime("session_%Y%m%dT%H%M%S_%f%z")
    session_started = datetime.now().astimezone()
    _append_jsonl(
        sessions_path,
        {
            "session_id": session_id,
            "event": "started",
            "timestamp": session_started.isoformat(),
            "resume": resume,
            "start_epoch": start_epoch,
            "target_epoch": training["epochs"],
            "entrypoint": str(Path(sys.argv[0]).resolve()),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
        },
    )

    print("experiment:", config["experiment_id"])
    print("device:", torch.cuda.get_device_name(0))
    print("precision: FP32 (autocast disabled, scaler=None)")
    print("scheduler:", scheduler_config)
    print("split counts:", frame["split"].value_counts().to_dict())
    print(f"epochs: {start_epoch}..{training['epochs']}")

    experiment_start = time.perf_counter()
    learning_rate_trace = []
    learning_rate_path = (
        experiment_dir / outputs["learning_rate_history"]
        if "learning_rate_history" in outputs
        else None
    )
    for epoch in range(start_epoch, training["epochs"] + 1):
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        train_loss, train_acc, skipped = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            lr_scheduler=lr_scheduler,
            learning_rate_trace=learning_rate_trace,
        )
        epoch_learning_rates = learning_rate_trace[-len(train_loader):]
        if learning_rate_path is not None:
            first_global_step = (epoch - 1) * len(train_loader) + 1
            _append_jsonl_rows(
                learning_rate_path,
                [
                    {
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "global_step": first_global_step + step - 1,
                        "learning_rate": value,
                    }
                    for step, value in enumerate(epoch_learning_rates, start=1)
                ],
            )
        val_loss, val_acc, cm, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        seconds = time.perf_counter() - epoch_start
        peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 ** 2
        class_rows = _per_class_rows(cm)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch

        rng_state = {
            "pythonRandom": random.getstate(),
            "numpyRandom": np.random.get_state(),
            "torchRandom": torch.get_rng_state(),
            "cudaRandom": torch.cuda.get_rng_state_all(),
            "loaderRandom": train_loader.generator.get_state(),
        }
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
            "scaler_state": None,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "rng_state": rng_state,
            "experiment_id": config["experiment_id"],
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "config": config,
            "manifest_path": str(MANIFEST),
            "manifest_sha256": manifest_sha256,
            "model_name": "EurosatCNN",
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "normalization_mean": TRAIN_MEAN,
            "normalization_std": TRAIN_STD,
            "precision": "fp32",
            "seed": training["seed"],
        }
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, best_path)

        metric_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rate_start": epoch_learning_rates[0],
            "learning_rate_end": optimizer.param_groups[0]["lr"],
            "seconds": seconds,
            "skipped_steps": skipped,
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "is_best": is_best,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
        }
        _append_jsonl(metrics_path, metric_row)
        _append_jsonl(per_class_path, {"epoch": epoch, "classes": class_rows})
        (confusion_dir / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "rows": "true",
                    "columns": "predicted",
                    "class_names": CLASS_NAMES,
                    "matrix": cm.cpu().tolist(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"best={best_val_acc:.4f}@{best_epoch} {seconds:.1f}s "
            f"peak={peak_allocated_mb:.0f}/{peak_reserved_mb:.0f}MB"
        )

    # 最终只对 best checkpoint 做一次详细评价和错误样本导出。
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    val_loss, val_acc, cm, labels, predictions, confidences = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )
    class_rows = _per_class_rows(cm)
    per_class_frame = pd.DataFrame(class_rows)
    per_class_frame.to_csv(
        final_dir / "per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    error_rows = []
    for image, label, prediction, confidence in zip(
        val_loader.dataset.images,
        labels,
        predictions,
        confidences,
        strict=True,
    ):
        if prediction != label:
            error_rows.append(
                {
                    "filepath": image,
                    "true_idx": label,
                    "true_class": CLASS_NAMES[label],
                    "pred_idx": prediction,
                    "pred_class": CLASS_NAMES[prediction],
                    "confidence": confidence,
                }
            )
    error_rows.sort(key=lambda row: row["confidence"], reverse=True)
    pd.DataFrame(error_rows).to_csv(
        final_dir / "error_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    f1_values = per_class_frame["f1"].to_numpy()
    true_counts = per_class_frame["true_count"].to_numpy()
    total_seconds = time.perf_counter() - experiment_start
    summary = {
        "experiment_id": config["experiment_id"],
        "best_epoch": best_checkpoint["epoch"],
        "last_epoch": training["epochs"],
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "macro_f1": float(f1_values.mean()),
        "weighted_f1": float((f1_values * true_counts).sum() / true_counts.sum()),
        "error_count": len(error_rows),
        "total_training_and_final_eval_seconds": total_seconds,
        "test_split_used": False,
        "precision": "fp32",
        "scheduler": scheduler_config,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    (final_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    visualizations = save_experiment_visualizations(experiment_dir)

    _append_jsonl(
        sessions_path,
        {
            "session_id": session_id,
            "event": "completed",
            "timestamp": datetime.now().astimezone().isoformat(),
            "completed_epoch": training["epochs"],
            "best_epoch": best_checkpoint["epoch"],
            "best_val_accuracy": val_acc,
            "total_seconds": total_seconds,
            "visualizations": visualizations,
        },
    )
    print("completed:", summary)
