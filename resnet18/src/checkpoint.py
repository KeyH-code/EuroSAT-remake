"""04 Checkpoint：保存完整训练状态，并在评价或续训时核对数据身份。"""

import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from .data import CLASS_NAMES, MANIFEST


def current_learning_rate(optimizer):
    """取优化器当前学习率；可学习率调度器会直接改写 param_groups。"""
    return optimizer.param_groups[0]["lr"]


def _rng_state(train_loader=None):
    """记录会影响下一轮采样和参数更新的随机状态。"""
    numpy_state = np.random.get_state()
    generator = getattr(train_loader, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gauss": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": generator.get_state() if generator is not None else None,
    }


def save_checkpoint(
    path, *, model, optimizer, scaler, epoch, best_epoch, best_val_accuracy, config,
    train_loader=None, lr_scheduler=None,
):
    """把模型、优化器、AMP、可学习率调度器和恢复训练所需的进度写入一个 checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_name": type(model).__name__,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "lr_scheduler_state": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_accuracy),
        "config": config,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "class_names": list(CLASS_NAMES),
        "rng_state": _rng_state(train_loader),
    }

    # 临时文件写完后才替换目标，避免中断时把旧 last/best 覆盖成半个文件。
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    return path


def load_checkpoint(
    path, *, model, optimizer=None, scaler=None, map_location="cpu",
    train_loader=None, restore_rng=False, lr_scheduler=None,
):
    """评价只恢复模型；续训再恢复优化器、scaler、学习率调度器与随机状态。"""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint["schema_version"] != 1:
        raise ValueError("未知 checkpoint 格式")
    if checkpoint["model_name"] != type(model).__name__:
        raise ValueError("checkpoint 模型类型与当前模型不一致")
    if checkpoint["manifest_sha256"] != hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
        raise ValueError("checkpoint 使用的 EuroSAT split 与当前清单不一致")
    if checkpoint["class_names"] != list(CLASS_NAMES):
        raise ValueError("checkpoint 类别顺序与当前数据不一致")

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None:
        if checkpoint["scaler_state"] is None:
            raise ValueError("checkpoint 没有 AMP scaler 状态")
        scaler.load_state_dict(checkpoint["scaler_state"])
    if lr_scheduler is not None:
        # 旧 checkpoint 没有调度器状态；此时按当前配置从头开始退火，避免静默用一个错误进度。
        scheduler_state = checkpoint.get("lr_scheduler_state")
        if scheduler_state is None:
            raise ValueError("checkpoint 没有学习率调度器状态，无法按原进度恢复")
        lr_scheduler.load_state_dict(scheduler_state)

    if restore_rng:
        rng = checkpoint["rng_state"]
        random.setstate(rng["python"])
        numpy_state = rng["numpy"]
        np.random.set_state((
            numpy_state["name"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gauss"],
        ))
        torch.set_rng_state(rng["torch_cpu"])
        if rng["torch_cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        generator = getattr(train_loader, "generator", None)
        if generator is not None and rng["loader_generator"] is not None:
            generator.set_state(rng["loader_generator"])
    return checkpoint
