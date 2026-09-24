"""EuroSAT 固定验证集评价入口。

直接复用学习者在 ``train_fit_loop.py`` 中完成的 evaluate、混淆矩阵、
逐类指标和错误样本导出逻辑，只补 checkpoint 加载与结果落盘外壳。
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from train_fit_loop import (  # noqa: E402
    CLASS_NAMES,
    MANIFEST,
    NUM_CLASSES,
    EurosatCNN,
    evaluate,
    make_loader,
    save_per_class_metrics,
)
from transforms import (  # noqa: E402
    TRAIN_MEAN,
    TRAIN_STD,
    build_eval_transform,
)
sys.path.insert(0, str(PROJECT_ROOT.parent))
from eurosat_paths import LEGACY_ROOT, RUNS_ROOT  # noqa: E402

DEFAULT_CHECKPOINT = LEGACY_ROOT / "checkpoints" / "best.pt"
EVALUATION_DIR = RUNS_ROOT / "cnn_evaluations" / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--per-class-csv",
        type=Path,
        default=EVALUATION_DIR / "per_class_metrics.csv",
    )
    parser.add_argument(
        "--errors-csv",
        type=Path,
        default=EVALUATION_DIR / "error_samples.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=EVALUATION_DIR / "summary.json",
    )
    args = parser.parse_args()

    # 新评价单独建目录，拒绝与任何已保存结果混写。
    output_paths = (args.per_class_csv, args.errors_csv, args.summary_json)
    if any(path.exists() for path in output_paths):
        raise FileExistsError("评价输出已存在，请指定新的文件路径")

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # evaluate 与训练必须共享同一数据、标签和预处理契约；不兼容时立即拒绝。
    manifest_sha256 = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if checkpoint["manifest_sha256"] != manifest_sha256:
        raise ValueError("当前 manifest 与 checkpoint 记录的版本不一致")
    if checkpoint["model_name"] != "EurosatCNN":
        raise ValueError("checkpoint 模型类型不是 EurosatCNN")
    if checkpoint["num_classes"] != NUM_CLASSES:
        raise ValueError("checkpoint 类别数量与当前工程不一致")
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("checkpoint 类别编号映射与当前工程不一致")
    if tuple(checkpoint["normalization_mean"]) != tuple(TRAIN_MEAN):
        raise ValueError("checkpoint 归一化 mean 与当前工程不一致")
    if tuple(checkpoint["normalization_std"]) != tuple(TRAIN_STD):
        raise ValueError("checkpoint 归一化 std 与当前工程不一致")

    val_loader = make_loader(
        "val",
        build_eval_transform(),
        args.batch_size,
        shuffle=False,
    )
    model = EurosatCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    val_loss, val_acc, cm, all_labels, all_preds, all_confidences = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )
    seconds = time.perf_counter() - start
    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 ** 2

    # 这段逐样本导出直接取自 train_fit_loop 的最终评价逻辑。
    error_rows = []
    for image, label, pred, confidence in zip(
        val_loader.dataset.images,
        all_labels,
        all_preds,
        all_confidences,
        strict=True,
    ):
        if pred != label:
            error_rows.append(
                {
                    "filepath": image,
                    "true_idx": label,
                    "true_class": CLASS_NAMES[label],
                    "pred_idx": pred,
                    "pred_class": CLASS_NAMES[pred],
                    "confidence": confidence,
                }
            )
    error_rows = sorted(
        error_rows,
        key=lambda row: row["confidence"],
        reverse=True,
    )

    for path in (args.per_class_csv, args.errors_csv, args.summary_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        error_rows,
        columns=[
            "filepath",
            "true_idx",
            "true_class",
            "pred_idx",
            "pred_class",
            "confidence",
        ],
    ).to_csv(args.errors_csv, index=False, encoding="utf-8-sig")
    per_class = save_per_class_metrics(cm, args.per_class_csv)

    true_count = per_class["true_count"].to_numpy()
    f1 = per_class["f1"].to_numpy()
    summary = {
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_run_id": checkpoint.get("run_id"),
        "manifest_path": str(MANIFEST),
        "manifest_sha256": manifest_sha256,
        "sample_count": len(val_loader.dataset),
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * true_count).sum() / true_count.sum()),
        "error_count": len(error_rows),
        "seconds": seconds,
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "per_class_metrics_path": str(args.per_class_csv),
        "error_samples_path": str(args.errors_csv),
    }
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("checkpoint epoch:", checkpoint["epoch"])
    print("errors:", len(error_rows))
    print(
        f"seconds={seconds:.1f} peak_alloc={peak_allocated_mb:.0f}MB "
        f"peak_reserved={peak_reserved_mb:.0f}MB"
    )
    print("summary:", args.summary_json)


if __name__ == "__main__":
    main()
