"""把正式实验的 JSONL 记录转换为表格和静态图。"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_experiment_visualizations(experiment_dir):
    """生成总体学习曲线、逐类 recall 热力图和便于查看的 CSV。"""
    experiment_dir = Path(experiment_dir)
    metrics = _read_jsonl(experiment_dir / "metrics.jsonl")
    per_class_epochs = _read_jsonl(experiment_dir / "per_class_metrics.jsonl")

    # JSONL 是机器追加记录；CSV 是同一批数据的表格视图，不重新计算指标。
    csv_path = experiment_dir / "epoch_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    epochs = [row["epoch"] for row in metrics]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        epochs,
        [row["train_loss"] for row in metrics],
        marker="o",
        label="train (online)",
    )
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in metrics],
        marker="o",
        label="val (post-epoch)",
    )
    axes[0].set(title="Loss by epoch", xlabel="epoch", ylabel="loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in metrics],
        marker="o",
        label="train (online)",
    )
    axes[1].plot(
        epochs,
        [row["val_accuracy"] for row in metrics],
        marker="o",
        label="val (post-epoch)",
    )
    best_row = max(metrics, key=lambda row: row["val_accuracy"])
    axes[1].scatter(
        [best_row["epoch"]],
        [best_row["val_accuracy"]],
        color="red",
        zorder=3,
        label=f"best={best_row['epoch']}",
    )
    axes[1].set(title="Accuracy by epoch", xlabel="epoch", ylabel="accuracy", ylim=(0, 1))
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(experiment_dir / "learning_curves.png", dpi=160)
    plt.close(figure)

    class_names = [row["class_name"] for row in per_class_epochs[0]["classes"]]
    recall_matrix = np.array([
        [class_row["recall"] for class_row in epoch_row["classes"]]
        for epoch_row in per_class_epochs
    ])
    figure, axis = plt.subplots(figsize=(11, 5.5))
    image = axis.imshow(recall_matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=35, ha="right")
    axis.set_yticks(range(len(epochs)), labels=epochs)
    axis.set_xlabel("class")
    axis.set_ylabel("epoch")
    axis.set_title("Per-class recall by epoch")
    figure.colorbar(image, ax=axis, label="recall")
    figure.tight_layout()
    figure.savefig(experiment_dir / "per_class_recall.png", dpi=160)
    plt.close(figure)

    outputs = {
        "epoch_metrics_csv": str(csv_path),
        "learning_curves": str(experiment_dir / "learning_curves.png"),
        "per_class_recall": str(experiment_dir / "per_class_recall.png"),
    }

    # warmup 实验额外画逐 optimizer step 学习率，避免 epoch 级记录掩盖爬升过程。
    learning_rate_path = experiment_dir / "learning_rate_history.jsonl"
    if learning_rate_path.is_file():
        learning_rate_rows = _read_jsonl(learning_rate_path)
        figure, axis = plt.subplots(figsize=(8, 4.2))
        axis.plot(
            [row["global_step"] for row in learning_rate_rows],
            [row["learning_rate"] for row in learning_rate_rows],
        )
        axis.set(
            title="Learning rate by optimizer step",
            xlabel="optimizer step",
            ylabel="learning rate",
        )
        axis.grid(alpha=0.3)
        figure.tight_layout()
        lr_figure_path = experiment_dir / "learning_rate_curve.png"
        figure.savefig(lr_figure_path, dpi=160)
        plt.close(figure)
        outputs["learning_rate_curve"] = str(lr_figure_path)

    return outputs


def save_experiment_comparison(reference_dir, candidate_dir, output_path):
    """把两个相同训练协议的 epoch 指标画在同一坐标系，便于直接比较。"""
    reference_dir = Path(reference_dir)
    candidate_dir = Path(candidate_dir)
    reference = _read_jsonl(reference_dir / "metrics.jsonl")
    candidate = _read_jsonl(candidate_dir / "metrics.jsonl")

    reference_name = reference_dir.name
    candidate_name = candidate_dir.name
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for rows, name in (
        (reference, reference_name),
        (candidate, candidate_name),
    ):
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(
            epochs,
            [row["val_accuracy"] for row in rows],
            marker="o",
            label=name,
        )
        axes[1].plot(
            epochs,
            [row["val_loss"] for row in rows],
            marker="o",
            label=name,
        )

    axes[0].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy")
    axes[1].set(title="Validation loss", xlabel="epoch", ylabel="loss")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return str(output_path)
