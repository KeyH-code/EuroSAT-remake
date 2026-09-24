"""03 实验图表：读取与旧 CNN 相同口径的 JSONL，生成 loss/accuracy 曲线。"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 文件输出，不依赖交互窗口。
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _read_jsonl(path):
    """每行一个 epoch；与旧实验的记录格式保持一致。"""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_experiment_visualizations(experiment_dir):
    """根据已记录的轮次生成 CSV、训练/验证曲线和逐类 recall 图。"""
    experiment_dir = Path(experiment_dir)
    metrics = _read_jsonl(experiment_dir / "metrics.jsonl")
    per_class_epochs = _read_jsonl(experiment_dir / "per_class_metrics.jsonl")
    if not metrics or len(metrics) != len(per_class_epochs):
        raise ValueError("整体与逐类 epoch 记录不完整")

    # CSV 只是同一批 JSONL 数值的表格视图，不重新计算任何指标。
    csv_path = experiment_dir / "epoch_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    epochs = [row["epoch"] for row in metrics]
    # 放宽画布，逐轮刻度和较密的纵轴刻度仍能保持可读。
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(epochs, [row["train_loss"] for row in metrics], marker="o", label="train (online)")
    axes[0].plot(epochs, [row["val_loss"] for row in metrics], marker="o", label="val (post-epoch)")
    # 聚焦后期 loss；超出 0.3 的早期数值仍保留在原始记录中。
    axes[0].set(title="Loss by epoch", xlabel="epoch", ylabel="loss", ylim=(0, 0.3))
    axes[0].set_yticks(np.arange(0, 0.301, 0.025))
    if any(row["train_loss"] > 0.3 or row["val_loss"] > 0.3 for row in metrics):
        axes[0].text(
            0.98, 0.04, "Early loss > 0.30 is clipped", transform=axes[0].transAxes,
            ha="right", va="bottom", fontsize=8, color="dimgray",
        )
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs, [row["train_accuracy"] for row in metrics], marker="o", label="train (online)"
    )
    axes[1].plot(
        epochs, [row["val_accuracy"] for row in metrics], marker="o", label="val (post-epoch)"
    )
    best_row = max(metrics, key=lambda row: row["val_accuracy"])
    axes[1].scatter(
        [best_row["epoch"]], [best_row["val_accuracy"]], color="red", zorder=3,
        label=f"best={best_row['epoch']}",
    )
    axes[1].set(title="Accuracy by epoch", xlabel="epoch", ylabel="accuracy", ylim=(0.8, 1))
    axes[1].set_yticks(np.arange(0.8, 1.001, 0.02))
    # 25 轮每轮都给一个横轴刻度，便于看相邻轮次的微小波动。
    for axis in axes:
        axis.set_xticks(epochs)
        axis.tick_params(axis="x", labelsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    curves_path = experiment_dir / "learning_curves.png"
    figure.savefig(curves_path, dpi=160)
    plt.close(figure)

    # 行是 epoch、列是类别；颜色表达每类召回率，便于发现某一类的回落。
    class_names = [row["class_name"] for row in per_class_epochs[0]["classes"]]
    recall_matrix = np.array([
        [class_row["recall"] for class_row in epoch_row["classes"]]
        for epoch_row in per_class_epochs
    ])
    figure, axis = plt.subplots(figsize=(11, 5.5))
    heatmap = axis.imshow(recall_matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=35, ha="right")
    axis.set_yticks(range(len(epochs)), labels=epochs)
    axis.set(xlabel="class", ylabel="epoch", title="Per-class recall by epoch")
    figure.colorbar(heatmap, ax=axis, label="recall")
    figure.tight_layout()
    recall_path = experiment_dir / "per_class_recall.png"
    figure.savefig(recall_path, dpi=160)
    plt.close(figure)

    return {
        "epoch_metrics_csv": str(csv_path),
        "learning_curves": str(curves_path),
        "per_class_recall": str(recall_path),
    }


def save_validation_visualizations(output_dir, confusion_matrix, class_rows):
    """对固定权重的一次 val 评价，画混淆矩阵和逐类 P/R/F1。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = [row["class_name"] for row in class_rows]
    matrix = confusion_matrix.cpu().numpy()

    # 行是真值、列是预测；保留原始计数以便与 CSV/JSON 审计对照。
    figure, axis = plt.subplots(figsize=(9, 7.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set(xlabel="predicted", ylabel="true", title="Validation confusion matrix")
    figure.colorbar(image, ax=axis, label="images")
    figure.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)

    positions = np.arange(len(class_names))
    figure, axis = plt.subplots(figsize=(11, 4.5))
    for offset, metric in ((-0.25, "precision"), (0, "recall"), (0.25, "f1")):
        axis.bar(positions + offset, [row[metric] for row in class_rows], width=0.24, label=metric)
    axis.set_xticks(positions, labels=class_names, rotation=35, ha="right")
    axis.set(xlabel="class", ylabel="score", ylim=(0, 1), title="Validation per-class scores")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    scores_path = output_dir / "per_class_scores.png"
    figure.savefig(scores_path, dpi=160)
    plt.close(figure)
    return {"confusion_matrix": str(confusion_path), "per_class_scores": str(scores_path)}
