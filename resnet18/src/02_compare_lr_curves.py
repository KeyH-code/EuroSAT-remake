"""02 将同为 20 轮的学习率实验画在一张验证曲线图上。

只读取本仓库 `runs/resnet18/` 下的新运行目录；历史快照不参与对照。
四个同预算实验需先在新仓库重训（参数见 docs/migration-record.md 第 3.2 节），
否则本脚本会明确报出缺失目录，而不会回退去读历史产物。
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无图形窗口也能稳定保存实验图。
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "resnet18"
OUTPUT_ROOT = RUN_ROOT / "analysis"
RUNS = (
    ("0.003", "resnet18_head_lr0p003_20ep_01"),
    ("0.004", "resnet18_head_lr0p004_20ep_01"),
    ("0.005", "resnet18_head_lr0p005_20ep_01"),
    ("0.006", "resnet18_head_lr0p006_20ep_01"),
)

missing = [str(RUN_ROOT / experiment_id) for _, experiment_id in RUNS if not (RUN_ROOT / experiment_id / "epoch_metrics.csv").is_file()]
if missing:
    raise FileNotFoundError(
        "缺少本仓库运行目录，无法对照；请先在 runs/resnet18/ 下重训这些实验：\n  "
        + "\n  ".join(missing)
    )

figure, (acc_axis, loss_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
for lr, experiment_id in RUNS:
    # 各组按自己的逐轮 CSV 读取；图上只比较固定 val 集。
    csv_path = RUN_ROOT / experiment_id / "epoch_metrics.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    epochs = [int(row["epoch"]) for row in rows]
    val_accuracy = [float(row["val_accuracy"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    acc_axis.plot(epochs, val_accuracy, marker=".", label=f"lr={lr}")
    loss_axis.plot(epochs, val_loss, marker=".", label=f"lr={lr}")

acc_axis.set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0.90, 0.96))
loss_axis.set(title="Validation loss", xlabel="Epoch", ylabel="Cross-entropy", ylim=(0.14, 0.25))
for axis in (acc_axis, loss_axis):
    axis.set_xticks((1, 5, 10, 15, 20))
    axis.grid(alpha=0.25)
    axis.legend()

figure.suptitle("EuroSAT ResNet-18 frozen backbone: 20-epoch LR comparison")
figure.tight_layout()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_ROOT / "lr_0p003_to_0p006_20ep_comparison.png"
if output_path.exists():
    raise FileExistsError(f"拒绝覆盖已有分析图：{output_path}")
figure.savefig(output_path, dpi=170)
plt.close(figure)
print(output_path)
