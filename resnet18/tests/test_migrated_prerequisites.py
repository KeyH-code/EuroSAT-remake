"""验证迁移后的真实 split 入口和模型无关的指标/图表契约。"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CLASS_NAMES, load_split_frame, make_train_val_loaders  # noqa: E402
from src.metrics import (  # noqa: E402
    append_epoch_record,
    evaluate_classifier,
    save_validation_details,
)
from src.visualization import (  # noqa: E402
    save_experiment_visualizations,
    save_validation_visualizations,
)


def test_existing_eurosat_split_reaches_new_loader():
    # 只读一批已复制的真实 val 图像，证明新目录复用原样清单，没有重新划分。
    train = load_split_frame("train")
    val = load_split_frame("val")
    assert len(train) == 18900 and len(val) == 4050
    assert set(train["filepath"]).isdisjoint(val["filepath"])
    assert all(CLASS_NAMES[index] == name for index, name in zip(val["label_idx"], val["label"]))

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_loader, val_loader = make_train_val_loaders(transform, transform, batch_size=2, seed=42)
    images, labels = next(iter(val_loader))
    assert images.shape == (2, 3, 64, 64)
    assert labels.dtype == torch.long
    assert val_loader.dataset.images[:2] == val["filepath"].tolist()[:2]
    assert len(train_loader.dataset) == 18900


def test_uneven_last_batch_metrics_and_artifacts(tmp_path):
    # 5 张假图分成 3+2 两批，检查 loss 按样本加权、混淆矩阵和落盘图表。
    labels = torch.tensor([0, 1, 2, 3, 4])
    logits = torch.full((5, 10), -2.0)
    logits[0, 0] = logits[1, 1] = logits[2, 2] = logits[3, 3] = 3.0
    logits[4, 0] = 3.0  # 最后一张故意把 true=4 判为 pred=0。

    class IndexedModel(nn.Module):
        def forward(self, indices):
            return logits[indices[:, 0].long()]

    loader = DataLoader(TensorDataset(torch.arange(5).float().view(-1, 1), labels), batch_size=3)
    criterion = nn.CrossEntropyLoss()
    result = evaluate_classifier(IndexedModel(), loader, criterion, torch.device("cpu"))
    assert result["accuracy"] == 4 / 5
    assert result["loss"] == pytest.approx(criterion(logits, labels).item())
    assert result["confusion_matrix"].sum().item() == 5
    assert result["confusion_matrix"][4, 0].item() == 1

    # 验证逐样本路径、指标 JSONL、CSV 与曲线都能在独立实验目录生成。
    val_dataset = type("ValDataset", (), {"images": [f"image_{i}.jpg" for i in range(5)]})()
    details = save_validation_details(tmp_path / "final_evaluation", result, val_dataset)
    assert details["error_count"] == 1
    final_plots = save_validation_visualizations(
        tmp_path / "final_evaluation", result["confusion_matrix"], details["per_class"]
    )
    append_epoch_record(tmp_path, 1, 0.8, 0.6, result)
    outputs = save_experiment_visualizations(tmp_path)
    assert all(Path(path).is_file() for path in (*outputs.values(), *final_plots.values()))
