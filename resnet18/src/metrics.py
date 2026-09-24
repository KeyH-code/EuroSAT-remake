"""02 评价与记录：复用小型 CNN 的样本累计、混淆矩阵和逐类口径。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import CLASS_NAMES, NUM_CLASSES


def evaluate_classifier(model, loader, criterion, device):
    """对任意输出 [B, 10] logits 的模型评价，返回整体及逐样本结果。"""
    model.eval()  # 固定 BN/Dropout 等层的评价行为；无梯度上下文不能代替此模式。
    loss_sum = 0.0
    sample_count = 0
    correct_count = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
    labels_all, predictions_all, confidences_all = [], [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if logits.ndim != 2 or logits.shape != (len(labels), NUM_CLASSES):
                raise ValueError("模型必须返回 [B, 10] 的原始 logits")

            # CrossEntropyLoss 接收原始 logits。按样本数加权，避免末批偏小导致偏差。
            loss = criterion(logits, labels)
            batch_size = len(labels)
            loss_sum += loss.item() * batch_size
            sample_count += batch_size

            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            correct_count += (predictions == labels).sum().item()

            labels_cpu = labels.cpu().long()
            predictions_cpu = predictions.cpu().long()
            counts = torch.bincount(
                labels_cpu * NUM_CLASSES + predictions_cpu,
                minlength=NUM_CLASSES * NUM_CLASSES,
            )
            confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)
            labels_all.extend(labels_cpu.tolist())
            predictions_all.extend(predictions_cpu.tolist())
            confidences_all.extend(confidences.cpu().tolist())

    if sample_count == 0:
        raise ValueError("评价 loader 没有样本")
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct_count / sample_count,
        "confusion_matrix": confusion,
        "labels": labels_all,
        "predictions": predictions_all,
        "confidences": confidences_all,
    }


def per_class_rows(confusion_matrix):
    """矩阵行是真值、列是预测；由此计算每类 precision/recall/F1。"""
    matrix = confusion_matrix.cpu().numpy()
    true_count = matrix.sum(axis=1)
    pred_count = matrix.sum(axis=0)
    correct = np.diag(matrix)
    recall = np.divide(correct, true_count, out=np.zeros(NUM_CLASSES), where=true_count > 0)
    precision = np.divide(correct, pred_count, out=np.zeros(NUM_CLASSES), where=pred_count > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(NUM_CLASSES),
        where=(precision + recall) > 0,
    )
    return [
        {
            "class_idx": index,
            "class_name": CLASS_NAMES[index],
            "true_count": int(true_count[index]),
            "pred_count": int(pred_count[index]),
            "correct": int(correct[index]),
            "recall": float(recall[index]),
            "precision": float(precision[index]),
            "f1": float(f1[index]),
        }
        for index in range(NUM_CLASSES)
    ]


def append_epoch_record(experiment_dir, epoch, train_loss, train_accuracy, val_result, extras=None):
    """追加一轮曲线数据、逐类指标和混淆矩阵；训练循环自行决定何时调用。"""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir = experiment_dir / "confusion_matrices"
    confusion_dir.mkdir(exist_ok=True)
    matrix = val_result["confusion_matrix"]

    # 字段沿用旧实验，图表代码可直接读取；不在这里选择 best 或保存 checkpoint。
    row = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "val_loss": float(val_result["loss"]),
        "val_accuracy": float(val_result["accuracy"]),
    }
    if extras:
        row.update(extras)  # 耗时、学习率和峰值显存等只作附加记录，不改指标口径。
    with (experiment_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (experiment_dir / "per_class_metrics.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as file:
        file.write(json.dumps({"epoch": epoch, "classes": per_class_rows(matrix)}) + "\n")
    (confusion_dir / f"epoch_{epoch:03d}.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "rows": "true",
                "columns": "predicted",
                "class_names": CLASS_NAMES,
                "matrix": matrix.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def save_validation_details(output_dir, val_result, val_dataset):
    """保存固定顺序 val loader 的逐类指标和错误图像清单。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = val_result["confusion_matrix"]
    rows = per_class_rows(matrix)
    pd.DataFrame(rows).to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    # val loader 必须 shuffle=False，dataset.images 才与评价收集的逐样本结果对齐。
    images = val_dataset.images
    labels = val_result["labels"]
    predictions = val_result["predictions"]
    confidences = val_result["confidences"]
    if not (len(images) == len(labels) == len(predictions) == len(confidences)):
        raise ValueError("验证样本路径与预测结果数量不一致")
    errors = [
        {
            "filepath": image,
            "true_idx": label,
            "true_class": CLASS_NAMES[label],
            "pred_idx": prediction,
            "pred_class": CLASS_NAMES[prediction],
            "confidence": confidence,
        }
        for image, label, prediction, confidence in zip(
            images, labels, predictions, confidences, strict=True
        )
        if prediction != label
    ]
    errors.sort(key=lambda row: row["confidence"], reverse=True)
    columns = ["filepath", "true_idx", "true_class", "pred_idx", "pred_class", "confidence"]
    pd.DataFrame(errors, columns=columns).to_csv(
        output_dir / "error_samples.csv", index=False, encoding="utf-8-sig"
    )
    return {"error_count": len(errors), "per_class": rows}
