"""锁定 config41 epoch17 权重，按同一口径计算固定 val/test 指标。"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "resnet18"
SRC = ROOT / "resnet18" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(PROJECT))
from data import CLASS_NAMES, DEVICE, MANIFEST, make_loader  # noqa: E402
from src.checkpoint import load_checkpoint  # noqa: E402
from transforms import make_transforms  # noqa: E402


CONFIG = ROOT / "resnet18/configs/config41_layer4_lr0p0001_warmup_cosine_bs64_flip_rotate_translate_crop_20ep.json"
CHECKPOINT = ROOT / "runs/resnet18/resnet18_layer4_lr0p0001_warmup_cosine_bs64_flip_rotate_translate_crop_20ep_01/checkpoints/best.pt"
EXPECTED_CONFIG_SHA256 = "f6c4da80f50b71f7af303672d06f95fe856617bc4c1233eee80abf6af29da848"
EXPECTED_CHECKPOINT_SHA256 = "8039bbce2884fa5058d939bf6973c7866492e50b714504442aba87e6af56d025"
EXPECTED_MANIFEST_SHA256 = "66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_plot(matrix, output, split):
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), constrained_layout=True)
    for axis, values, title, fmt in (
        (axes[0], matrix, f"{split.capitalize()} confusion matrix: counts", "d"),
        (axes[1], fractions * 100, f"{split.capitalize()} confusion matrix: row %", ".1f"),
    ):
        axis.imshow(values, cmap="Blues", vmin=0)
        axis.set_title(title)
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=55, ha="right")
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                value = int(values[i, j]) if fmt == "d" else values[i, j]
                color = "white" if values[i, j] > values.max() * 0.55 else "black"
                axis.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7, color=color)
    fig.savefig(output / "confusion_matrix.png", dpi=170)
    plt.close(fig)


def save_class_metric_plot(rows, output, split):
    names = [row["class"] for row in rows]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    axes[0].bar(positions, [row["f1"] for row in rows], color="#2a7f80")
    f1_lower = min(row["f1"] for row in rows)
    axes[0].set(title=f"{split.capitalize()} F1 by class", ylabel="F1", ylim=(min(0.94, f1_lower - 0.005), 1.0))
    axes[1].bar(positions, [row["roc_auc_ovr"] for row in rows], color="#4c6fa5")
    lower = min(row["roc_auc_ovr"] for row in rows)
    axes[1].set(title=f"{split.capitalize()} ROC AUC by class (one-vs-rest)", ylabel="ROC AUC", ylim=(min(0.95, lower - 0.005), 1.0))
    for axis in axes:
        axis.set_xticks(positions, names, rotation=55, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "per_class_f1_auc.png", dpi=170)
    plt.close(fig)


def main(split="val"):
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    output = ROOT / "final_result/resnet18" / ("validation" if split == "val" else "test")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    hashes = {
        "config_sha256": sha256(CONFIG),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "manifest_sha256": sha256(MANIFEST),
    }
    expected = {
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"Locked source hash mismatch: {hashes}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["loader"]["seed"] != 20260920:
        raise ValueError("Locked seed changed")
    from model import model  # noqa: E402

    model.to(DEVICE)
    checkpoint = load_checkpoint(CHECKPOINT, model=model, map_location="cpu")
    if (checkpoint["epoch"], checkpoint["best_epoch"]) != (17, 17):
        raise ValueError("Locked checkpoint is not epoch 17 best")
    if checkpoint["config"]["experiment_id"] != config["experiment_id"]:
        raise ValueError("Checkpoint experiment ID differs from config41")
    if checkpoint["config"]["transform"] != config["transform"]:
        raise ValueError("Checkpoint validation transform differs from config41")
    _, eval_transform = make_transforms(config)
    loader = make_loader(split, eval_transform, config["loader"]["batch_size"], shuffle=False)
    model.eval()
    probabilities, truths = [], []
    ce_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            ce_sum += torch.nn.functional.cross_entropy(logits, labels.to(DEVICE), reduction="sum").item()
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            truths.append(labels.numpy())
    probs = np.concatenate(probabilities).astype(np.float64)
    true = np.concatenate(truths).astype(np.int64)
    pred = probs.argmax(axis=1)
    count = len(true)
    if count != 4050 or len(loader.dataset.images) != count:
        raise ValueError(f"{split} size differs: {count}")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Invalid probability rows")
    accuracy = accuracy_score(true, pred)
    cross_entropy = ce_sum / count
    if split == "val":
        if abs(accuracy - checkpoint["best_val_accuracy"]) > 1e-9:
            raise ValueError("Fresh validation accuracy differs from checkpoint")
        recorded_loss = 0.061789271643408285
        if abs(cross_entropy - recorded_loss) > 1e-6:
            raise ValueError("Fresh validation loss differs from epoch 17 record")

    labels = np.arange(len(CLASS_NAMES))
    matrix = confusion_matrix(true, pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=labels, zero_division=0)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.int64)[true]
    roc_aucs = [roc_auc_score(one_hot[:, i], probs[:, i]) for i in labels]
    aps = [average_precision_score(one_hot[:, i], probs[:, i]) for i in labels]
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        tn = count - matrix[i, :].sum() - matrix[:, i].sum() + matrix[i, i]
        fp = matrix[:, i].sum() - matrix[i, i]
        per_class.append({
            "class": name,
            "support": int(support[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "specificity": float(tn / (tn + fp)),
            "f1": float(f1[i]),
            "roc_auc_ovr": float(roc_aucs[i]),
            "average_precision_ovr": float(aps[i]),
        })
    aggregate = {}
    for average in ("micro", "macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(true, pred, labels=labels, average=average, zero_division=0)
        aggregate[average] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    aggregate["macro"]["roc_auc_ovr"] = float(np.mean(roc_aucs))
    aggregate["weighted"]["roc_auc_ovr"] = float(np.average(roc_aucs, weights=support))
    aggregate["micro"]["roc_auc_ovr"] = float(roc_auc_score(one_hot.ravel(), probs.ravel()))
    aggregate["macro"]["average_precision_ovr"] = float(np.mean(aps))
    aggregate["weighted"]["average_precision_ovr"] = float(np.average(aps, weights=support))
    aggregate["micro"]["average_precision_ovr"] = float(average_precision_score(one_hot.ravel(), probs.ravel()))

    confidence = probs.max(axis=1)
    correct = pred == true
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece_15 = sum(
        (bins == i).mean() * abs(correct[bins == i].mean() - confidence[bins == i].mean())
        for i in range(15) if np.any(bins == i)
    )
    metrics = {
        "split": split,
        "test_used": split == "test",
        "checkpoint_epoch": 17,
        "sample_count": count,
        "correct_count": int(correct.sum()),
        "error_count": int((~correct).sum()),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "cross_entropy": float(cross_entropy),
        "top2_accuracy": float(np.mean(np.any(np.argsort(-probs, axis=1)[:, :2] == true[:, None], axis=1))),
        "top3_accuracy": float(np.mean(np.any(np.argsort(-probs, axis=1)[:, :3] == true[:, None], axis=1))),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece_15_equal_width": float(ece_15),
        "aggregates": aggregate,
        "per_class": per_class,
        "confusion_matrix_true_rows_pred_columns": matrix.tolist(),
        "class_names": list(CLASS_NAMES),
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "model_lock.json").write_text(json.dumps({
        "checkpoint": str(CHECKPOINT), "config": str(CONFIG), "manifest": str(MANIFEST),
        "experiment_id": config["experiment_id"], "seed": 20260920,
        "checkpoint_epoch": 17, **hashes, "split": split, "test_used": split == "test",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "per_class_metrics.csv", per_class, list(per_class[0]))
    write_csv(output / "confusion_matrix.csv", [
        {"true_class": CLASS_NAMES[i], **{CLASS_NAMES[j]: int(matrix[i, j]) for j in labels}}
        for i in labels
    ], ["true_class", *CLASS_NAMES])
    prediction_fields = ["filepath", "true_class", "pred_class", "confidence", *[f"p_{name}" for name in CLASS_NAMES]]
    write_csv(output / "predictions.csv", [
        {"filepath": str(loader.dataset.images[i]), "true_class": CLASS_NAMES[true[i]],
         "pred_class": CLASS_NAMES[pred[i]], "confidence": float(confidence[i]),
         **{f"p_{name}": float(probs[i, j]) for j, name in enumerate(CLASS_NAMES)}}
        for i in range(count)
    ], prediction_fields)
    save_confusion_plot(matrix, output, split)
    save_class_metric_plot(per_class, output, split)
    print(json.dumps({"output": str(output), "split": split, "accuracy": accuracy, "cross_entropy": cross_entropy,
                      "macro_f1": aggregate["macro"]["f1"], "macro_roc_auc_ovr": aggregate["macro"]["roc_auc_ovr"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
