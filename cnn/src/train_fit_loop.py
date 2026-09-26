r"""EuroSAT 小 CNN 的训练与评价核心。

本模块只提供可复用的训练/评价函数与固定契约（类别顺序、清单、归一化常数）：
    - ``EurosatCNN``、``make_loader``、``train_one_epoch``、``evaluate``
    - ``fix_seed``、``save_per_class_metrics``
调用方是 ``cnn/src/experiment_runner.py``（配置驱动正式训练）与 ``cnn/evaluate.py``、
``cnn/predict.py``。本模块自身不再是脚本入口。

教学期"自己写一遍训练循环"的独立入口已切走，完整原件见
``artifacts/legacy_teaching/cnn/src/train_fit_loop_original.py``。

``train_one_epoch`` 的 autocast 契约：只有传入 scaler 时才进入 autocast，FP32 走空上下文。
"""
import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同目录模块可导入
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 独立仓库的路径契约
from eurosat_paths import MANIFEST, map_manifest_images  # noqa: E402
from transforms import (  # noqa: E402
    TRAIN_MEAN,
    TRAIN_STD,
    build_eval_transform,
    build_train_transform,
)

NUM_CLASSES = 10   # EuroSAT 的 10 个类别
SEED = 20260920

# EuroSAT 类别名，顺序即 label_idx
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# 数据：读清单 -> 按 split 取子集 -> 交给既有的 EuroSATDataset
# 这一段的契约已在 C1 建立并通过夹具，这里只是把它接到训练上。
# ---------------------------------------------------------------------------
class EuroSATDataset(Dataset):
    """按索引读一张图 + 它的整数标签。C1 已通过契约验证。"""

    def __init__(self, images, labels, transform):
        self.images = images      # 图像路径列表
        self.labels = labels      # 与上面逐位对齐的 label_idx 列表
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")  # 按需读图，不预先缓存像素
        return self.transform(image), self.labels[idx]


def make_loader(split, transform, batch_size, shuffle, seed=None):
    """从固定的完整清单里取某个 split，组成 DataLoader。

    seed 给 shuffle 用独立随机源（C1 已确认的做法：不让"打乱顺序"和"初始化权重"
    抢同一个全局随机状态）。验证集 shuffle=False，不需要 seed。
    """
    frame = map_manifest_images(pd.read_csv(MANIFEST, encoding="utf-8"))
    frame = frame[frame["split"] == split].reset_index(drop=True)

    dataset = EuroSATDataset(
        images=frame["filepath"].tolist(),
        labels=frame["label_idx"].tolist(),
        transform=transform,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,      # 只有训练集打乱顺序；验证集顺序固定，便于对照
        num_workers=0,        # 先用 0：出错时能直接看到真实异常，不受 worker 影响
        generator=generator,  # 只影响 shuffle 的取样顺序
    )


# ---------------------------------------------------------------------------
# 模型：4 个 64x64 输入的小 CNN，输出 [B,10] logits
# 结构说明（本块不重讲卷积细节，只标出 shape 怎么走）：
#   输入     [B, 3, 64, 64]
#   block1   [B, 16,64, 64] -> maxpool -> [B, 16,32, 32]
#   block2   [B, 32,32, 32] -> maxpool -> [B, 32,16, 16]
#   block3   [B, 64,16, 16] -> maxpool -> [B, 64, 8,  8]
#   block4   [B,128, 8,  8] -> 自适应平均池化 -> [B,128, 1, 1]
#   head     flatten -> [B,128] -> Linear -> [B,10]
# 卷积用 bias=False：后面紧跟 BatchNorm，常数项由 BN 的 beta 承担，
# 这是 M2-B4 已经验证过的组合（当时检查过 bias 与 running statistics）。
# ---------------------------------------------------------------------------
class EurosatCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.block1 = self._make_block(3, 16)
        self.block2 = self._make_block(16, 32)
        self.block3 = self._make_block(32, 64)
        self.block4 = self._make_block(64, 128)

        # 自适应平均池化：不管前面留下来的是 8x8 还是别的尺寸，都压成 1x1
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = nn.Linear(in_features=128, out_features=num_classes)

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 每过一块，空间尺寸减半
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)                    # [B,128,1,1]
        x = torch.flatten(x, start_dim=1)   # [B,128]
        return self.head(x)                 # [B,10]


# ===========================================================================
# 两个关键函数
# ===========================================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    lr_scheduler=None,
    learning_rate_trace=None,
):
    """训练一个 epoch，返回 (本 epoch 的平均 loss, 样本准确率, skipped 步数)。"""
    model.train()
    loss_sum = 0
    sample_count = 0
    correct_count = 0
    skipped = 0          # 被 scaler 跳过的步数（scale 被调小就说明跳了一步）
    prev_scale = None

    for batch, label in loader:
        # 记录这一批参数更新实际使用的学习率；正式实验用它审计 warmup 曲线。
        if learning_rate_trace is not None:
            learning_rate_trace.append(optimizer.param_groups[0]["lr"])

        batch = batch.to(device)
        label = label.to(device)
        optimizer.zero_grad(set_to_none=True)

        # AMP 开启时才进入 autocast；纯 FP32 分支使用空上下文，确保前向不降精度。
        amp_context = (
            torch.autocast("cuda", dtype=torch.float16)
            if scaler is not None
            else nullcontext()
        )
        with amp_context:
            logits = model(batch)
            loss = criterion(logits, label)

        optimizer_updated = True
        if scaler is not None:                       # AMP：放大后反向，由 scaler 管更新
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            current_scale = scaler.get_scale()
            optimizer_updated = current_scale >= scale_before
            if not optimizer_updated:
                skipped += 1                         # update() 减半 = 检出 inf/nan，那一步没更新
            prev_scale = current_scale
        else:                                        # 纯 FP32：直接反向、直接更新
            loss.backward()
            optimizer.step()

        # 按 optimizer step 推进学习率；AMP 真正跳步时不推进 scheduler。
        if lr_scheduler is not None and optimizer_updated:
            lr_scheduler.step()

        sample_count += len(batch)
        loss_sum += loss.item() * len(batch)

        predictions = logits.argmax(dim=1)
        correct_count += (predictions == label).sum().item()

    loss_mean = loss_sum / sample_count
    accuracy = correct_count / sample_count
    print(f"train loss_mean={loss_mean:.4f} accuracy={accuracy:.4f} skipped={skipped}")
    return loss_mean, accuracy, skipped


def evaluate(model, loader, criterion, device):
    """在验证集上判卷，返回整体指标与逐样本标签、预测及置信度；不更新参数。"""
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    loss_sum = 0
    sample_count = 0
    correct_count = 0

    with torch.no_grad():
        for batch, label in loader:
            batch = batch.to(device)
            label = label.to(device)
            logits = model(batch)
            loss = criterion(logits, label)

            sample_count += len(batch)
            loss_sum += loss.item() * len(batch)

            probabilities = torch.softmax(logits,dim=1) # 类别维度做softmax
            confidence,predictions = torch.topk(
                probabilities,
                k=1,
                dim=1
            )
            confidence = confidence.squeeze(1)
            predictions = predictions.squeeze(1)

            correct_count += (predictions == label).sum().item()
            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(label.cpu().tolist())
            all_confidences.extend(confidence.cpu().tolist())

    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    cm.index_put_(
        (torch.tensor(all_labels, device=device),
         torch.tensor(all_preds, device=device)),
        torch.ones(len(all_labels), dtype=torch.long, device=device),
        accumulate=True,
    )

    loss_mean = loss_sum / sample_count
    accuracy = correct_count / sample_count
    print(f"eval loss_mean={loss_mean:.4f} accuracy={accuracy:.4f}")
    return loss_mean, accuracy, cm, all_labels,all_preds,all_confidences


# ===========================================================================
# 外壳：数据、模型、优化器和指标记录
# ===========================================================================
def fix_seed(seed):
    """固定本进程的随机起点：模型初始化（torch.manual_seed）与 numpy/random。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_per_class_metrics(cm, out_csv):
    """从混淆矩阵算 per-class 指标并写 CSV。"""
    cm_np = cm.cpu().numpy()                       # [10, 10] int64

    true_count = cm_np.sum(axis=1)                 # 行和：每类真实样本数
    pred_count = cm_np.sum(axis=0)                 # 列和：每类被预测次数
    correct = np.diag(cm_np)                       # 对角线：每类判对数

    recall = np.divide(correct, true_count,
                       out=np.zeros(NUM_CLASSES, dtype=float),
                       where=true_count > 0)
    precision = np.divide(correct, pred_count,
                          out=np.zeros(NUM_CLASSES, dtype=float),
                          where=pred_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(NUM_CLASSES, dtype=float),
                   where=(precision + recall) > 0)

    rows = []
    for i in range(NUM_CLASSES):
        rows.append({
            "class_idx":  i,
            "class_name": IDX_TO_CLASS[i],
            "true_count": int(true_count[i]),
            "pred_count": int(pred_count[i]),
            "correct":    int(correct[i]),
            "recall":     float(recall[i]),
            "precision":  float(precision[i]),
            "f1":         float(f1[i]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(df.to_string(index=False))
    print(f"\noverall acc = {correct.sum() / cm_np.sum():.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(cm_np)

    # 顺便返回几个量，方便外部用
    macro_f1 = float(f1.mean())
    weighted_f1 = float((f1 * true_count).sum() / true_count.sum())
    print(f"macro_f1 = {macro_f1:.4f}  weighted_f1 = {weighted_f1:.4f}")
    return df


