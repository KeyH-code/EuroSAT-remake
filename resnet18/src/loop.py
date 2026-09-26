"""ResNet-18 微调的训练循环。

只提供可复用函数，不含 CLI 与编排；调用方是 ``runner.py``。
"""

import torch
from torch import nn

from data import DEVICE


def train_one_epoch(model, train_loader, optimizer, scaler):
    """跑一个 epoch，返回按样本加权的平均 loss 与在线 accuracy。

    两个契约：
    - 冻结 backbone 时，其 BatchNorm 继续使用预训练的 ImageNet 运行统计量，
      因此每轮都把 BN 切到 eval；解冻 layer4 后这一行为保持不变。
    - 前向固定走 FP16 autocast，配合调用方的 GradScaler。
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    criterion = nn.CrossEntropyLoss()
    all_loss = []
    loss_sum = 0
    samples_sum = 0
    correct = 0
    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
        loss_sum += loss.item() * len(x)
        samples_sum += len(x)
        all_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return {
        "train_loss_ave": loss_sum / samples_sum,
        "train_all_loss": all_loss,
        "train_accuracy": correct / samples_sum,
    }
