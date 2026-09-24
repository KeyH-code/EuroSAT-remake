r"""第一段真实训练（骨架由 Agent 提供，两个关键函数留给学习者填写）。

它把 M2-C1 已经做完的两段接起来，只补中间缺的一环：

    my_split.csv ──> EuroSATDataset ──> DataLoader ──> 批次 ──┐
                                                               │
                                     模型 forward ──> loss ──> backward ──> step
                                                               │
                                     val 批次 ──> 推理 ──> 累计 loss / 正确数

要你填的只有两个函数，它们各回答一个问题：
  - `train_one_epoch`：一个 epoch 里，一批数据怎样变成一次参数更新？
  - `evaluate`：判卷时为什么不能更新参数，指标又该按什么口径累计？

三个容易做反的点（属于契约，不是风格问题）：
  1. `model.train()` / `model.eval()` 写在 epoch 循环里、调用函数之前，
     不要写在批循环里。模式是"这一段数据用来学还是用来判卷"的属性，
     不是"这一批"的属性；写在批循环里会让每个批次都切一次，代价白花。
  2. 要搬到卡上的是**每一批数据**，不是数据集。模型只在开始时搬一次。
  3. `predicted == target` 得到逐样本布尔向量；对它求和就是本批正确数，
     直接累计后除以总样本数，不要再乘 batch_size。批均值 loss 才需要先乘
     本批样本数还原为该批 loss 总和，再除以全体验证样本数。

用法（固定读取完整 my_split.csv）：
    conda run -n dl-reboot python src/train_fit_loop.py
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
from eurosat_paths import MANIFEST, RUNS_ROOT, map_manifest_images  # noqa: E402

from transforms import (  # noqa: E402
    TRAIN_MEAN,
    TRAIN_STD,
    build_eval_transform,
    build_train_transform,
)

# 清单是 C1 亲手划分的产物：四列 filepath,label,label_idx,split
PER_CLASS_METRICS_CSV = RUNS_ROOT / "cnn_manual" / "per_class_metrics.csv"
ERROR_SAMPLES_CSV = RUNS_ROOT / "cnn_manual" / "error_samples.csv"
RUNS_DIR = RUNS_ROOT / "cnn_manual"
NUM_CLASSES = 10   # EuroSAT 的 10 个类别，编号规则见 LABELS.md
SEED = 20260920

# EuroSAT 类别名，顺序即 label_idx（和 LABELS.md 一致）
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp", action="store_true", help="启用 autocast + GradScaler")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 last.pt 恢复；--epochs 表示恢复后的目标总轮次",
    )
    args = parser.parse_args()

    # 固定种子：本块只要求"同样的配置跑出同样的初始化与打乱顺序"
    fix_seed(SEED)
    device = torch.device("cuda")
    print("设备:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "| 参数:",
          f"epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"amp={args.amp}")
    print("清单:", MANIFEST)

    # train 用带随机增强的 transform（当前还没有增强，但两条路径已经分开），
    # val 用确定性 transform：同一个样本读两次必须逐位相同。
    train_loader = make_loader("train", build_train_transform(),
                               args.batch_size, shuffle=True, seed=SEED)
    val_loader = make_loader("val", build_eval_transform(),
                             args.batch_size, shuffle=False)
    print(f"train 批次数={len(train_loader)}  val 批次数={len(val_loader)}")

    model = EurosatCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # AMP 开关：不开就是 None，train_one_epoch 走纯 FP32 分支；
    # 开了则由 scaler 负责放大反向、unscale 更新和跳步决策。
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    print("precision:", "AMP (autocast fp16 + GradScaler)" if scaler else "FP32")

    # checkpoint
    last_checkpoint_path = RUNS_DIR / "checkpoints" / "last.pt"
    best_checkpoint_path = RUNS_DIR / "checkpoints" / "best.pt"
    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {}
    manifest_sha256 = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    checkpoint["manifest_sha256"] = manifest_sha256
    checkpoint["model_name"] = "EurosatCNN"
    checkpoint["num_classes"] = NUM_CLASSES
    checkpoint["manifest_path"] = str(MANIFEST)
    checkpoint["class_names"] = CLASS_NAMES
    checkpoint["normalization_mean"] = TRAIN_MEAN
    checkpoint["normalization_std"] = TRAIN_STD

    best_val_acc = float("-inf")
    best_epoch = 0
    start_epoch = 1
    resumed_from_epoch = None
    resumed_from_run_id = None

    # 恢复必须发生在训练循环之前。这里沿用已经绑定到 model 的 optimizer
    # 以及由 --amp 决定是否存在的 scaler，避免恢复到另一套运行对象上。
    if args.resume:
        checkpoint = torch.load(
            last_checkpoint_path,
            weights_only=False,
            # RNG 与 DataLoader generator 的状态必须保持为 CPU ByteTensor；
            # 模型和优化器在 load_state_dict 时会恢复到其运行参数所在设备。
            map_location="cpu",
        )

        # 连续训练要求会改变数据顺序或数值路径的配置保持一致。
        saved_args = checkpoint["args"]
        if saved_args["batch_size"] != args.batch_size:
            raise ValueError(
                "恢复训练时 batch-size 必须与 checkpoint 一致："
                f"saved={saved_args['batch_size']} current={args.batch_size}"
            )
        if saved_args["amp"] != args.amp:
            raise ValueError(
                "恢复训练时 AMP 模式必须与 checkpoint 一致："
                f"saved={saved_args['amp']} current={args.amp}"
            )
        if checkpoint["manifest_sha256"] != manifest_sha256:
            raise ValueError("当前 manifest 与 checkpoint 记录的版本不一致")

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        best_epoch = checkpoint["best_epoch"]
        resumed_from_epoch = checkpoint["epoch"]
        resumed_from_run_id = checkpoint.get("run_id")

        # 最后恢复随机状态，使下一次采样和随机运算接在保存点之后。
        random.setstate(checkpoint["rng_state"]["pythonRandom"])
        np.random.set_state(checkpoint["rng_state"]["numpyRandom"])
        torch.set_rng_state(checkpoint["rng_state"]["torchRandom"])
        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cudaRandom"])
        train_loader.generator.set_state(checkpoint["rng_state"]["loaderRandom"])

        if start_epoch > args.epochs:
            raise ValueError(
                f"checkpoint 已完成 epoch {checkpoint['epoch']}，"
                f"目标总轮次 --epochs={args.epochs} 没有可训练的轮次"
            )
        print(
            f"恢复 checkpoint: epoch={checkpoint['epoch']}，"
            f"将训练 epoch {start_epoch}..{args.epochs}"
        )

    # 每次脚本调用都是一次可审计运行。目录名包含到微秒的本地时间，
    # exist_ok=False 让极少数命名冲突显式失败，不静默覆盖旧证据。
    created_at = datetime.now().astimezone()
    run_id = created_at.strftime("run_%Y%m%dT%H%M%S_%f%z")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.jsonl"

    # 这是运行开始前即可确定的事实快照；Path、device 等对象先转成字符串，
    # 保证内容能被标准 JSON 直接读取，而不依赖 PyTorch。
    run_config = {
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "entrypoint": str(Path(sys.argv[0]).resolve()),
        "resume": args.resume,
        "resumed_from_epoch": resumed_from_epoch,
        "resumed_from_run_id": resumed_from_run_id,
        "resumed_from_checkpoint": str(last_checkpoint_path) if args.resume else None,
        "manifest_path": str(MANIFEST),
        "manifest_sha256": manifest_sha256,
        "model_name": "EurosatCNN",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "normalization_mean": TRAIN_MEAN,
        "normalization_std": TRAIN_STD,
        "seed": SEED,
        "batch_size": args.batch_size,
        # 恢复后以 optimizer 中的实际值为准，避免 CLI 默认值冒充真实学习率。
        "learning_rate": optimizer.param_groups[0]["lr"],
        "target_total_epochs": args.epochs,
        "start_epoch": start_epoch,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "precision": "amp_fp16" if scaler is not None else "fp32",
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "last_checkpoint_path": str(last_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "per_class_metrics_path": str(PER_CLASS_METRICS_CSV),
        "error_samples_path": str(ERROR_SAMPLES_CSV),
    }
    run_config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("运行目录:", run_dir)

    for epoch in range(start_epoch, args.epochs + 1):
        # 峰值按 epoch 单独计量，避免上一轮的历史峰值污染本轮记录。
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        train_loss, train_acc, skipped = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, cm, _,_,_ = evaluate(model, val_loader, criterion, device)
        seconds = time.perf_counter() - start
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 ** 2
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"skipped={skipped} | {seconds:.1f}s "
              f"peak_alloc={peak_mb:.0f}MB peak_reserved={peak_reserved_mb:.0f}MB")
        checkpoint["model_state"] = model.state_dict()
        checkpoint["optimizer_state"] = optimizer.state_dict()
        checkpoint["scaler_state"] = scaler.state_dict() if scaler is not None else None
        checkpoint["seed"] = SEED
        checkpoint["args"] = vars(args)
        checkpoint["epoch"] = epoch

        rng_state = {
            "pythonRandom":random.getstate(),
            "numpyRandom":np.random.get_state(),
            "torchRandom":torch.get_rng_state(),
            "cudaRandom":torch.cuda.get_rng_state_all(),
            "loaderRandom":train_loader.generator.get_state()
        }
        checkpoint["rng_state"] = rng_state
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch

        checkpoint["best_val_acc"] = best_val_acc
        checkpoint["best_epoch"] = best_epoch
        checkpoint["run_id"] = run_id
        checkpoint["run_dir"] = str(run_dir)

        torch.save(checkpoint, last_checkpoint_path)

        if is_best:
            torch.save(checkpoint, best_checkpoint_path)

        # JSON Lines 每行是一条独立 JSON；一轮完成后才追加，适合逐轮读取和追查。
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "seconds": seconds,
            "skipped_steps": skipped,
            "peak_allocated_mb": peak_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "is_best": is_best,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
        }
        with metrics_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")


    # 最后再判一次卷，用最新的模型算 per-class 指标
    val_loss, val_acc, cm, all_labels, all_preds, all_confidences = evaluate(
        model, val_loader, criterion, device
    )

    error_rows = []
    for image,label,pred,confidence in zip(val_loader.dataset.images,all_labels,all_preds,all_confidences,strict=True):
        if pred != label:
            error_rows.append(
                {
                    "filepath": image,
                    "true_idx": label,
                    "true_class": CLASS_NAMES[label],
                    "pred_idx": pred,
                    "pred_class": CLASS_NAMES[pred],
                    "confidence": confidence
                }
            )
    error_rows = sorted(error_rows,key=lambda d: d["confidence"],reverse=True)
    error_columns = [
        "filepath", "true_idx", "true_class",
        "pred_idx", "pred_class", "confidence",
    ]
    dataframe = pd.DataFrame(error_rows, columns=error_columns)
    dataframe.to_csv(ERROR_SAMPLES_CSV, index=False, encoding="utf-8-sig")
    print("错误样本总数：", len(error_rows))
    print("置信度最高的前 20 条：")
    print(dataframe.head(20).to_string(index=False))
    print("path：", ERROR_SAMPLES_CSV)

    save_per_class_metrics(cm, PER_CLASS_METRICS_CSV)

if __name__ == "__main__":
    main()
