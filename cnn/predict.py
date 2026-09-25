"""EuroSAT 无标签目录批量推理入口。

运行示例：
    conda run -n dl-reboot python predict.py --input-dir <图像目录>
"""

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

# 当前先复用已经验证的模型定义；正式拆分 train/evaluate 时再机械迁移到 models.py。
from inference import UnlabeledImageDataset, predict  # noqa: E402
from train_fit_loop import (  # noqa: E402
    CLASS_NAMES,
    MANIFEST,
    NUM_CLASSES,
    EurosatCNN,
)
from transforms import TRAIN_MEAN, TRAIN_STD, build_eval_transform  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT.parent))
from eurosat_paths import RUNS_ROOT  # noqa: E402

DEFAULT_OUTPUT = RUNS_ROOT / "cnn_predictions" / (datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f") + ".csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="本仓库 runs/ 下训练得到的 checkpoint；历史快照不参与预测",
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda")
    dataset = UnlabeledImageDataset(args.input_dir, build_eval_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # checkpoint 先留在 CPU；load_state_dict 会把权重复制到 GPU 模型。
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest_sha256 = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if checkpoint["manifest_sha256"] != manifest_sha256:
        raise ValueError("当前 manifest 与 checkpoint 记录的版本不一致")
    if checkpoint["num_classes"] != NUM_CLASSES:
        raise ValueError("checkpoint 类别数量与当前工程不一致")
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("checkpoint 类别编号映射与当前工程不一致")
    if tuple(checkpoint["normalization_mean"]) != tuple(TRAIN_MEAN):
        raise ValueError("checkpoint 归一化 mean 与当前工程不一致")
    if tuple(checkpoint["normalization_std"]) != tuple(TRAIN_STD):
        raise ValueError("checkpoint 归一化 std 与当前工程不一致")

    model = EurosatCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])

    filepaths, pred_indices, confidences = predict(model, loader, device)
    rows = [
        {
            "filepath": filepath,
            "pred_idx": pred_idx,
            "pred_class": CLASS_NAMES[pred_idx],
            "confidence": confidence,
        }
        for filepath, pred_idx, confidence in zip(
            filepaths, pred_indices, confidences, strict=True
        )
    ]
    if len(rows) != len(dataset):
        raise ValueError("预测输出条数与输入图像数不一致")
    if not all(0 <= row["pred_idx"] < NUM_CLASSES for row in rows):
        raise ValueError("预测类别编号超出标签映射范围")
    if not all(0.0 <= row["confidence"] <= 1.0 for row in rows):
        raise ValueError("预测置信度不在 [0,1] 范围内")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.out_csv.exists():
        raise FileExistsError(f"拒绝覆盖已有预测结果：{args.out_csv}")
    pd.DataFrame(rows).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("device:", torch.cuda.get_device_name(0))
    print("checkpoint:", args.checkpoint)
    print("input images:", len(dataset))
    print("predictions:", len(rows))
    print("output:", args.out_csv)


if __name__ == "__main__":
    main()
