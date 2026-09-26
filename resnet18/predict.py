"""ResNet18 无标签图片批量预测入口。

数据集与推理主流程在 ``src/inference.py``；本文件只做参数解析与结果落盘。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data import CLASS_NAMES, DEVICE, NUM_CLASSES, REPO_ROOT  # noqa: E402
from inference import UnlabeledImageDataset, predict  # noqa: E402
from transforms import make_transforms  # noqa: E402
sys.path.insert(0, str(PROJECT_DIR))
from src.checkpoint import load_checkpoint  # noqa: E402

PREDICTION_DIR = REPO_ROOT / "runs" / "resnet18" / "predictions"


def main():
    parser = argparse.ArgumentParser(description="用 ResNet18 checkpoint 预测图片目录")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, help="默认按时间写入 runs/resnet18/predictions")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, eval_transform = make_transforms(config)  # 预测绝不使用训练随机增强。
    dataset = UnlabeledImageDataset(args.input_dir, eval_transform)
    loader = DataLoader(dataset, batch_size=config["loader"]["batch_size"], shuffle=False)

    # 当前模型直接定义；延迟导入，缺配置时不会请求预训练权重。
    from model import model  # noqa: E402

    model.to(DEVICE)
    # checkpoint 校验清单哈希、类别顺序与模型结构。
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    if config["transform"] != checkpoint["config"]["transform"]:
        raise ValueError("预测预处理与 checkpoint 训练配置不一致")

    filepaths, pred_indices, confidences = predict(model, loader, DEVICE)
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

    # 与 cnn/predict.py 相同的输出契约：条数、类别范围与置信度范围都要核。
    if len(rows) != len(dataset):
        raise ValueError("预测输出条数与输入图像数不一致")
    if not all(0 <= row["pred_idx"] < NUM_CLASSES for row in rows):
        raise ValueError("预测类别编号超出标签映射范围")
    if not all(0.0 <= row["confidence"] <= 1.0 for row in rows):
        raise ValueError("预测置信度不在 [0,1] 范围内")

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
    output_csv = args.output_csv or (PREDICTION_DIR / f"{stamp}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        raise FileExistsError(f"拒绝覆盖已有预测结果：{output_csv}")
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("device:", torch.cuda.get_device_name(0))
    print("checkpoint:", args.checkpoint)
    print("input images:", len(dataset))
    print("predictions:", len(rows))
    print("output:", output_csv)


if __name__ == "__main__":
    main()
