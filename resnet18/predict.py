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

from architecture import build_resnet18  # noqa: E402
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
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path, help="原训练 checkpoint，需原样划分清单")
    weights.add_argument("--inference-weights", type=Path, help="独立推理权重，默认使用 CPU")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="覆盖默认推理设备")
    parser.add_argument("--output-csv", type=Path, help="默认按时间写入 runs/resnet18/predictions")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, eval_transform = make_transforms(config)  # 预测绝不使用训练随机增强。
    dataset = UnlabeledImageDataset(args.input_dir, eval_transform)
    loader = DataLoader(dataset, batch_size=config["loader"]["batch_size"], shuffle=False)

    if args.inference_weights:
        device = torch.device(args.device or "cpu")
        # 完整模型参数已在推理文件内，构建结构时无需下载初始化权重。
        model = build_resnet18()
        inference_weights = torch.load(args.inference_weights, map_location="cpu", weights_only=True)
        if (
            not isinstance(inference_weights, dict)
            or inference_weights.get("schema_version") != 1
            or inference_weights.get("class_names") != list(CLASS_NAMES)
            or "transform" not in inference_weights
            or "model_state" not in inference_weights
        ):
            raise ValueError("推理权重格式或类别顺序与 EuroSAT ResNet18 不一致")
        recorded_transform = inference_weights["transform"]
        model.load_state_dict(inference_weights["model_state"])
    else:
        device = torch.device(args.device) if args.device else DEVICE
        from model import model  # noqa: E402

        # 原训练 checkpoint 继续校验划分清单、类别顺序和模型结构。
        checkpoint = load_checkpoint(args.checkpoint, model=model, map_location="cpu")
        recorded_transform = checkpoint["config"]["transform"]

    if config["transform"] != recorded_transform:
        raise ValueError("预测预处理与训练配置不一致")
    model.to(device)

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

    print("device:", torch.cuda.get_device_name(device) if device.type == "cuda" else device)
    if args.inference_weights:
        print("inference weights:", args.inference_weights)
    else:
        print("checkpoint:", args.checkpoint)
    print("input images:", len(dataset))
    print("predictions:", len(rows))
    print("output:", output_csv)


if __name__ == "__main__":
    main()
