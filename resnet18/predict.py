"""01 ResNet18 无标签图片批量预测，结果写入独立运行目录。"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
from data import CLASS_NAMES, DEVICE, REPO_ROOT  # noqa: E402
from src.checkpoint import load_checkpoint  # noqa: E402
from transforms import make_transforms  # noqa: E402


class UnlabeledImages(Dataset):
    """固定文件名顺序，逐张解码，保证预测行能回溯原图。"""

    def __init__(self, directory, transform):
        self.paths = sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise ValueError(f"没有可预测的图片：{directory}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), str(self.paths[index])


def main():
    parser = argparse.ArgumentParser(description="用 ResNet18 checkpoint 预测图片目录")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, help="默认按时间写入 runs/resnet18/predictions")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, eval_transform = make_transforms(config)  # 预测绝不使用训练随机增强。
    dataset = UnlabeledImages(args.input_dir, eval_transform)
    loader = DataLoader(dataset, batch_size=config["loader"]["batch_size"], shuffle=False)

    # checkpoint 校验清单哈希、类别顺序和模型结构；eval 关闭训练态 BN/Dropout。
    from model import model  # noqa: E402

    model.to(DEVICE)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    if config["transform"] != checkpoint["config"]["transform"]:
        raise ValueError("预测预处理与 checkpoint 训练配置不一致")
    model.eval()
    rows = []
    with torch.no_grad():
        for images, paths in loader:
            probabilities = torch.softmax(model(images.to(DEVICE)), dim=1)
            confidences, indices = probabilities.max(dim=1)
            rows.extend(
                {
                    "filepath": path,
                    "pred_idx": int(index),
                    "pred_class": CLASS_NAMES[int(index)],
                    "confidence": float(confidence),
                }
                for path, index, confidence in zip(paths, indices.cpu(), confidences.cpu(), strict=True)
            )

    if len(rows) != len(dataset):
        raise ValueError("预测条数与输入图片数不一致")
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
    output_csv = args.output_csv or (REPO_ROOT / "runs" / "resnet18" / "predictions" / f"{stamp}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        raise FileExistsError(f"拒绝覆盖已有预测结果：{output_csv}")
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"预测 {len(rows)} 张，输出：{output_csv}")


if __name__ == "__main__":
    main()
