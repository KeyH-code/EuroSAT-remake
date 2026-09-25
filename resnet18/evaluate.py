"""05 独立验证入口：加载指定 checkpoint，在固定 val split 上评价并导出图表。"""

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))  # 兼容当前学习者模块的同目录导入写法。

from data import DEVICE, REPO_ROOT, make_loader  # noqa: E402
from src.checkpoint import load_checkpoint  # noqa: E402
from src.metrics import evaluate_classifier, save_validation_details  # noqa: E402
from src.visualization import (  # noqa: E402
    save_experiment_visualizations,
    save_validation_visualizations,
)
from transforms import make_transforms  # noqa: E402


OUTPUT_ROOT = REPO_ROOT / "runs" / "resnet18"


def main():
    parser = argparse.ArgumentParser(description="评价 ResNet-18 checkpoint；只使用既有 val split")
    parser.add_argument("--config", type=Path, required=True, help="该实验使用的 JSON 配置")
    parser.add_argument("--checkpoint", type=Path, required=True, help="该实验的 checkpoints/best.pt")
    parser.add_argument("--output-dir", type=Path, help="新评价目录；默认在 runs 下按时间创建")
    args = parser.parse_args()

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    # 只评价本仓库 runs/ 下的产物；历史快照不参与，旧实验按 docs/migration-record.md 重训。
    if not checkpoint_path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("checkpoint 必须位于本仓库 runs/resnet18 下的实验目录")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # 当前 model.py 在导入时创建 ResNet-18；这里延迟导入，--help 不会加载权重。
    from model import model  # noqa: E402

    model.to(DEVICE)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location="cpu")
    if config["transform"] != checkpoint["config"]["transform"]:
        raise ValueError("评价预处理与 checkpoint 训练配置不一致")
    _, val_transform = make_transforms(config)
    val_loader = make_loader(
        "val", val_transform, config["loader"]["batch_size"], shuffle=False
    )
    criterion = nn.CrossEntropyLoss()
    result = evaluate_classifier(model, val_loader, criterion, DEVICE)

    # 历史运行快照只读，任何新评价均写入独立 runs 目录；不会读取 test。
    run_dir = checkpoint_path.parent.parent
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
    output_dir = args.output_dir or (OUTPUT_ROOT / "evaluations" / f"{run_dir.name}_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=False)
    details = save_validation_details(output_dir, result, val_loader.dataset)
    final_plots = save_validation_visualizations(
        output_dir, result["confusion_matrix"], details["per_class"]
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "config": str(config_path),
        "split": "val",
        "val_loss": result["loss"],
        "val_accuracy": result["accuracy"],
        "error_count": details["error_count"],
        "test_split_used": False,
        "validation_plots": final_plots,
    }
    if run_dir.is_relative_to(OUTPUT_ROOT.resolve()) and (run_dir / "metrics.jsonl").is_file():
        summary["visualizations"] = save_experiment_visualizations(run_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
