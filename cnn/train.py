"""EuroSAT 配置驱动的正式训练入口。"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from experiment_runner import run_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="实验目录中的 config.json；所有产物都绑定到该目录",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从该实验目录的 checkpoints/last.pt 恢复并继续追加记录",
    )
    args = parser.parse_args()
    run_experiment(args.config.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
