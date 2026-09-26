"""ResNet18 训练入口：参数解析与训练主流程都在 ``src/train.py``。"""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from src.train import main  # noqa: E402


if __name__ == "__main__":
    main()
