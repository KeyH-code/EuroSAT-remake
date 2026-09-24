"""01 ResNet18 训练入口：委托给原有训练主流程。"""

import sys
from pathlib import Path


# 原训练核心位于 src/train.py；只调整模块搜索路径，不复制训练逻辑。
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from train import main  # noqa: E402


if __name__ == "__main__":
    main()

