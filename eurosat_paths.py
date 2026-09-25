"""01 独立仓库路径与原清单路径映射，不修改历史 CSV。

本模块只描述**新仓库自己**的数据、清单和运行目录。历史产物（checkpoint、旧配置、
旧指标）一律不在这里暴露：旧实验要复用就按 docs/migration-record.md 的参数重训。
"""

from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "EuroSAT_RGB" / "2750"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "my_split.csv"
RUNS_ROOT = PROJECT_ROOT / "runs"


def local_image_path(recorded_path, label):
    """按类别和文件名定位复制图像；绝不回退读取教学项目的旧绝对路径。"""
    original = PureWindowsPath(str(recorded_path))
    if original.parent.name != label or original.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"清单路径与类别不一致：{recorded_path}")
    return DATA_ROOT / label / original.name


def map_manifest_images(frame):
    """保留行顺序和标签，只替换内存中的图片路径。"""
    mapped = frame.copy()
    mapped["filepath"] = [
        str(local_image_path(path, label))
        for path, label in zip(frame["filepath"], frame["label"], strict=True)
    ]
    return mapped
