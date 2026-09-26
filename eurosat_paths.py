"""仓库路径契约：数据、清单与运行目录，不修改原始 CSV。

本模块是路径的唯一出口。清单里记录的旧绝对路径不参与运行时定位，只在内存中按
「类别 + 文件名」映射到 ``data/EuroSAT_RGB/2750/``。
"""

from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "EuroSAT_RGB" / "2750"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "my_split.csv"
RUNS_ROOT = PROJECT_ROOT / "runs"


def local_image_path(recorded_path, label):
    """按类别和文件名定位本仓库图像。"""
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
