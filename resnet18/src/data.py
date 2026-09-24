"""01 数据入口：复用已经验证的 EuroSAT 划分，不重新分配样本。"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # 统一使用独立仓库的数据与输出路径。
from eurosat_paths import MANIFEST, map_manifest_images  # noqa: E402
MANIFEST_SHA256 = "66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf"
CLASS_NAMES = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda")

# 这些数值来自既有训练 split，仅供选择预处理时参考；此模块不替学习者决定 transform。
EUROSAT_TRAIN_MEAN = (0.3438910908404892, 0.37992634527006, 0.40740581022786704)
EUROSAT_TRAIN_STD = (0.20255132079555324, 0.13690415097787975, 0.11545699660990988)

CONFIG_PATH = REPO_ROOT / "resnet18" / "configs"

class EuroSATDataset(Dataset):
    """按需读取清单中的 RGB 图像，返回 (transform 后的图像, 整数标签)。"""

    def __init__(self, images, labels, transform):
        self.images = list(images)
        self.labels = list(labels)
        self.transform = transform
        if transform is None:
            raise ValueError("必须显式提供当前实验的 transform")
        if len(self.images) != len(self.labels):
            raise ValueError("图像路径与标签数量不一致")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 路径与 label_idx 已在同一行清单中配对；这里不预先加载整套图像。
        with Image.open(self.images[index]) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), int(self.labels[index])


def load_split_frame(split):
    """读取既有清单中指定的一组；哈希防止无意中换成新划分。"""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"未知 split：{split}")
    actual_sha256 = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if actual_sha256 != MANIFEST_SHA256:
        raise ValueError("my_split.csv 与既有 EuroSAT 划分不一致")

    frame = pd.read_csv(MANIFEST, encoding="utf-8")
    expected_columns = ["filepath", "label", "label_idx", "split"]
    if list(frame.columns) != expected_columns:
        raise ValueError("my_split.csv 列契约发生变化")
    expected_counts = {"train": 18900, "val": 4050, "test": 4050}
    if frame["split"].value_counts().to_dict() != expected_counts:
        raise ValueError("my_split.csv 的 train/val/test 数量发生变化")

    # 保留 CSV 原有顺序。验证集不打乱，逐样本输出才能对应回 filepath。
    selected = frame.loc[frame["split"] == split].reset_index(drop=True)
    return map_manifest_images(selected)


def make_loader(split, transform, batch_size, *, shuffle, seed=None):
    """由固定清单构造 DataLoader；batch 和 transform 由微调方案传入。"""
    if split != "train" and shuffle:
        raise ValueError("验证或测试 loader 必须保持清单顺序")
    frame = load_split_frame(split)
    dataset = EuroSATDataset(frame["filepath"], frame["label_idx"], transform)

    # 独立 Generator 只控制采样顺序，不消耗模型初始化的全局随机状态。
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )


def make_train_val_loaders(train_transform, val_transform, batch_size, seed):
    """训练只组装 train/val；test 留到最终评价，不能参与选模。"""
    train_loader = make_loader("train", train_transform, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader("val", val_transform, batch_size, shuffle=False)
    return train_loader, val_loader
