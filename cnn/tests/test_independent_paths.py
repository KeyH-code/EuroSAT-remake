"""01 迁移契约：清单原样保留，读图只落在新仓库，且不依赖历史快照。"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eurosat_paths import DATA_ROOT, MANIFEST, map_manifest_images  # noqa: E402


# 清单字节哈希必须与历史 checkpoint 绑定的划分相同；这个常量是划分契约本身。
MANIFEST_SHA256 = "66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf"


def test_manifest_identity_and_split():
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA256
    frame = pd.read_csv(MANIFEST, encoding="utf-8")
    assert list(frame.columns) == ["filepath", "label", "label_idx", "split"]
    assert frame["split"].value_counts().to_dict() == {"train": 18900, "val": 4050, "test": 4050}

    # 真实像素从独立数据根读取，原始 CSV 内的旧路径永不作为运行时路径。
    mapped = map_manifest_images(frame.head(2))
    for path in mapped["filepath"]:
        image_path = Path(path)
        assert image_path.is_relative_to(DATA_ROOT)
        with Image.open(image_path) as image:
            assert image.size == (64, 64)


def test_manifest_path_is_the_only_manifest_the_code_can_reach():
    """data/ 下只有一份清单；快照副本属冻结档案，不在代码可达路径中。"""
    assert MANIFEST.is_file()
    assert sorted(path.name for path in MANIFEST.parent.iterdir()) == ["my_split.csv"]
    assert MANIFEST.is_relative_to(ROOT / "data")
