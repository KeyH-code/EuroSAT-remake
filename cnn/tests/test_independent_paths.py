"""01 迁移契约：旧 CSV 原样保留，读图只落在新仓库。"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eurosat_paths import DATA_ROOT, LEGACY_ROOT, MANIFEST, map_manifest_images  # noqa: E402


def test_manifest_identity_and_split():
    # 清单的字节哈希必须与历史 checkpoint 绑定的划分相同。
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == (
        "66730fa4825f2aa75366389b428023d2e50ed93e625b8448f6b9f45a690e54bf"
    )
    assert MANIFEST.read_bytes() == (LEGACY_ROOT / "my_split.csv").read_bytes()
    frame = pd.read_csv(MANIFEST, encoding="utf-8")
    assert frame["split"].value_counts().to_dict() == {"train": 18900, "val": 4050, "test": 4050}

    # 真实像素从独立数据根读取，原始 CSV 内的旧路径永不作为运行时路径。
    mapped = map_manifest_images(frame.head(2))
    for path in mapped["filepath"]:
        image_path = Path(path)
        assert image_path.is_relative_to(DATA_ROOT)
        with Image.open(image_path) as image:
            assert image.size == (64, 64)

