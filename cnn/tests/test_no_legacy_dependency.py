"""02 解耦守卫：源码与配置不得再引用历史快照。

历史产物（artifacts/legacy_eurosat/）保持冻结、可查阅，但**不能被运行代码读取**。
旧实验要复用就按 docs/migration-record.md 记录的参数重新训练。
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOTS = (ROOT / "cnn", ROOT / "resnet18", ROOT / "tools")
FORBIDDEN = ("legacy_eurosat", "LEGACY_ROOT", "archived_config_path", "deeplearning")


def _code_files():
    for base in (*CODE_ROOTS, ROOT):
        for path in sorted(base.glob("*.py")):
            if path.resolve() == Path(__file__).resolve():
                continue  # 本文件按定义要写出这些禁用词。
            if "__pycache__" in path.parts or "artifacts" in path.parts or "runs" in path.parts:
                continue
            yield path


def test_no_source_file_reads_legacy_artifacts():
    offenders = []
    for path in _code_files():
        text = path.read_text(encoding="utf-8")
        hits = [word for word in FORBIDDEN if word in text]
        if hits:
            offenders.append(f"{path.relative_to(ROOT)} -> {hits}")
    assert not offenders, "源码仍引用历史快照：" + "; ".join(offenders)


def test_legacy_paths_api_is_gone():
    sys.path.insert(0, str(ROOT))
    import eurosat_paths  # noqa: PLC0415

    exported = set(dir(eurosat_paths))
    assert "LEGACY_ROOT" not in exported
    assert "archived_config_path" not in exported
    assert eurosat_paths.PROJECT_ROOT == ROOT
    assert eurosat_paths.MANIFEST.is_file() and eurosat_paths.DATA_ROOT.is_dir()


def test_evaluate_and_predict_require_explicit_checkpoint():
    """CNN 两个推理入口不得再提供指向快照的默认 checkpoint。"""
    for name in ("evaluate.py", "predict.py"):
        text = (ROOT / "cnn" / name).read_text(encoding="utf-8")
        assert "DEFAULT_CHECKPOINT" not in text, f"cnn/{name} 仍有默认 checkpoint"
        assert 'required=True' in text
