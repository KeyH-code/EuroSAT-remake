"""Copy repeatedly misclassified validation images for manual review.

The source is the fixed seven-best-checkpoint cohort from reports 20 and 21.
Images are copied byte for byte; the original data split is never modified.
"""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "resnet18" / "analysis"
REVIEW = SOURCE / "error_review_20260927_01"
PATTERNS = SOURCE / "prediction_patterns_20260927_01"
DEST = ROOT / "error_samples"
DATA = ROOT / "data" / "EuroSAT_RGB" / "2750"
CLASS_NAMES = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)


def load_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    # Refuse to replace an existing manual-review collection.
    if DEST.exists():
        raise FileExistsError(f"Existing destination would be overwritten: {DEST}")
    experiments = json.loads((REVIEW / "experiment_summary.json").read_text(encoding="utf-8"))
    run_to_alias = {entry["run_id"]: entry["alias"] for entry in experiments}
    if len(run_to_alias) != 7 or len(set(run_to_alias.values())) != 7:
        raise ValueError("Expected seven distinct runs and aliases")
    consensus = load_csv(PATTERNS / "seven_run_correctness.csv")
    by_path = {row["filepath"]: row for row in consensus}
    if len(consensus) != 4050 or len(by_path) != 4050:
        raise ValueError("Expected 4050 unique validation paths")
    errors = load_csv(REVIEW / "errors_all_experiments.csv")
    observations = defaultdict(list)
    unique_predictions = set()
    for row in errors:
        run_id, path = row["run_id"], row["filepath"]
        if run_id not in run_to_alias or path not in by_path:
            raise ValueError("Error source is outside the frozen seven-run validation cohort")
        alias = run_to_alias[run_id]
        if (path, alias) in unique_predictions:
            raise ValueError(f"Duplicate experiment/sample pair: {alias} {path}")
        unique_predictions.add((path, alias))
        if row["true_class"] != by_path[path]["true_class"]:
            raise ValueError(f"True-label disagreement: {path}")
        if row["pred_class"] not in CLASS_NAMES or row["pred_class"] == row["true_class"]:
            raise ValueError(f"Unexpected wrong class: {path}")
        observations[path].append({"alias": alias, "run_id": run_id, "epoch": int(row["epoch"]),
                                   "pred_class": row["pred_class"], "confidence": float(row["confidence"])})
    for path, row in by_path.items():
        aliases = {observation["alias"] for observation in observations[path]}
        expected_aliases = set(filter(None, row["wrong_run_aliases"].split("|")))
        if len(aliases) != int(row["wrong_runs"]) or aliases != expected_aliases:
            raise ValueError(f"Consensus mismatch: {path}")
    selected = [row for row in consensus if int(row["wrong_runs"]) >= 2]
    if len(selected) != 98:
        raise ValueError(f"Unexpected number of repeatedly wrong images: {len(selected)}")

    # Check every source, label, and output name before creating the destination.
    planned = []
    used_names = set()
    for row in selected:
        source = Path(row["filepath"])
        true_class = row["true_class"]
        if source.resolve().parent != (DATA / true_class).resolve() or not source.is_file():
            raise ValueError(f"Image path outside the expected true-class directory: {source}")
        wrong = observations[str(source)]
        counts = Counter(item["pred_class"] for item in wrong)
        largest = max(counts.values())
        tied = [name for name in CLASS_NAMES if counts[name] == largest]
        current = next((item["pred_class"] for item in wrong if item["alias"] == "current_best"), None)
        chosen = current if current in tied else tied[0]
        # Windows forbids '>' in filenames. U+2192 is the closest usable arrow.
        output_name = f"{true_class}→{chosen}__{source.name}"
        destination = DEST / true_class / output_name
        if destination in used_names:
            raise ValueError(f"Output-name collision: {destination}")
        used_names.add(destination)
        planned.append({"source": source, "destination": destination, "true_class": true_class,
                        "chosen_wrong_class": chosen, "counts": counts, "observations": wrong,
                        "consensus": row, "tie": len(tied) > 1})

    DEST.mkdir(exist_ok=False)
    for class_name in CLASS_NAMES:
        (DEST / class_name).mkdir()
    index = []
    for item in planned:
        source, destination = item["source"], item["destination"]
        shutil.copyfile(source, destination)
        source_hash = file_sha256(source)
        if file_sha256(destination) != source_hash:
            raise ValueError(f"Copied bytes differ: {destination}")
        row = item["consensus"]
        index.append({"export_filepath": str(destination), "source_filepath": str(source),
                      "source_filename": source.name, "true_class": item["true_class"],
                      "filename_wrong_class": item["chosen_wrong_class"],
                      "wrong_runs": int(row["wrong_runs"]), "total_runs": 7,
                      "wrong_run_aliases": row["wrong_run_aliases"],
                      "wrong_class_counts": json.dumps(dict(sorted(item["counts"].items())), ensure_ascii=False),
                      "all_wrong_predictions": json.dumps(sorted(item["observations"], key=lambda r: r["alias"]), ensure_ascii=False),
                      "current_best_correct": row["current_best_correct"],
                      "wrong_class_frequency_tie": item["tie"], "sha256": source_hash})
    index.sort(key=lambda r: (CLASS_NAMES.index(r["true_class"]), r["source_filename"]))
    write_csv(DEST / "index.csv", index)
    by_class = Counter(row["true_class"] for row in index)
    by_frequency = Counter(row["wrong_runs"] for row in index)
    summary = {"definition": "val image misclassified by at least 2 of the fixed 7 best checkpoints from reports 20-21",
               "selected_unique_images": len(index), "source_error_observations": len(errors),
               "by_true_class": {name: by_class[name] for name in CLASS_NAMES},
               "by_wrong_run_count": dict(sorted(by_frequency.items())),
               "different_wrong_class_images": sum(len(row["counts"]) > 1 for row in planned),
               "most_frequent_wrong_class_ties": sum(row["tie"] for row in planned),
               "current_best_correct_among_selected": sum(row["current_best_correct"] == "True" for row in index),
               "naming": "<true_class>→<chosen_wrong_class>__<original_filename>",
               "wrong_class_choice": "most frequent among wrong predictions; tie goes to current_best if present, else fixed class order",
               "source_files": [str(REVIEW / "errors_all_experiments.csv"), str(PATTERNS / "seven_run_correctness.csv")],
               "no_test_images_used": True, "original_images_unchanged": True,
               "copy_bytes_match_source": True}
    (DEST / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 多次误判样本（人工查看）", "",
             "来源：报告20–21固定七组best checkpoint在val上的预测；至少两组误判入选，共98张不同原图。",
             "每个真实类别一个子目录，图片逐字节复制自data/，原图不变。没有读取test或重划分。", "",
             "Windows文件名不能含有 `>`，故用 Unicode 箭头 `→` 表示请求中的 `->`。",
             "格式：`真实类别→最常见误判类别__原文件名.jpg`；后缀保留原文件名以防同一类别对的图片互相覆盖。",
             "若最常见误判类别并列，优先选当前最佳模型的误判类别（若在并列组中），否则按固定类别顺序。",
             "一张图片只复制一次；若它在不同实验中被误判为不同类别，所有类别及次数见 `index.csv`。", "",
             "`index.csv`：原图与副本绝对路径、七组误判次数、每组预测和文件SHA256。",
             "`summary.json`：数量、各类别计数、命名口径。", ""]
    (DEST / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
