from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image


def write_image(path: Path, size: tuple[int, int] = (80, 60), color: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (color, color, color))
    image.save(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_mock_dataset(root: Path) -> Path:
    dataset = root / "data_v2.0"
    write_text(dataset / "classes.txt", "soldier\nvehicle\nfire\n")
    write_text(
        dataset / "data.yaml",
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [soldier, vehicle, fire]\n",
    )
    write_text(dataset / "manifest.csv", "image,split\ntrain_a.jpg,train\nval_a.jpg,val\n")

    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)

    write_image(dataset / "images/train/train_a.jpg", (100, 80), 100)
    write_text(
        dataset / "labels/train/train_a.txt",
        "0 0.50 0.50 0.05 0.05\n1 0.25 0.25 0.30 0.20\n",
    )
    write_image(dataset / "images/train/train_empty.jpg", (100, 80), 80)
    write_text(dataset / "labels/train/train_empty.txt", "")
    write_image(dataset / "images/val/val_a.jpg", (120, 90), 150)
    write_text(
        dataset / "labels/val/val_a.txt",
        "2 0.50 0.50 0.02 0.02\n2 0.50 0.50 0.02 0.02\n",
    )
    write_text(dataset / "labels/val/orphan.txt", "1 0.5 0.5 0.2 0.2\n")
    write_image(dataset / "images/test/test_missing_label.jpg", (64, 64), 200)
    return dataset


def test_eda_cli_generates_report_artifacts_and_quality_warnings(tmp_path: Path) -> None:
    dataset = build_mock_dataset(tmp_path)
    output_dir = tmp_path / "reports" / "data_v2_0_eda"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eda_yolo_dataset.py",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--sample-images",
            "4",
            "--seed",
            "7",
            "--include-empty-labels",
            "--max-warnings",
            "20",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    report = output_dir / "eda_report.md"
    warnings_csv = output_dir / "label_quality_warnings.csv"
    assert report.exists()
    assert warnings_csv.exists()

    report_text = report.read_text(encoding="utf-8")
    assert "# YOLO Dataset EDA Report: data_v2.0" in report_text
    assert "## 10. Connection to Model Metrics" in report_text
    assert "tiny" in report_text.lower()
    assert "sample_labeled_images_grid.png" in report_text

    expected_artifacts = [
        "class_distribution_overall.png",
        "bbox_size_category_distribution.png",
        "instances_per_image_distribution.png",
        "sample_labeled_images_grid.png",
    ]
    for artifact in expected_artifacts:
        assert (output_dir / artifact).exists(), artifact

    sample_images = list((output_dir / "sample_labeled_images").glob("*.png"))
    assert sample_images

    with warnings_csv.open(encoding="utf-8", newline="") as file_obj:
        warning_types = {row["warning_type"] for row in csv.DictReader(file_obj)}
    assert {"missing_label", "orphan_label", "duplicate_box"} <= warning_types
