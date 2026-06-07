from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image


def write_image(path: Path, color: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (80, 60), (color, color, color))
    image.save(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_flat_backup_export(root: Path) -> Path:
    source = root / "back_up_data"
    write_text(
        source / "docker_labeled_manifest.csv",
        "image,label\nsample_0.jpg,sample_0.txt\nsample_1.jpg,sample_1.txt\n",
    )

    for index in range(4):
        write_image(source / "images" / f"sample_{index}.jpg", color=80 + index)
        write_text(
            source / "labels" / f"sample_{index}.txt",
            f"{index % 3} 0.50 0.50 0.20 0.20\n",
        )

    write_image(source / "images" / "missing_label.jpg", color=180)
    write_text(source / "labels" / "orphan_label.txt", "0 0.50 0.50 0.20 0.20\n")
    write_text(source / "labels" / "invalid_label.txt", "99 0.50 0.50 0.20 0.20\n")
    return source


def test_prepare_data_version_accepts_flat_backup_with_cli_classes(tmp_path: Path) -> None:
    source = build_flat_backup_export(tmp_path)
    output = tmp_path / "data" / "versions" / "data_v3.0"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data_version.py",
            "--source",
            str(source),
            "--output",
            str(output),
            "--version",
            "data_v3.0",
            "--class-name",
            "soldier",
            "--class-name",
            "vehicle",
            "--class-name",
            "fire",
            "--train-ratio",
            "0.5",
            "--val-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
            "--seed",
            "42",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "classes.txt").read_text(encoding="utf-8") == "soldier\nvehicle\nfire\n"
    assert (output / "manifest.csv").read_text(encoding="utf-8").startswith("image,label\n")

    split_label_counts = {
        split: len(list((output / "labels" / split).glob("*.txt")))
        for split in ("train", "val", "test")
    }
    assert split_label_counts == {"train": 2, "val": 1, "test": 1}

    report_text = (output / "dataset_report.md").read_text(encoding="utf-8")
    assert "docker_labeled_manifest.csv" in report_text
    assert "missing_label.jpg" in report_text
    assert "orphan_label.txt" in report_text
