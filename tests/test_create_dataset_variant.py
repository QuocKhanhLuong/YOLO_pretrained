from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = "\n".join(
    [
        "soldier",
        "vehicle",
        "fire",
        "thermal_soldier",
        "thermal_vehicle",
        "thermal_fire",
    ]
) + "\n"


def write_sample(source: Path, split: str, name: str, label_text: str) -> None:
    image_path = source / "images" / split / f"{name}.jpg"
    label_path = source / "labels" / split / f"{name}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake image bytes")
    label_path.write_text(label_text, encoding="utf-8")


def build_source_dataset(root: Path) -> Path:
    source = root / "data_v3.1"
    (source / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
    (source / "classes.txt").write_text(CLASSES, encoding="utf-8")
    for split in ("train", "val", "test"):
        (source / "images" / split).mkdir(parents=True, exist_ok=True)
        (source / "labels" / split).mkdir(parents=True, exist_ok=True)
    return source


def test_rgb_variant_filters_thermal_and_upsamples_train_fire(tmp_path: Path) -> None:
    source = build_source_dataset(tmp_path)
    output = tmp_path / "data_v3.1_rgb_firex2"

    write_sample(
        source,
        "train",
        "rgb_fire",
        "2 0.50 0.50 0.20 0.20\n4 0.10 0.10 0.05 0.05\n2 bad 0.5 0.1 0.1\n",
    )
    write_sample(source, "train", "rgb_soldier", "0 0.40 0.40 0.10 0.10\n")
    write_sample(source, "train", "thermal_only", "3 0.30 0.30 0.10 0.10\n")
    write_sample(source, "val", "val_fire", "2 0.60 0.60 0.20 0.20\n")
    write_sample(source, "test", "test_vehicle", "1 0.50 0.50 0.30 0.30\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_dataset_variant.py",
            "--source",
            str(source),
            "--output",
            str(output),
            "--mode",
            "rgb",
            "--upsample-class",
            "fire",
            "--upsample-factor",
            "2",
            "--train-only-upsampling",
            "--seed",
            "123",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "classes.txt").read_text(encoding="utf-8") == "soldier\nvehicle\nfire\n"
    assert (output / "labels" / "train" / "rgb_fire.txt").read_text(encoding="utf-8") == (
        "2 0.50 0.50 0.20 0.20\n"
    )
    assert (output / "labels" / "train" / "rgb_fire_upsample1.txt").read_text(
        encoding="utf-8"
    ) == "2 0.50 0.50 0.20 0.20\n"
    assert not (output / "images" / "train" / "thermal_only.jpg").exists()
    assert not (output / "images" / "val" / "val_fire_upsample1.jpg").exists()

    manifest_rows = list(csv.DictReader((output / "manifest.csv").open(encoding="utf-8")))
    assert [row["split"] for row in manifest_rows].count("train") == 3
    assert any(row["is_upsampled"] == "true" for row in manifest_rows)

    report = (output / "variant_report.md").read_text(encoding="utf-8")
    assert "source dataset was not modified" in report
    assert "invalid label lines skipped" in report
    assert "thermal_only" in report


def test_thermal_variant_renames_thermal_classes(tmp_path: Path) -> None:
    source = build_source_dataset(tmp_path)
    output = tmp_path / "data_v3.1_thermal"

    write_sample(source, "train", "thermal_soldier", "3 0.50 0.50 0.20 0.20\n")
    write_sample(source, "val", "thermal_vehicle", "4 0.50 0.50 0.20 0.20\n")
    write_sample(source, "test", "thermal_fire", "5 0.50 0.50 0.20 0.20\n")
    write_sample(source, "test", "rgb_only", "0 0.50 0.50 0.20 0.20\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_dataset_variant.py",
            "--source",
            str(source),
            "--output",
            str(output),
            "--mode",
            "thermal",
            "--rename-thermal",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "classes.txt").read_text(encoding="utf-8") == "soldier\nvehicle\nfire\n"
    assert (output / "labels" / "train" / "thermal_soldier.txt").read_text(
        encoding="utf-8"
    ) == "0 0.50 0.50 0.20 0.20\n"
    assert (output / "labels" / "val" / "thermal_vehicle.txt").read_text(
        encoding="utf-8"
    ) == "1 0.50 0.50 0.20 0.20\n"
    assert (output / "labels" / "test" / "thermal_fire.txt").read_text(
        encoding="utf-8"
    ) == "2 0.50 0.50 0.20 0.20\n"
    assert not (output / "images" / "test" / "rgb_only.jpg").exists()
    assert "path: " in (output / "data.yaml").read_text(encoding="utf-8")
