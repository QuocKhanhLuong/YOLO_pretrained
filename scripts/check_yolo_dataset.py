#!/usr/bin/env python3
"""Validate a prepared YOLO dataset directory."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def read_classes(classes_path: Path, errors: list[str]) -> list[str]:
    """Read classes.txt and append validation errors instead of raising."""
    if not classes_path.exists():
        errors.append(f"Missing classes.txt: {classes_path}")
        return []

    try:
        names = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        errors.append(f"Could not read classes.txt: {exc}")
        return []

    if not names:
        errors.append(f"classes.txt has no class names: {classes_path}")
    return names


def is_finite_float(value: float) -> bool:
    return math.isfinite(value)


def validate_label_file(label_path: Path, num_classes: int, display_path: str) -> list[str]:
    """Validate one YOLO label file and return human-readable errors."""
    errors: list[str] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"{display_path}: could not read label file: {exc}"]

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"{display_path}:{line_number}: expected 5 values, found {len(parts)}"
            )
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"{display_path}:{line_number}: class_id is not an integer")
            continue

        if class_id < 0 or class_id >= num_classes:
            errors.append(
                f"{display_path}:{line_number}: class_id {class_id} outside [0, {num_classes - 1}]"
            )

        parsed_values: list[float] = []
        for value_name, raw_value in zip(
            ("x_center", "y_center", "width", "height"), parts[1:]
        ):
            try:
                value = float(raw_value)
            except ValueError:
                errors.append(f"{display_path}:{line_number}: {value_name} is not a float")
                parsed_values.append(float("nan"))
                continue
            if not is_finite_float(value):
                errors.append(f"{display_path}:{line_number}: {value_name} is not finite")
            parsed_values.append(value)

        if len(parsed_values) != 4:
            continue

        x_center, y_center, width, height = parsed_values
        if is_finite_float(x_center) and not 0.0 <= x_center <= 1.0:
            errors.append(f"{display_path}:{line_number}: x_center outside [0, 1]")
        if is_finite_float(y_center) and not 0.0 <= y_center <= 1.0:
            errors.append(f"{display_path}:{line_number}: y_center outside [0, 1]")
        if is_finite_float(width) and not 0.0 < width <= 1.0:
            errors.append(f"{display_path}:{line_number}: width outside (0, 1]")
        if is_finite_float(height) and not 0.0 < height <= 1.0:
            errors.append(f"{display_path}:{line_number}: height outside (0, 1]")

    return errors


def collect_images(images_dir: Path) -> dict[str, list[Path]]:
    """Collect image files by relative stem, preserving duplicate-stem evidence."""
    images_by_stem: dict[str, list[Path]] = {}
    if not images_dir.exists():
        return images_by_stem

    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel_stem = image_path.relative_to(images_dir).with_suffix("").as_posix()
        images_by_stem.setdefault(rel_stem, []).append(image_path)
    return images_by_stem


def collect_labels(labels_dir: Path) -> dict[str, Path]:
    """Collect label files by relative stem."""
    labels_by_stem: dict[str, Path] = {}
    if not labels_dir.exists():
        return labels_by_stem

    for label_path in sorted(labels_dir.rglob("*.txt")):
        rel_stem = label_path.relative_to(labels_dir).with_suffix("").as_posix()
        labels_by_stem[rel_stem] = label_path
    return labels_by_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a prepared YOLO dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Prepared YOLO dataset directory to validate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    errors: list[str] = []
    split_counts: dict[str, dict[str, int]] = {}

    if not dataset_dir.exists():
        print(f"ERROR: dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 1
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset path is not a directory: {dataset_dir}", file=sys.stderr)
        return 1

    if not (dataset_dir / "data.yaml").exists():
        errors.append(f"Missing data.yaml: {dataset_dir / 'data.yaml'}")

    class_names = read_classes(dataset_dir / "classes.txt", errors)
    num_classes = len(class_names)

    for split in SPLITS:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        if not images_dir.exists():
            errors.append(f"Missing images/{split}: {images_dir}")
        if not labels_dir.exists():
            errors.append(f"Missing labels/{split}: {labels_dir}")

        images_by_stem = collect_images(images_dir)
        labels_by_stem = collect_labels(labels_dir)

        duplicate_image_stems = {
            stem: paths for stem, paths in images_by_stem.items() if len(paths) > 1
        }
        for stem, paths in duplicate_image_stems.items():
            rel_paths = ", ".join(path.relative_to(images_dir).as_posix() for path in paths)
            errors.append(f"{split}: duplicate image stem '{stem}': {rel_paths}")

        for stem, paths in images_by_stem.items():
            if stem not in labels_by_stem:
                rel_path = paths[0].relative_to(images_dir).as_posix()
                errors.append(f"{split}: image has no matching label: images/{split}/{rel_path}")

        for stem, label_path in labels_by_stem.items():
            if stem not in images_by_stem:
                rel_path = label_path.relative_to(labels_dir).as_posix()
                errors.append(f"{split}: label has no matching image: labels/{split}/{rel_path}")
                continue
            display_path = f"labels/{split}/{label_path.relative_to(labels_dir).as_posix()}"
            if num_classes > 0:
                errors.extend(validate_label_file(label_path, num_classes, display_path))

        split_counts[split] = {
            "images": sum(len(paths) for paths in images_by_stem.values()),
            "labels": len(labels_by_stem),
            "duplicate_image_stems": len(duplicate_image_stems),
        }

    total_images = sum(counts["images"] for counts in split_counts.values())
    total_labels = sum(counts["labels"] for counts in split_counts.values())

    print(f"Dataset: {dataset_dir}")
    print(f"Classes: {num_classes}")
    for split in SPLITS:
        counts = split_counts.get(split, {"images": 0, "labels": 0, "duplicate_image_stems": 0})
        print(
            f"{split}: images={counts['images']} labels={counts['labels']} "
            f"duplicate_image_stems={counts['duplicate_image_stems']}"
        )
    print(f"Total: images={total_images} labels={total_labels} errors={len(errors)}")

    if errors:
        print("\nFirst 100 errors:")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} more errors not shown")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
