#!/usr/bin/env python3
"""Compare two prepared YOLO dataset versions."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def read_classes(dataset_dir: Path) -> list[str]:
    """Read class names from classes.txt."""
    classes_path = dataset_dir / "classes.txt"
    if not classes_path.exists():
        return []
    return [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def count_images(dataset_dir: Path, split: str) -> int:
    """Count supported image files for a split."""
    images_dir = dataset_dir / "images" / split
    if not images_dir.exists():
        return 0
    return sum(
        1
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_labels(dataset_dir: Path, split: str) -> int:
    """Count label files for a split."""
    labels_dir = dataset_dir / "labels" / split
    if not labels_dir.exists():
        return 0
    return sum(1 for path in labels_dir.rglob("*.txt") if path.is_file())


def count_instances(dataset_dir: Path, class_names: list[str]) -> Counter[int]:
    """Count YOLO class instances by class ID."""
    counts: Counter[int] = Counter()
    for split in SPLITS:
        labels_dir = dataset_dir / "labels" / split
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.rglob("*.txt"):
            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                counts[class_id] += 1
    for class_id in range(len(class_names)):
        counts.setdefault(class_id, 0)
    return counts


def summarize_dataset(dataset_dir: Path) -> dict[str, object]:
    """Collect summary counts for a prepared YOLO dataset."""
    class_names = read_classes(dataset_dir)
    images_per_split = {split: count_images(dataset_dir, split) for split in SPLITS}
    labels_per_split = {split: count_labels(dataset_dir, split) for split in SPLITS}
    instances = count_instances(dataset_dir, class_names)
    total_images = sum(images_per_split.values())
    total_instances = sum(instances.values())
    return {
        "classes": class_names,
        "images_per_split": images_per_split,
        "labels_per_split": labels_per_split,
        "instances": instances,
        "total_images": total_images,
        "total_instances": total_instances,
    }


def format_classes(class_names: list[str]) -> str:
    """Format class names for markdown."""
    if not class_names:
        return "None"
    return "\n".join(f"- {index}: {name}" for index, name in enumerate(class_names))


def format_counts(counts: dict[str, int] | Counter[int], class_names: list[str] | None = None) -> str:
    """Format split or class counts as markdown bullets."""
    if not counts:
        return "None"
    lines: list[str] = []
    for key in sorted(counts, key=str):
        if class_names is not None and isinstance(key, int):
            label = class_names[key] if 0 <= key < len(class_names) else "unknown"
            lines.append(f"- {key}: {label}: {counts[key]}")
        else:
            lines.append(f"- {key}: {counts[key]}")
    return "\n".join(lines)


def format_warnings(warnings: list[str]) -> str:
    """Format warning messages as markdown bullets."""
    if not warnings:
        return "None"
    return "\n".join(f"- {warning}" for warning in warnings)


def split_ratios(images_per_split: dict[str, int]) -> dict[str, float]:
    """Return image split ratios."""
    total = sum(images_per_split.values())
    if total == 0:
        return {split: 0.0 for split in SPLITS}
    return {split: images_per_split.get(split, 0) / total for split in SPLITS}


def split_ratio_warning(old_summary: dict[str, object], new_summary: dict[str, object]) -> str | None:
    """Warn if split ratios differ substantially."""
    old_ratios = split_ratios(old_summary["images_per_split"])  # type: ignore[arg-type]
    new_ratios = split_ratios(new_summary["images_per_split"])  # type: ignore[arg-type]
    max_delta = max(abs(old_ratios[split] - new_ratios[split]) for split in SPLITS)
    if math.isfinite(max_delta) and max_delta > 0.15:
        return f"split ratios differ by up to {max_delta:.3f}"
    return None


def write_report(old_dir: Path, new_dir: Path, output_path: Path) -> None:
    """Write the comparison report."""
    old_summary = summarize_dataset(old_dir)
    new_summary = summarize_dataset(new_dir)
    warnings: list[str] = []
    if old_summary["classes"] != new_summary["classes"]:
        warnings.append("class names differ")
    ratio_warning = split_ratio_warning(old_summary, new_summary)
    if ratio_warning:
        warnings.append(ratio_warning)

    old_total_images = int(old_summary["total_images"])
    new_total_images = int(new_summary["total_images"])
    old_total_instances = int(old_summary["total_instances"])
    new_total_instances = int(new_summary["total_instances"])

    content = f"""# Dataset Comparison: {old_dir.name} vs {new_dir.name}

## Inputs

- Old: `{old_dir}`
- New: `{new_dir}`

## Class Names

### Old

{format_classes(old_summary["classes"])}

### New

{format_classes(new_summary["classes"])}

## Images Per Split

### Old

{format_counts(old_summary["images_per_split"])}

### New

{format_counts(new_summary["images_per_split"])}

## Labels Per Split

### Old

{format_counts(old_summary["labels_per_split"])}

### New

{format_counts(new_summary["labels_per_split"])}

## Instance Count Per Class

### Old

{format_counts(old_summary["instances"], old_summary["classes"])}

### New

{format_counts(new_summary["instances"], new_summary["classes"])}

## Totals

- Old total images: {old_total_images}
- New total images: {new_total_images}
- Delta images: {new_total_images - old_total_images}
- Old total instances: {old_total_instances}
- New total instances: {new_total_instances}
- Delta instances: {new_total_instances - old_total_instances}

## Warnings

{format_warnings(warnings)}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two prepared YOLO dataset versions.")
    parser.add_argument("--old", required=True, help="Old dataset directory.")
    parser.add_argument("--new", required=True, help="New dataset directory.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old_dir = Path(args.old).expanduser().resolve()
    new_dir = Path(args.new).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    for label, dataset_dir in (("old", old_dir), ("new", new_dir)):
        if not dataset_dir.exists():
            print(f"ERROR: {label} dataset does not exist: {dataset_dir}", file=sys.stderr)
            return 1
        if not dataset_dir.is_dir():
            print(f"ERROR: {label} dataset is not a directory: {dataset_dir}", file=sys.stderr)
            return 1

    write_report(old_dir, new_dir, output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
