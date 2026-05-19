#!/usr/bin/env python3
"""Generate EDA summaries for a prepared YOLO detection dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def import_pillow():
    """Import Pillow lazily so CLI help works without optional dependencies."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image-size EDA: pip install pillow") from exc
    return Image


def import_matplotlib():
    """Import matplotlib lazily and configure a non-interactive backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plots: pip install matplotlib") from exc
    return plt


def read_classes(dataset_dir: Path) -> list[str]:
    """Read class names from classes.txt."""
    classes_path = dataset_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_path}")
    classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f"classes.txt has no class names: {classes_path}")
    return classes


def collect_images(images_dir: Path) -> dict[str, Path]:
    """Collect supported images by relative stem."""
    images: dict[str, Path] = {}
    if not images_dir.exists():
        return images
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = image_path.relative_to(images_dir).with_suffix("").as_posix()
            images[stem] = image_path
    return images


def collect_labels(labels_dir: Path) -> dict[str, Path]:
    """Collect YOLO label files by relative stem."""
    labels: dict[str, Path] = {}
    if not labels_dir.exists():
        return labels
    for label_path in sorted(labels_dir.rglob("*.txt")):
        stem = label_path.relative_to(labels_dir).with_suffix("").as_posix()
        labels[stem] = label_path
    return labels


def parse_label_file(label_path: Path, class_names: list[str]) -> tuple[list[dict[str, float | int]], list[str]]:
    """Parse a YOLO label file and return boxes plus human-readable errors."""
    boxes: list[dict[str, float | int]] = []
    errors: list[str] = []
    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_number}: expected 5 values, found {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_number}: non-numeric YOLO value")
            continue
        if class_id < 0 or class_id >= len(class_names):
            errors.append(f"{label_path}:{line_number}: class_id {class_id} out of range")
            continue
        if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
            errors.append(f"{label_path}:{line_number}: non-finite box value")
            continue
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{label_path}:{line_number}: normalized box outside YOLO bounds")
            continue
        boxes.append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "area": width * height,
            }
        )
    return boxes, errors


def quantiles(values: list[float]) -> dict[str, float | None]:
    """Return compact quantile stats for a numeric vector."""
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    ordered = sorted(values)

    def pick(percent: float) -> float:
        index = round((len(ordered) - 1) * percent)
        return ordered[index]

    return {
        "min": ordered[0],
        "p25": pick(0.25),
        "median": statistics.median(ordered),
        "p75": pick(0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable field order."""
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_plot(summary: dict[str, object], output_dir: Path) -> list[str]:
    """Generate EDA plots and return written plot paths."""
    plt = import_matplotlib()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    class_names: list[str] = summary["class_names"]  # type: ignore[assignment]
    class_counts: dict[str, dict[str, int]] = summary["class_counts"]  # type: ignore[assignment]
    total_by_class = [sum(split_counts.get(name, 0) for split_counts in class_counts.values()) for name in class_names]

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, total_by_class)
    plt.title("Instances by Class")
    plt.xlabel("Class")
    plt.ylabel("Instances")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = plots_dir / "instances_by_class.png"
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(str(path))

    image_rows: list[dict[str, object]] = summary["image_rows"]  # type: ignore[assignment]
    widths = [row["width"] for row in image_rows if row.get("width")]
    heights = [row["height"] for row in image_rows if row.get("height")]
    if widths and heights:
        plt.figure(figsize=(7, 6))
        plt.scatter(widths, heights, alpha=0.5, s=16)
        plt.title("Image Size Distribution")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = plots_dir / "image_size_scatter.png"
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(str(path))

    box_rows: list[dict[str, object]] = summary["box_rows"]  # type: ignore[assignment]
    box_widths = [row["box_width"] for row in box_rows]
    box_heights = [row["box_height"] for row in box_rows]
    if box_widths and box_heights:
        plt.figure(figsize=(7, 6))
        plt.scatter(box_widths, box_heights, alpha=0.35, s=12)
        plt.title("Normalized Box Width vs Height")
        plt.xlabel("Box width")
        plt.ylabel("Box height")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = plots_dir / "bbox_width_height_scatter.png"
        plt.savefig(path, dpi=160)
        plt.close()
        written.append(str(path))

    return written


def build_summary(dataset_dir: Path) -> dict[str, object]:
    """Build a full EDA summary for a prepared YOLO dataset."""
    Image = import_pillow()
    class_names = read_classes(dataset_dir)
    image_rows: list[dict[str, object]] = []
    box_rows: list[dict[str, object]] = []
    class_counts: dict[str, dict[str, int]] = {split: {name: 0 for name in class_names} for split in SPLITS}
    split_image_counts: dict[str, int] = {}
    split_label_counts: dict[str, int] = {}
    empty_label_counts: dict[str, int] = {}
    missing_labels: list[str] = []
    orphan_labels: list[str] = []
    label_errors: list[str] = []
    objects_per_image: dict[str, list[int]] = defaultdict(list)

    for split in SPLITS:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        images = collect_images(images_dir)
        labels = collect_labels(labels_dir)
        split_image_counts[split] = len(images)
        split_label_counts[split] = len(labels)

        for stem in sorted(set(images) - set(labels)):
            missing_labels.append(f"{split}/{stem}")
        for stem in sorted(set(labels) - set(images)):
            orphan_labels.append(f"{split}/{stem}")

        for stem, image_path in images.items():
            width = height = None
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as exc:  # Pillow raises format-specific exceptions.
                label_errors.append(f"{split}/{stem}: could not read image size: {exc}")

            label_path = labels.get(stem)
            boxes: list[dict[str, float | int]] = []
            if label_path:
                boxes, errors = parse_label_file(label_path, class_names)
                label_errors.extend(errors)
            if label_path and not boxes and label_path.stat().st_size == 0:
                empty_label_counts[split] = empty_label_counts.get(split, 0) + 1

            objects_per_image[split].append(len(boxes))
            image_rows.append(
                {
                    "split": split,
                    "relative_stem": stem,
                    "image_path": image_path.relative_to(dataset_dir).as_posix(),
                    "width": width,
                    "height": height,
                    "aspect_ratio": (width / height) if width and height else None,
                    "objects": len(boxes),
                }
            )

            for box in boxes:
                class_id = int(box["class_id"])
                class_name = class_names[class_id]
                class_counts[split][class_name] += 1
                box_rows.append(
                    {
                        "split": split,
                        "relative_stem": stem,
                        "class_id": class_id,
                        "class_name": class_name,
                        "box_width": box["width"],
                        "box_height": box["height"],
                        "box_area": box["area"],
                    }
                )

    widths = [float(row["width"]) for row in image_rows if row.get("width")]
    heights = [float(row["height"]) for row in image_rows if row.get("height")]
    aspect_ratios = [float(row["aspect_ratio"]) for row in image_rows if row.get("aspect_ratio")]
    box_widths = [float(row["box_width"]) for row in box_rows]
    box_heights = [float(row["box_height"]) for row in box_rows]
    box_areas = [float(row["box_area"]) for row in box_rows]

    return {
        "dataset": str(dataset_dir),
        "class_names": class_names,
        "split_image_counts": split_image_counts,
        "split_label_counts": split_label_counts,
        "empty_label_counts": {split: empty_label_counts.get(split, 0) for split in SPLITS},
        "class_counts": class_counts,
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "label_errors": label_errors,
        "image_size_stats": {
            "width": quantiles(widths),
            "height": quantiles(heights),
            "aspect_ratio": quantiles(aspect_ratios),
        },
        "box_stats": {
            "width": quantiles(box_widths),
            "height": quantiles(box_heights),
            "area": quantiles(box_areas),
        },
        "objects_per_image_stats": {
            split: quantiles([float(value) for value in values])
            for split, values in objects_per_image.items()
        },
        "image_rows": image_rows,
        "box_rows": box_rows,
    }


def write_report(summary: dict[str, object], output_dir: Path, plot_paths: list[str]) -> None:
    """Write Markdown, JSON, and CSV EDA artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    image_rows: list[dict[str, object]] = summary["image_rows"]  # type: ignore[assignment]
    box_rows: list[dict[str, object]] = summary["box_rows"]  # type: ignore[assignment]
    write_csv(
        output_dir / "image_summary.csv",
        image_rows,
        ["split", "relative_stem", "image_path", "width", "height", "aspect_ratio", "objects"],
    )
    write_csv(
        output_dir / "box_summary.csv",
        box_rows,
        ["split", "relative_stem", "class_id", "class_name", "box_width", "box_height", "box_area"],
    )

    class_counts: dict[str, dict[str, int]] = summary["class_counts"]  # type: ignore[assignment]
    class_names: list[str] = summary["class_names"]  # type: ignore[assignment]
    class_lines = ["| Split | " + " | ".join(class_names) + " |", "|---|" + "|".join(["---:"] * len(class_names)) + "|"]
    for split in SPLITS:
        class_lines.append(
            "| "
            + split
            + " | "
            + " | ".join(str(class_counts.get(split, {}).get(name, 0)) for name in class_names)
            + " |"
        )

    content = f"""# YOLO Dataset EDA

Dataset: `{summary["dataset"]}`

## Split Counts

| Split | Images | Labels | Empty Labels |
|---|---:|---:|---:|
"""
    split_image_counts: dict[str, int] = summary["split_image_counts"]  # type: ignore[assignment]
    split_label_counts: dict[str, int] = summary["split_label_counts"]  # type: ignore[assignment]
    empty_label_counts: dict[str, int] = summary["empty_label_counts"]  # type: ignore[assignment]
    for split in SPLITS:
        content += f"| {split} | {split_image_counts.get(split, 0)} | {split_label_counts.get(split, 0)} | {empty_label_counts.get(split, 0)} |\n"

    content += "\n## Class Counts\n\n" + "\n".join(class_lines) + "\n"
    content += "\n## Image Size Stats\n\n```json\n"
    content += json.dumps(summary["image_size_stats"], indent=2)
    content += "\n```\n\n## Bounding Box Stats\n\n```json\n"
    content += json.dumps(summary["box_stats"], indent=2)
    content += "\n```\n\n## Pairing And Label Issues\n\n"
    for key, label in (
        ("missing_labels", "Images missing labels"),
        ("orphan_labels", "Labels missing images"),
        ("label_errors", "Invalid labels or unreadable images"),
    ):
        items: list[str] = summary[key]  # type: ignore[assignment]
        content += f"### {label}\n\n"
        if items:
            for item in items[:100]:
                content += f"- {item}\n"
            if len(items) > 100:
                content += f"- ... {len(items) - 100} more not shown\n"
        else:
            content += "None\n"
        content += "\n"

    content += "## Plots\n\n"
    if plot_paths:
        for path in plot_paths:
            content += f"- `{path}`\n"
    else:
        content += "No plots generated.\n"

    (output_dir / "eda_report.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA for a prepared YOLO dataset.")
    parser.add_argument("--dataset", required=True, help="Prepared YOLO dataset directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for EDA report artifacts.")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plot generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not dataset_dir.exists():
        print(f"ERROR: dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 1
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset path is not a directory: {dataset_dir}", file=sys.stderr)
        return 1

    try:
        summary = build_summary(dataset_dir)
        plot_paths = [] if args.no_plots else maybe_plot(summary, output_dir)
        write_report(summary, output_dir, plot_paths)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"EDA report: {output_dir / 'eda_report.md'}")
    print(f"Image summary: {output_dir / 'image_summary.csv'}")
    print(f"Box summary: {output_dir / 'box_summary.csv'}")
    print(f"JSON summary: {output_dir / 'eda_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
