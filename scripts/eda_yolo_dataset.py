#!/usr/bin/env python3
"""Generate EDA reports for YOLO detection datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
SIZE_CATEGORIES = ("small", "medium", "large")
DEFAULT_CLASS_NAMES = ["soldier", "vehicle", "fire"]


def import_matplotlib():
    """Import matplotlib lazily so --help works without plotting dependencies."""
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "yolo_eda_matplotlib_cache"),
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for EDA plots: pip install matplotlib") from exc
    return plt


def import_cv2():
    """Import OpenCV if available."""
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def import_pillow():
    """Import Pillow if available."""
    try:
        from PIL import Image, ImageDraw, ImageStat
    except ImportError:
        return None, None, None
    return Image, ImageDraw, ImageStat


def read_classes(dataset_dir: Path, warnings: list[dict[str, object]]) -> list[str]:
    """Read class names from classes.txt, falling back to the known data_v2.0 names."""
    classes_path = dataset_dir / "classes.txt"
    if not classes_path.exists():
        add_warning(
            warnings,
            "",
            "classes.txt",
            "",
            "missing_classes",
            "classes.txt is missing; using default data_v2.0 class names.",
        )
        return DEFAULT_CLASS_NAMES[:]

    try:
        classes = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        add_warning(
            warnings,
            "",
            "classes.txt",
            "",
            "classes_read_error",
            f"Could not read classes.txt: {exc}; using default data_v2.0 class names.",
        )
        return DEFAULT_CLASS_NAMES[:]

    if not classes:
        add_warning(
            warnings,
            "",
            "classes.txt",
            "",
            "empty_classes",
            "classes.txt has no class names; using default data_v2.0 class names.",
        )
        return DEFAULT_CLASS_NAMES[:]
    return classes


def add_warning(
    warnings: list[dict[str, object]],
    split: str,
    file_path: str,
    line_number: int | str,
    warning_type: str,
    message: str,
    severity: str = "warning",
) -> None:
    warnings.append(
        {
            "severity": severity,
            "split": split,
            "file_path": file_path,
            "line_number": line_number,
            "warning_type": warning_type,
            "message": message,
        }
    )


def collect_images(images_dir: Path) -> dict[str, list[Path]]:
    """Collect supported images by relative stem, preserving duplicate-stem evidence."""
    images: dict[str, list[Path]] = {}
    if not images_dir.exists():
        return images
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = image_path.relative_to(images_dir).with_suffix("").as_posix()
            images.setdefault(stem, []).append(image_path)
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


def safe_relpath(path: Path, base: Path) -> str:
    """Return a POSIX relative path for Markdown links."""
    return Path(os.path.relpath(path, base)).as_posix()


def manifest_row_count(dataset_dir: Path, warnings: list[dict[str, object]]) -> int | None:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open(encoding="utf-8", newline="") as file_obj:
            rows = list(csv.reader(file_obj))
    except OSError as exc:
        add_warning(
            warnings,
            "",
            "manifest.csv",
            "",
            "manifest_read_error",
            f"Could not read manifest.csv: {exc}",
        )
        return None
    if not rows:
        return 0
    return max(0, len(rows) - 1)


def read_image_info(
    image_path: Path,
    display_path: str,
    split: str,
    cv2_module,
    pil_image,
    pil_stat,
    warnings: list[dict[str, object]],
) -> dict[str, object]:
    """Read cheap image metadata without failing the whole EDA."""
    info: dict[str, object] = {
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "file_size_bytes": None,
        "brightness_mean": None,
        "brightness_std": None,
    }
    try:
        info["file_size_bytes"] = image_path.stat().st_size
    except OSError as exc:
        add_warning(warnings, split, display_path, "", "image_stat_error", f"Could not stat image: {exc}")

    if cv2_module is not None:
        try:
            image = cv2_module.imread(str(image_path), cv2_module.IMREAD_COLOR)
            if image is not None:
                height, width = image.shape[:2]
                info["width"] = int(width)
                info["height"] = int(height)
                info["aspect_ratio"] = float(width / height) if height else None
                gray = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2GRAY)
                mean, std = cv2_module.meanStdDev(gray)
                info["brightness_mean"] = float(mean[0][0])
                info["brightness_std"] = float(std[0][0])
                return info
        except Exception as exc:  # OpenCV can raise codec-specific errors.
            add_warning(warnings, split, display_path, "", "image_read_error", f"cv2 could not read image: {exc}")

    if pil_image is not None:
        try:
            with pil_image.open(image_path) as image:
                width, height = image.size
                info["width"] = int(width)
                info["height"] = int(height)
                info["aspect_ratio"] = float(width / height) if height else None
                if pil_stat is not None:
                    gray = image.convert("L")
                    stats = pil_stat.Stat(gray)
                    info["brightness_mean"] = float(stats.mean[0])
                    info["brightness_std"] = float(stats.stddev[0])
                return info
        except Exception as exc:  # Pillow can raise format-specific errors.
            add_warning(warnings, split, display_path, "", "image_read_error", f"Pillow could not read image: {exc}")

    add_warning(
        warnings,
        split,
        display_path,
        "",
        "image_unreadable",
        "Image size could not be read with cv2 or Pillow.",
    )
    return info


def parse_label_file(
    label_path: Path,
    display_path: str,
    split: str,
    class_names: list[str],
    warnings: list[dict[str, object]],
) -> tuple[list[dict[str, float | int]], bool]:
    """Parse one YOLO label file, returning valid boxes and whether it had content."""
    boxes: list[dict[str, float | int]] = []
    seen_boxes: set[tuple[int, float, float, float, float]] = set()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        add_warning(warnings, split, display_path, "", "label_read_error", f"Could not read label file: {exc}")
        return boxes, False

    has_content = any(line.strip() for line in lines)
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            add_warning(
                warnings,
                split,
                display_path,
                line_number,
                "invalid_label_line",
                f"Expected 5 YOLO values, found {len(parts)}.",
            )
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            add_warning(
                warnings,
                split,
                display_path,
                line_number,
                "invalid_class_id",
                "Class id is not an integer.",
            )
            continue
        if class_id < 0 or class_id >= len(class_names):
            add_warning(
                warnings,
                split,
                display_path,
                line_number,
                "class_id_out_of_range",
                f"Class id {class_id} outside [0, {len(class_names) - 1}].",
            )
            continue

        values: list[float] = []
        valid = True
        for value_name, raw_value in zip(("x_center", "y_center", "width", "height"), parts[1:]):
            try:
                value = float(raw_value)
            except ValueError:
                add_warning(
                    warnings,
                    split,
                    display_path,
                    line_number,
                    "invalid_numeric_value",
                    f"{value_name} is not numeric.",
                )
                valid = False
                value = float("nan")
            if not math.isfinite(value):
                add_warning(
                    warnings,
                    split,
                    display_path,
                    line_number,
                    "non_finite_value",
                    f"{value_name} is not finite.",
                )
                valid = False
            values.append(value)
        if not valid:
            continue

        x_center, y_center, width, height = values
        if not 0.0 <= x_center <= 1.0:
            add_warning(warnings, split, display_path, line_number, "x_out_of_range", "x_center outside [0, 1].")
            valid = False
        if not 0.0 <= y_center <= 1.0:
            add_warning(warnings, split, display_path, line_number, "y_out_of_range", "y_center outside [0, 1].")
            valid = False
        if not 0.0 < width <= 1.0:
            add_warning(warnings, split, display_path, line_number, "width_out_of_range", "width outside (0, 1].")
            valid = False
        if not 0.0 < height <= 1.0:
            add_warning(warnings, split, display_path, line_number, "height_out_of_range", "height outside (0, 1].")
            valid = False
        if not valid:
            continue

        area = width * height
        aspect_ratio = width / height
        key = (class_id, round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6))
        if key in seen_boxes:
            add_warning(
                warnings,
                split,
                display_path,
                line_number,
                "duplicate_box",
                "Duplicate box in the same label file.",
            )
        seen_boxes.add(key)

        if area > 0.8:
            add_warning(
                warnings,
                split,
                display_path,
                line_number,
                "extremely_large_box",
                f"Normalized box area {area:.6f} is greater than 0.8.",
            )
        if area < 0.0001:
            add_warning(
                warnings,
                split,
                display_path,
                line_number,
                "extremely_tiny_box",
                f"Normalized box area {area:.6f} is less than 0.0001.",
            )
        if aspect_ratio > 10.0 or aspect_ratio < 0.1:
            add_warning(
                warnings,
                split,
                display_path,
                line_number,
                "unusual_aspect_ratio",
                f"Box aspect ratio {aspect_ratio:.4f} is outside [0.1, 10].",
            )

        boxes.append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "area": area,
                "aspect_ratio": aspect_ratio,
            }
        )
    return boxes, has_content


def quantiles(values: list[float]) -> dict[str, float | None]:
    """Return compact stats for a numeric vector."""
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}

    def pick(percent: float) -> float:
        index = round((len(clean) - 1) * percent)
        return clean[index]

    return {
        "count": len(clean),
        "min": clean[0],
        "p25": pick(0.25),
        "median": statistics.median(clean),
        "p75": pick(0.75),
        "max": clean[-1],
        "mean": statistics.fmean(clean),
    }


def pct(part: int | float, total: int | float) -> float:
    return (float(part) / float(total) * 100.0) if total else 0.0


def size_category(area: float, small_threshold: float, medium_threshold: float) -> str:
    if area < small_threshold:
        return "small"
    if area < medium_threshold:
        return "medium"
    return "large"


def build_summary(
    dataset_dir: Path,
    small_area_threshold: float,
    medium_area_threshold: float,
) -> dict[str, object]:
    """Build the full EDA summary from a YOLO dataset directory."""
    warnings: list[dict[str, object]] = []
    class_names = read_classes(dataset_dir, warnings)
    manifest_rows = manifest_row_count(dataset_dir, warnings)
    cv2_module = import_cv2()
    pil_image, pil_draw, pil_stat = import_pillow()

    image_rows: list[dict[str, object]] = []
    box_rows: list[dict[str, object]] = []
    split_summary: dict[str, dict[str, object]] = {}
    parsed_labels_by_split: dict[str, dict[str, tuple[list[dict[str, float | int]], bool]]] = {}
    class_counts_by_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    image_class_counts_by_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    size_counts = Counter({category: 0 for category in SIZE_CATEGORIES})
    size_counts_by_class: dict[str, Counter[str]] = {
        class_name: Counter({category: 0 for category in SIZE_CATEGORIES}) for class_name in class_names
    }
    image_boxes: dict[str, list[dict[str, object]]] = defaultdict(list)

    for split in SPLITS:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        if not images_dir.exists():
            add_warning(
                warnings,
                split,
                f"images/{split}",
                "",
                "missing_images_dir",
                f"Missing images directory: {images_dir}",
            )
        if not labels_dir.exists():
            add_warning(
                warnings,
                split,
                f"labels/{split}",
                "",
                "missing_labels_dir",
                f"Missing labels directory: {labels_dir}",
            )

        images = collect_images(images_dir)
        labels = collect_labels(labels_dir)
        parsed_labels: dict[str, tuple[list[dict[str, float | int]], bool]] = {}
        empty_label_files = 0
        for stem, label_path in labels.items():
            display_label = f"labels/{split}/{label_path.relative_to(labels_dir).as_posix()}"
            boxes, has_content = parse_label_file(label_path, display_label, split, class_names, warnings)
            parsed_labels[stem] = (boxes, has_content)
            if not has_content:
                empty_label_files += 1
        parsed_labels_by_split[split] = parsed_labels

        duplicate_image_stems = 0
        for stem, paths in images.items():
            if len(paths) <= 1:
                continue
            duplicate_image_stems += 1
            rel_paths = ", ".join(f"images/{split}/{path.relative_to(images_dir).as_posix()}" for path in paths)
            add_warning(
                warnings,
                split,
                f"images/{split}/{stem}",
                "",
                "duplicate_image_stem",
                f"Multiple images share the same relative stem: {rel_paths}",
            )

        missing_label_files = 0
        for stem, paths in images.items():
            if stem in labels:
                continue
            missing_label_files += len(paths)
            for image_path in paths:
                display_image = f"images/{split}/{image_path.relative_to(images_dir).as_posix()}"
                add_warning(
                    warnings,
                    split,
                    display_image,
                    "",
                    "missing_label",
                    "Image has no matching label file.",
                )

        for stem, label_path in labels.items():
            if stem in images:
                continue
            display_label = f"labels/{split}/{label_path.relative_to(labels_dir).as_posix()}"
            add_warning(
                warnings,
                split,
                display_label,
                "",
                "orphan_label",
                "Label file has no matching image.",
            )

        objects_per_image: list[int] = []
        split_instances = 0
        split_image_count = sum(len(paths) for paths in images.values())

        for stem, paths in images.items():
            label_boxes, _ = parsed_labels.get(stem, ([], False))
            for image_index, image_path in enumerate(paths):
                display_image = f"images/{split}/{image_path.relative_to(images_dir).as_posix()}"
                image_key = f"{split}/{image_path.relative_to(images_dir).as_posix()}"
                image_info = read_image_info(
                    image_path,
                    display_image,
                    split,
                    cv2_module,
                    pil_image,
                    pil_stat,
                    warnings,
                )
                width = image_info["width"]
                height = image_info["height"]
                image_class_ids: set[int] = set()

                for box in label_boxes:
                    class_id = int(box["class_id"])
                    class_name = class_names[class_id]
                    area = float(box["area"])
                    category = size_category(area, small_area_threshold, medium_area_threshold)
                    pixel_box_width = float(box["width"]) * int(width) if width else None
                    pixel_box_height = float(box["height"]) * int(height) if height else None
                    pixel_box_area = (
                        pixel_box_width * pixel_box_height
                        if pixel_box_width is not None and pixel_box_height is not None
                        else None
                    )
                    row = {
                        "split": split,
                        "image_path": display_image,
                        "class_id": class_id,
                        "class_name": class_name,
                        "x_center": box["x_center"],
                        "y_center": box["y_center"],
                        "width": box["width"],
                        "height": box["height"],
                        "area": area,
                        "aspect_ratio": box["aspect_ratio"],
                        "size_category": category,
                        "image_width": width,
                        "image_height": height,
                        "pixel_width": pixel_box_width,
                        "pixel_height": pixel_box_height,
                        "pixel_area": pixel_box_area,
                    }
                    box_rows.append(row)
                    image_boxes[image_key].append(row)
                    class_counts_by_split[split][class_name] += 1
                    size_counts[category] += 1
                    size_counts_by_class[class_name][category] += 1
                    image_class_ids.add(class_id)
                    split_instances += 1

                for class_id in image_class_ids:
                    image_class_counts_by_split[split][class_names[class_id]] += 1

                object_count = len(label_boxes)
                objects_per_image.append(object_count)
                image_rows.append(
                    {
                        "split": split,
                        "image_path": display_image,
                        "image_key": image_key,
                        "label_path": f"labels/{split}/{stem}.txt" if stem in labels else "",
                        "has_label": stem in labels,
                        "object_count": object_count,
                        "width": width,
                        "height": height,
                        "aspect_ratio": image_info["aspect_ratio"],
                        "file_size_bytes": image_info["file_size_bytes"],
                        "brightness_mean": image_info["brightness_mean"],
                        "brightness_std": image_info["brightness_std"],
                        "duplicate_stem_index": image_index if len(paths) > 1 else "",
                    }
                )

        split_summary[split] = {
            "images": split_image_count,
            "label_files": len(labels),
            "empty_label_files": empty_label_files,
            "missing_label_files": missing_label_files,
            "orphan_label_files": len(set(labels) - set(images)),
            "duplicate_image_stems": duplicate_image_stems,
            "instances": split_instances,
            "avg_instances_per_image": statistics.fmean(objects_per_image) if objects_per_image else 0.0,
            "median_instances_per_image": statistics.median(objects_per_image) if objects_per_image else 0.0,
        }

    total_images = sum(int(split_summary[split]["images"]) for split in SPLITS)
    total_labels = sum(int(split_summary[split]["label_files"]) for split in SPLITS)
    total_instances = len(box_rows)
    class_counts_overall = {
        class_name: sum(class_counts_by_split[split][class_name] for split in SPLITS)
        for class_name in class_names
    }
    image_class_counts_overall = {
        class_name: sum(image_class_counts_by_split[split][class_name] for split in SPLITS)
        for class_name in class_names
    }
    split_ratios = {
        split: {
            "image_ratio_percent": pct(int(split_summary[split]["images"]), total_images),
            "instance_ratio_percent": pct(int(split_summary[split]["instances"]), total_instances),
        }
        for split in SPLITS
    }
    class_percentages = {
        class_name: pct(count, total_instances)
        for class_name, count in class_counts_overall.items()
    }

    return {
        "dataset": str(dataset_dir),
        "dataset_name": dataset_dir.name,
        "class_names": class_names,
        "num_classes": len(class_names),
        "manifest_rows": manifest_rows,
        "small_area_threshold": small_area_threshold,
        "medium_area_threshold": medium_area_threshold,
        "total_images": total_images,
        "total_labels": total_labels,
        "total_instances": total_instances,
        "split_summary": split_summary,
        "split_ratios": split_ratios,
        "class_counts_overall": class_counts_overall,
        "class_counts_by_split": {
            split: {class_name: class_counts_by_split[split][class_name] for class_name in class_names}
            for split in SPLITS
        },
        "image_class_counts_overall": image_class_counts_overall,
        "image_class_counts_by_split": {
            split: {class_name: image_class_counts_by_split[split][class_name] for class_name in class_names}
            for split in SPLITS
        },
        "class_percentages": class_percentages,
        "size_counts": dict(size_counts),
        "size_counts_by_class": {
            class_name: {category: size_counts_by_class[class_name][category] for category in SIZE_CATEGORIES}
            for class_name in class_names
        },
        "image_rows": image_rows,
        "box_rows": box_rows,
        "image_boxes": dict(image_boxes),
        "warnings": warnings,
        "image_stats": {
            "width": quantiles([float(row["width"]) for row in image_rows if row["width"]]),
            "height": quantiles([float(row["height"]) for row in image_rows if row["height"]]),
            "aspect_ratio": quantiles([float(row["aspect_ratio"]) for row in image_rows if row["aspect_ratio"]]),
            "file_size_kb": quantiles(
                [float(row["file_size_bytes"]) / 1024.0 for row in image_rows if row["file_size_bytes"]]
            ),
            "brightness_mean": quantiles(
                [float(row["brightness_mean"]) for row in image_rows if row["brightness_mean"] is not None]
            ),
        },
        "box_stats": {
            "width": quantiles([float(row["width"]) for row in box_rows]),
            "height": quantiles([float(row["height"]) for row in box_rows]),
            "area": quantiles([float(row["area"]) for row in box_rows]),
            "aspect_ratio": quantiles([float(row["aspect_ratio"]) for row in box_rows]),
            "pixel_area": quantiles(
                [float(row["pixel_area"]) for row in box_rows if row["pixel_area"] is not None]
            ),
        },
        "instances_per_image_stats": {
            split: quantiles(
                [float(row["object_count"]) for row in image_rows if row["split"] == split]
            )
            for split in SPLITS
        },
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def plot_bar(plt, names: list[str], values: list[int | float], title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_grouped_bars(
    plt,
    groups: list[str],
    series_names: list[str],
    values_by_series: list[list[int | float]],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    positions = list(range(len(groups)))
    width = 0.8 / max(1, len(series_names))
    plt.figure(figsize=(11, 6))
    for index, series in enumerate(series_names):
        offsets = [position - 0.4 + width / 2 + index * width for position in positions]
        plt.bar(offsets, values_by_series[index], width=width, label=series)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(positions, groups, rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_plots(summary: dict[str, object], output_dir: Path) -> list[Path]:
    """Generate all requested matplotlib PNG plots."""
    plt = import_matplotlib()
    artifacts: list[Path] = []
    class_names: list[str] = summary["class_names"]  # type: ignore[assignment]
    class_counts_overall: dict[str, int] = summary["class_counts_overall"]  # type: ignore[assignment]
    class_counts_by_split: dict[str, dict[str, int]] = summary["class_counts_by_split"]  # type: ignore[assignment]
    split_summary: dict[str, dict[str, object]] = summary["split_summary"]  # type: ignore[assignment]
    box_rows: list[dict[str, object]] = summary["box_rows"]  # type: ignore[assignment]
    image_rows: list[dict[str, object]] = summary["image_rows"]  # type: ignore[assignment]
    size_counts_by_class: dict[str, dict[str, int]] = summary["size_counts_by_class"]  # type: ignore[assignment]

    path = output_dir / "class_distribution_overall.png"
    plot_bar(
        plt,
        class_names,
        [class_counts_overall.get(class_name, 0) for class_name in class_names],
        "Class Distribution Overall",
        "Instances",
        path,
    )
    artifacts.append(path)

    path = output_dir / "class_distribution_by_split.png"
    plot_grouped_bars(
        plt,
        class_names,
        list(SPLITS),
        [[class_counts_by_split[split].get(class_name, 0) for class_name in class_names] for split in SPLITS],
        "Class Distribution by Split",
        "Instances",
        path,
    )
    artifacts.append(path)

    path = output_dir / "split_image_count.png"
    plot_bar(
        plt,
        list(SPLITS),
        [int(split_summary[split]["images"]) for split in SPLITS],
        "Image Count by Split",
        "Images",
        path,
    )
    artifacts.append(path)

    path = output_dir / "split_instance_count.png"
    plot_bar(
        plt,
        list(SPLITS),
        [int(split_summary[split]["instances"]) for split in SPLITS],
        "Instance Count by Split",
        "Instances",
        path,
    )
    artifacts.append(path)

    path = output_dir / "split_class_distribution.png"
    plot_grouped_bars(
        plt,
        list(SPLITS),
        class_names,
        [[class_counts_by_split[split].get(class_name, 0) for split in SPLITS] for class_name in class_names],
        "Split Class Distribution",
        "Instances",
        path,
    )
    artifacts.append(path)

    areas = [float(row["area"]) for row in box_rows]
    if areas:
        path = output_dir / "bbox_area_distribution.png"
        plt.figure(figsize=(10, 5))
        plt.hist(areas, bins=40)
        plt.title("Normalized Bounding Box Area Distribution")
        plt.xlabel("Normalized area")
        plt.ylabel("Boxes")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

        path = output_dir / "bbox_area_distribution_by_class.png"
        plt.figure(figsize=(10, 5))
        for class_name in class_names:
            values = [float(row["area"]) for row in box_rows if row["class_name"] == class_name]
            if values:
                plt.hist(values, bins=30, alpha=0.45, label=class_name)
        plt.title("Normalized Box Area by Class")
        plt.xlabel("Normalized area")
        plt.ylabel("Boxes")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

        path = output_dir / "bbox_width_height_scatter.png"
        plt.figure(figsize=(7, 6))
        plt.scatter(
            [float(row["width"]) for row in box_rows],
            [float(row["height"]) for row in box_rows],
            alpha=0.45,
            s=18,
        )
        plt.title("Normalized Box Width vs Height")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

        path = output_dir / "bbox_aspect_ratio_distribution.png"
        plt.figure(figsize=(10, 5))
        plt.hist([float(row["aspect_ratio"]) for row in box_rows], bins=40)
        plt.title("Bounding Box Aspect Ratio Distribution")
        plt.xlabel("Width / height")
        plt.ylabel("Boxes")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

        pixel_areas = [float(row["pixel_area"]) for row in box_rows if row["pixel_area"] is not None]
        if pixel_areas:
            path = output_dir / "bbox_pixel_area_distribution.png"
            plt.figure(figsize=(10, 5))
            plt.hist(pixel_areas, bins=40)
            plt.title("Pixel Bounding Box Area Distribution")
            plt.xlabel("Pixel area")
            plt.ylabel("Boxes")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            artifacts.append(path)

        path = output_dir / "bbox_center_heatmap.png"
        plt.figure(figsize=(7, 6))
        plt.hist2d(
            [float(row["x_center"]) for row in box_rows],
            [float(row["y_center"]) for row in box_rows],
            bins=30,
            range=[[0, 1], [0, 1]],
        )
        plt.title("Bounding Box Center Heatmap")
        plt.xlabel("x center")
        plt.ylabel("y center")
        plt.colorbar(label="Boxes")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

        path = output_dir / "bbox_size_category_distribution.png"
        plot_grouped_bars(
            plt,
            class_names,
            list(SIZE_CATEGORIES),
            [[size_counts_by_class[class_name][category] for class_name in class_names] for category in SIZE_CATEGORIES],
            "Bounding Box Size Categories by Class",
            "Boxes",
            path,
        )
        artifacts.append(path)

    readable_images = [row for row in image_rows if row["width"] and row["height"]]
    if readable_images:
        path = output_dir / "image_resolution_distribution.png"
        plt.figure(figsize=(7, 6))
        plt.scatter(
            [float(row["width"]) for row in readable_images],
            [float(row["height"]) for row in readable_images],
            alpha=0.5,
            s=20,
        )
        plt.title("Image Resolution Distribution")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

        path = output_dir / "image_aspect_ratio_distribution.png"
        plt.figure(figsize=(10, 5))
        plt.hist([float(row["aspect_ratio"]) for row in readable_images if row["aspect_ratio"]], bins=30)
        plt.title("Image Aspect Ratio Distribution")
        plt.xlabel("Width / height")
        plt.ylabel("Images")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

    file_sizes = [float(row["file_size_bytes"]) / 1024.0 for row in image_rows if row["file_size_bytes"]]
    if file_sizes:
        path = output_dir / "image_file_size_distribution.png"
        plt.figure(figsize=(10, 5))
        plt.hist(file_sizes, bins=30)
        plt.title("Image File Size Distribution")
        plt.xlabel("File size (KB)")
        plt.ylabel("Images")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

    brightness = [float(row["brightness_mean"]) for row in image_rows if row["brightness_mean"] is not None]
    if brightness:
        path = output_dir / "image_brightness_distribution.png"
        plt.figure(figsize=(10, 5))
        plt.hist(brightness, bins=30)
        plt.title("Image Brightness Mean Distribution")
        plt.xlabel("Brightness mean")
        plt.ylabel("Images")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

    instance_counts = [int(row["object_count"]) for row in image_rows]
    if instance_counts:
        path = output_dir / "instances_per_image_distribution.png"
        max_count = max(instance_counts)
        bins = list(range(max_count + 2))
        plt.figure(figsize=(10, 5))
        plt.hist(instance_counts, bins=bins, align="left", rwidth=0.85)
        plt.title("Instances per Image")
        plt.xlabel("Objects per image")
        plt.ylabel("Images")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(path)

    return artifacts


def draw_sample_images(
    dataset_dir: Path,
    output_dir: Path,
    summary: dict[str, object],
    sample_images: int,
    seed: int,
    include_empty_labels: bool,
) -> list[Path]:
    """Save sampled images with ground-truth boxes drawn on top."""
    if sample_images <= 0:
        return []
    sample_dir = output_dir / "sample_labeled_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    cv2_module = import_cv2()
    pil_image, pil_draw, _ = import_pillow()
    if cv2_module is None and pil_image is None:
        warnings: list[dict[str, object]] = summary["warnings"]  # type: ignore[assignment]
        add_warning(
            warnings,
            "",
            "sample_labeled_images",
            "",
            "sample_render_unavailable",
            "cv2 and Pillow are unavailable; sample images were not rendered.",
        )
        return []

    rng = random.Random(seed)
    image_rows: list[dict[str, object]] = summary["image_rows"]  # type: ignore[assignment]
    image_boxes: dict[str, list[dict[str, object]]] = summary["image_boxes"]  # type: ignore[assignment]
    class_names: list[str] = summary["class_names"]  # type: ignore[assignment]
    selected_keys: list[str] = []

    for class_name in class_names:
        candidates = [
            row["image_key"]
            for row in image_rows
            if any(box["class_name"] == class_name for box in image_boxes.get(str(row["image_key"]), []))
        ]
        if candidates:
            rng.shuffle(candidates)
            selected_keys.append(str(candidates[0]))

    pool = [
        str(row["image_key"])
        for row in image_rows
        if include_empty_labels or int(row["object_count"]) > 0
    ]
    rng.shuffle(pool)
    for key in pool:
        selected_keys.append(key)

    deduped_keys: list[str] = []
    seen: set[str] = set()
    for key in selected_keys:
        if key in seen:
            continue
        seen.add(key)
        deduped_keys.append(key)
        if len(deduped_keys) >= sample_images:
            break

    row_by_key = {str(row["image_key"]): row for row in image_rows}
    rendered: list[Path] = []
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]
    warnings: list[dict[str, object]] = summary["warnings"]  # type: ignore[assignment]

    for key in deduped_keys:
        row = row_by_key.get(key)
        if not row:
            continue
        image_path = dataset_dir / str(row["image_path"])
        output_name = key.replace("/", "__").replace("\\", "__")
        output_path = sample_dir / f"{Path(output_name).stem}.png"
        boxes = image_boxes.get(key, [])
        try:
            if cv2_module is not None:
                image = cv2_module.imread(str(image_path), cv2_module.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("cv2.imread returned None")
                img_h, img_w = image.shape[:2]
                for box in boxes:
                    class_id = int(box["class_id"])
                    color = colors[class_id % len(colors)]
                    x_center = float(box["x_center"]) * img_w
                    y_center = float(box["y_center"]) * img_h
                    box_w = float(box["width"]) * img_w
                    box_h = float(box["height"]) * img_h
                    x1 = max(0, int(round(x_center - box_w / 2)))
                    y1 = max(0, int(round(y_center - box_h / 2)))
                    x2 = min(img_w - 1, int(round(x_center + box_w / 2)))
                    y2 = min(img_h - 1, int(round(y_center + box_h / 2)))
                    cv2_module.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2_module.putText(
                        image,
                        str(box["class_name"]),
                        (x1, max(12, y1 - 4)),
                        cv2_module.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2_module.LINE_AA,
                    )
                cv2_module.imwrite(str(output_path), image)
            else:
                with pil_image.open(image_path) as source:
                    image = source.convert("RGB")
                draw = pil_draw.Draw(image)
                img_w, img_h = image.size
                for box in boxes:
                    class_id = int(box["class_id"])
                    color = colors[class_id % len(colors)]
                    x_center = float(box["x_center"]) * img_w
                    y_center = float(box["y_center"]) * img_h
                    box_w = float(box["width"]) * img_w
                    box_h = float(box["height"]) * img_h
                    x1 = max(0, int(round(x_center - box_w / 2)))
                    y1 = max(0, int(round(y_center - box_h / 2)))
                    x2 = min(img_w - 1, int(round(x_center + box_w / 2)))
                    y2 = min(img_h - 1, int(round(y_center + box_h / 2)))
                    draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                    draw.text((x1, max(0, y1 - 12)), str(box["class_name"]), fill=color)
                image.save(output_path)
            rendered.append(output_path)
        except Exception as exc:
            add_warning(
                warnings,
                str(row["split"]),
                str(row["image_path"]),
                "",
                "sample_render_error",
                f"Could not render sample image: {exc}",
            )

    grid_path = output_dir / "sample_labeled_images_grid.png"
    if rendered and pil_image is not None:
        try:
            create_image_grid(rendered, grid_path, pil_image)
            rendered.append(grid_path)
        except Exception as exc:
            add_warning(
                warnings,
                "",
                "sample_labeled_images_grid.png",
                "",
                "sample_grid_error",
                f"Could not render sample image grid: {exc}",
            )
    return rendered


def create_image_grid(image_paths: list[Path], output_path: Path, pil_image) -> None:
    max_images = min(16, len(image_paths))
    selected = image_paths[:max_images]
    thumb_w, thumb_h = 240, 180
    cols = min(4, max_images)
    rows = math.ceil(max_images / cols)
    canvas = pil_image.new("RGB", (cols * thumb_w, rows * thumb_h), (255, 255, 255))
    for index, image_path in enumerate(selected):
        with pil_image.open(image_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((thumb_w, thumb_h))
            x = (index % cols) * thumb_w + (thumb_w - thumb.width) // 2
            y = (index // cols) * thumb_h + (thumb_h - thumb.height) // 2
            canvas.paste(thumb, (x, y))
    canvas.save(output_path)


def warning_summary_rows(warnings: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter(str(row["warning_type"]) for row in warnings)
    return [{"warning_type": warning_type, "count": count} for warning_type, count in sorted(counts.items())]


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index, _ in enumerate(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4g}"
    return str(value)


def build_insights(summary: dict[str, object]) -> dict[str, list[str]]:
    class_names: list[str] = summary["class_names"]  # type: ignore[assignment]
    class_counts: dict[str, int] = summary["class_counts_overall"]  # type: ignore[assignment]
    split_summary: dict[str, dict[str, object]] = summary["split_summary"]  # type: ignore[assignment]
    split_ratios: dict[str, dict[str, float]] = summary["split_ratios"]  # type: ignore[assignment]
    class_counts_by_split: dict[str, dict[str, int]] = summary["class_counts_by_split"]  # type: ignore[assignment]
    size_counts: dict[str, int] = summary["size_counts"]  # type: ignore[assignment]
    size_counts_by_class: dict[str, dict[str, int]] = summary["size_counts_by_class"]  # type: ignore[assignment]
    image_rows: list[dict[str, object]] = summary["image_rows"]  # type: ignore[assignment]
    warnings: list[dict[str, object]] = summary["warnings"]  # type: ignore[assignment]
    total_instances = int(summary["total_instances"])
    total_images = int(summary["total_images"])

    insights: dict[str, list[str]] = {
        "key_findings": [],
        "class_distribution": [],
        "split": [],
        "bbox": [],
        "image_quality": [],
        "instances": [],
        "label_quality": [],
        "metrics": [],
        "recommendations": [],
    }

    if total_instances == 0:
        insights["key_findings"].append("No valid labeled instances were found; training metrics would be unreliable.")
        insights["recommendations"].append("Fix dataset labeling or paths before training or comparing YOLO runs.")
        return insights

    counts = list(class_counts.values())
    largest = max(counts) if counts else 0
    smallest = min(counts) if counts else 0
    if smallest == 0 or (smallest > 0 and largest / smallest > 2.0):
        insights["class_distribution"].append(
            "Class imbalance is present; the largest class has more than 2x the smallest class or a class is absent."
        )
        insights["key_findings"].append("Class imbalance can suppress recall and class-level AP for rare classes.")
        insights["recommendations"].append("Add or oversample examples for underrepresented classes before tuning more epochs.")

    few_threshold = max(20, int(total_instances * 0.01))
    rare_classes = [class_name for class_name, count in class_counts.items() if count < few_threshold]
    if rare_classes:
        insights["class_distribution"].append(
            f"Very low-instance classes detected ({', '.join(rare_classes)}); class-level recall/AP may be unstable."
        )
        insights["recommendations"].append("Audit rare classes first; decide whether to add data, merge labels, or filter the task.")

    train_total = sum(class_counts_by_split["train"].values())
    for split in ("val", "test"):
        split_total = sum(class_counts_by_split[split].values())
        if train_total == 0 or split_total == 0:
            continue
        drifted: list[str] = []
        for class_name in class_names:
            train_pct = class_counts_by_split["train"].get(class_name, 0) / train_total
            split_pct = class_counts_by_split[split].get(class_name, 0) / split_total
            if abs(train_pct - split_pct) > 0.20:
                drifted.append(class_name)
        if drifted:
            insights["split"].append(
                f"{split} class distribution differs from train for {', '.join(drifted)}; validation/test may measure a harder distribution."
            )
            insights["key_findings"].append(f"{split} split drift can make recall or AP look worse than train-like validation.")
            insights["recommendations"].append("Review whether split-by-video/frame grouping intentionally made val/test harder.")
    insights["split"].append(
        "If data_v2.0 is split by video or scene, harder validation/test metrics can be realistic rather than a pure training failure."
    )

    small_count = int(size_counts.get("small", 0))
    small_percent = pct(small_count, total_instances)
    if small_percent >= 30.0:
        insights["bbox"].append(
            f"{small_percent:.1f}% of boxes are small by the configured threshold; tiny UAV objects can lower recall and mAP50-95."
        )
        insights["key_findings"].append("Many small boxes make missed detections and localization errors more likely.")
        insights["recommendations"].append("Consider higher imgsz, tiling/cropping, or class/size-aware sampling for small objects.")
    for class_name in class_names:
        class_total = sum(size_counts_by_class[class_name].values())
        if class_total and pct(size_counts_by_class[class_name].get("small", 0), class_total) >= 50.0:
            insights["bbox"].append(
                f"{class_name} is dominated by small boxes; this class may need higher input size, tiling, or label audit."
            )
    insights["bbox"].append(
        "A large gap between mAP50 and mAP50-95 often points to localization difficulty from tiny boxes or inconsistent box tightness."
    )

    resolutions = {
        (row["width"], row["height"])
        for row in image_rows
        if row["width"] is not None and row["height"] is not None
    }
    aspect_ratios = {
        round(float(row["aspect_ratio"]), 3)
        for row in image_rows
        if row["aspect_ratio"] is not None
    }
    if len(resolutions) > 1 or len(aspect_ratios) > 1:
        insights["image_quality"].append(
            "Multiple image resolutions or aspect ratios exist; resizing/padding can change tiny-object pixel size."
        )
    small_files = [
        row
        for row in image_rows
        if row["file_size_bytes"] is not None and float(row["file_size_bytes"]) < 50 * 1024
    ]
    if total_images and pct(len(small_files), total_images) >= 20.0:
        insights["image_quality"].append(
            "Many images are below 50 KB; compression or low quality may hurt recall and localization."
        )
        insights["recommendations"].append("Inspect low-file-size samples and consider image preprocessing or a mixed enhanced dataset.")
    insights["image_quality"].append("Low-quality UAV imagery can reduce both objectness recall and tight localization.")

    one_object_images = [row for row in image_rows if int(row["object_count"]) == 1]
    many_object_images = [row for row in image_rows if int(row["object_count"]) >= 10]
    if total_images and pct(len(one_object_images), total_images) >= 60.0:
        insights["instances"].append("Most images have one object; the model may have limited multi-object examples.")
    if total_images and pct(len(many_object_images), total_images) >= 10.0:
        insights["instances"].append(
            "Many images contain crowded scenes; audit NMS, confidence threshold, and small-object false negatives."
        )

    warning_counts = Counter(str(row["warning_type"]) for row in warnings)
    invalid_count = sum(
        warning_counts[name]
        for name in (
            "invalid_label_line",
            "invalid_class_id",
            "class_id_out_of_range",
            "invalid_numeric_value",
            "non_finite_value",
            "x_out_of_range",
            "y_out_of_range",
            "width_out_of_range",
            "height_out_of_range",
        )
    )
    suspicious_count = sum(
        warning_counts[name]
        for name in (
            "duplicate_box",
            "extremely_large_box",
            "extremely_tiny_box",
            "unusual_aspect_ratio",
            "missing_label",
            "orphan_label",
        )
    )
    if invalid_count:
        insights["label_quality"].append(
            f"{invalid_count} invalid label warnings were found; metrics and training behavior may be unreliable."
        )
        insights["recommendations"].append("Fix invalid label rows before using data_v2.0 for final model comparison.")
    if suspicious_count:
        insights["label_quality"].append(
            f"{suspicious_count} suspicious label/pairing warnings were found; manual label audit is recommended."
        )
        insights["recommendations"].append("Review duplicate, tiny, unusual-aspect, missing-label, and orphan-label samples manually.")

    insights["metrics"].append(
        "High YOLO11m precision with low recall can mean the model is conservative, confidence is too high, or tiny/hard objects are underrepresented."
    )
    insights["metrics"].append(
        "mAP50 improving while mAP50-95 stays much lower usually means detections are roughly correct but boxes are not tight enough at stricter IoU thresholds."
    )
    insights["recommendations"].append("Tune inference confidence threshold during review if recall remains low.")
    insights["recommendations"].append("Run controlled low-augmentation experiments if class confusion or tiny-object distortion appears.")

    for key, values in insights.items():
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        insights[key] = deduped
    return insights


def write_outputs(
    summary: dict[str, object],
    output_dir: Path,
    plot_paths: list[Path],
    sample_paths: list[Path],
    run_dir: Path | None,
    pred_dir: Path | None,
    max_warnings: int,
) -> None:
    """Write CSV, JSON, and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, object]] = summary["image_rows"]  # type: ignore[assignment]
    box_rows: list[dict[str, object]] = summary["box_rows"]  # type: ignore[assignment]
    warnings: list[dict[str, object]] = summary["warnings"]  # type: ignore[assignment]
    class_names: list[str] = summary["class_names"]  # type: ignore[assignment]
    split_summary: dict[str, dict[str, object]] = summary["split_summary"]  # type: ignore[assignment]
    split_ratios: dict[str, dict[str, float]] = summary["split_ratios"]  # type: ignore[assignment]
    class_counts_overall: dict[str, int] = summary["class_counts_overall"]  # type: ignore[assignment]
    class_counts_by_split: dict[str, dict[str, int]] = summary["class_counts_by_split"]  # type: ignore[assignment]
    image_class_counts: dict[str, int] = summary["image_class_counts_overall"]  # type: ignore[assignment]
    class_percentages: dict[str, float] = summary["class_percentages"]  # type: ignore[assignment]
    size_counts: dict[str, int] = summary["size_counts"]  # type: ignore[assignment]
    size_counts_by_class: dict[str, dict[str, int]] = summary["size_counts_by_class"]  # type: ignore[assignment]
    insights = build_insights(summary)

    summary_for_json = dict(summary)
    summary_for_json.pop("image_boxes", None)
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary_for_json, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "image_summary.csv",
        image_rows,
        [
            "split",
            "image_path",
            "label_path",
            "has_label",
            "object_count",
            "width",
            "height",
            "aspect_ratio",
            "file_size_bytes",
            "brightness_mean",
            "brightness_std",
            "duplicate_stem_index",
        ],
    )
    write_csv(
        output_dir / "box_summary.csv",
        box_rows,
        [
            "split",
            "image_path",
            "class_id",
            "class_name",
            "x_center",
            "y_center",
            "width",
            "height",
            "area",
            "aspect_ratio",
            "size_category",
            "image_width",
            "image_height",
            "pixel_width",
            "pixel_height",
            "pixel_area",
        ],
    )
    class_rows = [
        {
            "class_id": index,
            "class_name": class_name,
            "instances": class_counts_overall[class_name],
            "images_containing_class": image_class_counts[class_name],
            "percentage": class_percentages[class_name],
            **{f"{split}_instances": class_counts_by_split[split][class_name] for split in SPLITS},
        }
        for index, class_name in enumerate(class_names)
    ]
    write_csv(
        output_dir / "class_distribution.csv",
        class_rows,
        ["class_id", "class_name", "instances", "images_containing_class", "percentage"]
        + [f"{split}_instances" for split in SPLITS],
    )
    split_rows = [
        {
            "split": split,
            **split_summary[split],
            **split_ratios[split],
        }
        for split in SPLITS
    ]
    write_csv(
        output_dir / "split_summary.csv",
        split_rows,
        [
            "split",
            "images",
            "label_files",
            "empty_label_files",
            "missing_label_files",
            "orphan_label_files",
            "duplicate_image_stems",
            "instances",
            "avg_instances_per_image",
            "median_instances_per_image",
            "image_ratio_percent",
            "instance_ratio_percent",
        ],
    )
    write_csv(
        output_dir / "label_quality_warnings.csv",
        warnings,
        ["severity", "split", "file_path", "line_number", "warning_type", "message"],
    )

    artifact_paths = [
        output_dir / "eda_report.md",
        output_dir / "eda_summary.json",
        output_dir / "image_summary.csv",
        output_dir / "box_summary.csv",
        output_dir / "class_distribution.csv",
        output_dir / "split_summary.csv",
        output_dir / "label_quality_warnings.csv",
        *plot_paths,
        *sample_paths,
    ]
    report = build_markdown_report(
        summary,
        insights,
        output_dir,
        plot_paths,
        sample_paths,
        artifact_paths,
        run_dir,
        pred_dir,
        max_warnings,
    )
    (output_dir / "eda_report.md").write_text(report, encoding="utf-8")


def build_markdown_report(
    summary: dict[str, object],
    insights: dict[str, list[str]],
    output_dir: Path,
    plot_paths: list[Path],
    sample_paths: list[Path],
    artifact_paths: list[Path],
    run_dir: Path | None,
    pred_dir: Path | None,
    max_warnings: int,
) -> str:
    class_names: list[str] = summary["class_names"]  # type: ignore[assignment]
    split_summary: dict[str, dict[str, object]] = summary["split_summary"]  # type: ignore[assignment]
    split_ratios: dict[str, dict[str, float]] = summary["split_ratios"]  # type: ignore[assignment]
    class_counts_overall: dict[str, int] = summary["class_counts_overall"]  # type: ignore[assignment]
    class_counts_by_split: dict[str, dict[str, int]] = summary["class_counts_by_split"]  # type: ignore[assignment]
    image_class_counts: dict[str, int] = summary["image_class_counts_overall"]  # type: ignore[assignment]
    class_percentages: dict[str, float] = summary["class_percentages"]  # type: ignore[assignment]
    size_counts: dict[str, int] = summary["size_counts"]  # type: ignore[assignment]
    size_counts_by_class: dict[str, dict[str, int]] = summary["size_counts_by_class"]  # type: ignore[assignment]
    warnings: list[dict[str, object]] = summary["warnings"]  # type: ignore[assignment]
    image_rows: list[dict[str, object]] = summary["image_rows"]  # type: ignore[assignment]
    box_rows: list[dict[str, object]] = summary["box_rows"]  # type: ignore[assignment]

    plot_lookup = {path.name: safe_relpath(path, output_dir) for path in plot_paths}
    lines: list[str] = [f"# YOLO Dataset EDA Report: {summary['dataset_name']}", ""]

    lines.extend(
        [
            "## 1. Executive Summary",
            f"- dataset path: `{summary['dataset']}`",
            f"- total images: {summary['total_images']}",
            f"- total instances: {summary['total_instances']}",
            f"- classes: {', '.join(class_names)}",
        ]
    )
    manifest_rows = summary["manifest_rows"]
    if manifest_rows is not None:
        lines.append(f"- manifest.csv rows: {manifest_rows}")
    lines.append("- key findings:")
    for finding in insights["key_findings"] or ["No high-risk dataset findings were triggered by the automatic checks."]:
        lines.append(f"  - {finding}")
    lines.append("")

    lines.extend(["## 2. Dataset Overview", ""])
    lines.append(
        markdown_table(
            [
                "Split",
                "Images",
                "Label files",
                "Empty labels",
                "Missing labels",
                "Instances",
                "Avg inst/img",
                "Median inst/img",
            ],
            [
                [
                    split,
                    split_summary[split]["images"],
                    split_summary[split]["label_files"],
                    split_summary[split]["empty_label_files"],
                    split_summary[split]["missing_label_files"],
                    split_summary[split]["instances"],
                    split_summary[split]["avg_instances_per_image"],
                    split_summary[split]["median_instances_per_image"],
                ]
                for split in SPLITS
            ],
        )
    )
    lines.append("")
    lines.append(
        markdown_table(
            ["Metric", "Value"],
            [
                ["Total images", summary["total_images"]],
                ["Total labels", summary["total_labels"]],
                ["Total instances", summary["total_instances"]],
                ["Number of classes", summary["num_classes"]],
                ["Class names", ", ".join(class_names)],
                ["manifest.csv rows", manifest_rows if manifest_rows is not None else "not available"],
            ],
        )
    )
    lines.append("")

    lines.extend(["## 3. Class Distribution", ""])
    lines.append(
        markdown_table(
            ["Class", "Instances", "Images containing class", "Percent", *[f"{split} instances" for split in SPLITS]],
            [
                [
                    class_name,
                    class_counts_overall[class_name],
                    image_class_counts[class_name],
                    class_percentages[class_name],
                    *[class_counts_by_split[split][class_name] for split in SPLITS],
                ]
                for class_name in class_names
            ],
        )
    )
    lines.append("")
    add_plot(lines, "class_distribution_overall.png", "Class distribution overall", plot_lookup)
    add_plot(lines, "class_distribution_by_split.png", "Class distribution by split", plot_lookup)
    add_bullets(lines, insights["class_distribution"])

    lines.extend(["## 4. Split Analysis", ""])
    lines.append(
        markdown_table(
            ["Split", "Image ratio %", "Instance ratio %"],
            [
                [
                    split,
                    split_ratios[split]["image_ratio_percent"],
                    split_ratios[split]["instance_ratio_percent"],
                ]
                for split in SPLITS
            ],
        )
    )
    lines.append("")
    add_plot(lines, "split_image_count.png", "Split image count", plot_lookup)
    add_plot(lines, "split_instance_count.png", "Split instance count", plot_lookup)
    add_plot(lines, "split_class_distribution.png", "Split class distribution", plot_lookup)
    add_bullets(lines, insights["split"])

    lines.extend(["## 5. Bounding Box Size and Shape Analysis", ""])
    lines.append(
        markdown_table(
            ["Size category", "Count", "Percent"],
            [[category, size_counts.get(category, 0), pct(size_counts.get(category, 0), int(summary["total_instances"]))] for category in SIZE_CATEGORIES],
        )
    )
    lines.append("")
    lines.append(
        markdown_table(
            ["Class", "Small", "Medium", "Large", "Small %", "Medium %", "Large %"],
            [
                [
                    class_name,
                    size_counts_by_class[class_name].get("small", 0),
                    size_counts_by_class[class_name].get("medium", 0),
                    size_counts_by_class[class_name].get("large", 0),
                    pct(
                        size_counts_by_class[class_name].get("small", 0),
                        sum(size_counts_by_class[class_name].values()),
                    ),
                    pct(
                        size_counts_by_class[class_name].get("medium", 0),
                        sum(size_counts_by_class[class_name].values()),
                    ),
                    pct(
                        size_counts_by_class[class_name].get("large", 0),
                        sum(size_counts_by_class[class_name].values()),
                    ),
                ]
                for class_name in class_names
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Small/medium/large thresholds: small `< {summary['small_area_threshold']}`, "
        f"medium `< {summary['medium_area_threshold']}`, large `>= {summary['medium_area_threshold']}` normalized area."
    )
    lines.append("")
    for name, title in [
        ("bbox_area_distribution.png", "Bounding box area distribution"),
        ("bbox_area_distribution_by_class.png", "Bounding box area by class"),
        ("bbox_width_height_scatter.png", "Bounding box width/height scatter"),
        ("bbox_aspect_ratio_distribution.png", "Bounding box aspect ratio distribution"),
        ("bbox_pixel_area_distribution.png", "Bounding box pixel area distribution"),
        ("bbox_center_heatmap.png", "Bounding box center heatmap"),
        ("bbox_size_category_distribution.png", "Bounding box size category distribution"),
    ]:
        add_plot(lines, name, title, plot_lookup)
    lines.append("Box stats:")
    lines.append("```json")
    lines.append(json.dumps(summary["box_stats"], indent=2))
    lines.append("```")
    lines.append("")
    add_bullets(lines, insights["bbox"])

    lines.extend(["## 6. Image Resolution and Quality Proxy", ""])
    lines.append("Image stats:")
    lines.append("```json")
    lines.append(json.dumps(summary["image_stats"], indent=2))
    lines.append("```")
    lines.append("")
    for name, title in [
        ("image_resolution_distribution.png", "Image resolution distribution"),
        ("image_aspect_ratio_distribution.png", "Image aspect ratio distribution"),
        ("image_file_size_distribution.png", "Image file size distribution"),
        ("image_brightness_distribution.png", "Image brightness distribution"),
    ]:
        add_plot(lines, name, title, plot_lookup)
    add_bullets(lines, insights["image_quality"])

    lines.extend(["## 7. Instances per Image", ""])
    lines.append(
        markdown_table(
            ["Split", "Min", "Median", "Mean", "Max"],
            [
                [
                    split,
                    summary["instances_per_image_stats"][split]["min"],  # type: ignore[index]
                    summary["instances_per_image_stats"][split]["median"],  # type: ignore[index]
                    summary["instances_per_image_stats"][split]["mean"],  # type: ignore[index]
                    summary["instances_per_image_stats"][split]["max"],  # type: ignore[index]
                ]
                for split in SPLITS
            ],
        )
    )
    lines.append("")
    zero_images = sum(1 for row in image_rows if int(row["object_count"]) == 0)
    many_images = sum(1 for row in image_rows if int(row["object_count"]) >= 10)
    lines.append(f"- images with 0 objects: {zero_images}")
    lines.append(f"- images with 10+ objects: {many_images}")
    lines.append("")
    add_plot(lines, "instances_per_image_distribution.png", "Instances per image distribution", plot_lookup)
    add_bullets(lines, insights["instances"])

    lines.extend(["## 8. Label Quality Checks", ""])
    lines.append(
        markdown_table(
            ["Warning type", "Count"],
            [[row["warning_type"], row["count"]] for row in warning_summary_rows(warnings)],
        )
        if warnings
        else "No label quality warnings were generated."
    )
    lines.append("")
    if warnings:
        lines.append(f"First {min(max_warnings, len(warnings))} warnings:")
        lines.append(
            markdown_table(
                ["Severity", "Split", "File", "Line", "Type", "Message"],
                [
                    [
                        row["severity"],
                        row["split"],
                        row["file_path"],
                        row["line_number"],
                        row["warning_type"],
                        row["message"],
                    ]
                    for row in warnings[:max_warnings]
                ],
            )
        )
        if len(warnings) > max_warnings:
            lines.append(f"\n{len(warnings) - max_warnings} more warnings are in `label_quality_warnings.csv`.")
    lines.append("")
    add_bullets(lines, insights["label_quality"])

    lines.extend(["## 9. Sample Ground Truth Images", ""])
    grid_path = output_dir / "sample_labeled_images_grid.png"
    if grid_path in sample_paths:
        lines.append(f"![Sample labeled images]({safe_relpath(grid_path, output_dir)})")
    sample_only = [path for path in sample_paths if path.name != "sample_labeled_images_grid.png"]
    if sample_only:
        lines.append("")
        lines.append("Sample image files:")
        for path in sample_only[:40]:
            lines.append(f"- [{path.name}]({safe_relpath(path, output_dir)})")
    else:
        lines.append("No sample labeled images were rendered.")
    lines.append("")

    lines.extend(["## 10. Connection to Model Metrics", ""])
    add_model_context(lines, output_dir, run_dir, pred_dir)
    add_bullets(lines, insights["metrics"])
    if box_rows:
        small_boxes = int(size_counts.get("small", 0))
        lines.append(
            f"- EDA small-object ratio: {pct(small_boxes, int(summary['total_instances'])):.1f}% "
            "of valid boxes are small by the configured normalized-area threshold."
        )
    lines.append(
        "- If a class has fewer or tinier objects than the others, expect lower class-level recall/AP until data or tiling improves."
    )
    lines.append(
        "- If val/test distribution differs from train, lower metrics may reflect a harder or more realistic split instead of only bad optimization."
    )
    lines.append("")

    lines.extend(["## 11. Recommendations", ""])
    add_bullets(lines, insights["recommendations"] or ["No automatic recommendations were triggered."])

    lines.extend(["## 12. Generated Artifacts", ""])
    for path in artifact_paths:
        lines.append(f"- `{safe_relpath(path, output_dir)}`")
    lines.append("")
    return "\n".join(lines)


def add_plot(lines: list[str], name: str, title: str, plot_lookup: dict[str, str]) -> None:
    if name in plot_lookup:
        lines.append(f"![{title}]({plot_lookup[name]})")
        lines.append("")


def add_bullets(lines: list[str], bullets: list[str]) -> None:
    if bullets:
        for bullet in bullets:
            lines.append(f"- {bullet}")
        lines.append("")


def add_model_context(lines: list[str], output_dir: Path, run_dir: Path | None, pred_dir: Path | None) -> None:
    if run_dir is not None:
        lines.append(f"- run-dir: `{safe_relpath(run_dir, output_dir)}` ({'exists' if run_dir.exists() else 'missing'})")
        for artifact in ("results.csv", "args.yaml", "experiment_metadata.json", "weights/best.pt"):
            path = run_dir / artifact
            if path.exists():
                lines.append(f"  - `{safe_relpath(path, output_dir)}`")
    else:
        lines.append("- run-dir: not provided")

    if pred_dir is not None:
        lines.append(f"- pred-dir: `{safe_relpath(pred_dir, output_dir)}` ({'exists' if pred_dir.exists() else 'missing'})")
        if pred_dir.exists():
            prediction_images = [
                path
                for path in sorted(pred_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ][:6]
            prediction_txts = [path for path in pred_dir.rglob("*.txt") if path.is_file()]
            if prediction_txts:
                lines.append(
                    f"- prediction label txt files found: {len(prediction_txts)}; this EDA does not run full TP/FP/FN matching."
                )
            if prediction_images:
                lines.append("")
                lines.append("Prediction samples:")
                for image_path in prediction_images:
                    lines.append(f"![Prediction sample]({safe_relpath(image_path, output_dir)})")
    else:
        lines.append("- pred-dir: not provided")
    lines.append("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA for a prepared YOLO dataset.")
    parser.add_argument("--dataset", required=True, help="YOLO dataset directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for EDA report artifacts.")
    parser.add_argument("--sample-images", type=int, default=30, help="Maximum labeled sample images to render.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample image selection.")
    parser.add_argument(
        "--small-area-threshold",
        type=float,
        default=0.01,
        help="Normalized bbox area threshold for small objects.",
    )
    parser.add_argument(
        "--medium-area-threshold",
        type=float,
        default=0.05,
        help="Normalized bbox area threshold for medium objects.",
    )
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Include empty-label/zero-object images in sample selection.",
    )
    parser.add_argument("--max-warnings", type=int, default=100, help="Maximum warnings to show in the Markdown report.")
    parser.add_argument("--run-dir", help="Optional model run directory to reference in the report.")
    parser.add_argument("--pred-dir", help="Optional prediction output directory to reference in the report.")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plot generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.small_area_threshold <= 0:
        print("ERROR: --small-area-threshold must be > 0", file=sys.stderr)
        return 1
    if args.medium_area_threshold <= args.small_area_threshold:
        print("ERROR: --medium-area-threshold must be greater than --small-area-threshold", file=sys.stderr)
        return 1
    if args.sample_images < 0:
        print("ERROR: --sample-images must be >= 0", file=sys.stderr)
        return 1
    if args.max_warnings < 0:
        print("ERROR: --max-warnings must be >= 0", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 1
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset path is not a directory: {dataset_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = build_summary(dataset_dir, args.small_area_threshold, args.medium_area_threshold)
        plot_paths = [] if args.no_plots else generate_plots(summary, output_dir)
        sample_paths = draw_sample_images(
            dataset_dir,
            output_dir,
            summary,
            args.sample_images,
            args.seed,
            args.include_empty_labels,
        )
        run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
        pred_dir = Path(args.pred_dir).expanduser().resolve() if args.pred_dir else None
        write_outputs(summary, output_dir, plot_paths, sample_paths, run_dir, pred_dir, args.max_warnings)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"EDA report: {output_dir / 'eda_report.md'}")
    print(f"Image summary: {output_dir / 'image_summary.csv'}")
    print(f"Box summary: {output_dir / 'box_summary.csv'}")
    print(f"Class distribution: {output_dir / 'class_distribution.csv'}")
    print(f"Split summary: {output_dir / 'split_summary.csv'}")
    print(f"Label warnings: {output_dir / 'label_quality_warnings.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
