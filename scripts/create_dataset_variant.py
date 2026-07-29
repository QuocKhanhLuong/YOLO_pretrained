#!/usr/bin/env python3
"""Create RGB-only or thermal-only YOLO dataset variants from data_v3.1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RGB_CLASSES = ("soldier", "vehicle", "fire")
THERMAL_CLASSES = ("thermal_soldier", "thermal_vehicle", "thermal_fire")


@dataclass(frozen=True)
class LabelBox:
    class_id: int
    line: str


@dataclass(frozen=True)
class VariantSample:
    split: str
    source_image: Path
    source_label: Path | None
    rel_image: Path
    rel_label: Path
    boxes: tuple[LabelBox, ...]
    is_upsampled: bool = False
    upsample_index: int = 0


def read_classes(dataset_dir: Path) -> list[str]:
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


def build_data_yaml(dataset_dir: Path, class_names: list[str]) -> str:
    lines = [
        f"path: {json.dumps(str(dataset_dir.resolve()), ensure_ascii=False)}",
        'train: "images/train"',
        'val: "images/val"',
        'test: "images/test"',
        "",
        "names:",
    ]
    for index, name in enumerate(class_names):
        lines.append(f"  {index}: {json.dumps(name, ensure_ascii=False)}")
    lines.append("")
    return "\n".join(lines)


def class_mapping(source_classes: list[str], mode: str) -> tuple[dict[int, int], list[str], dict[str, str]]:
    if mode == "rgb":
        kept_names = RGB_CLASSES
        output_names = list(RGB_CLASSES)
    else:
        kept_names = THERMAL_CLASSES
        output_names = list(THERMAL_CLASSES)

    missing = [class_name for class_name in kept_names if class_name not in source_classes]
    if missing:
        raise ValueError(
            "source classes.txt is missing required classes for "
            f"{mode} mode: {', '.join(missing)}"
        )

    mapping = {
        source_classes.index(class_name): new_index
        for new_index, class_name in enumerate(kept_names)
    }
    mapping_names = {
        f"{source_id} {source_classes[source_id]}": f"{target_id} {output_names[target_id]}"
        for source_id, target_id in mapping.items()
    }
    return mapping, output_names, mapping_names


def parse_label_line(
    raw_line: str,
    label_path: Path,
    line_number: int,
    source_classes: list[str],
    warnings: list[str],
) -> tuple[int, list[str]] | None:
    line = raw_line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) != 5:
        warnings.append(
            f"{label_path}:{line_number}: invalid label lines skipped; expected 5 values, found {len(parts)}"
        )
        return None

    try:
        class_id = int(parts[0])
    except ValueError:
        warnings.append(f"{label_path}:{line_number}: invalid label lines skipped; class id is not an integer")
        return None

    if class_id < 0 or class_id >= len(source_classes):
        warnings.append(
            f"{label_path}:{line_number}: invalid label lines skipped; class id {class_id} outside "
            f"[0, {len(source_classes) - 1}]"
        )
        return None

    values: list[float] = []
    for value_name, raw_value in zip(("x_center", "y_center", "width", "height"), parts[1:]):
        try:
            value = float(raw_value)
        except ValueError:
            warnings.append(
                f"{label_path}:{line_number}: invalid label lines skipped; {value_name} is not a float"
            )
            return None
        if not math.isfinite(value):
            warnings.append(
                f"{label_path}:{line_number}: invalid label lines skipped; {value_name} is not finite"
            )
            return None
        values.append(value)

    x_center, y_center, width, height = values
    if not 0.0 <= x_center <= 1.0:
        warnings.append(f"{label_path}:{line_number}: invalid label lines skipped; x_center outside [0, 1]")
        return None
    if not 0.0 <= y_center <= 1.0:
        warnings.append(f"{label_path}:{line_number}: invalid label lines skipped; y_center outside [0, 1]")
        return None
    if not 0.0 < width <= 1.0:
        warnings.append(f"{label_path}:{line_number}: invalid label lines skipped; width outside (0, 1]")
        return None
    if not 0.0 < height <= 1.0:
        warnings.append(f"{label_path}:{line_number}: invalid label lines skipped; height outside (0, 1]")
        return None

    return class_id, parts[1:]


def collect_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_for_image(source_dir: Path, split: str, image_path: Path) -> Path:
    rel_stem = image_path.relative_to(source_dir / "images" / split).with_suffix("")
    return source_dir / "labels" / split / rel_stem.with_suffix(".txt")


def remap_label_file(
    label_path: Path | None,
    source_classes: list[str],
    mapping: dict[int, int],
    original_counts: Counter[int],
    warnings: list[str],
) -> tuple[LabelBox, ...]:
    if label_path is None or not label_path.exists():
        return ()

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"{label_path}: could not read label file; skipped image labels: {exc}")
        return ()

    boxes: list[LabelBox] = []
    for line_number, raw_line in enumerate(lines, start=1):
        parsed = parse_label_line(raw_line, label_path, line_number, source_classes, warnings)
        if parsed is None:
            continue
        source_class_id, bbox_values = parsed
        original_counts[source_class_id] += 1
        if source_class_id not in mapping:
            continue
        remapped_class_id = mapping[source_class_id]
        line = " ".join([str(remapped_class_id), *bbox_values])
        boxes.append(LabelBox(class_id=remapped_class_id, line=line))
    return tuple(boxes)


def rel_with_suffix(rel_path: Path, suffix: str, new_extension: str | None = None) -> Path:
    extension = rel_path.suffix if new_extension is None else new_extension
    return rel_path.with_name(f"{rel_path.stem}{suffix}{extension}")


def copy_sample(sample: VariantSample, output_dir: Path, manifest_rows: list[dict[str, str]]) -> bool:
    dest_image = output_dir / "images" / sample.split / sample.rel_image
    dest_label = output_dir / "labels" / sample.split / sample.rel_label

    if dest_image.exists() or dest_label.exists():
        return False

    dest_image.parent.mkdir(parents=True, exist_ok=True)
    dest_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.source_image, dest_image)
    dest_label.write_text(
        "".join(f"{box.line}\n" for box in sample.boxes),
        encoding="utf-8",
    )
    manifest_rows.append(
        {
            "split": sample.split,
            "image": (Path("images") / sample.split / sample.rel_image).as_posix(),
            "label": (Path("labels") / sample.split / sample.rel_label).as_posix(),
            "source_image": str(sample.source_image),
            "source_label": str(sample.source_label or ""),
            "is_upsampled": "true" if sample.is_upsampled else "false",
            "upsample_index": str(sample.upsample_index),
            "classes": " ".join(str(box.class_id) for box in sample.boxes),
        }
    )
    return True


def build_upsampled_samples(
    samples: list[VariantSample],
    output_dir: Path,
    upsample_class_id: int | None,
    upsample_factor: int,
    train_only: bool,
    rng: random.Random,
) -> list[VariantSample]:
    if upsample_class_id is None or upsample_factor <= 1:
        return []

    candidates = [
        sample
        for sample in samples
        if any(box.class_id == upsample_class_id for box in sample.boxes)
        and (sample.split == "train" or not train_only)
    ]
    rng.shuffle(candidates)

    upsampled: list[VariantSample] = []
    reserved_images = {
        (output_dir / "images" / sample.split / sample.rel_image).resolve()
        for sample in samples
    }
    reserved_labels = {
        (output_dir / "labels" / sample.split / sample.rel_label).resolve()
        for sample in samples
    }

    for sample in candidates:
        for duplicate_index in range(1, upsample_factor):
            suffix_index = duplicate_index
            while True:
                suffix = f"_upsample{suffix_index}"
                rel_image = rel_with_suffix(sample.rel_image, suffix)
                rel_label = rel_with_suffix(sample.rel_label, suffix, ".txt")
                dest_image = (output_dir / "images" / sample.split / rel_image).resolve()
                dest_label = (output_dir / "labels" / sample.split / rel_label).resolve()
                if dest_image not in reserved_images and dest_label not in reserved_labels:
                    reserved_images.add(dest_image)
                    reserved_labels.add(dest_label)
                    upsampled.append(
                        VariantSample(
                            split=sample.split,
                            source_image=sample.source_image,
                            source_label=sample.source_label,
                            rel_image=rel_image,
                            rel_label=rel_label,
                            boxes=sample.boxes,
                            is_upsampled=True,
                            upsample_index=suffix_index,
                        )
                    )
                    break
                suffix_index += 1
    return upsampled


def count_output_labels(samples: list[VariantSample]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        counts.update(box.class_id for box in sample.boxes)
    return counts


def split_counts(samples: list[VariantSample]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        split: {"images": 0, "labels": 0, "instances": 0} for split in SPLITS
    }
    for sample in samples:
        counts[sample.split]["images"] += 1
        counts[sample.split]["labels"] += 1
        counts[sample.split]["instances"] += len(sample.boxes)
    return counts


def format_counter(counter: Counter[int], class_names: list[str]) -> str:
    lines = ["| Class ID | Class Name | Count |", "| --- | --- | --- |"]
    max_id = max([*counter.keys(), len(class_names) - 1], default=-1)
    for class_id in range(max_id + 1):
        class_name = class_names[class_id] if class_id < len(class_names) else "unknown"
        lines.append(f"| {class_id} | {class_name} | {counter.get(class_id, 0)} |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    source_dir: Path,
    mode: str,
    mapping_names: dict[str, str],
    original_counts: Counter[int],
    source_classes: list[str],
    output_counts: Counter[int],
    output_names: list[str],
    samples: list[VariantSample],
    upsampled_count: int,
    skipped_empty_count: int,
    warnings: list[str],
) -> None:
    counts = split_counts(samples)
    mapping_lines = [f"- `{source}` -> `{target}`" for source, target in mapping_names.items()]
    split_lines = ["| Split | Images | Label Files | Instances |", "| --- | ---: | ---: | ---: |"]
    for split in SPLITS:
        split_lines.append(
            f"| {split} | {counts[split]['images']} | {counts[split]['labels']} | {counts[split]['instances']} |"
        )

    warning_lines = warnings[:200]
    if len(warnings) > len(warning_lines):
        warning_lines.append(f"... {len(warnings) - len(warning_lines)} more warnings omitted")

    report = [
        f"# Dataset Variant Report: {output_dir.name}",
        "",
        f"- Source dataset: `{source_dir.resolve()}`",
        f"- Output dataset: `{output_dir.resolve()}`",
        f"- Mode: `{mode}`",
        "- Note: source dataset was not modified.",
        "",
        "## Class Mapping",
        "",
        "\n".join(mapping_lines) if mapping_lines else "- None",
        "",
        "## Original Class Counts",
        "",
        format_counter(original_counts, source_classes),
        "",
        "## Output Class Counts",
        "",
        format_counter(output_counts, output_names),
        "",
        "## Split Counts",
        "",
        "\n".join(split_lines),
        "",
        "## Upsampling and Filtering",
        "",
        f"- Upsampled image count: {upsampled_count}",
        f"- Skipped empty-after-filter count: {skipped_empty_count}",
        "",
        "## Warnings",
        "",
        "\n".join(f"- {warning}" for warning in warning_lines) if warning_lines else "- None",
        "",
    ]
    output_dir.joinpath("variant_report.md").write_text("\n".join(report), encoding="utf-8")


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "split",
        "image",
        "label",
        "source_image",
        "source_label",
        "is_upsampled",
        "upsample_index",
        "classes",
    ]
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create RGB-only or thermal-only YOLO dataset variants from data_v3.1."
    )
    parser.add_argument("--source", required=True, help="Source YOLO dataset directory.")
    parser.add_argument("--output", required=True, help="Output variant dataset directory.")
    parser.add_argument("--mode", choices=("rgb", "thermal"), required=True)
    parser.add_argument("--upsample-class", help="Output class name to upsample, for example fire.")
    parser.add_argument("--upsample-factor", type=int, default=1)
    parser.add_argument("--train-only-upsampling", action="store_true")
    parser.add_argument(
        "--rename-thermal",
        action="store_true",
        help="For thermal mode, write classes as soldier/vehicle/fire instead of thermal_* names.",
    )
    parser.add_argument(
        "--keep-thermal-names",
        action="store_true",
        help="For thermal mode, keep thermal_soldier/thermal_vehicle/thermal_fire in classes.txt.",
    )
    parser.add_argument("--keep-empty", action="store_true", help="Keep images whose labels become empty.")
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it exists.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    rng = random.Random(args.seed)
    warnings: list[str] = []

    if args.upsample_factor < 1:
        print("ERROR: --upsample-factor must be >= 1", file=sys.stderr)
        return 1
    if args.rename_thermal and args.keep_thermal_names:
        print("ERROR: use either --rename-thermal or --keep-thermal-names, not both", file=sys.stderr)
        return 1
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"ERROR: source dataset directory does not exist: {source_dir}", file=sys.stderr)
        return 1
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        print(
            "ERROR: source and output directories must be separate, non-overlapping paths.",
            file=sys.stderr,
        )
        return 1
    if output_dir.exists():
        if not args.force:
            print(f"ERROR: output already exists; pass --force to replace it: {output_dir}", file=sys.stderr)
            return 1
        shutil.rmtree(output_dir)

    try:
        source_classes = read_classes(source_dir)
        mapping, output_names, mapping_names = class_mapping(source_classes, args.mode)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.mode == "thermal" and args.rename_thermal:
        output_names = list(RGB_CLASSES)
        mapping_names = {
            f"{source_id} {source_classes[source_id]}": f"{target_id} {output_names[target_id]}"
            for source_id, target_id in mapping.items()
        }

    upsample_class_id: int | None = None
    if args.upsample_class:
        if args.upsample_class not in output_names:
            print(
                f"ERROR: --upsample-class {args.upsample_class!r} is not in output classes: "
                f"{', '.join(output_names)}",
                file=sys.stderr,
            )
            return 1
        upsample_class_id = output_names.index(args.upsample_class)

    samples: list[VariantSample] = []
    original_counts: Counter[int] = Counter()
    skipped_empty_count = 0

    for split in SPLITS:
        images_dir = source_dir / "images" / split
        labels_dir = source_dir / "labels" / split
        if not images_dir.exists():
            warnings.append(f"missing source images directory: {images_dir}")
            continue
        if not labels_dir.exists():
            warnings.append(f"missing source labels directory: {labels_dir}")

        images = collect_images(images_dir)
        labels_seen: set[Path] = set()
        for image_path in images:
            rel_image = image_path.relative_to(images_dir)
            label_path = label_for_image(source_dir, split, image_path)
            if label_path.exists():
                labels_seen.add(label_path.resolve())
            else:
                warnings.append(f"{image_path}: missing label file; treating as empty after filter")
            boxes = remap_label_file(
                label_path if label_path.exists() else None,
                source_classes,
                mapping,
                original_counts,
                warnings,
            )
            if not boxes and not args.keep_empty:
                skipped_empty_count += 1
                warnings.append(f"{image_path}: skipped empty-after-filter sample")
                continue
            samples.append(
                VariantSample(
                    split=split,
                    source_image=image_path,
                    source_label=label_path if label_path.exists() else None,
                    rel_image=rel_image,
                    rel_label=rel_image.with_suffix(".txt"),
                    boxes=boxes,
                )
            )

        if labels_dir.exists():
            for label_path in sorted(labels_dir.rglob("*.txt")):
                if label_path.resolve() not in labels_seen:
                    warnings.append(f"{label_path}: orphan label file without matching image")

    upsampled_samples = build_upsampled_samples(
        samples,
        output_dir,
        upsample_class_id,
        args.upsample_factor,
        args.train_only_upsampling,
        rng,
    )
    all_samples = [*samples, *upsampled_samples]

    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for sample in all_samples:
        copied = copy_sample(sample, output_dir, manifest_rows)
        if not copied:
            warnings.append(
                f"{output_dir / 'images' / sample.split / sample.rel_image}: destination exists; sample not overwritten"
            )

    (output_dir / "classes.txt").write_text("\n".join(output_names) + "\n", encoding="utf-8")
    (output_dir / "data.yaml").write_text(build_data_yaml(output_dir, output_names), encoding="utf-8")
    write_manifest(output_dir, manifest_rows)
    output_counts = count_output_labels(all_samples)
    write_report(
        output_dir=output_dir,
        source_dir=source_dir,
        mode=args.mode,
        mapping_names=mapping_names,
        original_counts=original_counts,
        source_classes=source_classes,
        output_counts=output_counts,
        output_names=output_names,
        samples=all_samples,
        upsampled_count=len(upsampled_samples),
        skipped_empty_count=skipped_empty_count,
        warnings=warnings,
    )

    counts = split_counts(all_samples)
    print(f"Wrote dataset variant: {output_dir}")
    print(f"Mode: {args.mode}")
    for split in SPLITS:
        print(
            f"{split}: images={counts[split]['images']} labels={counts[split]['labels']} "
            f"instances={counts[split]['instances']}"
        )
    print(f"Upsampled images: {len(upsampled_samples)}")
    print(f"Skipped empty-after-filter: {skipped_empty_count}")
    print(f"Warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
