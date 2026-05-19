#!/usr/bin/env python3
"""Create a filtered YOLO dataset version with selected classes remapped to 0..N."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def yaml_scalar(value: str | Path) -> str:
    """Return a YAML-safe scalar using JSON string syntax."""
    return json.dumps(str(value), ensure_ascii=False)


def build_data_yaml(dataset_dir: Path, class_names: list[str]) -> str:
    """Build YOLO data.yaml content for the filtered dataset."""
    lines = [
        f"path: {yaml_scalar(dataset_dir)}",
        f"train: {yaml_scalar('images/train')}",
        f"val: {yaml_scalar('images/val')}",
        f"test: {yaml_scalar('images/test')}",
        "",
        "names:",
    ]
    for index, name in enumerate(class_names):
        lines.append(f"  {index}: {yaml_scalar(name)}")
    lines.append("")
    return "\n".join(lines)


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


def parse_and_filter_label(
    label_path: Path,
    old_to_new: dict[int, int],
) -> tuple[list[str], int, list[str]]:
    """Filter one YOLO label file and remap kept class IDs."""
    kept_lines: list[str] = []
    removed_count = 0
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
            old_class_id = int(parts[0])
        except ValueError:
            errors.append(f"{label_path}:{line_number}: class_id is not an integer")
            continue
        if old_class_id not in old_to_new:
            removed_count += 1
            continue
        kept_lines.append(" ".join([str(old_to_new[old_class_id]), *parts[1:]]))
    return kept_lines, removed_count, errors


def prepare_output(output_dir: Path, force: bool) -> None:
    """Create a clean output dataset directory."""
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"output directory already exists: {output_dir}. Re-run with --force to replace it."
            )
        if not output_dir.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def is_relative_to(child: Path, parent: Path) -> bool:
    """Return True when child is equal to or under parent."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def maybe_write_filtered_manifest(source_manifest: Path, output_manifest: Path, kept_stems: set[str]) -> str:
    """Write a best-effort filtered manifest if a filename-like column exists."""
    if not source_manifest.exists():
        return "source manifest.csv not found"

    try:
        with source_manifest.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            fieldnames = reader.fieldnames or []
            filename_column = next(
                (
                    column
                    for column in fieldnames
                    if column.lower() in {"filename", "file", "image", "image_path", "path"}
                ),
                None,
            )
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError):
        shutil.copy2(source_manifest, output_manifest)
        return "copied original manifest.csv because it could not be parsed as CSV"

    if not fieldnames or filename_column is None:
        shutil.copy2(source_manifest, output_manifest)
        return "copied original manifest.csv because no filename-like column was found"

    filtered_rows = []
    for row in rows:
        raw_name = row.get(filename_column, "")
        stem = Path(raw_name).with_suffix("").as_posix()
        if stem in kept_stems or Path(stem).name in {Path(item).name for item in kept_stems}:
            filtered_rows.append(row)

    with output_manifest.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    return f"filtered manifest.csv rows: {len(filtered_rows)}"


def write_dataset_guide(output_dir: Path, version: str, class_names: list[str]) -> None:
    """Write a short guide inside the filtered dataset directory."""
    content = f"""# Guide: {version}

This dataset version was filtered from another prepared YOLO dataset.

## Classes

{chr(10).join(f"- {index}: {name}" for index, name in enumerate(class_names))}

Images with only removed classes are kept as empty-label background samples by
default. If this version was created with `--drop-empty-after-filter`, those
samples were removed.

## Validate

```bash
python scripts/check_yolo_dataset.py --dataset {output_dir}
```

## EDA

```bash
python scripts/eda_yolo_dataset.py --dataset {output_dir} --output-dir reports/{version}/eda
```
"""
    (output_dir / "GUIDE.md").write_text(content, encoding="utf-8")


def write_report(
    output_dir: Path,
    source_dir: Path,
    version: str,
    kept_classes: list[str],
    split_counts: dict[str, int],
    removed_instances: dict[str, int],
    empty_after_filter: dict[str, int],
    dropped_after_filter: dict[str, int],
    label_errors: list[str],
    manifest_note: str,
) -> None:
    """Write filter report into the output dataset."""
    content = f"""# Filtered Dataset Report: {version}

- Source dataset: `{source_dir}`
- Output dataset: `{output_dir}`
- Kept classes: {", ".join(f"`{name}`" for name in kept_classes)}
- Manifest handling: {manifest_note}

## Split Counts

| Split | Images Kept | Removed Instances | Empty After Filter | Dropped After Filter |
|---|---:|---:|---:|---:|
"""
    for split in SPLITS:
        content += (
            f"| {split} | {split_counts.get(split, 0)} | {removed_instances.get(split, 0)} | "
            f"{empty_after_filter.get(split, 0)} | {dropped_after_filter.get(split, 0)} |\n"
        )

    content += "\n## Label Errors\n\n"
    if label_errors:
        for error in label_errors[:100]:
            content += f"- {error}\n"
        if len(label_errors) > 100:
            content += f"- ... {len(label_errors) - 100} more not shown\n"
    else:
        content += "None\n"

    (output_dir / "dataset_report.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a prepared YOLO dataset to selected classes and remap labels."
    )
    parser.add_argument("--source-dataset", required=True, help="Prepared source YOLO dataset.")
    parser.add_argument("--output", required=True, help="Output dataset version directory.")
    parser.add_argument("--version", required=True, help="Output dataset version name.")
    parser.add_argument(
        "--include-classes",
        nargs="+",
        required=True,
        help="Class names to keep, in the desired new class order.",
    )
    parser.add_argument(
        "--drop-empty-after-filter",
        action="store_true",
        help="Drop images whose labels become empty after removed classes are filtered out.",
    )
    parser.add_argument("--force", action="store_true", help="Replace output directory if it exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dataset).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not source_dir.exists():
        print(f"ERROR: source dataset does not exist: {source_dir}", file=sys.stderr)
        return 1
    if not source_dir.is_dir():
        print(f"ERROR: source dataset is not a directory: {source_dir}", file=sys.stderr)
        return 1

    try:
        source_classes = read_classes(source_dir)
        if output_dir == source_dir or is_relative_to(output_dir, source_dir):
            raise ValueError(
                "output directory must not be the source dataset directory or inside it"
            )
        if is_relative_to(source_dir, output_dir):
            raise ValueError("source dataset must not be inside the output directory")
        missing_classes = [name for name in args.include_classes if name not in source_classes]
        if missing_classes:
            raise ValueError(f"requested classes not found in source classes.txt: {', '.join(missing_classes)}")
        old_to_new = {
            source_classes.index(class_name): new_index
            for new_index, class_name in enumerate(args.include_classes)
        }
        prepare_output(output_dir, args.force)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    split_counts: dict[str, int] = {}
    removed_instances: dict[str, int] = {}
    empty_after_filter: dict[str, int] = {}
    dropped_after_filter: dict[str, int] = {}
    label_errors: list[str] = []
    kept_stems: set[str] = set()

    for split in SPLITS:
        images_dir = source_dir / "images" / split
        labels_dir = source_dir / "labels" / split
        images = collect_images(images_dir)
        split_counts[split] = 0
        removed_instances[split] = 0
        empty_after_filter[split] = 0
        dropped_after_filter[split] = 0

        for stem, image_path in images.items():
            label_path = labels_dir / f"{stem}.txt"
            if not label_path.exists():
                label_errors.append(f"{split}/{stem}: image has no matching label")
                continue

            try:
                kept_lines, removed_count, errors = parse_and_filter_label(label_path, old_to_new)
            except OSError as exc:
                label_errors.append(f"{split}/{stem}: could not read label: {exc}")
                continue

            label_errors.extend(errors)
            removed_instances[split] += removed_count
            if not kept_lines:
                empty_after_filter[split] += 1
                if args.drop_empty_after_filter:
                    dropped_after_filter[split] += 1
                    continue

            image_rel = image_path.relative_to(images_dir)
            output_image = output_dir / "images" / split / image_rel
            output_label = output_dir / "labels" / split / image_rel.with_suffix(".txt")
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_image)
            output_label.write_text(("\n".join(kept_lines) + "\n") if kept_lines else "", encoding="utf-8")
            split_counts[split] += 1
            kept_stems.add(stem)
            kept_stems.add(Path(stem).name)

    (output_dir / "classes.txt").write_text("\n".join(args.include_classes) + "\n", encoding="utf-8")
    (output_dir / "data.yaml").write_text(build_data_yaml(output_dir, args.include_classes), encoding="utf-8")

    source_manifest = source_dir / "manifest.csv"
    manifest_note = maybe_write_filtered_manifest(source_manifest, output_dir / "manifest.csv", kept_stems)
    write_dataset_guide(output_dir, args.version, args.include_classes)
    write_report(
        output_dir=output_dir,
        source_dir=source_dir,
        version=args.version,
        kept_classes=args.include_classes,
        split_counts=split_counts,
        removed_instances=removed_instances,
        empty_after_filter=empty_after_filter,
        dropped_after_filter=dropped_after_filter,
        label_errors=label_errors,
        manifest_note=manifest_note,
    )

    print(f"Filtered dataset version: {args.version}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    for split in SPLITS:
        print(
            f"{split}: images={split_counts[split]} removed_instances={removed_instances[split]} "
            f"empty_after_filter={empty_after_filter[split]} dropped={dropped_after_filter[split]}"
        )
    print(f"Report: {output_dir / 'dataset_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
