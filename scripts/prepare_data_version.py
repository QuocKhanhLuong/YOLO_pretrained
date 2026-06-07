#!/usr/bin/env python3
"""Prepare a clean, versioned YOLO dataset from Labeling System exports.

The script never modifies the source export directory. It copies valid
image-label pairs into a prepared dataset version and records skipped files in
dataset_report.md.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import shutil
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
MANIFEST_CANDIDATES = (
    "manifest.csv",
    "docker_labeled_manifest.csv",
    "labeled_images_manifest.csv",
    "docker_labeled_in_backup_manifest.csv",
)


def yaml_scalar(value: str | Path) -> str:
    """Return a YAML-safe scalar using JSON string syntax."""
    return json.dumps(str(value), ensure_ascii=False)


def build_data_yaml(dataset_dir: Path, class_names: list[str]) -> str:
    """Build deterministic YOLO data.yaml content."""
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


def read_classes(classes_path: Path) -> list[str]:
    """Read class names from classes.txt, ignoring blank lines."""
    names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"classes.txt has no class names: {classes_path}")
    return names


def read_cli_classes(class_names: list[str], class_names_csv: str | None) -> list[str]:
    """Read class names from CLI args while preserving the caller's order."""
    names: list[str] = []
    if class_names_csv:
        names.extend(name.strip() for name in class_names_csv.split(","))
    names.extend(name.strip() for name in class_names)
    names = [name for name in names if name]
    if not names:
        raise ValueError("class names are empty")
    return names


def count_manifest_rows(manifest_path: Path) -> int | None:
    """Count manifest.csv rows for reporting; return None if it cannot be parsed."""
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as file_obj:
            return sum(1 for _ in csv.reader(file_obj))
    except (OSError, csv.Error, UnicodeDecodeError):
        return None


def is_relative_to(child: Path, parent: Path) -> bool:
    """Compatibility helper for checking path containment."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split ratios before any output is written."""
    ratios = {
        "train-ratio": train_ratio,
        "val-ratio": val_ratio,
        "test-ratio": test_ratio,
    }
    for name, ratio in ratios.items():
        if ratio < 0:
            raise ValueError(f"--{name} must be >= 0")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"split ratios must sum to 1.0; got {ratio_sum:.6f}"
        )


def resolve_manifest_path(source_dir: Path, manifest_arg: str | None) -> Path:
    """Resolve an explicit or known manifest file inside the source export."""
    if manifest_arg:
        manifest_path = Path(manifest_arg).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = source_dir / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest file does not exist: {manifest_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest path is not a file: {manifest_path}")
        return manifest_path

    for candidate in MANIFEST_CANDIDATES:
        manifest_path = source_dir / candidate
        if manifest_path.is_file():
            return manifest_path

    raise FileNotFoundError(
        "source directory is missing a supported manifest file: "
        + ", ".join(MANIFEST_CANDIDATES)
    )


def validate_source(
    source_dir: Path,
    cli_class_names: list[str],
    class_names_csv: str | None,
    manifest_arg: str | None,
) -> tuple[Path, Path, list[str], str, Path]:
    """Validate the required Labeling System export structure."""
    if not source_dir.exists():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source path is not a directory: {source_dir}")

    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    classes_path = source_dir / "classes.txt"

    missing = [
        str(path)
        for path in (images_dir, labels_dir)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "source directory is missing required paths: " + ", ".join(missing)
        )
    if not images_dir.is_dir():
        raise NotADirectoryError(f"source images path is not a directory: {images_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"source labels path is not a directory: {labels_dir}")

    if classes_path.exists():
        class_names = read_classes(classes_path)
        class_source = str(classes_path)
    else:
        class_names = read_cli_classes(cli_class_names, class_names_csv)
        class_source = "CLI --class-name/--class-names"

    manifest_path = resolve_manifest_path(source_dir, manifest_arg)
    return images_dir, labels_dir, class_names, class_source, manifest_path


def collect_images(images_dir: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    """Collect supported images by relative stem and record unsupported files."""
    images_by_stem: dict[str, list[Path]] = {}
    unsupported_images: list[Path] = []

    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            unsupported_images.append(image_path)
            continue
        rel_stem = image_path.relative_to(images_dir).with_suffix("").as_posix()
        images_by_stem.setdefault(rel_stem, []).append(image_path)

    return images_by_stem, unsupported_images


def collect_labels(labels_dir: Path) -> dict[str, Path]:
    """Collect label files by relative stem."""
    labels_by_stem: dict[str, Path] = {}
    for label_path in sorted(labels_dir.rglob("*.txt")):
        rel_stem = label_path.relative_to(labels_dir).with_suffix("").as_posix()
        labels_by_stem[rel_stem] = label_path
    return labels_by_stem


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
            if not math.isfinite(value):
                errors.append(f"{display_path}:{line_number}: {value_name} is not finite")
            parsed_values.append(value)

        if len(parsed_values) != 4:
            continue

        x_center, y_center, width, height = parsed_values
        if math.isfinite(x_center) and not 0.0 <= x_center <= 1.0:
            errors.append(f"{display_path}:{line_number}: x_center outside [0, 1]")
        if math.isfinite(y_center) and not 0.0 <= y_center <= 1.0:
            errors.append(f"{display_path}:{line_number}: y_center outside [0, 1]")
        if math.isfinite(width) and not 0.0 < width <= 1.0:
            errors.append(f"{display_path}:{line_number}: width outside (0, 1]")
        if math.isfinite(height) and not 0.0 < height <= 1.0:
            errors.append(f"{display_path}:{line_number}: height outside (0, 1]")

    return errors


def split_samples(
    samples: list[dict[str, Path | str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Path | str]]]:
    """Split valid samples reproducibly into train/val/test."""
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def copy_sample(sample: dict[str, Path | str], output_dir: Path, split: str) -> None:
    """Copy one image-label pair into the prepared dataset split."""
    image_path = sample["image_path"]
    label_path = sample["label_path"]
    image_rel = sample["image_rel"]
    label_rel = sample["label_rel"]

    assert isinstance(image_path, Path)
    assert isinstance(label_path, Path)
    assert isinstance(image_rel, Path)
    assert isinstance(label_rel, Path)

    image_output = output_dir / "images" / split / image_rel
    label_output = output_dir / "labels" / split / label_rel
    image_output.parent.mkdir(parents=True, exist_ok=True)
    label_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_output)
    shutil.copy2(label_path, label_output)


def format_list(items: list[str], empty_text: str = "None") -> str:
    """Format a markdown bullet list."""
    if not items:
        return empty_text
    return "\n".join(f"- {item}" for item in items)


def write_dataset_report(
    output_dir: Path,
    source_dir: Path,
    version: str,
    class_names: list[str],
    class_source: str,
    manifest_path: Path,
    manifest_rows: int | None,
    ratios: tuple[float, float, float],
    seed: int,
    split_map: dict[str, list[dict[str, Path | str]]],
    report_items: dict[str, list[str]],
) -> None:
    """Write dataset_report.md inside the prepared dataset directory."""
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    train_ratio, val_ratio, test_ratio = ratios
    split_lines = [
        f"- {split}: {len(split_map.get(split, []))} samples" for split in SPLITS
    ]
    split_detail_lines: list[str] = []
    for split in SPLITS:
        for sample in split_map.get(split, []):
            split_detail_lines.append(f"- {split}: {sample['stem']}")

    content = f"""# Dataset Report: {version}

Generated: {now}

## Inputs

- Source: `{source_dir}`
- Output: `{output_dir}`
- Version: `{version}`
- Classes: {len(class_names)}
- Class source: `{class_source}`
- Manifest: `{manifest_path}`
- Manifest rows: {manifest_rows if manifest_rows is not None else "unreadable"}
- Train ratio: {train_ratio}
- Val ratio: {val_ratio}
- Test ratio: {test_ratio}
- Seed: {seed}

## Split Summary

{chr(10).join(split_lines)}

## Classes

{format_list([f"{index}: {name}" for index, name in enumerate(class_names)])}

## Skipped Or Problematic Files

### Images Without Labels

{format_list(report_items.get("images_without_labels", []))}

### Orphan Labels Without Images

{format_list(report_items.get("orphan_labels", []))}

### Invalid Label Lines

{format_list(report_items.get("invalid_label_errors", []))}

### Duplicate Image Stems

{format_list(report_items.get("duplicate_image_stems", []))}

### Unsupported Files Under images/

{format_list(report_items.get("unsupported_images", []))}

## Split Assignments

{format_list(split_detail_lines)}
"""
    (output_dir / "dataset_report.md").write_text(content, encoding="utf-8")


def write_dataset_guide(output_dir: Path, version: str) -> None:
    """Write a short GUIDE.md inside the prepared dataset version."""
    content = f"""# Guide: {version}

This directory is a prepared YOLO dataset version. It was generated from a
Labeling System export without modifying the source data.

## Structure

- `images/train`, `images/val`, `images/test`: split image files.
- `labels/train`, `labels/val`, `labels/test`: matching YOLO `.txt` labels.
- `classes.txt`: class names copied from the source export.
- `manifest.csv`: manifest copied from the source export.
- `data.yaml`: YOLO training config with an absolute `path`.
- `dataset_report.md`: preparation summary and skipped/problematic files.

## Validate

From the repository root:

```bash
python scripts/check_yolo_dataset.py --dataset {output_dir}
```

## Regenerate data.yaml

```bash
python scripts/create_yolo_yaml.py --dataset {output_dir}
```

## Train With Ultralytics

```bash
yolo detect train data={output_dir / "data.yaml"} model=yolov10n.pt epochs=100 imgsz=640
```
"""
    (output_dir / "GUIDE.md").write_text(content, encoding="utf-8")


def prepare_output_dir(source_dir: Path, output_dir: Path, force: bool) -> None:
    """Create a clean output directory without risking the source tree."""
    if output_dir == source_dir or is_relative_to(output_dir, source_dir):
        raise ValueError(
            "output directory must not be the source directory or inside the source directory"
        )
    if is_relative_to(source_dir, output_dir):
        raise ValueError("source directory must not be inside the output directory")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a clean, versioned YOLO dataset from Labeling System output."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Labeling System output directory containing images/ and labels/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset version directory, for example data/versions/data_v1.0.",
    )
    parser.add_argument("--version", required=True, help="Dataset version name, for example data_v1.0.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-name",
        action="append",
        default=[],
        help=(
            "Class name in YOLO class-id order. Repeat when the source export "
            "does not contain classes.txt."
        ),
    )
    parser.add_argument(
        "--class-names",
        help=(
            "Comma-separated class names in YOLO class-id order, for example "
            "soldier,vehicle,fire. Used only when classes.txt is missing."
        ),
    )
    parser.add_argument(
        "--manifest",
        help=(
            "Optional manifest path. Relative paths are resolved under --source. "
            "If omitted, known manifest names are auto-detected."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    try:
        validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
        images_dir, labels_dir, class_names, class_source, manifest_path = validate_source(
            source_dir=source_dir,
            cli_class_names=args.class_name,
            class_names_csv=args.class_names,
            manifest_arg=args.manifest,
        )
        prepare_output_dir(source_dir, output_dir, args.force)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    manifest_rows = count_manifest_rows(manifest_path)
    images_by_stem, unsupported_images = collect_images(images_dir)
    labels_by_stem = collect_labels(labels_dir)

    report_items: dict[str, list[str]] = {
        "images_without_labels": [],
        "orphan_labels": [],
        "invalid_label_errors": [],
        "duplicate_image_stems": [],
        "unsupported_images": [
            path.relative_to(images_dir).as_posix() for path in unsupported_images
        ],
    }
    valid_samples: list[dict[str, Path | str]] = []

    for stem, image_paths in sorted(images_by_stem.items()):
        if len(image_paths) > 1:
            rel_paths = ", ".join(path.relative_to(images_dir).as_posix() for path in image_paths)
            report_items["duplicate_image_stems"].append(f"{stem}: {rel_paths}")
            continue

        image_path = image_paths[0]
        label_path = labels_by_stem.get(stem)
        if label_path is None:
            report_items["images_without_labels"].append(
                image_path.relative_to(images_dir).as_posix()
            )
            continue

        label_rel = label_path.relative_to(labels_dir)
        display_label = label_rel.as_posix()
        label_errors = validate_label_file(label_path, len(class_names), display_label)
        if label_errors:
            report_items["invalid_label_errors"].extend(label_errors)
            continue

        valid_samples.append(
            {
                "stem": stem,
                "image_path": image_path,
                "label_path": label_path,
                "image_rel": image_path.relative_to(images_dir),
                "label_rel": label_rel,
            }
        )

    for stem, label_path in sorted(labels_by_stem.items()):
        if stem not in images_by_stem:
            report_items["orphan_labels"].append(label_path.relative_to(labels_dir).as_posix())

    split_map = split_samples(
        valid_samples,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    for split, samples in split_map.items():
        for sample in samples:
            copy_sample(sample, output_dir, split)

    (output_dir / "classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(manifest_path, output_dir / "manifest.csv")
    (output_dir / "data.yaml").write_text(
        build_data_yaml(output_dir, class_names),
        encoding="utf-8",
    )
    write_dataset_report(
        output_dir=output_dir,
        source_dir=source_dir,
        version=args.version,
        class_names=class_names,
        class_source=class_source,
        manifest_path=manifest_path,
        manifest_rows=manifest_rows,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        split_map=split_map,
        report_items=report_items,
    )
    write_dataset_guide(output_dir, args.version)

    total_problem_count = sum(len(items) for items in report_items.values())
    print(f"Prepared YOLO dataset version: {args.version}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Valid samples copied: {len(valid_samples)}")
    for split in SPLITS:
        print(f"{split}: {len(split_map.get(split, []))}")
    print(f"Skipped/problematic entries: {total_problem_count}")
    print(f"Report: {output_dir / 'dataset_report.md'}")
    print(f"Data YAML: {output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
