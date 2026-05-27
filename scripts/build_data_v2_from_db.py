#!/usr/bin/env python3
"""Build a prepared YOLO dataset version from a labeling DB backup."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from db_copy_parser import load_tables


DB_IMAGE_PREFIX = "/data/label-system"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
EXPECTED_PROJECT_CLASSES = {5: ["soldier", "vehicle", "fire"]}
COPY_TABLES = {"public.projects", "public.frames", "public.annotations"}
MANIFEST_COLUMNS = [
    "split",
    "image_path",
    "label_path",
    "frame_id",
    "annotation_id",
    "video_id",
    "frame_index",
    "source_image_path",
    "resolved_source_image_path",
    "labeler_id",
    "review_status",
    "submitted_at",
    "version",
    "width",
    "height",
    "status",
]


def yaml_scalar(value: str | Path) -> str:
    """Return a YAML-safe scalar using JSON string syntax."""
    return json.dumps(str(value), ensure_ascii=False)


def build_data_yaml(dataset_dir: Path, class_names: list[str]) -> str:
    """Build deterministic YOLO data.yaml content."""
    lines = [
        f"path: {yaml_scalar(dataset_dir.resolve())}",
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


def parse_int(value: object, default: int = 0) -> int:
    """Parse an integer-like DB value for sorting and reporting."""
    if value is None:
        return default
    try:
        return int(str(value))
    except ValueError:
        try:
            return int(float(str(value)))
        except ValueError:
            return default


def parse_float(value: object, default: float = -1.0) -> float:
    """Parse a numeric DB value for sorting."""
    if value is None:
        return default
    try:
        parsed = float(str(value))
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def normalize_status(value: object) -> str:
    """Normalize review/status strings from the DB."""
    return "" if value is None else str(value).strip().lower()


def is_nonempty_text(value: object) -> bool:
    """Return true when a DB text value contains non-whitespace content."""
    return value is not None and bool(str(value).strip())


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split ratios before output is written."""
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
        raise ValueError(f"split ratios must sum to 1.0; got {ratio_sum:.6f}")


def table_rows(
    tables: dict[str, list[dict[str, str | None]]],
    table_name: str,
) -> list[dict[str, str | None]]:
    """Return rows for a table, accepting schema-qualified or unqualified names."""
    for key, rows in tables.items():
        if key == table_name or key.rsplit(".", 1)[-1] == table_name:
            return rows
    return []


def read_image_roots(
    overrides: list[str] | None,
    roots_file: str | None,
) -> list[Path]:
    """Collect image root overrides from repeated CLI args and an optional file."""
    raw_roots: list[str] = []
    if overrides:
        raw_roots.extend(overrides)
    if roots_file:
        path = Path(roots_file).expanduser()
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                raw_roots.append(line)

    roots: list[Path] = []
    seen: set[str] = set()
    for raw_root in raw_roots:
        root = Path(raw_root).expanduser()
        key = str(root)
        if key in seen:
            continue
        roots.append(root)
        seen.add(key)
    return roots


def read_classes_file(classes_path: Path) -> list[str]:
    """Read classes.txt, accepting either plain names or 'id name' rows."""
    class_rows: list[tuple[int, str]] = []
    plain_names: list[str] = []
    for index, raw_line in enumerate(classes_path.read_text(encoding="utf-8").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            class_rows.append((int(parts[0]), parts[1].strip()))
        else:
            plain_names.append(line)

    if class_rows and not plain_names:
        return [name for _, name in sorted(class_rows)]
    return plain_names


def names_from_dict_map(value: dict[str, Any]) -> list[str]:
    """Extract names from a dict keyed by numeric class IDs."""
    numeric_items: list[tuple[int, str]] = []
    for key, item in value.items():
        if not str(key).isdigit():
            continue
        if isinstance(item, str):
            numeric_items.append((int(str(key)), item))
        elif isinstance(item, dict):
            name = item.get("name") or item.get("label") or item.get("value")
            if isinstance(name, str):
                numeric_items.append((int(str(key)), name))
    return [name for _, name in sorted(numeric_items)]


def names_from_list(value: list[Any]) -> list[str]:
    """Extract ordered class names from a JSON list."""
    if all(isinstance(item, str) for item in value):
        return [str(item) for item in value]

    rows: list[tuple[int, str]] = []
    fallback: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("label") or item.get("value") or item.get("title")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_id = item.get("id", item.get("class_id", item.get("classId", item.get("index", index))))
        rows.append((parse_int(raw_id, index), name.strip()))
        fallback.append(name.strip())
    if rows:
        return [name for _, name in sorted(rows)]
    return fallback


def extract_class_names(value: Any) -> list[str]:
    """Extract class names from likely label_config_json shapes."""
    if isinstance(value, list):
        return names_from_list(value)
    if not isinstance(value, dict):
        return []

    mapped_names = names_from_dict_map(value)
    if mapped_names:
        return mapped_names

    for key in ("classes", "labels", "names", "categories"):
        if key in value:
            names = extract_class_names(value[key])
            if names:
                return names

    for nested in value.values():
        names = extract_class_names(nested)
        if names:
            return names
    return []


def class_names_from_backup(
    backup_root: Path,
    project_row: dict[str, str | None] | None,
    project_id: int,
) -> tuple[list[str], str, list[str]]:
    """Load class names from backup classes.txt, project JSON, or known context."""
    warnings: list[str] = []
    classes_path = backup_root / "labels_working" / f"project_{project_id}" / "classes.txt"
    if classes_path.exists():
        names = read_classes_file(classes_path)
        if names:
            return names, str(classes_path), warnings
        warnings.append(f"{classes_path} exists but has no class names")

    if project_row:
        raw_config = project_row.get("label_config_json")
        if raw_config:
            try:
                parsed = json.loads(raw_config)
            except json.JSONDecodeError as exc:
                warnings.append(f"Could not parse projects.label_config_json: {exc}")
            else:
                names = extract_class_names(parsed)
                if names:
                    return names, "projects.label_config_json", warnings
                warnings.append("No class names found in projects.label_config_json")

    if project_id in EXPECTED_PROJECT_CLASSES:
        warnings.append("Using expected class names from project context")
        return EXPECTED_PROJECT_CLASSES[project_id], "project-context fallback", warnings
    return [], "unavailable", warnings


def annotation_is_selected(annotation: dict[str, str | None], include_draft: bool) -> bool:
    """Return true when an annotation passes review-state selection."""
    if include_draft:
        return True
    return normalize_status(annotation.get("review_status")) == "submitted" or bool(
        annotation.get("submitted_at")
    )


def annotation_has_allowed_label(
    annotation: dict[str, str | None],
    allow_empty_labels: bool,
) -> bool:
    """Return true when the annotation label content should be considered."""
    return allow_empty_labels or is_nonempty_text(annotation.get("yolo_text"))


def best_annotation_key(annotation: dict[str, str | None]) -> tuple[int, int, float, str, int]:
    """Sort key for selecting the best annotation for a frame."""
    return (
        1 if annotation.get("submitted_at") else 0,
        1 if normalize_status(annotation.get("review_status")) == "submitted" else 0,
        parse_float(annotation.get("version")),
        str(annotation.get("updated_at") or ""),
        parse_int(annotation.get("id")),
    )


def select_best_annotations(
    annotations: list[dict[str, str | None]],
    include_draft: bool,
    allow_empty_labels: bool,
) -> dict[str, dict[str, str | None]]:
    """Select the best eligible annotation for each frame."""
    by_frame: dict[str, list[dict[str, str | None]]] = defaultdict(list)
    for annotation in annotations:
        frame_id = annotation.get("frame_id")
        if frame_id is None:
            continue
        if not annotation_is_selected(annotation, include_draft):
            continue
        if not annotation_has_allowed_label(annotation, allow_empty_labels):
            continue
        by_frame[str(frame_id)].append(annotation)

    return {
        frame_id: max(frame_annotations, key=best_annotation_key)
        for frame_id, frame_annotations in by_frame.items()
        if frame_annotations
    }


def validate_yolo_text(
    yolo_text: str | None,
    num_classes: int,
    display_name: str,
) -> tuple[list[str], Counter[int]]:
    """Validate YOLO text and count class instances."""
    if not is_nonempty_text(yolo_text):
        return [], Counter()

    errors: list[str] = []
    class_counts: Counter[int] = Counter()
    for line_number, raw_line in enumerate(str(yolo_text).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{display_name}:{line_number}: expected 5 values, found {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"{display_name}:{line_number}: class_id is not an integer")
            continue

        if class_id < 0 or class_id >= num_classes:
            errors.append(
                f"{display_name}:{line_number}: class_id {class_id} outside [0, {num_classes - 1}]"
            )
        else:
            class_counts[class_id] += 1

        values: list[float] = []
        for value_name, raw_value in zip(
            ("x_center", "y_center", "width", "height"), parts[1:]
        ):
            try:
                value = float(raw_value)
            except ValueError:
                errors.append(f"{display_name}:{line_number}: {value_name} is not a float")
                values.append(float("nan"))
                continue
            if not math.isfinite(value):
                errors.append(f"{display_name}:{line_number}: {value_name} is not finite")
            values.append(value)

        if len(values) != 4:
            continue
        x_center, y_center, width, height = values
        if math.isfinite(x_center) and not 0.0 <= x_center <= 1.0:
            errors.append(f"{display_name}:{line_number}: x_center outside [0, 1]")
        if math.isfinite(y_center) and not 0.0 <= y_center <= 1.0:
            errors.append(f"{display_name}:{line_number}: y_center outside [0, 1]")
        if math.isfinite(width) and not 0.0 < width <= 1.0:
            errors.append(f"{display_name}:{line_number}: width outside (0, 1]")
        if math.isfinite(height) and not 0.0 < height <= 1.0:
            errors.append(f"{display_name}:{line_number}: height outside (0, 1]")

    return errors, class_counts


def label_text_for_write(yolo_text: str | None) -> str:
    """Preserve label text content and ensure a final newline."""
    if not is_nonempty_text(yolo_text):
        return ""
    return str(yolo_text).rstrip("\r\n") + "\n"


def resolve_image_path(
    db_image_path: str | None,
    image_roots: list[Path],
) -> tuple[Path | None, str, str | None]:
    """Resolve a DB image path using direct lookup and configured root overrides."""
    if not db_image_path:
        return None, "missing-db-path", None

    direct_path = Path(db_image_path).expanduser()
    if direct_path.exists():
        return direct_path.resolve(), "direct", None

    if db_image_path.startswith(DB_IMAGE_PREFIX):
        suffix = db_image_path[len(DB_IMAGE_PREFIX) :].lstrip("/")
        for root in image_roots:
            candidate = root / suffix
            if candidate.exists():
                return candidate.resolve(), "override", str(root)

    return None, "missing", None


def make_output_image_name(
    source_image_path: str,
    frame_id: str,
    used_names: set[str],
) -> str:
    """Create a stable unique image filename from the DB image path."""
    posix_path = PurePosixPath(source_image_path)
    video_name = posix_path.parent.name or "video_unknown"
    frame_stem = posix_path.stem or f"frame_{frame_id}"
    suffix = posix_path.suffix or ".jpg"

    candidate = f"{video_name}_{frame_stem}{suffix}"
    if candidate in used_names:
        candidate = f"{video_name}_{frame_stem}_frame{frame_id}{suffix}"

    counter = 2
    base_candidate = candidate
    while candidate in used_names:
        candidate = f"{PurePosixPath(base_candidate).stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate)
    return candidate


def split_items(
    items: list[Any],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Any]]:
    """Split items reproducibly, keeping non-empty splits where possible."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total == 0:
        return {split: [] for split in SPLITS}

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    counts = {"train": train_count, "val": val_count, "test": test_count}

    positive_splits = [split for split, ratio in zip(SPLITS, (train_ratio, val_ratio, 1.0 - train_ratio - val_ratio)) if ratio > 0]
    if total >= len(positive_splits):
        for split in positive_splits:
            if counts[split] == 0:
                donor = max(counts, key=counts.get)
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[split] += 1

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def split_samples(
    samples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    split_by_video: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Split samples randomly or by video ID."""
    if not split_by_video:
        return split_items(samples, train_ratio, val_ratio, seed)

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_video[str(sample["frame"].get("video_id") or "video_unknown")].append(sample)

    video_split = split_items(list(by_video), train_ratio, val_ratio, seed)
    split_map: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for split, video_ids in video_split.items():
        for video_id in video_ids:
            split_map[split].extend(by_video[video_id])
        split_map[split].sort(
            key=lambda sample: (
                parse_int(sample["frame"].get("video_id")),
                parse_int(sample["frame"].get("frame_index")),
                parse_int(sample["frame"].get("id")),
            )
        )
    return split_map


def ensure_output_dir(output_dir: Path, backup_root: Path, force: bool) -> None:
    """Create a clean output directory without risking source data or data_v1.0."""
    if output_dir.name == "data_v1.0":
        raise ValueError("refusing to write to data_v1.0")
    if output_dir == backup_root or backup_root in output_dir.parents:
        raise ValueError("output directory must not be the backup root or inside it")
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


def format_count_map(counter: dict[Any, int] | Counter[Any], empty_text: str = "None") -> str:
    """Format a dict or Counter as markdown bullets."""
    if not counter:
        return empty_text
    return "\n".join(f"- {key}: {counter[key]}" for key in sorted(counter, key=str))


def format_list(items: list[str], limit: int, empty_text: str = "None") -> str:
    """Format a limited markdown bullet list."""
    if not items:
        return empty_text
    lines = [f"- {item}" for item in items[:limit]]
    if len(items) > limit:
        lines.append(f"- ... {len(items) - limit} more not shown")
    return "\n".join(lines)


def write_manifest(output_dir: Path, split_map: dict[str, list[dict[str, Any]]]) -> None:
    """Write manifest.csv for copied samples."""
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for split in SPLITS:
            for sample in split_map.get(split, []):
                frame = sample["frame"]
                annotation = sample["annotation"]
                writer.writerow(
                    {
                        "split": split,
                        "image_path": sample["output_image_rel"],
                        "label_path": sample["output_label_rel"],
                        "frame_id": frame.get("id") or "",
                        "annotation_id": annotation.get("id") or "",
                        "video_id": frame.get("video_id") or "",
                        "frame_index": frame.get("frame_index") or "",
                        "source_image_path": frame.get("image_path") or "",
                        "resolved_source_image_path": str(sample["resolved_image_path"]),
                        "labeler_id": annotation.get("labeler_id") or "",
                        "review_status": annotation.get("review_status") or "",
                        "submitted_at": annotation.get("submitted_at") or "",
                        "version": annotation.get("version") or "",
                        "width": frame.get("width") or "",
                        "height": frame.get("height") or "",
                        "status": frame.get("status") or "",
                    }
                )


def write_dataset_guide(
    output_dir: Path,
    version: str,
    backup_root: Path,
    project_id: int,
) -> None:
    """Write a dataset-local guide for validation and training."""
    content = f"""# Guide: {version}

This YOLO dataset version was generated from the labeling-system database backup,
not from the old `output/` export and not from `dataset_raw` template labels.

## Source

- Backup root: `{backup_root}`
- DB dump: `{backup_root / "labeling_db.sql.gz"}`
- Project ID: `{project_id}`
- Labels: `public.annotations.yolo_text`
- Image mapping: `public.annotations.frame_id -> public.frames.id -> public.frames.image_path`

## Validate

```bash
python scripts/create_yolo_yaml.py --dataset {output_dir}
python scripts/check_yolo_dataset.py --dataset {output_dir}
```

## Train

```bash
python scripts/train_yolo.py \\
  --data {output_dir / "data.yaml"} \\
  --weights pretrained_weights/yolo11s.pt \\
  --epochs 80 \\
  --imgsz 1280 \\
  --batch 4 \\
  --device 0 \\
  --project runs \\
  --name yolo11s_data_v2_0_img1280_b4 \\
  --optimizer AdamW \\
  --lr0 0.0005 \\
  --patience 20 \\
  --workers 2 \\
  --close-mosaic 20
```
"""
    (output_dir / "GUIDE.md").write_text(content, encoding="utf-8")


def write_dataset_report(
    output_dir: Path,
    version: str,
    backup_root: Path,
    db_path: Path,
    project_row: dict[str, str | None] | None,
    class_names: list[str],
    class_source: str,
    class_warnings: list[str],
    stats: dict[str, Any],
    split_map: dict[str, list[dict[str, Any]]],
    split_by_video: bool,
    missing_examples: list[str],
    invalid_examples: list[str],
    copied_examples: list[str],
    status: str,
) -> None:
    """Write dataset_report.md."""
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    project_name = project_row.get("name") if project_row else "unknown"
    project_status = project_row.get("status") if project_row else "unknown"
    split_counts = {split: len(split_map.get(split, [])) for split in SPLITS}
    total_copied = sum(split_counts.values())
    split_ratios = {
        split: (count / total_copied if total_copied else 0.0)
        for split, count in split_counts.items()
    }
    videos_per_split: dict[str, int] = {}
    if split_by_video:
        for split, samples in split_map.items():
            videos_per_split[split] = len(
                {str(sample["frame"].get("video_id") or "") for sample in samples}
            )

    content = f"""# Dataset Report: {version}

Generated: {now}

Build status: {status}

## Source

- Dataset version: `{version}`
- Backup root: `{backup_root}`
- DB path: `{db_path}`
- Project: `{stats.get("project_id")}` / `{project_name}`
- Project status: `{project_status}`
- Source logic: DB annotations + DB frames
- `output/` ignored because it is an old export
- `dataset_raw/*.zip` labels ignored because they are raw empty templates
- Class source: `{class_source}`

## Classes

{format_list([f"{index}: {name}" for index, name in enumerate(class_names)], 100)}

## Class Warnings

{format_list(class_warnings, 100)}

## Counts

- Total project frames: {stats.get("total_project_frames", 0)}
- Joined annotations: {stats.get("joined_annotations", 0)}
- Selected annotations: {stats.get("selected_annotations", 0)}
- Selected non-empty annotations: {stats.get("selected_non_empty_annotations", 0)}
- Best frame-level selected samples: {stats.get("best_selected_samples", 0)}
- Valid samples copied: {total_copied}
- Missing images: {stats.get("missing_images", 0)}
- Missing image ratio: {stats.get("missing_image_ratio", 0.0):.6f}
- Invalid labels: {stats.get("invalid_labels", 0)}
- Invalid label ratio: {stats.get("invalid_label_ratio", 0.0):.6f}

## Frame Status Counts

{format_count_map(stats.get("frame_status_counts", Counter()))}

## Annotation Review Status Counts

{format_count_map(stats.get("annotation_review_counts", Counter()))}

## Split Counts

{format_count_map(split_counts)}

## Split Ratios

{format_count_map({split: f"{ratio:.4f}" for split, ratio in split_ratios.items()})}

## Instance Count Per Class

{format_count_map(stats.get("instance_counts", Counter()))}

## Videos Per Split

{format_count_map(videos_per_split) if split_by_video else "Not split by video"}

## Missing Image Examples

{format_list(missing_examples, 100)}

## Invalid Label Examples

{format_list(invalid_examples, 100)}

## Copied Sample Examples

{format_list(copied_examples, 20)}
"""
    (output_dir / "dataset_report.md").write_text(content, encoding="utf-8")


def copy_split_samples(output_dir: Path, split_map: dict[str, list[dict[str, Any]]]) -> None:
    """Copy images and write labels for all split samples."""
    used_names: set[str] = set()
    for split in SPLITS:
        for sample in split_map.get(split, []):
            frame = sample["frame"]
            annotation = sample["annotation"]
            source_image_path = str(frame.get("image_path") or "")
            frame_id = str(frame.get("id") or "")
            image_name = make_output_image_name(source_image_path, frame_id, used_names)
            label_name = f"{PurePosixPath(image_name).stem}.txt"

            image_output = output_dir / "images" / split / image_name
            label_output = output_dir / "labels" / split / label_name
            if image_output.exists() or label_output.exists():
                raise FileExistsError(f"refusing to overwrite output sample: {image_name}")

            shutil.copy2(sample["resolved_image_path"], image_output)
            label_output.write_text(
                label_text_for_write(annotation.get("yolo_text")),
                encoding="utf-8",
            )
            sample["output_image_rel"] = image_output.relative_to(output_dir).as_posix()
            sample["output_label_rel"] = label_output.relative_to(output_dir).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data_v2.0 from labeling DB annotations and frame image paths."
    )
    parser.add_argument("--backup-root", required=True, help="Backup root containing labeling_db.sql.gz.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    parser.add_argument("--version", default="data_v2.0")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split-by-video", action="store_true")
    parser.add_argument("--include-draft", action="store_true")
    parser.add_argument("--allow-empty-labels", action="store_true")
    parser.add_argument("--image-root-override", action="append", default=[])
    parser.add_argument("--image-roots-file")
    parser.add_argument("--max-missing-ratio", type=float, default=0.03)
    parser.add_argument("--max-invalid-ratio", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backup_root = Path(args.backup_root).expanduser().resolve()
    db_path = backup_root / "labeling_db.sql.gz"
    output_dir = Path(args.output).expanduser().resolve()

    try:
        validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
        if not db_path.exists():
            raise FileNotFoundError(f"DB dump does not exist: {db_path}")
        image_roots = read_image_roots(args.image_root_override, args.image_roots_file)
        tables = load_tables(db_path, COPY_TABLES)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    projects = table_rows(tables, "projects")
    frames = table_rows(tables, "frames")
    annotations = table_rows(tables, "annotations")
    project_row = next(
        (row for row in projects if parse_int(row.get("id"), -1) == args.project_id),
        None,
    )

    class_names, class_source, class_warnings = class_names_from_backup(
        backup_root,
        project_row,
        args.project_id,
    )
    if not class_names:
        print("ERROR: could not determine class names", file=sys.stderr)
        return 1

    project_frames = [
        frame for frame in frames if parse_int(frame.get("project_id"), -1) == args.project_id
    ]
    frame_by_id = {str(frame.get("id")): frame for frame in project_frames if frame.get("id")}
    project_annotations = [
        annotation
        for annotation in annotations
        if annotation.get("frame_id") is not None and str(annotation.get("frame_id")) in frame_by_id
    ]

    selected_annotations = [
        annotation
        for annotation in project_annotations
        if annotation_is_selected(annotation, args.include_draft)
    ]
    selected_non_empty_annotations = [
        annotation for annotation in selected_annotations if is_nonempty_text(annotation.get("yolo_text"))
    ]
    best_annotations = select_best_annotations(
        project_annotations,
        args.include_draft,
        args.allow_empty_labels,
    )

    valid_samples: list[dict[str, Any]] = []
    missing_examples: list[str] = []
    invalid_examples: list[str] = []
    instance_counts: Counter[int] = Counter()
    invalid_label_samples = 0
    missing_images = 0

    for frame_id, annotation in sorted(best_annotations.items(), key=lambda item: parse_int(item[0])):
        frame = frame_by_id[frame_id]
        annotation_id = str(annotation.get("id") or "")
        display_name = f"annotation {annotation_id} frame {frame_id}"
        label_errors, label_counts = validate_yolo_text(
            annotation.get("yolo_text"),
            len(class_names),
            display_name,
        )
        if label_errors:
            invalid_label_samples += 1
            invalid_examples.extend(label_errors)
            continue

        resolved_image_path, method, root = resolve_image_path(frame.get("image_path"), image_roots)
        if resolved_image_path is None:
            missing_images += 1
            missing_examples.append(
                f"frame_id={frame_id} annotation_id={annotation_id} image_path={frame.get('image_path')}"
            )
            continue

        sample = {
            "frame": frame,
            "annotation": annotation,
            "resolved_image_path": resolved_image_path,
            "resolution_method": method,
            "resolution_root": root or "",
            "class_counts": label_counts,
        }
        valid_samples.append(sample)
        instance_counts.update(label_counts)

    denominator = max(len(best_annotations), 1)
    missing_image_ratio = missing_images / denominator
    invalid_label_ratio = invalid_label_samples / denominator

    stats: dict[str, Any] = {
        "project_id": args.project_id,
        "total_project_frames": len(project_frames),
        "frame_status_counts": Counter(str(frame.get("status") or "") for frame in project_frames),
        "joined_annotations": len(project_annotations),
        "annotation_review_counts": Counter(
            str(annotation.get("review_status") or "") for annotation in project_annotations
        ),
        "selected_annotations": len(selected_annotations),
        "selected_non_empty_annotations": len(selected_non_empty_annotations),
        "best_selected_samples": len(best_annotations),
        "missing_images": missing_images,
        "missing_image_ratio": missing_image_ratio,
        "invalid_labels": invalid_label_samples,
        "invalid_label_ratio": invalid_label_ratio,
        "instance_counts": {
            f"{class_id}: {class_names[class_id]}": instance_counts.get(class_id, 0)
            for class_id in range(len(class_names))
        },
    }

    threshold_errors: list[str] = []
    if len(best_annotations) == 0:
        threshold_errors.append("no selected annotations found")
    if missing_image_ratio > args.max_missing_ratio:
        threshold_errors.append(
            f"missing image ratio {missing_image_ratio:.6f} exceeds {args.max_missing_ratio:.6f}"
        )
    if invalid_label_ratio > args.max_invalid_ratio:
        threshold_errors.append(
            f"invalid label ratio {invalid_label_ratio:.6f} exceeds {args.max_invalid_ratio:.6f}"
        )

    try:
        ensure_output_dir(output_dir, backup_root, args.force)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if threshold_errors:
        write_dataset_report(
            output_dir=output_dir,
            version=args.version,
            backup_root=backup_root,
            db_path=db_path,
            project_row=project_row,
            class_names=class_names,
            class_source=class_source,
            class_warnings=class_warnings + threshold_errors,
            stats=stats,
            split_map={split: [] for split in SPLITS},
            split_by_video=args.split_by_video,
            missing_examples=missing_examples,
            invalid_examples=invalid_examples,
            copied_examples=[],
            status="failed",
        )
        print("ERROR: dataset build failed safety thresholds", file=sys.stderr)
        for error in threshold_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Report: {output_dir / 'dataset_report.md'}")
        return 1

    split_map = split_samples(
        valid_samples,
        args.train_ratio,
        args.val_ratio,
        args.seed,
        args.split_by_video,
    )

    try:
        copy_split_samples(output_dir, split_map)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    (output_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output_dir / "data.yaml").write_text(build_data_yaml(output_dir, class_names), encoding="utf-8")
    write_manifest(output_dir, split_map)
    write_dataset_guide(output_dir, args.version, backup_root, args.project_id)

    copied_examples: list[str] = []
    for split in SPLITS:
        for sample in split_map.get(split, [])[:20]:
            copied_examples.append(
                f"{split}: frame_id={sample['frame'].get('id')} "
                f"annotation_id={sample['annotation'].get('id')} "
                f"{sample['output_image_rel']} <- {sample['frame'].get('image_path')}"
            )

    write_dataset_report(
        output_dir=output_dir,
        version=args.version,
        backup_root=backup_root,
        db_path=db_path,
        project_row=project_row,
        class_names=class_names,
        class_source=class_source,
        class_warnings=class_warnings,
        stats=stats,
        split_map=split_map,
        split_by_video=args.split_by_video,
        missing_examples=missing_examples,
        invalid_examples=invalid_examples,
        copied_examples=copied_examples,
        status="completed",
    )

    print(f"Prepared YOLO dataset version: {args.version}")
    print(f"Backup root: {backup_root}")
    print(f"Output: {output_dir}")
    print(f"Best selected samples: {len(best_annotations)}")
    print(f"Valid samples copied: {sum(len(split_map[split]) for split in SPLITS)}")
    for split in SPLITS:
        print(f"{split}: {len(split_map[split])}")
    print(f"Missing images: {missing_images} ({missing_image_ratio:.6f})")
    print(f"Invalid labels: {invalid_label_samples} ({invalid_label_ratio:.6f})")
    print(f"Report: {output_dir / 'dataset_report.md'}")
    print(f"Data YAML: {output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
