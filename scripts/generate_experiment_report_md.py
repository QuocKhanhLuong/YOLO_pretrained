#!/usr/bin/env python3
"""Generate a rich Markdown report for an Ultralytics YOLO experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METRICS = [
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
]
CONFIG_KEYS = [
    "model",
    "data",
    "epochs",
    "imgsz",
    "batch",
    "optimizer",
    "lr0",
    "lrf",
    "cos_lr",
    "patience",
    "close_mosaic",
    "mosaic",
    "scale",
    "translate",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "fliplr",
    "flipud",
    "weight_decay",
    "warmup_epochs",
    "seed",
    "deterministic",
    "device",
    "amp",
    "workers",
    "save_dir",
]
TRAIN_IMAGE_ARTIFACTS = [
    "results.png",
    "labels.jpg",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "train_batch0.jpg",
    "train_batch1.jpg",
    "train_batch2.jpg",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
    "val_batch1_labels.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_labels.jpg",
    "val_batch2_pred.jpg",
]
TEST_IMAGE_ARTIFACTS = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
    "val_batch1_labels.jpg",
    "val_batch1_pred.jpg",
]
EXPECTED_NON_IMAGE_ARTIFACTS = [
    "args.yaml",
    "results.csv",
    "weights/best.pt",
    "weights/last.pt",
    "experiment_metadata.json",
]
PREDICTION_EXCLUDE_TOKENS = (
    "curve",
    "confusion",
    "labels",
    "results",
    "batch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown experiment report for a YOLO training run."
    )
    parser.add_argument("--run-dir", required=True, help="Ultralytics training run directory.")
    parser.add_argument("--dataset", required=True, help="YOLO dataset directory.")
    parser.add_argument("--output", required=True, help="Output Markdown report path.")
    parser.add_argument("--test-dir", help="Optional validation/test output directory.")
    parser.add_argument("--pred-dir", help="Optional prediction directory, usually conf=0.10.")
    parser.add_argument(
        "--pred-dir-conf025",
        help="Optional second prediction directory, usually conf=0.25.",
    )
    parser.add_argument("--run-name", help="Optional display name for the run.")
    parser.add_argument("--notes", help="Optional report notes.")
    parser.add_argument("--max-pred-images", type=int, default=12)
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read CSV rows and strip column names and values."""
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows: list[dict[str, str]] = []
            for row in reader:
                clean_row: dict[str, str] = {}
                for key, value in row.items():
                    clean_key = (key or "").strip()
                    if not clean_key:
                        continue
                    clean_row[clean_key] = "" if value is None else str(value).strip()
                rows.append(clean_row)
            return rows
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def parse_float(value: object) -> float | None:
    """Parse a finite float from a CSV/config value."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text == "-":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_yaml_scalar(raw_value: str) -> object:
    """Parse a simple YAML scalar without requiring PyYAML."""
    value = raw_value.strip()
    if value == "":
        return ""
    if value in {"null", "Null", "NULL", "~", "None"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        if any(char in value for char in (".", "e", "E")):
            parsed_float = float(value)
            return parsed_float if math.isfinite(parsed_float) else value
        return int(value)
    except ValueError:
        return value


def read_simple_yaml(yaml_path: Path) -> dict[str, object]:
    """Read simple top-level key: value YAML files such as Ultralytics args.yaml."""
    if not yaml_path.exists():
        return {}
    values: dict[str, object] = {}
    try:
        lines = yaml_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return values

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = parse_yaml_scalar(raw_value)
    return values


def read_json_file(json_path: Path) -> dict[str, object]:
    """Read a JSON object if available."""
    if not json_path.exists():
        return {}
    try:
        value = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def best_epoch_row(rows: list[dict[str, str]]) -> tuple[dict[str, str], str]:
    """Return the best row and ranking metric name."""
    for metric_name in ("metrics/mAP50-95(B)", "metrics/mAP50(B)"):
        best_row: dict[str, str] | None = None
        best_value: float | None = None
        for row in rows:
            value = parse_float(row.get(metric_name))
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
                best_row = row
        if best_row is not None:
            return best_row, metric_name
    return {}, "-"


def format_value(value: object) -> str:
    """Format a value for Markdown tables."""
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_metric(value: object) -> str:
    """Format a numeric metric if possible."""
    parsed = parse_float(value)
    if parsed is None:
        return "-"
    if abs(parsed) >= 100:
        return f"{parsed:.2f}"
    return f"{parsed:.4f}"


def markdown_escape(value: object) -> str:
    """Escape a value for use inside Markdown tables."""
    text = format_value(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Build a Markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(markdown_escape(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def metric_table(row: dict[str, str]) -> str:
    """Build a Markdown metric table for one results.csv row."""
    return markdown_table(
        ["Metric", "Value"],
        [[f"`{metric}`", format_metric(row.get(metric))] for metric in METRICS],
    )


def rel_markdown_path(target: Path, output_path: Path) -> str:
    """Return a URL-escaped path relative to the output Markdown file."""
    try:
        relative = os.path.relpath(target.resolve(), start=output_path.parent.resolve())
    except OSError:
        relative = os.path.relpath(target, start=output_path.parent)
    return quote(Path(relative).as_posix(), safe="/._-")


def image_markdown(image_path: Path, output_path: Path, alt: str) -> str:
    """Return Markdown image syntax with a relative path."""
    return f"![{alt}]({rel_markdown_path(image_path, output_path)})"


def read_classes(dataset_dir: Path) -> list[str]:
    """Read dataset classes.txt."""
    classes_path = dataset_dir / "classes.txt"
    if not classes_path.exists():
        return []
    try:
        return [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError):
        return []


def count_images(images_dir: Path) -> int:
    """Count supported image files under a directory."""
    if not images_dir.exists():
        return 0
    return sum(
        1
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_stats(labels_dir: Path) -> tuple[int, int, Counter[int]]:
    """Count label files, empty label files, and class instances."""
    label_files = 0
    empty_files = 0
    class_counts: Counter[int] = Counter()
    if not labels_dir.exists():
        return label_files, empty_files, class_counts

    for label_path in labels_dir.rglob("*.txt"):
        if not label_path.is_file():
            continue
        label_files += 1
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if not non_empty_lines:
            empty_files += 1
            continue
        for line in non_empty_lines:
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            class_counts[class_id] += 1
    return label_files, empty_files, class_counts


def count_manifest_rows(manifest_path: Path) -> int | None:
    """Count data rows in manifest.csv if available."""
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.reader(file_obj)
            rows = list(reader)
    except (OSError, csv.Error, UnicodeDecodeError):
        return None
    if not rows:
        return 0
    return max(len(rows) - 1, 0)


def summarize_dataset(dataset_dir: Path) -> dict[str, object]:
    """Summarize YOLO dataset splits, labels, and instances."""
    classes = read_classes(dataset_dir)
    split_rows: list[dict[str, int | str]] = []
    total_images = 0
    total_labels = 0
    total_empty_labels = 0
    total_instances = 0
    class_counts: Counter[int] = Counter()

    for split in SPLITS:
        images = count_images(dataset_dir / "images" / split)
        labels, empty_labels, split_class_counts = label_stats(
            dataset_dir / "labels" / split
        )
        instances = sum(split_class_counts.values())
        split_rows.append(
            {
                "split": split,
                "images": images,
                "labels": labels,
                "empty_labels": empty_labels,
                "instances": instances,
            }
        )
        total_images += images
        total_labels += labels
        total_empty_labels += empty_labels
        total_instances += instances
        class_counts.update(split_class_counts)

    for class_id in range(len(classes)):
        class_counts.setdefault(class_id, 0)

    return {
        "classes": classes,
        "split_rows": split_rows,
        "class_counts": class_counts,
        "total_images": total_images,
        "total_labels": total_labels,
        "total_empty_labels": total_empty_labels,
        "total_instances": total_instances,
        "manifest_rows": count_manifest_rows(dataset_dir / "manifest.csv"),
        "dataset_report": dataset_dir / "dataset_report.md",
    }


def metadata_parameters(metadata: dict[str, object]) -> dict[str, object]:
    """Return experiment metadata parameters."""
    params = metadata.get("parameters")
    return params if isinstance(params, dict) else {}


def config_value(
    key: str,
    args_yaml: dict[str, object],
    metadata: dict[str, object],
) -> object:
    """Read a config value from args.yaml, then experiment metadata."""
    params = metadata_parameters(metadata)
    if key in args_yaml and args_yaml[key] not in (None, ""):
        return args_yaml[key]
    if key == "model":
        return args_yaml.get("model") or metadata.get("weights") or "-"
    if key == "data":
        return args_yaml.get("data") or metadata.get("data") or "-"
    if key == "save_dir":
        return args_yaml.get("save_dir") or metadata.get("run_dir") or "-"
    return params.get(key, "-")


def config_table(args_yaml: dict[str, object], metadata: dict[str, object]) -> str:
    """Build the training configuration table."""
    return markdown_table(
        ["Key", "Value"],
        [[f"`{key}`", config_value(key, args_yaml, metadata)] for key in CONFIG_KEYS],
    )


def dataset_tables(dataset_summary: dict[str, object]) -> tuple[str, str]:
    """Build dataset split and class-count tables."""
    split_rows = dataset_summary["split_rows"]
    assert isinstance(split_rows, list)
    split_table = markdown_table(
        ["Split", "Images", "Labels", "Empty Label Files", "Instances"],
        [
            [
                row["split"],
                row["images"],
                row["labels"],
                row["empty_labels"],
                row["instances"],
            ]
            for row in split_rows
        ],
    )

    classes = dataset_summary["classes"]
    class_counts = dataset_summary["class_counts"]
    assert isinstance(classes, list)
    assert isinstance(class_counts, Counter)
    class_table = markdown_table(
        ["Class ID", "Class Name", "Instances"],
        [
            [class_id, classes[class_id] if class_id < len(classes) else "unknown", count]
            for class_id, count in sorted(class_counts.items())
        ],
    )
    return split_table, class_table


def artifact_section(
    title: str,
    artifact_dir: Path,
    filenames: list[str],
    output_path: Path,
    missing: list[str],
) -> str:
    """Build an image artifact section and record missing files."""
    lines = [f"### {title}", ""]
    found = False
    for filename in filenames:
        artifact_path = artifact_dir / filename
        if artifact_path.exists():
            found = True
            lines.append(f"**{filename}**")
            lines.append("")
            lines.append(image_markdown(artifact_path, output_path, filename))
            lines.append("")
        else:
            missing.append(str(artifact_path))
    if not found:
        lines.append("No expected image artifacts found.")
        lines.append("")
    return "\n".join(lines)


def collect_missing_training_artifacts(run_dir: Path) -> list[str]:
    """Collect expected training artifacts that are not present."""
    missing: list[str] = []
    for filename in [*EXPECTED_NON_IMAGE_ARTIFACTS, *TRAIN_IMAGE_ARTIFACTS]:
        path = run_dir / filename
        if not path.exists():
            missing.append(str(path))
    return missing


def is_prediction_image(path: Path) -> bool:
    """Return true for likely prediction sample images."""
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    lower_name = path.name.lower()
    return not any(token in lower_name for token in PREDICTION_EXCLUDE_TOKENS)


def prediction_samples_section(
    title: str,
    pred_dir: Path | None,
    output_path: Path,
    max_images: int,
    missing: list[str],
) -> str:
    """Build a qualitative prediction sample section."""
    lines = [f"### {title}", ""]
    if pred_dir is None:
        lines.append("Prediction directory not provided.")
        lines.append("")
        return "\n".join(lines)
    if not pred_dir.exists():
        lines.append(f"Prediction directory not found: `{pred_dir}`")
        lines.append("")
        missing.append(str(pred_dir))
        return "\n".join(lines)

    images = sorted(path for path in pred_dir.rglob("*") if is_prediction_image(path))
    if not images:
        lines.append("No qualitative prediction images found.")
        lines.append("")
        return "\n".join(lines)

    for image_path in images[: max(max_images, 0)]:
        lines.append(f"**{image_path.name}**")
        lines.append("")
        lines.append(image_markdown(image_path, output_path, image_path.name))
        lines.append("")
    if len(images) > max_images:
        lines.append(f"_Showing {max_images} of {len(images)} prediction images._")
        lines.append("")
    return "\n".join(lines)


def final_metric(row: dict[str, str], metric_name: str) -> float | None:
    """Return a parsed final metric."""
    return parse_float(row.get(metric_name))


def loss_gap_text(row: dict[str, str]) -> tuple[str, bool]:
    """Summarize train-vs-val loss gap at the final epoch."""
    train_losses = [
        final_metric(row, "train/box_loss"),
        final_metric(row, "train/cls_loss"),
        final_metric(row, "train/dfl_loss"),
    ]
    val_losses = [
        final_metric(row, "val/box_loss"),
        final_metric(row, "val/cls_loss"),
        final_metric(row, "val/dfl_loss"),
    ]
    if any(value is None for value in train_losses + val_losses):
        return "Loss gap cannot be computed because one or more train/val loss columns are missing.", False

    train_total = sum(value for value in train_losses if value is not None)
    val_total = sum(value for value in val_losses if value is not None)
    if train_total <= 0:
        return "Loss gap cannot be computed because total train loss is zero or invalid.", False

    ratio = val_total / train_total
    text = (
        f"Final train loss sum is {train_total:.4f}; final validation loss sum is "
        f"{val_total:.4f}; val/train ratio is {ratio:.2f}."
    )
    if ratio >= 1.5:
        text += " Validation loss is much larger than train loss."
        return text, True
    text += " No large train-validation loss gap is apparent from the final row."
    return text, False


def class_imbalance_observation(dataset_summary: dict[str, object]) -> str | None:
    """Return a class-imbalance observation if counts are uneven."""
    classes = dataset_summary["classes"]
    class_counts = dataset_summary["class_counts"]
    assert isinstance(classes, list)
    assert isinstance(class_counts, Counter)
    if not class_counts:
        return None
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    if max_count <= 0:
        return "No labeled object instances were counted in the dataset labels."
    if min_count <= max_count * 0.5:
        rare_classes = [
            f"{class_id} {classes[class_id] if class_id < len(classes) else 'unknown'} ({count})"
            for class_id, count in sorted(class_counts.items())
            if count == min_count
        ]
        return "Class imbalance exists; fewer instances were found for: " + ", ".join(
            rare_classes
        )
    return None


def observations(
    final_row: dict[str, str],
    dataset_summary: dict[str, object],
    overfit_gap: bool,
) -> list[str]:
    """Generate automatic observations from metrics and dataset counts."""
    items: list[str] = []
    recall = final_metric(final_row, "metrics/recall(B)")
    map50 = final_metric(final_row, "metrics/mAP50(B)")
    map5095 = final_metric(final_row, "metrics/mAP50-95(B)")

    if recall is not None and recall < 0.5:
        items.append("Recall is below 0.5, so the object miss rate may still be high.")
    if map50 is not None and map5095 is not None and (map50 - map5095) >= 0.20:
        items.append(
            "mAP50-95 is much lower than mAP50; localization is likely hard, "
            "possibly due to small UAV objects or loose/tight label variation."
        )
    imbalance = class_imbalance_observation(dataset_summary)
    if imbalance:
        items.append(imbalance)
    if overfit_gap:
        items.append(
            "Final validation loss is much larger than train loss, which can indicate "
            "overfitting, domain gap, or label noise."
        )
    items.append("Compare against previous run manually.")
    return items


def interpretation(final_row: dict[str, str], best_row: dict[str, str]) -> str:
    """Generate a short executive interpretation."""
    best_map5095 = final_metric(best_row, "metrics/mAP50-95(B)")
    final_recall = final_metric(final_row, "metrics/recall(B)")
    if best_map5095 is None:
        return "Numeric mAP50-95 is unavailable; inspect plots and validation outputs manually."
    if best_map5095 >= 0.5 and (final_recall is None or final_recall >= 0.5):
        return "The run shows usable baseline signal; inspect per-class errors before selecting it."
    if best_map5095 >= 0.25:
        return "The run has partial detection quality but likely needs error analysis and data cleanup."
    return "The run appears weak by strict mAP; prioritize labels, split quality, and prediction review."


def bullet_list(items: list[str], empty_text: str = "None") -> str:
    """Format a Markdown bullet list."""
    if not items:
        return empty_text
    return "\n".join(f"- {item}" for item in items)


def test_results_section(test_dir: Path | None, output_path: Path, missing: list[str]) -> str:
    """Build validation/test results section."""
    lines = ["## 6. Validation/Test Results", ""]
    if test_dir is None:
        lines.append("Test evaluation has not been provided.")
        lines.append("")
        return "\n".join(lines)
    if not test_dir.exists():
        lines.append(f"Test directory was provided but does not exist: `{test_dir}`")
        lines.append("")
        missing.append(str(test_dir))
        return "\n".join(lines)

    rows = read_csv_rows(test_dir / "results.csv")
    if rows:
        final_row = rows[-1]
        best_row, metric_name = best_epoch_row(rows)
        lines.append(f"Best test/validation row selected by `{metric_name}`:")
        lines.append("")
        lines.append(metric_table(best_row))
        lines.append("")
        lines.append("Final test/validation row:")
        lines.append("")
        lines.append(metric_table(final_row))
        lines.append("")
    else:
        lines.append(
            "No `results.csv` found in the test directory; numerical test metrics "
            "should be read from validation console output or generated artifacts."
        )
        lines.append("")

    lines.append(
        artifact_section("Validation/test plots", test_dir, TEST_IMAGE_ARTIFACTS, output_path, missing)
    )
    return "\n".join(lines)


def build_report(
    run_dir: Path,
    dataset_dir: Path,
    output_path: Path,
    test_dir: Path | None,
    pred_dir: Path | None,
    pred_dir_conf025: Path | None,
    run_name: str | None,
    notes: str | None,
    max_pred_images: int,
) -> str:
    """Build the Markdown report content."""
    rows = read_csv_rows(run_dir / "results.csv")
    final_row = rows[-1] if rows else {}
    best_row, best_metric_name = best_epoch_row(rows)
    args_yaml = read_simple_yaml(run_dir / "args.yaml")
    metadata = read_json_file(run_dir / "experiment_metadata.json")
    dataset_summary = summarize_dataset(dataset_dir)
    split_table, class_table = dataset_tables(dataset_summary)
    display_name = run_name or run_dir.name
    missing = collect_missing_training_artifacts(run_dir)
    gap_text, overfit_gap = loss_gap_text(final_row)
    dataset_report = dataset_summary["dataset_report"]
    manifest_rows = dataset_summary["manifest_rows"]

    assert isinstance(dataset_report, Path)
    report_reference = (
        f"[`{dataset_report}`]({rel_markdown_path(dataset_report, output_path)})"
        if dataset_report.exists()
        else "-"
    )

    executive_rows = [
        ["Run directory", f"`{run_dir}`"],
        ["Dataset", f"`{dataset_dir}`"],
        ["Model", config_value("model", args_yaml, metadata)],
        ["Best epoch", best_row.get("epoch", "-")],
        ["Best mAP50", format_metric(best_row.get("metrics/mAP50(B)"))],
        ["Best mAP50-95", format_metric(best_row.get("metrics/mAP50-95(B)"))],
        ["Final mAP50", format_metric(final_row.get("metrics/mAP50(B)"))],
        ["Final mAP50-95", format_metric(final_row.get("metrics/mAP50-95(B)"))],
        ["Precision", format_metric(final_row.get("metrics/precision(B)"))],
        ["Recall", format_metric(final_row.get("metrics/recall(B)"))],
        ["Interpretation", interpretation(final_row, best_row)],
    ]

    content = [
        f"# YOLO Experiment Report: {display_name}",
        "",
        "## 1. Executive Summary",
        "",
        markdown_table(["Item", "Value"], executive_rows),
        "",
    ]
    if notes:
        content.extend(["**Notes**", "", notes, ""])

    content.extend(
        [
            f"Best epoch selection metric: `{best_metric_name}`.",
            "",
            "## 2. Training Configuration",
            "",
            config_table(args_yaml, metadata),
            "",
            "## 3. Dataset Summary",
            "",
            f"- Manifest rows: {manifest_rows if manifest_rows is not None else '-'}",
            f"- Dataset report: {report_reference}",
            f"- Total images: {dataset_summary['total_images']}",
            f"- Total labels: {dataset_summary['total_labels']}",
            f"- Total instances: {dataset_summary['total_instances']}",
            f"- Total empty label files: {dataset_summary['total_empty_labels']}",
            "",
            "### Split Counts",
            "",
            split_table,
            "",
            "### Class Instance Counts",
            "",
            class_table,
            "",
            "## 4. Training Metrics",
            "",
            "### Best Epoch Metrics",
            "",
            metric_table(best_row),
            "",
            "### Final Epoch Metrics",
            "",
            metric_table(final_row),
            "",
            "### Generalization Gap",
            "",
            gap_text,
            "",
            "## 5. Training Curves and Built-in Ultralytics Plots",
            "",
            artifact_section(
                "Training run plots",
                run_dir,
                TRAIN_IMAGE_ARTIFACTS,
                output_path,
                missing,
            ),
            "",
            test_results_section(test_dir, output_path, missing),
            "",
            "## 7. Qualitative Prediction Samples",
            "",
            prediction_samples_section(
                "Qualitative prediction samples, confidence 0.10",
                pred_dir,
                output_path,
                max_pred_images,
                missing,
            ),
            prediction_samples_section(
                "Qualitative prediction samples, confidence 0.25",
                pred_dir_conf025,
                output_path,
                max_pred_images,
                missing,
            ),
            "## 8. Observations",
            "",
            bullet_list(observations(final_row, dataset_summary, overfit_gap)),
            "",
            "## 9. Recommended Next Steps",
            "",
            bullet_list(
                [
                    "Evaluate on the test split.",
                    "Inspect the confusion matrix.",
                    "Review prediction samples at conf=0.10 and conf=0.25.",
                    "Audit false positives and false negatives.",
                    "Try a low-augmentation run if class confusion is high.",
                    "Try an image preprocessing experiment if image quality is poor.",
                    "Consider data_v2.1 label cleanup if label errors are found.",
                ]
            ),
            "",
            "## 10. Missing Artifacts",
            "",
            bullet_list(sorted(set(missing))),
            "",
        ]
    )
    return "\n".join(content)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    test_dir = Path(args.test_dir).expanduser().resolve() if args.test_dir else None
    pred_dir = Path(args.pred_dir).expanduser().resolve() if args.pred_dir else None
    pred_dir_conf025 = (
        Path(args.pred_dir_conf025).expanduser().resolve()
        if args.pred_dir_conf025
        else None
    )

    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        output_path=output_path,
        test_dir=test_dir,
        pred_dir=pred_dir,
        pred_dir_conf025=pred_dir_conf025,
        run_name=args.run_name,
        notes=args.notes,
        max_pred_images=args.max_pred_images,
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote experiment report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
