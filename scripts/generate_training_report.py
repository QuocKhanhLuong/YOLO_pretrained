#!/usr/bin/env python3
"""Generate a Markdown training report from Ultralytics YOLO run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


TRACKED_METRICS = [
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
]


def read_csv_rows(results_csv: Path) -> list[dict[str, str]]:
    """Read results.csv rows with stripped column names."""
    if not results_csv.exists():
        return []
    with results_csv.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({(key or "").strip(): value for key, value in row.items()})
    return rows


def read_metadata(run_dir: Path) -> dict[str, object]:
    """Read experiment metadata if the training script wrote it."""
    metadata_path = run_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_dataset_version(dataset_dir: Path) -> str:
    """Infer dataset version from dataset directory name."""
    return dataset_dir.name


def best_epoch(rows: list[dict[str, str]]) -> tuple[str, str] | None:
    """Return best epoch and value by mAP50-95 if available."""
    metric = "metrics/mAP50-95(B)"
    best: tuple[float, str, str] | None = None
    for index, row in enumerate(rows, start=1):
        raw_value = (row.get(metric) or "").strip()
        if raw_value == "":
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        epoch = (row.get("epoch") or str(index)).strip()
        if best is None or value > best[0]:
            best = (value, epoch, raw_value)
    if best is None:
        return None
    return best[1], best[2]


def format_table(metrics: dict[str, str]) -> str:
    """Format tracked metrics as a Markdown table."""
    lines = ["| Metric | Final Value |", "|---|---:|"]
    for metric in TRACKED_METRICS:
        lines.append(f"| `{metric}` | {metrics.get(metric, 'unavailable')} |")
    return "\n".join(lines)


def list_plot_paths(report_output: Path) -> list[Path]:
    """Find likely plot files near the report output path."""
    plots_dir = report_output.parent / "plots"
    if not plots_dir.exists():
        return []
    return sorted(plots_dir.glob("*.png"))


def metadata_param(metadata: dict[str, object], key: str) -> object:
    """Read a parameter from experiment metadata."""
    params = metadata.get("parameters")
    if isinstance(params, dict):
        return params.get(key, "unavailable")
    return "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO training Markdown report.")
    parser.add_argument("--run-dir", required=True, help="Ultralytics run directory.")
    parser.add_argument("--dataset", required=True, help="Prepared dataset directory.")
    parser.add_argument("--output", required=True, help="Output Markdown report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 1
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 1

    results_csv = run_dir / "results.csv"
    rows = read_csv_rows(results_csv)
    final_metrics = rows[-1] if rows else {}
    metadata = read_metadata(run_dir)
    best_epoch_info = best_epoch(rows)
    data_yaml = dataset_dir / "data.yaml"
    best_model = run_dir / "weights" / "best.pt"
    last_model = run_dir / "weights" / "last.pt"
    plot_paths = list_plot_paths(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# YOLO Training Report

## Experiment

- Experiment name: `{run_dir.name}`
- Dataset path: `{dataset_dir}`
- Dataset version: `{read_dataset_version(dataset_dir)}`
- data.yaml path: `{data_yaml}`
- Weights used: `{metadata.get("weights", "unavailable")}`
- Run directory: `{run_dir}`

## Training Configuration

- Epochs: {metadata_param(metadata, "epochs")}
- Image size: {metadata_param(metadata, "imgsz")}
- Batch size: {metadata_param(metadata, "batch")}
- Optimizer: {metadata_param(metadata, "optimizer")}
- Learning rate: {metadata_param(metadata, "lr0")}
- Best model path: `{best_model if best_model.exists() else "not found"}`
- Last model path: `{last_model if last_model.exists() else "not found"}`

## Final Metrics

{format_table(final_metrics)}

## Best Epoch

"""
    if best_epoch_info:
        content += (
            f"- Best epoch by `metrics/mAP50-95(B)`: {best_epoch_info[0]} "
            f"with value {best_epoch_info[1]}\n"
        )
    else:
        content += "- Best epoch by `metrics/mAP50-95(B)`: unavailable\n"

    content += "\n## Plots\n\n"
    if plot_paths:
        for plot_path in plot_paths:
            content += f"- `{plot_path}`\n"
    else:
        content += "- No generated plot files found next to this report.\n"

    content += """
## Small-Object UAV Training Notes

- Try larger `imgsz` values such as 960 or 1280 when small objects are missed.
- Use tiling in a later phase if very large UAV images make objects too small after resizing.
- Check label quality carefully, especially for tiny or partially occluded objects.
- Inspect false positives and false negatives manually before changing model size.
- Review class imbalance and add examples for underperforming classes when needed.

## Phase 3 TODO

- Export the selected `best.pt` checkpoint to ONNX.
- Benchmark ONNX and TensorRT candidates after model quality is acceptable.
"""

    output_path.write_text(content, encoding="utf-8")
    print(f"Wrote training report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
