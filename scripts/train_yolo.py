#!/usr/bin/env python3
"""Train an Ultralytics YOLO detection model from pretrained weights."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


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


def import_yolo():
    """Import Ultralytics YOLO with a clear installation error."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc
    return YOLO


def model_reference(value: str) -> str:
    """Return an existing weight path or allow a plain Ultralytics model name."""
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())

    looks_like_path = path.is_absolute() or "/" in value or "\\" in value
    if looks_like_path:
        raise FileNotFoundError(f"weights path does not exist: {path}")

    if value.endswith(".pt"):
        return value
    raise FileNotFoundError(
        f"weights path does not exist and does not look like an Ultralytics .pt model name: {value}"
    )


def read_final_metrics(results_csv: Path) -> dict[str, str]:
    """Read the final row of an Ultralytics results.csv file."""
    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)
    if not rows:
        return {}
    return {key.strip(): value for key, value in rows[-1].items()}


def print_metric_summary(run_dir: Path) -> None:
    """Print tracked metrics from the final training row, if available."""
    metrics = read_final_metrics(run_dir / "results.csv")
    if not metrics:
        print("Metrics summary: results.csv not found or empty.")
        return

    print("Final metrics:")
    for metric in TRACKED_METRICS:
        if metric in metrics and metrics[metric] != "":
            print(f"  {metric}: {metrics[metric]}")


def package_version(package_name: str) -> str:
    """Return an installed package version, or unknown if unavailable."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def write_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    """Write experiment metadata to the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "experiment_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO from pretrained weights.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to .pt weights or Ultralytics model name, for example yolo11s.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="yolo11s_data_v1_0_baseline")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cos-lr", dest="cos_lr", action="store_true", default=True)
    parser.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and write metadata without starting training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    project_dir = Path(args.project).expanduser().resolve()
    run_dir = project_dir / args.name

    if not data_path.exists():
        print(f"ERROR: data.yaml does not exist: {data_path}", file=sys.stderr)
        return 1

    try:
        weights_ref = model_reference(args.weights)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    project_dir.mkdir(parents=True, exist_ok=True)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "experiment_name": args.name,
        "started_at": started_at,
        "status": "dry_run" if args.dry_run else "started",
        "data": str(data_path),
        "weights": weights_ref,
        "project": str(project_dir),
        "run_dir": str(run_dir),
        "parameters": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "patience": args.patience,
            "workers": args.workers,
            "cos_lr": args.cos_lr,
            "resume": args.resume,
            "amp": args.amp,
            "seed": args.seed,
            "close_mosaic": args.close_mosaic,
            "single_cls": args.single_cls,
        },
        "software": {
            "python": platform.python_version(),
            "ultralytics": package_version("ultralytics"),
            "torch": package_version("torch"),
        },
        "phase_3_todo": "Export best.pt to ONNX/TensorRT in Phase 3.",
    }
    write_metadata(run_dir, metadata)

    print("Starting YOLO training")
    print(f"Data: {data_path}")
    print(f"Weights: {weights_ref}")
    print(f"Run directory: {run_dir}")
    print(f"Metadata: {run_dir / 'experiment_metadata.json'}")

    if args.dry_run:
        print("Dry run requested; training was not started.")
        return 0

    try:
        YOLO = import_yolo()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    train_kwargs: dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(project_dir),
        "name": args.name,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "patience": args.patience,
        "workers": args.workers,
        "cos_lr": args.cos_lr,
        "resume": args.resume,
        "amp": args.amp,
        "seed": args.seed,
        "close_mosaic": args.close_mosaic,
        "single_cls": args.single_cls,
        "exist_ok": True,
    }

    try:
        model = YOLO(weights_ref)
        model.train(**train_kwargs)
    except Exception as exc:
        print(f"ERROR: YOLO training failed: {exc}", file=sys.stderr)
        return 1

    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    final_metrics = read_final_metrics(run_dir / "results.csv")
    metadata.update(
        {
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "completed",
            "best_model": str(best_path) if best_path.exists() else None,
            "last_model": str(last_path) if last_path.exists() else None,
            "final_metrics": {
                metric: final_metrics.get(metric)
                for metric in TRACKED_METRICS
                if metric in final_metrics
            },
        }
    )
    write_metadata(run_dir, metadata)

    print(f"best.pt: {best_path if best_path.exists() else 'not found yet'}")
    print(f"last.pt: {last_path if last_path.exists() else 'not found yet'}")
    print_metric_summary(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
