#!/usr/bin/env python3
"""Run Ultralytics YOLO validation on a prepared dataset split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def import_yolo():
    """Import Ultralytics YOLO with a clear installation error."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc
    return YOLO


def metric_value(metrics: object, attr_path: str) -> float | None:
    """Read a nested metric attribute if available."""
    current = metrics
    for attr in attr_path.split("."):
        current = getattr(current, attr, None)
        if current is None:
            return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO model on val or test split.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="yolo11s_data_v1_0_val")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print output path without running validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    project_dir = Path(args.project).expanduser().resolve()
    output_dir = project_dir / args.name

    if not weights_path.exists():
        print(f"ERROR: weights file does not exist: {weights_path}", file=sys.stderr)
        return 1
    if not data_path.exists():
        print(f"ERROR: data.yaml does not exist: {data_path}", file=sys.stderr)
        return 1

    print("YOLO validation")
    print(f"Weights: {weights_path}")
    print(f"Data: {data_path}")
    print(f"Split: {args.split}")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        print("Dry run requested; validation was not started.")
        return 0

    try:
        YOLO = import_yolo()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(data_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split=args.split,
            project=str(project_dir),
            name=args.name,
            plots=True,
            save_json=True,
            exist_ok=True,
        )
    except Exception as exc:
        print(f"ERROR: YOLO validation failed: {exc}", file=sys.stderr)
        return 1

    print("Metrics summary:")
    for label, attr_path in (
        ("precision", "box.mp"),
        ("recall", "box.mr"),
        ("mAP50", "box.map50"),
        ("mAP50-95", "box.map"),
    ):
        value = metric_value(metrics, attr_path)
        print(f"  {label}: {value if value is not None else 'unavailable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
