#!/usr/bin/env python3
"""Run YOLO inference visualization on images, folders, or videos."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generic YOLO prediction visualization on images or videos."
    )
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--source", required=True, help="Image, folder, video, or glob source.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="yolo11s_data_v1_0_test_predictions")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print output path without running prediction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    source_arg = args.source
    source_path = Path(source_arg).expanduser()
    project_dir = Path(args.project).expanduser().resolve()
    output_dir = project_dir / args.name

    if not weights_path.exists():
        print(f"ERROR: weights file does not exist: {weights_path}", file=sys.stderr)
        return 1
    if not any(char in source_arg for char in "*?[]") and not source_path.exists():
        print(f"ERROR: source does not exist: {source_path}", file=sys.stderr)
        return 1

    print("YOLO prediction visualization")
    print(f"Weights: {weights_path}")
    print(f"Source: {source_arg}")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        print("Dry run requested; prediction was not started.")
        return 0

    try:
        YOLO = import_yolo()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        model = YOLO(str(weights_path))
        model.predict(
            source=source_arg,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            project=str(project_dir),
            name=args.name,
            max_det=args.max_det,
            save=True,
            exist_ok=True,
        )
    except Exception as exc:
        print(f"ERROR: YOLO prediction failed: {exc}", file=sys.stderr)
        return 1

    print(f"Predicted images/videos saved under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
