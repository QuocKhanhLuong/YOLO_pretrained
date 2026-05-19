#!/usr/bin/env python3
"""Prepare Ultralytics YOLO pretrained weights for local training runs."""

from __future__ import annotations

import argparse
import shutil
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


def find_downloaded_weight(model_name: str, model_obj: object) -> Path | None:
    """Best-effort lookup for a weight file downloaded by Ultralytics."""
    candidates: list[Path] = []

    model_path = Path(model_name).expanduser()
    candidates.append(model_path)
    candidates.append(Path.cwd() / model_name)

    for attr_name in ("ckpt_path", "pt_path"):
        attr_value = getattr(model_obj, attr_name, None)
        if attr_value:
            candidates.append(Path(attr_value).expanduser())

    inner_model = getattr(model_obj, "model", None)
    if inner_model is not None:
        for attr_name in ("pt_path", "yaml_file"):
            attr_value = getattr(inner_model, attr_name, None)
            if attr_value:
                candidates.append(Path(attr_value).expanduser())

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_file() and resolved.suffix == ".pt":
            return resolved
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or copy an Ultralytics pretrained YOLO weight."
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Ultralytics model name or local .pt path. Default: yolo11s.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="pretrained_weights",
        help="Directory where the prepared weight is stored.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_arg = args.model
    model_path = Path(model_arg).expanduser()
    output_name = model_path.name
    if not output_name.endswith(".pt"):
        output_name = f"{output_name}.pt"
    output_path = output_dir / output_name

    if output_path.exists():
        print(f"Pretrained weight already exists: {output_path}")
        print("No download or copy needed.")
        return 0

    if model_path.exists():
        source_weight = model_path.resolve()
        print(f"Copying local pretrained weight: {source_weight}")
    else:
        try:
            YOLO = import_yolo()
            print(f"Loading Ultralytics model to trigger weight download: {model_arg}")
            model = YOLO(model_arg)
            source_weight = find_downloaded_weight(model_arg, model)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # Ultralytics raises package-specific exceptions.
            print(f"ERROR: failed to load/download model '{model_arg}': {exc}", file=sys.stderr)
            return 1

        if source_weight is None:
            print(
                "ERROR: Ultralytics loaded the model but the downloaded .pt file "
                "could not be found.",
                file=sys.stderr,
            )
            return 1

    if source_weight.resolve() != output_path.resolve():
        shutil.copy2(source_weight, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model: {model_arg}")
    print(f"Source weight: {source_weight}")
    print(f"Prepared weight: {output_path}")
    print(f"Size: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
