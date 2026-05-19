#!/usr/bin/env python3
"""Regenerate a YOLO data.yaml file from a prepared dataset directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_classes(classes_path: Path) -> list[str]:
    """Read class names from classes.txt, ignoring blank lines."""
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found: {classes_path}")

    names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"classes.txt has no class names: {classes_path}")
    return names


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate data.yaml for a prepared YOLO dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Prepared YOLO dataset directory containing classes.txt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()

    if not dataset_dir.exists():
        print(f"ERROR: dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        return 1
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset path is not a directory: {dataset_dir}", file=sys.stderr)
        return 1

    try:
        class_names = read_classes(dataset_dir / "classes.txt")
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    data_yaml_path = dataset_dir / "data.yaml"
    data_yaml_path.write_text(build_data_yaml(dataset_dir, class_names), encoding="utf-8")

    print(f"Wrote {data_yaml_path}")
    print(f"Classes: {len(class_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
