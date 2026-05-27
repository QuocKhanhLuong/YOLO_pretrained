#!/usr/bin/env python3
"""Package a prepared YOLO dataset version into a tar.gz archive."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path


def infer_archive_name(dataset_arg: Path, dataset_path: Path) -> str:
    """Infer the path to store inside the archive."""
    if not dataset_arg.is_absolute():
        return dataset_arg.as_posix().rstrip("/")

    parts = dataset_path.parts
    for index in range(len(parts) - 2):
        if parts[index] == "data" and parts[index + 1] == "versions":
            return Path(*parts[index:]).as_posix()
    return dataset_path.name


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a prepared YOLO dataset version.")
    parser.add_argument("--dataset", required=True, help="Dataset directory, for example data/versions/data_v2.0.")
    parser.add_argument("--output-dir", required=True, help="Directory for archive outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_arg = Path(args.dataset)
    dataset_path = dataset_arg.expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not dataset_path.exists():
        print(f"ERROR: dataset path does not exist: {dataset_path}", file=sys.stderr)
        return 1
    if not dataset_path.is_dir():
        print(f"ERROR: dataset path is not a directory: {dataset_path}", file=sys.stderr)
        return 1
    if not (dataset_path / "data.yaml").exists():
        print(f"ERROR: refusing to package without data.yaml: {dataset_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    version = dataset_path.name
    archive_path = output_dir / f"{version}_yolo_dataset.tar.gz"
    sha_path = output_dir / f"{version}_yolo_dataset.tar.gz.sha256"
    arcname = infer_archive_name(dataset_arg, dataset_path)

    with tarfile.open(archive_path, "w:gz") as tar_obj:
        tar_obj.add(dataset_path, arcname=arcname)

    digest = sha256_file(archive_path)
    sha_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")

    print(f"Archive: {archive_path}")
    print(f"Size bytes: {archive_path.stat().st_size}")
    print(f"SHA256: {digest}")
    print(f"SHA256 file: {sha_path}")
    print(f"Archive root: {arcname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
