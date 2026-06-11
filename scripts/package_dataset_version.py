#!/usr/bin/env python3
"""Package a prepared YOLO dataset version into a tar.gz archive."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import tarfile
import time
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


def format_duration(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS, or a placeholder before ETA is known."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    rounded = int(seconds + 0.5)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(
    label: str,
    current: int,
    total: int,
    started_at: float,
    *,
    unit: str,
    force_newline: bool = False,
) -> None:
    """Print periodic progress with elapsed time and estimated time remaining."""
    elapsed = time.monotonic() - started_at
    ratio = current / total if total else 1.0
    eta = elapsed * (1.0 - ratio) / ratio if current else None
    end = "\n" if force_newline else "\r"
    print(
        f"{label}: {current}/{total} {unit} ({ratio * 100:.1f}%) "
        f"elapsed {format_duration(elapsed)} ETA {format_duration(eta)}",
        end=end,
        flush=True,
    )


def sha256_file(path: Path, progress_interval: float) -> str:
    """Return the SHA256 digest for a file."""
    digest = hashlib.sha256()
    total_size = path.stat().st_size
    processed = 0
    started_at = time.monotonic()
    last_update = 0.0
    print_progress("Checksum progress", 0, total_size, started_at, unit="bytes")
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
            processed += len(chunk)
            now = time.monotonic()
            if now - last_update >= progress_interval or processed == total_size:
                print_progress(
                    "Checksum progress",
                    processed,
                    total_size,
                    started_at,
                    unit="bytes",
                    force_newline=processed == total_size,
                )
                last_update = now
    if total_size == 0:
        print_progress("Checksum progress", 0, total_size, started_at, unit="bytes", force_newline=True)
    return digest.hexdigest()


def collect_entries(dataset_path: Path) -> tuple[list[Path], list[Path], int]:
    """Collect directories, files, and total file bytes for packaging progress."""
    directories = [dataset_path]
    files: list[Path] = []
    total_size = 0
    for path in sorted(dataset_path.rglob("*")):
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            files.append(path)
            total_size += path.stat().st_size
    return directories, files, total_size


def add_dataset_to_archive(
    tar_obj: tarfile.TarFile,
    dataset_path: Path,
    arcname: str,
    progress_interval: float,
) -> None:
    """Add the dataset tree to an archive while printing byte-level ETA."""
    directories, files, total_size = collect_entries(dataset_path)
    processed = 0
    started_at = time.monotonic()
    last_update = 0.0
    archive_root = Path(arcname)

    print_progress("Package progress", 0, total_size, started_at, unit="bytes")
    for directory in directories:
        rel_path = directory.relative_to(dataset_path)
        tar_obj.add(directory, arcname=(archive_root / rel_path).as_posix(), recursive=False)

    for file_path in files:
        rel_path = file_path.relative_to(dataset_path)
        tar_obj.add(file_path, arcname=(archive_root / rel_path).as_posix(), recursive=False)
        processed += file_path.stat().st_size
        now = time.monotonic()
        if now - last_update >= progress_interval or processed == total_size:
            print_progress(
                "Package progress",
                processed,
                total_size,
                started_at,
                unit="bytes",
                force_newline=processed == total_size,
            )
            last_update = now

    if total_size == 0:
        print_progress("Package progress", 0, total_size, started_at, unit="bytes", force_newline=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a prepared YOLO dataset version.")
    parser.add_argument("--dataset", required=True, help="Dataset directory, for example data/versions/data_v2.0.")
    parser.add_argument("--output-dir", required=True, help="Directory for archive outputs.")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between package and checksum progress updates.",
    )
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

    progress_interval = max(args.progress_interval, 0.0)
    with tarfile.open(archive_path, "w:gz") as tar_obj:
        add_dataset_to_archive(tar_obj, dataset_path, arcname, progress_interval)

    digest = sha256_file(archive_path, progress_interval)
    sha_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")

    print(f"Archive: {archive_path}")
    print(f"Size bytes: {archive_path.stat().st_size}")
    print(f"SHA256: {digest}")
    print(f"SHA256 file: {sha_path}")
    print(f"Archive root: {arcname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
