from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_package_dataset_version_prints_progress_eta(tmp_path: Path) -> None:
    dataset = tmp_path / "data" / "versions" / "data_v3.0"
    write_text(dataset / "data.yaml", "path: .\n")
    write_text(dataset / "images" / "train" / "a.jpg", "image-bytes")
    write_text(dataset / "labels" / "train" / "a.txt", "0 0.5 0.5 0.2 0.2\n")
    output_dir = tmp_path / "archives"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/package_dataset_version.py",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Package progress:" in result.stdout
    assert "ETA" in result.stdout
    assert (output_dir / "data_v3.0_yolo_dataset.tar.gz").exists()
    assert (output_dir / "data_v3.0_yolo_dataset.tar.gz.sha256").exists()
