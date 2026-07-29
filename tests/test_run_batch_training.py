from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_batch_training_dry_run_does_not_require_dataset_or_weights(tmp_path: Path) -> None:
    missing_dataset = tmp_path / "missing_dataset"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_batch_training.py",
            "--dataset",
            str(missing_dataset),
            "--models",
            "yolo11s.pt",
            "yolo11m.pt",
            "--run-prefix",
            "rgb_firex2",
            "--dry-run",
            "--project",
            str(tmp_path / "runs"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert "prepare_pretrained_weights.py" in result.stdout
    assert "train_yolo.py" in result.stdout
    assert "validate_yolo.py" in result.stdout
    assert "predict_yolo.py" in result.stdout
    assert "generate_experiment_report_md.py" in result.stdout
    assert "yolo11s_missing_dataset_img1280_b8_e100" in result.stdout
    assert "yolo11m_missing_dataset_img1280_b8_e100" in result.stdout
    assert "pred_conf010" in result.stdout
    assert "pred_conf025" in result.stdout
    assert not missing_dataset.exists()
