#!/usr/bin/env python3
"""Run a safe YOLO training/reporting batch over multiple pretrained models."""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path


PYTHON = sys.executable


@dataclass
class CommandResult:
    label: str
    command: list[str]
    returncode: int | None = None
    status: str = "pending"


@dataclass
class ModelStatus:
    model: str
    run_name: str
    prepared_weight: str = "pending"
    train_status: str = "pending"
    test_status: str = "pending"
    prediction_status: str = "pending"
    report_path: str = "-"
    archive_path: str = "-"
    upload_status: str = "not requested"
    error: str = ""
    commands: list[CommandResult] = field(default_factory=list)


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def model_id(model: str) -> str:
    return Path(model).name.removesuffix(".pt")


def dataset_tag(dataset_dir: Path) -> str:
    return "".join(
        char if char.isalnum() else "_"
        for char in dataset_dir.name
    ).strip("_")


def confidence_tag(confidence: float) -> str:
    return f"conf{int(round(confidence * 100)):03d}"


def run_name_for(model: str, dataset_dir: Path, imgsz: int, batch: int, epochs: int) -> str:
    return f"{model_id(model)}_{dataset_tag(dataset_dir)}_img{imgsz}_b{batch}_e{epochs}"


def run_command(label: str, command: list[str], dry_run: bool) -> CommandResult:
    result = CommandResult(label=label, command=command)
    print(f"== {label} ==")
    print(shell_join(command))
    if dry_run:
        result.status = "dry-run"
        return result

    completed = subprocess.run(command, shell=False)
    result.returncode = completed.returncode
    result.status = "ok" if completed.returncode == 0 else "failed"
    return result


def ensure_dataset_ready(dataset_dir: Path, dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        return True, ""
    if not dataset_dir.exists():
        return False, f"dataset directory does not exist: {dataset_dir}"
    if not dataset_dir.is_dir():
        return False, f"dataset path is not a directory: {dataset_dir}"
    return True, ""


def create_archive(archive_path: Path, paths: list[Path]) -> tuple[bool, str]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return False, "no output paths exist to archive"

    try:
        with tarfile.open(archive_path, "w:gz") as tar_obj:
            for path in existing_paths:
                tar_obj.add(path, arcname=path.name)
    except OSError as exc:
        return False, str(exc)
    return True, ""


def write_summary(
    summary_path: Path,
    statuses: list[ModelStatus],
    args: argparse.Namespace,
    dry_run: bool,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Batch Training Summary: {args.run_prefix}",
        "",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"- Dataset: `{Path(args.dataset).expanduser().resolve()}`",
        f"- Models: {', '.join(args.models)}",
        f"- Dry run: {'true' if dry_run else 'false'}",
        f"- Image size: {args.imgsz}",
        f"- Batch: {args.batch}",
        f"- Epochs: {args.epochs}",
        f"- Upload remote: `{args.upload_remote}`" if args.upload_remote else "- Upload remote: not requested",
        "",
        "| Model | Run | Weight | Train | Test | Predictions | Report | Archive | Upload |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for status in statuses:
        lines.append(
            "| "
            + " | ".join(
                [
                    status.model,
                    status.run_name,
                    status.prepared_weight,
                    status.train_status,
                    status.test_status,
                    status.prediction_status,
                    status.report_path,
                    status.archive_path,
                    status.upload_status,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failed Commands", ""])
    failed = [
        (status, command)
        for status in statuses
        for command in status.commands
        if command.status == "failed"
    ]
    if failed:
        for status, command in failed:
            lines.append(
                f"- `{status.run_name}` {command.label} returned {command.returncode}: "
                f"`{shell_join(command.command)}`"
            )
    else:
        lines.append("- None")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def append_and_check(status: ModelStatus, result: CommandResult) -> bool:
    status.commands.append(result)
    return result.status in {"ok", "dry-run"}


def report_prediction_args(
    pred_dirs_by_conf: dict[float, Path],
    report_cmd: list[str],
) -> list[str]:
    if 0.10 in pred_dirs_by_conf:
        report_cmd.extend(["--pred-dir", str(pred_dirs_by_conf[0.10])])
    elif pred_dirs_by_conf:
        first_conf = sorted(pred_dirs_by_conf)[0]
        report_cmd.extend(["--pred-dir", str(pred_dirs_by_conf[first_conf])])

    if 0.25 in pred_dirs_by_conf:
        report_cmd.extend(["--pred-dir-conf025", str(pred_dirs_by_conf[0.25])])
    elif len(pred_dirs_by_conf) > 1:
        second_conf = sorted(pred_dirs_by_conf)[1]
        report_cmd.extend(["--pred-dir-conf025", str(pred_dirs_by_conf[second_conf])])
    return report_cmd


def run_one_model(args: argparse.Namespace, dataset_dir: Path, data_yaml: Path, model: str) -> ModelStatus:
    run_name = run_name_for(model, dataset_dir, args.imgsz, args.batch, args.epochs)
    run_dir = Path(args.project).expanduser().resolve() / run_name
    best_weight = run_dir / "weights" / "best.pt"
    weight_path = Path("pretrained_weights").resolve() / Path(model).name
    test_name = f"{run_name}_test"
    test_dir = Path(args.project).expanduser().resolve() / test_name
    report_dir = Path("reports").resolve() / run_name
    report_path = report_dir / "experiment_report.md"
    archive_path = Path("archives").resolve() / f"{run_name}_report_outputs.tar.gz"
    pred_dirs_by_conf: dict[float, Path] = {}
    status = ModelStatus(model=model, run_name=run_name)

    if args.skip_existing and best_weight.exists() and report_path.exists() and archive_path.exists():
        status.prepared_weight = str(weight_path)
        status.train_status = "skipped existing"
        status.test_status = "skipped existing"
        status.prediction_status = "skipped existing"
        status.report_path = str(report_path)
        status.archive_path = str(archive_path)
        return status

    if args.prepare_weights:
        prepare_cmd = [
            PYTHON,
            "scripts/prepare_pretrained_weights.py",
            "--model",
            model,
            "--output-dir",
            "pretrained_weights",
        ]
        result = run_command(f"Prepare weight: {model}", prepare_cmd, args.dry_run)
        if not append_and_check(status, result):
            status.prepared_weight = "failed"
            status.error = f"prepare weight failed for {model}"
            return status
    else:
        print(f"== Prepare weight: {model} ==")
        print("Skipped by --no-prepare-weights")

    if args.dry_run:
        status.prepared_weight = str(weight_path)
    elif weight_path.exists():
        status.prepared_weight = str(weight_path)
    else:
        status.prepared_weight = "missing"
        status.error = f"prepared weight not found: {weight_path}"
        print(f"ERROR: {status.error}", file=sys.stderr)
        return status

    for label, command in (
        (
            "Regenerate data.yaml",
            [PYTHON, "scripts/create_yolo_yaml.py", "--dataset", str(dataset_dir)],
        ),
        (
            "Validate dataset",
            [PYTHON, "scripts/check_yolo_dataset.py", "--dataset", str(dataset_dir)],
        ),
    ):
        result = run_command(label, command, args.dry_run)
        if not append_and_check(status, result):
            status.error = f"{label} failed"
            return status

    train_cmd = [
        PYTHON,
        "scripts/train_yolo.py",
        "--data",
        str(data_yaml),
        "--weights",
        str(weight_path),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--project",
        str(Path(args.project).expanduser().resolve()),
        "--name",
        run_name,
        "--optimizer",
        args.optimizer,
        "--lr0",
        str(args.lr0),
        "--patience",
        str(args.patience),
        "--workers",
        str(args.workers),
        "--close-mosaic",
        str(args.close_mosaic),
    ]
    result = run_command(f"Train: {run_name}", train_cmd, args.dry_run)
    if append_and_check(status, result):
        status.train_status = "dry-run" if args.dry_run else "ok"
    else:
        status.train_status = f"failed rc={result.returncode}"
        status.error = "training failed; if this was CUDA OOM, rerun with --batch 4"
        return status

    if not args.dry_run and not best_weight.exists():
        status.error = f"best.pt not found after training: {best_weight}"
        status.test_status = "blocked"
        print(f"ERROR: {status.error}", file=sys.stderr)
        return status

    validate_cmd = [
        PYTHON,
        "scripts/validate_yolo.py",
        "--weights",
        str(best_weight),
        "--data",
        str(data_yaml),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--split",
        "test",
        "--project",
        str(Path(args.project).expanduser().resolve()),
        "--name",
        test_name,
    ]
    result = run_command(f"Validate test split: {run_name}", validate_cmd, args.dry_run)
    if append_and_check(status, result):
        status.test_status = "dry-run" if args.dry_run else "ok"
    else:
        status.test_status = f"failed rc={result.returncode}"
        status.error = "test validation failed"

    prediction_failures = 0
    for confidence in args.conf_list:
        pred_name = f"{run_name}_pred_{confidence_tag(confidence)}"
        pred_dir = Path(args.project).expanduser().resolve() / pred_name
        pred_dirs_by_conf[confidence] = pred_dir
        predict_cmd = [
            PYTHON,
            "scripts/predict_yolo.py",
            "--weights",
            str(best_weight),
            "--source",
            str(dataset_dir / "images" / "test"),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(confidence),
            "--iou",
            "0.5",
            "--device",
            str(args.device),
            "--project",
            str(Path(args.project).expanduser().resolve()),
            "--name",
            pred_name,
            "--max-det",
            "300",
        ]
        result = run_command(f"Predict test split {confidence:.2f}: {run_name}", predict_cmd, args.dry_run)
        if not append_and_check(status, result):
            prediction_failures += 1
    if args.dry_run:
        status.prediction_status = "dry-run"
    elif prediction_failures:
        status.prediction_status = f"{prediction_failures} failed"
        if not status.error:
            status.error = "one or more prediction commands failed"
    else:
        status.prediction_status = "ok"

    if not args.dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
    report_cmd = [
        PYTHON,
        "scripts/generate_experiment_report_md.py",
        "--run-dir",
        str(run_dir),
        "--dataset",
        str(dataset_dir),
        "--test-dir",
        str(test_dir),
        "--run-name",
        run_name,
        "--notes",
        (
            f"Batch run prefix: {args.run_prefix}; model: {model}; dataset: {dataset_dir.name}; "
            f"imgsz: {args.imgsz}; batch: {args.batch}; epochs: {args.epochs}."
        ),
        "--output",
        str(report_path),
    ]
    result = run_command("Generate markdown report", report_prediction_args(pred_dirs_by_conf, report_cmd), args.dry_run)
    if append_and_check(status, result):
        status.report_path = str(report_path) if not args.dry_run else f"dry-run: {report_path}"
    else:
        status.report_path = f"failed rc={result.returncode}"
        if not status.error:
            status.error = "report generation failed"

    print(f"== Archive outputs: {run_name} ==")
    print(f"create tar.gz {archive_path}")
    if args.dry_run:
        status.archive_path = f"dry-run: {archive_path}"
    else:
        ok, error = create_archive(
            archive_path,
            [report_dir, run_dir, test_dir, *pred_dirs_by_conf.values()],
        )
        if ok:
            status.archive_path = str(archive_path)
        else:
            status.archive_path = f"failed: {error}"
            if not status.error:
                status.error = f"archive failed: {error}"

    if args.upload_remote:
        remote_run = f"{args.upload_remote.rstrip('/')}/{run_name}"
        upload_commands = [
            ["rclone", "mkdir", remote_run],
            ["rclone", "copy", str(archive_path), remote_run],
            ["rclone", "copy", str(report_path), remote_run],
            ["rclone", "copy", str(best_weight), remote_run],
        ]
        upload_failures = 0
        for command in upload_commands:
            result = run_command("Upload artifact", command, args.dry_run)
            if not append_and_check(status, result):
                upload_failures += 1
        if args.dry_run:
            status.upload_status = "dry-run"
        elif upload_failures:
            status.upload_status = f"{upload_failures} failed"
            if not status.error:
                status.error = "upload failed"
        else:
            status.upload_status = "ok"

    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiple YOLO models and generate validation, prediction, report, archive, and upload artifacts."
    )
    parser.add_argument("--dataset", required=True, help="Prepared YOLO dataset directory.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names or local .pt paths.")
    parser.add_argument("--run-prefix", required=True, help="Prefix used in the batch summary filename.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--conf-list", nargs="+", type=float, default=[0.10, 0.25])
    parser.add_argument("--upload-remote", help="Optional rclone remote folder.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue to the next model after a model failure. Default: true.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip a run if best.pt, report, and archive already exist.")
    parser.add_argument(
        "--prepare-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run prepare_pretrained_weights.py before each model. Default: true.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    data_yaml = dataset_dir / "data.yaml"
    statuses: list[ModelStatus] = []

    ok, error = ensure_dataset_ready(dataset_dir, args.dry_run)
    if not ok:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("DRY RUN: commands will not be executed." if args.dry_run else "Starting batch training.")
    for model in args.models:
        status = run_one_model(args, dataset_dir, data_yaml, model)
        statuses.append(status)
        if status.error:
            print(f"ERROR: {status.run_name}: {status.error}", file=sys.stderr)
            if not args.continue_on_error:
                break

    summary_path = Path("reports").resolve() / f"batch_training_{args.run_prefix}_summary.md"
    if args.dry_run:
        print(f"Dry-run summary path: {summary_path}")
    else:
        write_summary(summary_path, statuses, args, dry_run=False)
        print(f"Wrote batch summary: {summary_path}")

    failed = [status for status in statuses if status.error]
    if failed:
        print(f"Completed with {len(failed)} model failure(s); see summary/status output.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
