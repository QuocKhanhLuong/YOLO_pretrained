#!/usr/bin/env python3
"""Plot Ultralytics YOLO training metrics from results.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def import_matplotlib():
    """Import matplotlib with a clear installation error."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        ) from exc
    return plt


def read_results_csv(results_path: Path) -> dict[str, list[float | None]]:
    """Read numeric columns from an Ultralytics results.csv file."""
    with results_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        columns = [name.strip() for name in (reader.fieldnames or [])]
        data: dict[str, list[float | None]] = {name: [] for name in columns}
        for row in reader:
            normalized = {key.strip(): value for key, value in row.items()}
            for column in columns:
                raw_value = (normalized.get(column) or "").strip()
                if raw_value == "":
                    data[column].append(None)
                    continue
                try:
                    data[column].append(float(raw_value))
                except ValueError:
                    data[column].append(None)
    return data


def available(data: dict[str, list[float | None]], *columns: str) -> bool:
    """Return True when all requested columns exist."""
    return all(column in data for column in columns)


def plot_columns(
    plt,
    data: dict[str, list[float | None]],
    columns: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    """Generate one metric plot if all columns are available."""
    if not available(data, *columns):
        missing = [column for column in columns if column not in data]
        print(f"Skipping {output_path.name}; missing columns: {', '.join(missing)}")
        return False

    epoch_count = max((len(data[column]) for column in columns), default=0)
    epochs = list(range(1, epoch_count + 1))

    plt.figure(figsize=(10, 6))
    for column in columns:
        plt.plot(epochs[: len(data[column])], data[column], marker="o", linewidth=1.5, label=column)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Wrote {output_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot YOLO training metrics.")
    parser.add_argument("--results", required=True, help="Path to Ultralytics results.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG plots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_path = Path(args.results).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not results_path.exists():
        print(f"ERROR: results.csv does not exist: {results_path}", file=sys.stderr)
        return 1

    try:
        plt = import_matplotlib()
        data = read_results_csv(results_path)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        (
            ["train/box_loss", "val/box_loss"],
            "Train vs Val Box Loss",
            "Box Loss",
            "train_val_box_loss.png",
        ),
        (
            ["train/cls_loss", "val/cls_loss"],
            "Train vs Val Class Loss",
            "Class Loss",
            "train_val_cls_loss.png",
        ),
        (
            ["train/dfl_loss", "val/dfl_loss"],
            "Train vs Val DFL Loss",
            "DFL Loss",
            "train_val_dfl_loss.png",
        ),
        (
            ["metrics/mAP50(B)", "metrics/mAP50-95(B)"],
            "mAP50 and mAP50-95",
            "mAP",
            "map50_map5095.png",
        ),
        (
            ["metrics/precision(B)", "metrics/recall(B)"],
            "Precision and Recall",
            "Score",
            "precision_recall.png",
        ),
    ]

    generated = 0
    for columns, title, ylabel, filename in plot_specs:
        if plot_columns(plt, data, columns, title, ylabel, output_dir / filename):
            generated += 1

    lr_columns = [column for column in data if column.startswith("lr/") or column.startswith("x/lr")]
    if lr_columns:
        if plot_columns(plt, data, lr_columns, "Learning Rate", "Learning Rate", output_dir / "learning_rate.png"):
            generated += 1
    else:
        print("Skipping learning_rate.png; no learning-rate columns found.")

    print(f"Generated plots: {generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
