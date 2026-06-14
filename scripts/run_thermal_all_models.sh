#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# If OOM occurs, rerun with --batch 4 by directly calling run_batch_training.py.

set -x
python scripts/create_dataset_variant.py \
  --source data/versions/data_v3.1 \
  --output data/versions/data_v3.1_thermal \
  --mode thermal \
  --rename-thermal \
  --force

python scripts/run_batch_training.py \
  --dataset data/versions/data_v3.1_thermal \
  --models yolo11s.pt yolo11m.pt yolo26s.pt yolo26m.pt \
  --run-prefix thermal \
  --imgsz 1280 \
  --batch 8 \
  --epochs 100 \
  --device 0 \
  --optimizer AdamW \
  --lr0 0.0005 \
  --patience 25 \
  --workers 4 \
  --close-mosaic 20 \
  --conf-list 0.10 0.25 \
  --upload-remote Khanhdrive:YOLO_DVC_Backup/results_v3.1_thermal
