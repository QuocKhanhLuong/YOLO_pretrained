#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-data/versions/data_v3.0}"
DATA="${DATA:-$DATASET/data.yaml}"
WEIGHTS_DIR="${WEIGHTS_DIR:-pretrained_weights}"
RUNS_DIR="${RUNS_DIR:-runs}"
REPORTS_DIR="${REPORTS_DIR:-reports/waruav_v3}"
WANDB_PROJECT="${WANDB_PROJECT:-waruav}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"

EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-1280}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-2}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
LR0="${LR0:-0.0005}"
PATIENCE="${PATIENCE:-20}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-20}"
SEED="${SEED:-42}"

BATCH_11S="${BATCH_11S:-4}"
BATCH_11M="${BATCH_11M:-2}"
BATCH_26S="${BATCH_26S:-4}"
BATCH_26M="${BATCH_26M:-2}"

mkdir -p "$WEIGHTS_DIR" "$RUNS_DIR" "$REPORTS_DIR" "$WANDB_DIR"
export WANDB_PROJECT WANDB_MODE WANDB_DIR

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: data.yaml not found: $DATA" >&2
  exit 1
fi

python -c "import wandb" >/dev/null 2>&1 || {
  echo "ERROR: wandb is not installed. Run: python -m pip install -U wandb" >&2
  exit 1
}

yolo settings wandb=True >/dev/null 2>&1 || echo "WARNING: could not set Ultralytics wandb=True; continuing with WANDB_PROJECT=$WANDB_PROJECT" >&2

MODEL_SPECS=(
  "yolo11s.pt:$BATCH_11S"
  "yolo11m.pt:$BATCH_11M"
  "yolo26s.pt:$BATCH_26S"
  "yolo26m.pt:$BATCH_26M"
)

for spec in "${MODEL_SPECS[@]}"; do
  model="${spec%%:*}"
  batch="${spec##*:}"
  model_id="${model%.pt}"
  run_name="${model_id}_data_v3_0_img${IMGSZ}_b${batch}_waruav"
  run_dir="$RUNS_DIR/$run_name"
  report_dir="$REPORTS_DIR/$run_name"
  weight_path="$WEIGHTS_DIR/$model"
  best_weight="$run_dir/weights/best.pt"
  test_name="${run_name}_test"
  pred_conf010_name="${run_name}_pred_conf010"
  pred_conf025_name="${run_name}_pred_conf025"

  mkdir -p "$report_dir"
  export WANDB_NAME="$run_name"

  echo "== Preparing weight: $model =="
  python scripts/prepare_pretrained_weights.py --model "$model" --output-dir "$WEIGHTS_DIR"

  echo "== Training: $run_name =="
  python scripts/train_yolo.py --data "$DATA" --weights "$weight_path" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$batch" --device "$DEVICE" --project "$RUNS_DIR" --name "$run_name" --optimizer "$OPTIMIZER" --lr0 "$LR0" --patience "$PATIENCE" --workers "$WORKERS" --close-mosaic "$CLOSE_MOSAIC" --seed "$SEED"

  if [[ ! -f "$best_weight" ]]; then
    echo "ERROR: best.pt not found after training: $best_weight" >&2
    exit 1
  fi

  echo "== Test validation: $run_name =="
  python scripts/validate_yolo.py --weights "$best_weight" --data "$DATA" --imgsz "$IMGSZ" --batch "$batch" --device "$DEVICE" --split test --project "$RUNS_DIR" --name "$test_name"

  echo "== Predict test split at conf=0.10: $run_name =="
  python scripts/predict_yolo.py --weights "$best_weight" --source "$DATASET/images/test" --imgsz "$IMGSZ" --conf 0.10 --iou 0.5 --device "$DEVICE" --project "$RUNS_DIR" --name "$pred_conf010_name"

  echo "== Predict test split at conf=0.25: $run_name =="
  python scripts/predict_yolo.py --weights "$best_weight" --source "$DATASET/images/test" --imgsz "$IMGSZ" --conf 0.25 --iou 0.5 --device "$DEVICE" --project "$RUNS_DIR" --name "$pred_conf025_name"

  echo "== Generate plots and reports: $run_name =="
  python scripts/plot_training_metrics.py --results "$run_dir/results.csv" --output-dir "$report_dir/plots"
  python scripts/generate_training_report.py --run-dir "$run_dir" --dataset "$DATASET" --output "$report_dir/training_report.md"
  python scripts/generate_experiment_report_md.py --run-dir "$run_dir" --dataset "$DATASET" --test-dir "$RUNS_DIR/$test_name" --pred-dir "$RUNS_DIR/$pred_conf010_name" --pred-dir-conf025 "$RUNS_DIR/$pred_conf025_name" --run-name "$run_name" --notes "W&B project: $WANDB_PROJECT; dataset: data_v3.0; model: $model; imgsz: $IMGSZ; batch: $batch" --output "$report_dir/experiment_report.md"

  echo "== Completed: $run_name =="
  echo "Report: $report_dir/experiment_report.md"
done

echo "All waruav data_v3.0 runs completed."
