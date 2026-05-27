# Phase 2 Training Guide

Phase 2 fine-tunes a pretrained Ultralytics YOLO detection model on the prepared
YOLO dataset from Phase 1. The scope is generic UAV object/person detection for
model development, metric tracking, visualization, and export preparation.

Excluded from this repo: military targeting, weapon engagement, autonomous
navigation, evasion, surveillance deployment guidance, or real-time operational
decisioning.

## Why Use Pretrained YOLO

Pretrained YOLO weights already contain general visual features learned from
large detection datasets. Fine-tuning from pretrained weights usually converges
faster and performs better than training from scratch, especially when the custom
dataset is small or has limited class variety.

Preferred baseline:

```text
yolo11s.pt
```

Also supported:

```text
yolo11n.pt
yolo11m.pt
custom .pt path
```

## Is Extra Preprocessing Needed?

If `data_v1.0` is already in YOLO detection format with normalized labels, no
extra manual preprocessing is required before training.

Ultralytics automatically handles image resizing, normalization, batching, and
default augmentations during training.

Extra preprocessing is only needed for special cases such as:

- very large UAV images
- tiny objects that become too small after resizing
- duplicated frames
- invalid labels
- severe class imbalance

For small UAV objects, start by trying larger image sizes such as `960` or
`1280`. If objects are still too small, add tiling in a later phase.

## Dataset Requirement

The baseline command assumes:

```text
data/versions/data_v1.0/data.yaml
```

Validate the prepared dataset first:

```bash
python scripts/check_yolo_dataset.py \
  --dataset data/versions/data_v1.0
```

## Prepare Pretrained Weights

```bash
python scripts/prepare_pretrained_weights.py \
  --model yolo11s.pt \
  --output-dir pretrained_weights
```

Expected output:

```text
pretrained_weights/yolo11s.pt
```

## Train Baseline

```bash
python scripts/train_yolo.py \
  --data data/versions/data_v1.0/data.yaml \
  --weights pretrained_weights/yolo11s.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --project runs \
  --name yolo11s_data_v1_0_baseline \
  --optimizer AdamW \
  --lr0 0.001 \
  --patience 30 \
  --workers 4
```

The script writes experiment metadata to:

```text
runs/yolo11s_data_v1_0_baseline/experiment_metadata.json
```

It also prints the expected checkpoint paths:

```text
runs/yolo11s_data_v1_0_baseline/weights/best.pt
runs/yolo11s_data_v1_0_baseline/weights/last.pt
```

Use `--dry-run` to validate paths and write metadata without starting training.

## Monitor Training Metrics

Ultralytics writes `results.csv` under the run directory. Watch these columns:

- `train/box_loss`
- `train/cls_loss`
- `train/dfl_loss`
- `val/box_loss`
- `val/cls_loss`
- `val/dfl_loss`
- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`

## Plot Metrics

```bash
python scripts/plot_training_metrics.py \
  --results runs/yolo11s_data_v1_0_baseline/results.csv \
  --output-dir reports/yolo11s_data_v1_0_baseline/plots
```

Generated plots, when the corresponding CSV columns exist:

- `train_val_box_loss.png`
- `train_val_cls_loss.png`
- `train_val_dfl_loss.png`
- `map50_map5095.png`
- `precision_recall.png`
- `learning_rate.png`

## Validate best.pt

Validation split:

```bash
python scripts/validate_yolo.py \
  --weights runs/yolo11s_data_v1_0_baseline/weights/best.pt \
  --data data/versions/data_v1.0/data.yaml \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --split val \
  --project runs \
  --name yolo11s_data_v1_0_val
```

Test split:

```bash
python scripts/validate_yolo.py \
  --weights runs/yolo11s_data_v1_0_baseline/weights/best.pt \
  --data data/versions/data_v1.0/data.yaml \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --split test \
  --project runs \
  --name yolo11s_data_v1_0_test
```

## Run Test Prediction Visualization

```bash
python scripts/predict_yolo.py \
  --weights runs/yolo11s_data_v1_0_baseline/weights/best.pt \
  --source data/versions/data_v1.0/images/test \
  --imgsz 960 \
  --conf 0.25 \
  --iou 0.5 \
  --device 0 \
  --project runs \
  --name yolo11s_data_v1_0_test_predictions \
  --max-det 300
```

This saves annotated prediction images or videos. It does not implement
tracking, targeting, or deployment logic.

## Generate Training Report

```bash
python scripts/generate_training_report.py \
  --run-dir runs/yolo11s_data_v1_0_baseline \
  --dataset data/versions/data_v1.0 \
  --output reports/yolo11s_data_v1_0_baseline/training_report.md
```

## Generate Markdown Experiment Report

For `data_v2.0` runs, generate test artifacts and prediction samples before
building the richer Markdown experiment report:

```bash
RUN=yolo11s_data_v2_0_img1280_b12_e100

python scripts/validate_yolo.py \
  --weights runs/$RUN/weights/best.pt \
  --data data/versions/data_v2.0/data.yaml \
  --imgsz 1280 \
  --batch 12 \
  --device 0 \
  --split test \
  --project runs \
  --name ${RUN}_test

python scripts/predict_yolo.py \
  --weights runs/$RUN/weights/best.pt \
  --source data/versions/data_v2.0/images/test \
  --imgsz 1280 \
  --conf 0.10 \
  --iou 0.5 \
  --device 0 \
  --project runs \
  --name ${RUN}_pred_conf010 \
  --max-det 300

python scripts/predict_yolo.py \
  --weights runs/$RUN/weights/best.pt \
  --source data/versions/data_v2.0/images/test \
  --imgsz 1280 \
  --conf 0.25 \
  --iou 0.5 \
  --device 0 \
  --project runs \
  --name ${RUN}_pred_conf025 \
  --max-det 300

python scripts/generate_experiment_report_md.py \
  --run-dir runs/$RUN \
  --test-dir runs/${RUN}_test \
  --pred-dir runs/${RUN}_pred_conf010 \
  --pred-dir-conf025 runs/${RUN}_pred_conf025 \
  --dataset data/versions/data_v2.0 \
  --output reports/$RUN/experiment_report.md
```

The report reads `results.csv`, `args.yaml`, `experiment_metadata.json`, dataset
label counts, Ultralytics plots, optional test plots, and qualitative prediction
images. Missing artifacts are listed in the report instead of failing the run.

## Metric Interpretation

- `box_loss`: localization error for predicted bounding boxes. Lower is better.
- `cls_loss`: classification error. Lower is better.
- `dfl_loss`: Distribution Focal Loss for box localization quality. Lower is better.
- `precision`: of predicted detections, how many are correct. Low precision means too many false positives.
- `recall`: of real labeled objects, how many were found. Low recall means too many missed objects.
- `mAP50`: mean Average Precision at IoU 0.50. Useful for a permissive quality view.
- `mAP50-95`: mean Average Precision across IoU thresholds 0.50 to 0.95. Stricter and more important for final model comparison.

## Recommended Small-Object Settings

- Start with `imgsz 960`.
- Try `imgsz 1280` if GPU memory allows.
- Adjust `batch` to fit GPU memory.
- Keep `close_mosaic` around `10` to `15`.
- Reduce heavy geometry and color augmentation when tiny objects become blurred,
  cropped away, or visually unrealistic.
- Inspect labels carefully, especially for tiny objects.
- Keep a separate test split for final model selection.

## Low-Augmentation Recipes

Use these recipes as controlled experiments, not as replacements for label and
split checks. Keep the same dataset, weights, image size, seed, and train/val
split when comparing augmentation settings.

### Small UAV Objects

Tiny objects can be harmed by strong mosaic, scale, translation, or color jitter
because the object may become even smaller or visually inconsistent. Start with
a lighter augmentation run:

```bash
python scripts/train_yolo.py \
  --data data/versions/data_v1.0/data.yaml \
  --weights pretrained_weights/yolo11s.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 4 \
  --device 0 \
  --project runs \
  --name yolo11s_data_v1_0_small_uav_low_aug \
  --optimizer AdamW \
  --lr0 0.0005 \
  --patience 30 \
  --workers 4 \
  --mosaic 0.2 \
  --scale 0.2 \
  --translate 0.05 \
  --hsv-h 0.005 \
  --hsv-s 0.25 \
  --hsv-v 0.20 \
  --degrees 0.0 \
  --fliplr 0.5 \
  --flipud 0.0 \
  --close-mosaic 10 \
  --weight-decay 0.0005 \
  --warmup-epochs 3.0
```

If validation recall improves but precision drops, review false positives before
changing the recipe. If both precision and recall stay poor, inspect labels and
consider tiling/cropping instead of raising augmentation strength.

### Very Small or Noisy Datasets

When the dataset is small, duplicated, or has inconsistent labels, reduce
augmentation further so the model first learns the actual annotation policy:

```bash
python scripts/train_yolo.py \
  --data data/versions/data_v1.0/data.yaml \
  --weights pretrained_weights/yolo11s.pt \
  --epochs 80 \
  --imgsz 1280 \
  --batch 4 \
  --device 0 \
  --project runs \
  --name yolo11s_data_v1_0_min_aug_label_audit \
  --optimizer AdamW \
  --lr0 0.0003 \
  --patience 20 \
  --workers 4 \
  --mosaic 0.0 \
  --scale 0.1 \
  --translate 0.02 \
  --hsv-h 0.0 \
  --hsv-s 0.15 \
  --hsv-v 0.15 \
  --degrees 0.0 \
  --fliplr 0.5 \
  --flipud 0.0 \
  --close-mosaic 0 \
  --freeze 10 \
  --weight-decay 0.0007 \
  --warmup-epochs 4.0
```

Use this run to find mislabeled samples, missing boxes, duplicated frames across
splits, or classes that are not visually separable at the current resolution.
`--freeze 10` is a transfer-learning control for small datasets; compare it
against an unfrozen run before keeping it.

### Class-Confusion Reduction

If the confusion matrix shows visually similar classes being swapped, first
reduce transforms that change appearance or crop context. Keep horizontal flips
only when left/right orientation does not change the label:

```bash
python scripts/train_yolo.py \
  --data data/versions/data_v1.0/data.yaml \
  --weights pretrained_weights/yolo11s.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 4 \
  --device 0 \
  --project runs \
  --name yolo11s_data_v1_0_class_confusion_low_aug \
  --optimizer AdamW \
  --lr0 0.0003 \
  --patience 30 \
  --workers 4 \
  --mosaic 0.1 \
  --scale 0.15 \
  --translate 0.03 \
  --hsv-h 0.003 \
  --hsv-s 0.20 \
  --hsv-v 0.20 \
  --degrees 0.0 \
  --fliplr 0.5 \
  --flipud 0.0 \
  --close-mosaic 15 \
  --weight-decay 0.0007 \
  --warmup-epochs 4.0
```

If class confusion remains high, compare per-class examples side by side and
fix the dataset before tuning more hyperparameters. For a temporary objectness
sanity check only, use `--single-cls`; do not use that setting for final
multi-class model comparison.

## Two-Class Retraining Path

If the target task is only `vehicle` and `soldier`, create a new filtered dataset
version before retraining:

```bash
python scripts/filter_yolo_classes.py \
  --source-dataset data/versions/data_v1.0 \
  --output data/versions/data_v1.1_vehicle_soldier \
  --version data_v1.1_vehicle_soldier \
  --include-classes vehicle soldier
```

Run EDA before and after filtering:

```bash
python scripts/eda_yolo_dataset.py \
  --dataset data/versions/data_v1.0 \
  --output-dir reports/data_v1.0/eda

python scripts/eda_yolo_dataset.py \
  --dataset data/versions/data_v1.1_vehicle_soldier \
  --output-dir reports/data_v1.1_vehicle_soldier/eda
```

Recommended follow-up training command:

```bash
python scripts/train_yolo.py \
  --data data/versions/data_v1.1_vehicle_soldier/data.yaml \
  --weights pretrained_weights/yolo11s.pt \
  --epochs 80 \
  --imgsz 1280 \
  --batch 4 \
  --device 0 \
  --project runs \
  --name yolo11s_data_v1_1_vehicle_soldier_img1280_lr3e4 \
  --optimizer AdamW \
  --lr0 0.0003 \
  --patience 20 \
  --workers 4 \
  --close-mosaic 10
```

If this still misses small objects, move to a tiling/crop dataset preparation
phase instead of only increasing epochs.

## If Overfitting Happens

- Check whether training loss keeps dropping while validation mAP stalls or falls.
- Add more data or stronger diversity if available.
- Reduce epochs or rely on `patience`.
- Try a smaller model such as `yolo11n.pt`.
- Inspect train/val split quality and remove duplicate near-identical frames across splits.

## If Recall Is Low

- Lower confidence threshold during prediction review.
- Add more examples of missed object types.
- Increase `imgsz` for small objects.
- Inspect labels for missing boxes.
- Consider tiling very large UAV images in a later phase.

## If Precision Is Low

- Inspect false positives in prediction visualizations.
- Add more negative/background examples.
- Check whether classes are ambiguous or mislabeled.
- Increase confidence threshold for visualization after training.

## If Small Objects Are Missed

- Train with `imgsz 960` or `1280`.
- Inspect annotation tightness and consistency.
- Avoid excessive downscaling in future preprocessing.
- Add tiling as a Phase 3 or later data-preparation extension if needed.

## Phase 3 TODO

- Export `best.pt` to ONNX.
- Benchmark ONNX and TensorRT candidates.
- Keep export scripts separate from training so model quality checks remain clear.
