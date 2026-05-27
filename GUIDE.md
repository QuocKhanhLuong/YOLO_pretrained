# YOLO Dataset Preparation Guide

This repo prepares exported Labeling System data for YOLO training and DVC
versioning. The scripts are reusable: they do not assume that the real dataset
exists locally, and they do not hard-code the source dataset path.

## Real Server Source

The real server source path is:

```text
/home/linhdang/workspace2/binhanworkspace/label-img/data/label-system/output
```

Expected source structure:

```text
images/
labels/
classes.txt
manifest.csv
```

## Local Mock Run

Use a temporary mock dataset when developing locally:

```bash
mkdir -p /tmp/yolo_mock_dataset/images /tmp/yolo_mock_dataset/labels
printf "class_0\nclass_1\n" > /tmp/yolo_mock_dataset/classes.txt
printf "filename,split\nsample_1.jpg,\nsample_2.jpg,\n" > /tmp/yolo_mock_dataset/manifest.csv
touch /tmp/yolo_mock_dataset/images/sample_1.jpg
touch /tmp/yolo_mock_dataset/images/sample_2.jpg
printf "0 0.5 0.5 0.4 0.4\n" > /tmp/yolo_mock_dataset/labels/sample_1.txt
: > /tmp/yolo_mock_dataset/labels/sample_2.txt

python scripts/prepare_data_version.py \
  --source /tmp/yolo_mock_dataset \
  --output data/versions/data_v1.0 \
  --version data_v1.0 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --force
```

## Server Preparation Command

Run this from the repo root on the server:

```bash
python scripts/prepare_data_version.py \
  --source /home/linhdang/workspace2/binhanworkspace/label-img/data/label-system/output \
  --output data/versions/data_v1.0 \
  --version data_v1.0 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

If the repo root is `/home/linhdang/workspace2/binhanworkspace/label-img`, the
prepared dataset resolves to:

```text
/home/linhdang/workspace2/binhanworkspace/label-img/data/versions/data_v1.0
```

Use `--force` only when you intentionally want to replace an existing prepared
version directory.

## Validate A Prepared Dataset

```bash
python scripts/check_yolo_dataset.py \
  --dataset data/versions/data_v1.0
```

The checker validates required folders, `data.yaml`, `classes.txt`, image-label
pairing, and YOLO label format. It prints summary counts and the first 100
errors.

## Invalid Label Policy

`prepare_data_version.py` does not copy samples with invalid label files into the
prepared dataset. It records invalid lines, missing labels, orphan labels,
duplicate image stems, and unsupported image files in `dataset_report.md`.
Valid empty label files are copied and kept.

## Regenerate data.yaml

```bash
python scripts/create_yolo_yaml.py \
  --dataset data/versions/data_v1.0
```

The generated `data.yaml` stores the dataset `path` as an absolute path.
`create_yolo_yaml.py` overwrites `data.yaml` in the dataset directory.

## DVC Workflow

DVC commands are documented only; the Python scripts never run DVC automatically.

```bash
dvc init
dvc add data/versions/data_v1.0
git add .dvc .dvcignore data/versions/data_v1.0.dvc scripts docs GUIDE.md CHANGELOG.md
git commit -m "Add data_v1.0 dataset preparation toolkit"
git tag data_v1.0
```

Configure a remote when storage is ready:

```bash
dvc remote add -d dataset-storage <remote-url-or-path>
dvc push
git push origin main --tags
```

On another machine:

```bash
git pull
dvc pull
```

## Dataset Version Convention

- `data_v1.0`: original exported dataset.
- `data_v1.1`: label-fix version.
- `data_v2.0`: added-data version.

## Phase data_v2.0: DB Backup Workflow

`data_v2.0` is built from `backup_root/labeling_db.sql.gz` using DB frames and
DB annotation `yolo_text`, then tracked with DVC, packaged, transferred with
rclone, and trained on the 5060Ti machine.

Start with the full workflow:

```bash
docs/V2_DB_DATASET_WORKFLOW.md
```

Core server commands:

```bash
python scripts/inspect_db_annotations.py \
  --backup-root /home/linhdang/workspace2/binhanworkspace/back_up_data/20260526_155221 \
  --project-id 5 \
  --image-root-override /home/linhdang/workspace2/binhanworkspace/label-img/data/label-system

python scripts/build_data_v2_from_db.py \
  --backup-root /home/linhdang/workspace2/binhanworkspace/back_up_data/20260526_155221 \
  --output data/versions/data_v2.0 \
  --version data_v2.0 \
  --project-id 5 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --split-by-video \
  --image-root-override /home/linhdang/workspace2/binhanworkspace/label-img/data/label-system \
  --max-missing-ratio 0.03 \
  --force
```

Do not use `backup_root/output` or `backup_root/dataset_raw/*.zip` labels for
this phase.

## EDA And Two-Class Dataset Version

After the first baseline run, use EDA before training again:

```bash
python scripts/eda_yolo_dataset.py \
  --dataset data/versions/data_v1.0 \
  --output-dir reports/data_v1.0/eda
```

Create a two-class version for `vehicle` and `soldier`:

```bash
python scripts/filter_yolo_classes.py \
  --source-dataset data/versions/data_v1.0 \
  --output data/versions/data_v1.1_vehicle_soldier \
  --version data_v1.1_vehicle_soldier \
  --include-classes vehicle soldier
```

Validate and inspect the new version:

```bash
python scripts/check_yolo_dataset.py \
  --dataset data/versions/data_v1.1_vehicle_soldier

python scripts/eda_yolo_dataset.py \
  --dataset data/versions/data_v1.1_vehicle_soldier \
  --output-dir reports/data_v1.1_vehicle_soldier/eda
```

The class order above remaps labels to `0 vehicle` and `1 soldier`. Images that
only had removed classes are kept as empty-label background samples by default.
See `docs/DATASET_EDA_AND_FILTERING.md` for details.

## YOLO Training Example

```bash
yolo detect train \
  data=data/versions/data_v1.0/data.yaml \
  model=yolo11s.pt \
  epochs=100 \
  imgsz=960
```

## Phase 2 YOLO Fine-Tuning Pipeline

Phase 2 fine-tunes an Ultralytics YOLO model on the prepared dataset. The scope
is generic object/person detection training, metric tracking, visualization, and
export preparation.

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Prepare pretrained weights:

```bash
python scripts/prepare_pretrained_weights.py \
  --model yolo11s.pt \
  --output-dir pretrained_weights
```

Train the baseline:

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

Train the two-class follow-up:

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

Plot metrics:

```bash
python scripts/plot_training_metrics.py \
  --results runs/yolo11s_data_v1_0_baseline/results.csv \
  --output-dir reports/yolo11s_data_v1_0_baseline/plots
```

Validate `best.pt`:

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

Run test prediction visualization:

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

Generate the training report:

```bash
python scripts/generate_training_report.py \
  --run-dir runs/yolo11s_data_v1_0_baseline \
  --dataset data/versions/data_v1.0 \
  --output reports/yolo11s_data_v1_0_baseline/training_report.md
```

For details, see `docs/TRAINING_GUIDE.md`.

## ONNX Export Example

```bash
yolo export \
  model=runs/yolo11s_data_v1_0_baseline/weights/best.pt \
  format=onnx \
  opset=12
```
