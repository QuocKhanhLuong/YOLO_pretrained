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

## YOLO Training Example

```bash
yolo detect train \
  data=data/versions/data_v1.0/data.yaml \
  model=yolov10n.pt \
  epochs=100 \
  imgsz=640
```

## ONNX Export Example

```bash
yolo export \
  model=runs/detect/train/weights/best.pt \
  format=onnx \
  opset=12
```
