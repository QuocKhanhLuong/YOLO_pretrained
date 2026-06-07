# data_v3.0 Flat Backup Dataset Workflow

This workflow builds `data_v3.0` from a flat backup export that already has
YOLO image and label folders:

```text
back_up_data/
  images/
  labels/
  docker_labeled_manifest.csv
  labeled_images_manifest.csv
```

The source export may not contain `classes.txt`. In that case, pass the class
names in YOLO class-id order. Confirm this order before building; a wrong order
will make the labels look valid but semantically wrong.

## On 4070

```bash
cd /home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained
conda activate yolo
git pull

SRC=/home/linhdang/workspace2/binhanworkspace/back_up_data
VERSION=data_v3.0
OUT=data/versions/$VERSION
CLASS_NAMES=soldier,vehicle,fire
```

## Preflight Counts

```bash
du -sh "$SRC"

find "$SRC/images" -type f \( \
  -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o \
  -iname '*.bmp' -o -iname '*.webp' \
\) | wc -l

find "$SRC/labels" -type f -name '*.txt' | wc -l

find "$SRC/labels" -type f -name '*.txt' \
  -exec awk 'NF>=5 {print $1}' {} + \
  | sort -n | uniq -c
```

## Build data_v3.0

```bash
python scripts/prepare_data_version.py \
  --source "$SRC" \
  --output "$OUT" \
  --version "$VERSION" \
  --class-names "$CLASS_NAMES" \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  --force
```

The script auto-detects known manifest names such as
`docker_labeled_manifest.csv` and copies the selected manifest to
`data/versions/data_v3.0/manifest.csv`.

## Validate Labels

```bash
python scripts/create_yolo_yaml.py --dataset "$OUT"
python scripts/check_yolo_dataset.py --dataset "$OUT"

python scripts/eda_yolo_dataset.py \
  --dataset "$OUT" \
  --output-dir reports/data_v3_0_eda \
  --sample-images 40 \
  --seed 42
```

Review these files before training:

```text
data/versions/data_v3.0/dataset_report.md
reports/data_v3_0_eda/eda_report.md
reports/data_v3_0_eda/label_quality_warnings.csv
```

If `prepare_data_version.py` prints a non-zero `Skipped/problematic entries`
count, inspect `dataset_report.md` before continuing to DVC/package/upload.

## Compare With data_v2.0

```bash
python scripts/compare_dataset_versions.py \
  --old data/versions/data_v2.0 \
  --new "$OUT" \
  --output reports/dataset_v2_vs_v3.md
```

## Track With DVC

```bash
dvc add "$OUT"
git add data/versions/data_v3.0.dvc data/versions/.gitignore
git add scripts/prepare_data_version.py tests/test_prepare_data_version.py docs/V3_FLAT_BACKUP_WORKFLOW.md
git add reports/dataset_v2_vs_v3.md
git commit -m "Add data_v3.0 flat backup dataset"
git tag -a data_v3.0 -m "YOLO dataset v3.0 from flat backup export"
```

Push DVC data if the configured DVC remote is available:

```bash
dvc push
```

## Package And Upload To Drive

This uses the same packaging shape as `data_v2.0`: a `.tar.gz` archive plus a
`.sha256` file.

```bash
python scripts/package_dataset_version.py \
  --dataset "$OUT" \
  --output-dir archives

rclone mkdir Khanhdrive:YOLO_DVC_Backup/data_v3.0
rclone copy archives/data_v3.0_yolo_dataset.tar.gz Khanhdrive:YOLO_DVC_Backup/data_v3.0/ --progress
rclone copy archives/data_v3.0_yolo_dataset.tar.gz.sha256 Khanhdrive:YOLO_DVC_Backup/data_v3.0/ --progress
rclone ls Khanhdrive:YOLO_DVC_Backup/data_v3.0/
```

Verify the local archive checksum:

```bash
sha256sum archives/data_v3.0_yolo_dataset.tar.gz
cat archives/data_v3.0_yolo_dataset.tar.gz.sha256
```
