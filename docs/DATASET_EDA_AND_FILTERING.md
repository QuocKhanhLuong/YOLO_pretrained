# Dataset EDA And Class Filtering

This guide covers the post-baseline cleanup path after reviewing the
`runs/yolo11s_data_v1_0_baseline_cuda121` training artifacts.

The baseline run showed two important findings:

- The `111/111` training log count is batch count, not image count. With
  `batch=8`, `111 * 8 = 888` train images.
- The original `data_v1.0` run used three classes: `soldier`, `vehicle`, and
  `fire`. If the next model should only detect `vehicle` and `soldier`, create a
  new dataset version instead of modifying `data_v1.0`.

## Run EDA On data_v1.0

```bash
python scripts/eda_yolo_dataset.py \
  --dataset data/versions/data_v1.0 \
  --output-dir reports/data_v1.0/eda
```

Generated artifacts:

- `reports/data_v1.0/eda/eda_report.md`
- `reports/data_v1.0/eda/eda_summary.json`
- `reports/data_v1.0/eda/image_summary.csv`
- `reports/data_v1.0/eda/box_summary.csv`
- `reports/data_v1.0/eda/plots/`

Use this to inspect:

- image width/height/aspect-ratio distribution
- object counts by class and split
- normalized bbox width/height/area distribution
- empty label counts
- missing image-label pairs
- invalid label rows

## Create A Two-Class Dataset

Create `data_v1.1_vehicle_soldier` from `data_v1.0`:

```bash
python scripts/filter_yolo_classes.py \
  --source-dataset data/versions/data_v1.0 \
  --output data/versions/data_v1.1_vehicle_soldier \
  --version data_v1.1_vehicle_soldier \
  --include-classes vehicle soldier
```

The class order above produces:

```text
0 vehicle
1 soldier
```

Images that only contained removed classes, such as `fire`, are kept with empty
labels by default. This gives the model negative/background examples. To drop
those images instead:

```bash
python scripts/filter_yolo_classes.py \
  --source-dataset data/versions/data_v1.0 \
  --output data/versions/data_v1.1_vehicle_soldier \
  --version data_v1.1_vehicle_soldier \
  --include-classes vehicle soldier \
  --drop-empty-after-filter \
  --force
```

## Validate The Two-Class Dataset

```bash
python scripts/check_yolo_dataset.py \
  --dataset data/versions/data_v1.1_vehicle_soldier
```

Regenerate `data.yaml` if the repo was pulled to a new absolute path:

```bash
python scripts/create_yolo_yaml.py \
  --dataset data/versions/data_v1.1_vehicle_soldier
```

Run EDA again:

```bash
python scripts/eda_yolo_dataset.py \
  --dataset data/versions/data_v1.1_vehicle_soldier \
  --output-dir reports/data_v1.1_vehicle_soldier/eda
```

## DVC Track The New Dataset Version

```bash
dvc add data/versions/data_v1.1_vehicle_soldier
git add data/versions/data_v1.1_vehicle_soldier.dvc data/versions/.gitignore
git add scripts docs GUIDE.md CHANGELOG.md requirements.txt
git commit -m "Add EDA and two-class dataset filtering workflow"
git tag data_v1.1_vehicle_soldier
dvc push
git push origin main --tags
```

## Recommended Retrain Command

Start with a lower learning rate and larger image size:

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

If GPU memory is limited, lower `imgsz` to `960` or keep `imgsz=1280` and lower
`batch` further.

## What To Check After Retraining

- Did `metrics/mAP50-95(B)` improve over the baseline best value?
- Did recall improve for `vehicle`, which was weak in the baseline confusion matrix?
- Are many true `vehicle` or `soldier` instances still going to background?
- Do false positives happen mostly on removed `fire` images?
- Do tiny objects require tiling/crop preparation in a later dataset version?
