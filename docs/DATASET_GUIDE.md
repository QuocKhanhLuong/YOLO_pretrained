# Dataset Guide

Prepared datasets use this structure:

```text
data/versions/data_v1.0/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── classes.txt
├── manifest.csv
├── data.yaml
├── dataset_report.md
└── GUIDE.md
```

## images/

`images/` contains image files split into:

- `images/train/`
- `images/val/`
- `images/test/`

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`,
case-insensitive.

## labels/

`labels/` contains matching YOLO `.txt` label files split into:

- `labels/train/`
- `labels/val/`
- `labels/test/`

Each image must have a matching label with the same relative path and basename.
For example:

```text
images/train/example.jpg
labels/train/example.txt
```

An empty label file is valid and means the image has no labeled objects.

## classes.txt

`classes.txt` contains one class name per line. Class IDs in label files are
zero-based indexes into this file.

Example:

```text
helmet
no_helmet
person
```

## manifest.csv

`manifest.csv` is copied from the Labeling System export. It records source-side
metadata and should be kept with the prepared dataset for auditability.

## data.yaml

`data.yaml` is the YOLO training config generated from `classes.txt`.

Example:

```yaml
path: "/absolute/path/to/data/versions/data_v1.0"
train: "images/train"
val: "images/val"
test: "images/test"

names:
  0: "helmet"
  1: "no_helmet"
  2: "person"
```

## dataset_report.md

`dataset_report.md` is generated during preparation. It records:

- source and output paths
- dataset version
- split ratios and seed
- class list
- train/val/test sample counts
- images skipped because labels were missing
- orphan labels without images
- invalid label lines
- duplicate image stems
- unsupported files under `images/`

## GUIDE.md

The dataset-level `GUIDE.md` explains how to validate that specific prepared
version, regenerate `data.yaml`, and launch a YOLO training command.

## YOLO Label Format

Each non-empty label line must contain exactly five values:

```text
class_id x_center y_center width height
```

Rules:

- `class_id` is an integer.
- `class_id` is in `[0, num_classes - 1]`.
- `x_center` and `y_center` are floats in `[0, 1]`.
- `width` and `height` are floats greater than `0` and less than or equal to `1`.
- All coordinates are normalized to `[0, 1]` relative to image width and height.
