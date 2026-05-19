# DVC Guide

## Why DVC

DVC keeps large dataset files out of Git while preserving reproducible dataset
versions. Git tracks code, scripts, docs, and `.dvc` pointer files. DVC tracks
the actual dataset contents and can push or pull them from shared storage.

## Initialize DVC

Run from the repo root:

```bash
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

## Add data_v1.0

Prepare the dataset first:

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

Then add the prepared version to DVC:

```bash
dvc add data/versions/data_v1.0
git add data/versions/data_v1.0.dvc .gitignore
git commit -m "Track data_v1.0 with DVC"
```

## Configure A DVC Remote

Choose a remote that matches your server setup. Examples:

```bash
dvc remote add -d dataset-storage /mnt/shared/dvc-storage
```

```bash
dvc remote add -d dataset-storage s3://your-bucket/path
```

Commit the remote config if it is safe to share:

```bash
git add .dvc/config
git commit -m "Configure DVC remote"
```

Do not commit secrets. Use local DVC config or environment credentials for
private tokens and cloud keys.

## Push And Pull Data

Push dataset files after `dvc add`:

```bash
dvc push
git push origin main
```

Pull dataset files on another machine:

```bash
git pull
dvc pull
```

## Tag Git Version data_v1.0

Tag the exact Git commit that contains the script version, guide updates, and
`data/versions/data_v1.0.dvc` pointer:

```bash
git tag data_v1.0
git push origin data_v1.0
```

## Future Data Versions

Use a new output directory for each dataset version:

```bash
python scripts/prepare_data_version.py \
  --source /home/linhdang/workspace2/binhanworkspace/label-img/data/label-system/output \
  --output data/versions/data_v1.1 \
  --version data_v1.1 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42

dvc add data/versions/data_v1.1
git add data/versions/data_v1.1.dvc .gitignore
git commit -m "Track data_v1.1 with DVC"
git tag data_v1.1
dvc push
git push origin main --tags
```

Version convention:

- `data_v1.0`: original exported dataset.
- `data_v1.1`: label-fix version.
- `data_v2.0`: added-data version.
