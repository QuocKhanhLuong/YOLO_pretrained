# Phase data_v2.0 DB Dataset Workflow

## A. Background

`data_v2.0` is built from the labeling-system database backup, not from the old
`output/` export.

DB source:

```text
backup_root/labeling_db.sql.gz
```

Image root on 4070:

```text
/home/linhdang/workspace2/binhanworkspace/label-img/data/label-system
```

Mapping:

```text
annotations.frame_id -> frames.id
frames.image_path -> image
annotations.yolo_text -> label
```

The old `backup_root/output` export is ignored. Labels inside
`backup_root/dataset_raw/*.zip` are ignored because they are raw empty template
labels. Final labels come from `public.annotations.yolo_text`.

## B. On Development Machine

Implement and review the scripts and docs in Git. Do not run the data build when
the backup path is unavailable on the development machine.

Recommended local checks:

```bash
python -m py_compile scripts/db_copy_parser.py
python -m py_compile scripts/inspect_db_annotations.py
python -m py_compile scripts/build_data_v2_from_db.py
python -m py_compile scripts/package_dataset_version.py
python -m py_compile scripts/compare_dataset_versions.py

python scripts/inspect_db_annotations.py --help
python scripts/build_data_v2_from_db.py --help
python scripts/package_dataset_version.py --help
python scripts/compare_dataset_versions.py --help
```

Push the code and documentation to Git, then run the build on the 4070 server.

## C. On 4070

```bash
cd /home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained
conda activate yolo
git pull
```

Inspect the DB annotations and image-path resolution first:

```bash
python scripts/inspect_db_annotations.py \
  --backup-root /home/linhdang/workspace2/binhanworkspace/back_up_data/20260526_155221 \
  --project-id 5 \
  --image-root-override /home/linhdang/workspace2/binhanworkspace/label-img/data/label-system
```

Build `data_v2.0`:

```bash
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

Validate the prepared dataset:

```bash
python scripts/create_yolo_yaml.py --dataset data/versions/data_v2.0
python scripts/check_yolo_dataset.py --dataset data/versions/data_v2.0
```

Compare against `data_v1.0`:

```bash
python scripts/compare_dataset_versions.py \
  --old data/versions/data_v1.0 \
  --new data/versions/data_v2.0 \
  --output reports/dataset_v1_vs_v2.md
```

Track the dataset with DVC and commit the metadata:

```bash
dvc add data/versions/data_v2.0
git add data/versions/data_v2.0.dvc data/versions/.gitignore scripts/ docs/ GUIDE.md reports/dataset_v1_vs_v2.md
git commit -m "Add data_v2.0 from DB submitted annotations"
git tag -a data_v2.0 -m "YOLO dataset v2.0 from DB submitted annotations"
```

Package the dataset:

```bash
python scripts/package_dataset_version.py \
  --dataset data/versions/data_v2.0 \
  --output-dir archives
```

Upload the package manually with rclone:

```bash
rclone mkdir Khanhdrive:YOLO_DVC_Backup/data_v2.0
rclone copy archives/data_v2.0_yolo_dataset.tar.gz Khanhdrive:YOLO_DVC_Backup/data_v2.0/ --progress
rclone copy archives/data_v2.0_yolo_dataset.tar.gz.sha256 Khanhdrive:YOLO_DVC_Backup/data_v2.0/ --progress
rclone ls Khanhdrive:YOLO_DVC_Backup/data_v2.0/
```

## D. On 5060Ti

SSH:

```bash
ssh -p 14980 bkcs@0.tcp.ap.ngrok.io
```

Commands:

```bash
cd /data1/bkcs/warlabel/YOLO_pretrained
git pull
mkdir -p archives
```

Download the package:

```bash
rclone copy Khanhdrive:YOLO_DVC_Backup/data_v2.0/data_v2.0_yolo_dataset.tar.gz archives/ --progress
rclone copy Khanhdrive:YOLO_DVC_Backup/data_v2.0/data_v2.0_yolo_dataset.tar.gz.sha256 archives/ --progress
```

Verify and unpack:

```bash
sha256sum archives/data_v2.0_yolo_dataset.tar.gz
cat archives/data_v2.0_yolo_dataset.tar.gz.sha256

tar -xzf archives/data_v2.0_yolo_dataset.tar.gz
```

Regenerate and validate `data.yaml` for the 5060Ti absolute path:

```bash
python scripts/create_yolo_yaml.py --dataset data/versions/data_v2.0
python scripts/check_yolo_dataset.py --dataset data/versions/data_v2.0
```

Check CUDA:

```bash
conda activate /data1/bkcs/conda_envs/yolo
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Prepare pretrained weights:

```bash
python scripts/prepare_pretrained_weights.py \
  --model yolo11s.pt \
  --output-dir pretrained_weights
```

Train:

```bash
python scripts/train_yolo.py \
  --data data/versions/data_v2.0/data.yaml \
  --weights pretrained_weights/yolo11s.pt \
  --epochs 80 \
  --imgsz 1280 \
  --batch 4 \
  --device 0 \
  --project runs \
  --name yolo11s_data_v2_0_img1280_b4 \
  --optimizer AdamW \
  --lr0 0.0005 \
  --patience 20 \
  --workers 2 \
  --close-mosaic 20
```

## E. After Training

Use the existing scripts:

```bash
python scripts/plot_training_metrics.py --help
python scripts/validate_yolo.py --help
python scripts/predict_yolo.py --help
python scripts/generate_training_report.py --help
```

Then run them against the completed `runs/yolo11s_data_v2_0_img1280_b4`
artifacts.
