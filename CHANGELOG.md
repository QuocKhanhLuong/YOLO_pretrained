# Changelog

## 2026-05-19

### Added

- Added `scripts/prepare_data_version.py` to validate Labeling System exports,
  split valid image-label pairs into train/val/test, generate YOLO `data.yaml`,
  and write dataset-level `dataset_report.md` and `GUIDE.md`.
- Added `scripts/check_yolo_dataset.py` to validate prepared YOLO dataset
  structure, image-label pairing, and label format.
- Added `scripts/create_yolo_yaml.py` to regenerate `data.yaml` from
  `classes.txt`.
- Added root `GUIDE.md` with local mock usage, server preparation command, DVC
  workflow, YOLO training, and ONNX export examples.
- Added `docs/DVC_GUIDE.md` and `docs/DATASET_GUIDE.md` for DVC and dataset
  structure documentation.
- Added `docs/meetings/20260519T060033Z-deepdive.md` with the ATeam deepdive
  findings and follow-up recommendations for tests and reproducibility.
