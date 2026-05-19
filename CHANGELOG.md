# Changelog

## 2026-05-20

### Added

- Added `scripts/eda_yolo_dataset.py` to generate YOLO dataset EDA reports,
  image summaries, box summaries, JSON stats, and optional plots.
- Added `scripts/filter_yolo_classes.py` to create filtered/remapped dataset
  versions such as `data_v1.1_vehicle_soldier` from `data_v1.0` without
  modifying the source dataset.
- Added `docs/DATASET_EDA_AND_FILTERING.md` with the EDA, two-class filtering,
  DVC tracking, and recommended retraining workflow.
- Updated `GUIDE.md`, `docs/TRAINING_GUIDE.md`, and `docs/DATASET_GUIDE.md`
  with the data_v1.0 EDA and vehicle/soldier retraining commands.
- Added `pillow` to `requirements.txt` for image-size EDA.

## 2026-05-19

### Added

- Added Phase 2 YOLO fine-tuning scripts:
  `scripts/prepare_pretrained_weights.py`, `scripts/train_yolo.py`,
  `scripts/plot_training_metrics.py`, `scripts/validate_yolo.py`,
  `scripts/predict_yolo.py`, and `scripts/generate_training_report.py`.
- Added `docs/TRAINING_GUIDE.md` with baseline training, validation,
  prediction visualization, metric interpretation, preprocessing guidance, and
  small-object UAV recommendations.
- Added `docs/meetings/20260519T114357Z-deepdive.md` with Phase 2 ATeam
  findings and follow-up recommendations.
- Added `requirements.txt` with Phase 2 runtime dependencies.
- Updated root `GUIDE.md` with Phase 2 server commands.
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
