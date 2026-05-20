# YOLO Dataset EDA

Dataset: `/home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained/data/versions/data_v1.0`

## Split Counts

| Split | Images | Labels | Empty Labels |
|---|---:|---:|---:|
| train | 888 | 888 | 0 |
| val | 111 | 111 | 0 |
| test | 112 | 112 | 0 |

## Class Counts

| Split | soldier | vehicle | fire |
|---|---:|---:|---:|
| train | 700 | 318 | 415 |
| val | 81 | 37 | 60 |
| test | 103 | 37 | 42 |

## Image Size Stats

```json
{
  "width": {
    "min": 234.0,
    "p25": 1280.0,
    "median": 1280.0,
    "p75": 1280.0,
    "max": 3840.0,
    "mean": 1293.6255625562555
  },
  "height": {
    "min": 352.0,
    "p25": 720.0,
    "median": 720.0,
    "p75": 960.0,
    "max": 2160.0,
    "mean": 794.3294329432944
  },
  "aspect_ratio": {
    "min": 0.525,
    "p25": 1.7777777777777777,
    "median": 1.7777777777777777,
    "p75": 1.7777777777777777,
    "max": 2.409090909090909,
    "mean": 1.6632430411910781
  }
}
```

## Bounding Box Stats

```json
{
  "width": {
    "min": 0.000781,
    "p25": 0.042188,
    "median": 0.082031,
    "p75": 0.185938,
    "max": 0.988095,
    "mean": 0.1373068834355828
  },
  "height": {
    "min": 0.002778,
    "p25": 0.070833,
    "median": 0.136111,
    "p75": 0.300866,
    "max": 0.997845,
    "mean": 0.2134945861684328
  },
  "area": {
    "min": 2.169618e-06,
    "p25": 0.0031901151040000003,
    "median": 0.010987376736,
    "p75": 0.05437921475699999,
    "max": 0.82317151777,
    "mean": 0.04950241797425767
  }
}
```

## Pairing And Label Issues

### Images missing labels

None

### Labels missing images

None

### Invalid labels or unreadable images

None

## Plots

- `/home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained/reports/data_v1.0/eda/plots/instances_by_class.png`
- `/home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained/reports/data_v1.0/eda/plots/image_size_scatter.png`
- `/home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained/reports/data_v1.0/eda/plots/bbox_width_height_scatter.png`
