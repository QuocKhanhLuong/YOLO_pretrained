# YOLO Dataset EDA

Dataset: `/home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained/data/versions/data_v1.1_vehicle_soldier`

## Split Counts

| Split | Images | Labels | Empty Labels |
|---|---:|---:|---:|
| train | 888 | 888 | 283 |
| val | 111 | 111 | 41 |
| test | 112 | 112 | 29 |

## Class Counts

| Split | vehicle | soldier |
|---|---:|---:|
| train | 318 | 700 |
| val | 37 | 81 |
| test | 37 | 103 |

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
    "p25": 0.035156,
    "median": 0.075781,
    "p75": 0.173437,
    "max": 0.963443,
    "mean": 0.13187191144200627
  },
  "height": {
    "min": 0.002778,
    "p25": 0.068056,
    "median": 0.1322825,
    "p75": 0.307812,
    "max": 0.997845,
    "mean": 0.21798042633228842
  },
  "area": {
    "min": 2.169618e-06,
    "p25": 0.0028027124999999995,
    "median": 0.009727694835000002,
    "p75": 0.05403527417199999,
    "max": 0.82317151777,
    "mean": 0.050077723546752356
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

- `/home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained/reports/data_v1.1_vehicle_soldier/eda/plots/instances_by_class.png`
- `/home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained/reports/data_v1.1_vehicle_soldier/eda/plots/image_size_scatter.png`
- `/home/linhdang/workspace/quockhanh_workspace/YOLO_pretrained/reports/data_v1.1_vehicle_soldier/eda/plots/bbox_width_height_scatter.png`
