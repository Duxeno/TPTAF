# Data preparation

Dataset files are not redistributed. The validation helpers expect aligned infrared and visible images to share the same filename stem.

```text
DATASET/
├── train/
│   ├── ir/
│   ├── vi/
│   └── labels/
└── val/
    ├── ir/
    ├── vi/
    └── labels/
```

Optional detection labels use normalized YOLO format: `class_id center_x center_y width height`.

M3FD class order: people, car, bus, light, motorcycle, truck. Users must obtain all datasets separately and follow their original licenses.
