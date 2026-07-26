# Parameter status

## Confirmed by the paper

| Item | Value |
|---|---:|
| M3FD image pairs | 4,200 |
| M3FD train/test split | 3,360 / 840 |
| Source resolution | 768 × 1024 |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| beta1 / beta2 | 0.9 / 0.999 |
| Weight decay | 0.0005 |
| C1 / C2 / C3 | 1e-4 / 9e-4 / 0.01 |
| Detection backbone | YOLOv8s |
| Segmentation evaluator | SegFormer-B1 |
| Fusion metrics | SSIM, VIF, QAB/F, SCD, CC, FMI_dct |
| Detection metric | class-wise AP and mAP50-95 |

## Intentionally unresolved

Patch policy, epoch count, channel widths, detector feature layer, attention heads, sampling points, window size, task-prior strength, decoder depth, NMS settings, and UWL clamp bounds remain `null` in the public template until verified from the final experiment.
