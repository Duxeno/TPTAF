# TPTAF

**TPTAF: Task-Prior Tripartite Attention for Infrared and Visible Image Fusion**  
Xuyang Du, Xiwen Yao, Ankang Zang, and Gong Cheng  
Accepted by **IEEE Transactions on Image Processing**, 2026  
DOI: [`10.1109/TIP.2026.3709518`](https://doi.org/10.1109/TIP.2026.3709518)

## Release scope

This repository releases the **core method implementation** of TPTAF for research reference. It is intentionally organized as a method-oriented code release rather than an end-to-end reproduction package.

The released code covers:

- decoupled structural-discriminative feature organization;
- the phase-coherent structure block (PCSB);
- the discriminative detail enhancement block (DDEB);
- detection-semantic extraction and detection-prior generation;
- the tripartite attention module (TAM);
- uncertainty-weighted learning (UWL) over fusion and detection losses;
- validation utilities for the fusion and object-detection protocols reported in the paper.

Datasets, trained checkpoints, exact final training schedules, and third-party downstream evaluation environments are not included. Architecture-dependent values that cannot be recovered reliably are left configurable or documented as pseudocode instead of being presented as confirmed settings.

## Method overview

TPTAF establishes a detection-prior-guided infrared-visible image fusion framework by coupling detection semantics with cross-modal representation learning at the intermediate feature level.

The method contains two parallel feature pathways. The decoupled encoder organizes infrared and visible features into a structural space and a discriminative space. The structural space constrains cross-modal layout consistency, while the discriminative space preserves complementary modality-specific details. In parallel, detection semantics extracted from hybrid infrared-visible features are transformed into a detection-conditioned prior. The tripartite attention module integrates the prior with structural and discriminative representations so that target-related evidence regulates cross-modal feature selection. During joint optimization, UWL balances the fusion and detection objectives without manually fixing all loss weights.

The task prior in TPTAF is derived from **detection supervision**. Semantic segmentation is used as a downstream evaluation setting in the paper; it is not the source of the task prior.

## Code map

| Paper component | Code |
|---|---|
| Overall TPTAF network | `src/tptaf/model.py` |
| Shared building blocks | `src/tptaf/modules.py` |
| Decoupled Encoder (DE) | `src/tptaf/model.py::DecoupledEncoder` |
| PCSB | `src/tptaf/modules.py::PhaseCoherentStructureBlock` |
| DDEB | `src/tptaf/modules.py::DiscriminativeDetailEnhancementBlock` |
| Detection-semantic branch | `src/tptaf/detector.py::YOLOv8SemanticsBranch` |
| Detection Prior Generator (DPG) | `src/tptaf/modules.py::DetectionPriorGenerator` |
| Tripartite Attention Module (TAM) | `src/tptaf/modules.py::TripartiteAttention` |
| Joint fusion-detection interface | `src/tptaf/joint.py` |
| Fusion losses and six-term UWL | `src/tptaf/losses.py` |
| Validation pipeline | `src/tptaf/validation.py`, `tools/val.py`, `val.py` |
| Fusion metrics | `src/tptaf/metrics.py` |
| Detection metrics | `src/tptaf/detection_metrics.py` |

## Validation code

The validation code is kept complete because it defines the evaluation logic used around the released method. It contains:

- strict filename-based pairing of infrared and visible images;
- full-resolution tiled fusion;
- SSIM, VIF, QAB/F, SCD, CC, and FMIdct;
- class-wise AP and mAP over IoU thresholds from 0.50 to 0.95;
- reporting of the six UWL components and their effective weights;
- evaluation of pre-generated fused-image directories;
- per-image CSV, per-class CSV, and JSON summaries.

The last decimal of metric values can depend on the exact third-party metric implementation used in the original experiment. The provided validation code documents the protocol and core calculations, but the repository does not claim exact numerical reproduction of the published tables.

## Paper evaluation scope

The paper evaluates fusion quality on **M3FD, RoadScene, AVMS, and MSRS**. Downstream object detection uses **YOLOv8s**, and semantic segmentation evaluation uses **SegFormer-B1**. The reported results show stable fusion quality across different scene distributions and improved usefulness of the fused images for object detection and semantic segmentation evaluation.

## Parameter notes

`configs/tptaf_template.yaml` contains paper-confirmed settings and leaves uncertain architecture-dependent values as `null`. The logic of the unresolved parts is documented in:

- `docs/PSEUDOCODE.md`
- `docs/PARAMETER_STATUS.md`

These files distinguish paper-confirmed operations from implementation details that still require recovery from the final experiment.

## Repository structure

```text
TPTAF/
├── README.md
├── README_zh.md
├── CITATION.cff
├── LICENSE
├── LICENSE_NOTICE.md
├── requirements.txt
├── val.py
├── configs/
│   └── tptaf_template.yaml
├── data/
│   └── README.md
├── docs/
│   ├── PARAMETER_STATUS.md
│   └── PSEUDOCODE.md
├── src/tptaf/
│   ├── model.py
│   ├── modules.py
│   ├── detector.py
│   ├── joint.py
│   ├── losses.py
│   ├── validation.py
│   ├── metrics.py
│   └── detection_metrics.py
└── tools/
    └── val.py
```

## Citation

```bibtex
@article{du2026tptaf,
  author  = {Xuyang Du and Xiwen Yao and Ankang Zang and Gong Cheng},
  title   = {TPTAF: Task-Prior Tripartite Attention for Infrared and Visible Image Fusion},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026},
  doi     = {10.1109/TIP.2026.3709518}
}
```

## License and third-party code

The original TPTAF code in this repository is released under the MIT License. Third-party dependencies remain subject to their own licenses. See `LICENSE_NOTICE.md`.

## Contact

For questions about the paper or the released method code, please contact the authors through the email addresses listed in the paper.
