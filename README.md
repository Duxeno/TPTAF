# TPTAF

**TPTAF: Task-Prior Tripartite Attention for Infrared and Visible Image Fusion**  
Accepted by **IEEE Transactions on Image Processing (TIP)**, 2026  
[Paper](https://ieeexplore.ieee.org/document/11602761) · [DOI](https://doi.org/10.1109/TIP.2026.3709518) · [中文说明](README_zh.md)

## Introduction

Infrared images provide stable thermal cues under challenging illumination, while visible images preserve rich textures and scene structures. TPTAF introduces detection semantics as task-prior guidance for representation-level cross-modal interaction. A Decoupled Encoder (DE) organizes each modality into structural and discriminative spaces. Detection-relevant semantics are transformed into a task prior, and the Tripartite Attention Module (TAM) coordinates structural constraints, discriminative details, and detection prior information during cross-modal feature interaction. During joint training, Uncertainty-Weighted Learning (UWL) balances fusion and detection objectives.

## Overall framework

<p align="center">
  <img src="assets/tptaf_framework.jpg" alt="Overall framework of TPTAF" width="100%">
</p>

## Core components

- **DE (Decoupled Encoder):** organizes each modality into structural and discriminative representations.
- **PCSB:** enhances phase-coherent structural information.
- **DDEB:** strengthens discriminative high-frequency details.
- **Detection Semantics Extractor:** extracts task-relevant detection semantics from infrared-visible features.
- **DPG (Detection Prior Generator):** transforms detection semantics into the task prior.
- **TAM (Tripartite Attention Module):** integrates structural features, discriminative features, and the detection prior.
- **Decoder:** reconstructs the final fused image.
- **UWL (Uncertainty-Weighted Learning):** balances three fusion losses and three detection losses.

## Release scope

This repository contains the core implementation of TPTAF, including:

- network modules for DE, PCSB, DDEB, DPG, TAM, Decoder, and UWL;
- the joint fusion-detection interface;
- detector-side preprocessing and external detector interface definitions;
- paired-data utilities, inference utilities, validation utilities, fusion metrics, and detection metrics;
- configuration templates and pseudocode notes.

## Code map

| Paper component | Released code |
|---|---|
| Overall TPTAF network | `src/tptaf/model.py::TPTAF` |
| Decoupled Encoder (DE) | `src/tptaf/model.py::DecoupledEncoder` |
| Phase Coherent Structure Block (PCSB) | `src/tptaf/modules.py::PhaseCoherentStructureBlock` |
| Discriminative Detail Enhancement Block (DDEB) | `src/tptaf/modules.py::DiscriminativeDetailEnhancementBlock` |
| Detection preprocessing | `src/tptaf/detector.py::DetectionInputPreprocessor` |
| External detector contract | `src/tptaf/detector.py::DetectionSemanticsProvider` |
| Detection Prior Generator (DPG) | `src/tptaf/modules.py::DetectionPriorGenerator` |
| Tripartite Attention Module (TAM) | `src/tptaf/modules.py::TripartiteAttention` |
| Joint fusion-detection interface | `src/tptaf/joint.py::JointTPTAF` |
| Fusion losses and six-term UWL | `src/tptaf/losses.py` |
| Validation pipeline | `src/tptaf/validation.py`, `tools/val.py`, `val.py` |
| Fusion metrics | `src/tptaf/metrics.py` |
| Detection metrics | `src/tptaf/detection_metrics.py` |

## Validation

The repository provides a complete protocol-oriented validation pipeline, including image pairing, fusion metrics, detection metrics, and summary export utilities.

Example:

```bash
python tools/val.py \
  --fused-dir /path/to/fused \
  --data-root /path/to/dataset \
  --split val \
  --output-dir outputs/tptaf_validation
```

## Paper evaluation scope

The paper evaluates fusion quality on **M3FD, RoadScene, AVMS, and MSRS**. **YOLOv8s** and **SegFormer-B1** are used for downstream object detection and semantic segmentation evaluation, respectively.

## Repository structure

```text
TPTAF/
├── assets/tptaf_framework.svg
├── configs/tptaf_template.yaml
├── data/README.md
├── docs/
│   ├── PARAMETER_STATUS.md
│   └── PSEUDOCODE.md
├── src/tptaf/
│   ├── config.py
│   ├── data.py
│   ├── detection_metrics.py
│   ├── detector.py
│   ├── factory.py
│   ├── inference.py
│   ├── joint.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── modules.py
│   └── validation.py
├── tools/val.py
├── val.py
├── CITATION.cff
├── LICENSE
├── LICENSE_NOTICE.md
├── README.md
├── README_zh.md
└── requirements.txt
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

## License

The original TPTAF code in this repository is released under the [MIT License](LICENSE). Third-party dependencies, datasets, detector implementations, and pretrained weights remain subject to their own licenses.
