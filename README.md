# TPTAF

**TPTAF: Task-Prior Tripartite Attention for Infrared and Visible Image Fusion**  
Accepted by **IEEE Transactions on Image Processing (TIP)**, 2026  
[Paper](https://ieeexplore.ieee.org/document/11602761) · [DOI](https://doi.org/10.1109/TIP.2026.3709518) · [中文说明](README_zh.md)

## Overall framework

<p align="center">
  <img src="assets/tptaf_framework.jpg" alt="Overall framework of TPTAF" width="100%">
</p>

## Introduction

Infrared images provide stable thermal cues under challenging illumination, while visible images preserve rich textures and scene structures. TPTAF introduces detection semantics as task-prior guidance for representation-level cross-modal interaction. A Decoupled Encoder (DE) organizes each modality into structural and discriminative spaces. Detection-relevant semantics are transformed into a task prior, and the Tripartite Attention Module (TAM) coordinates structural constraints, discriminative details, and detection prior information during cross-modal feature interaction. During joint training, Uncertainty-Weighted Learning (UWL) balances fusion and detection objectives.

## Release scope

This repository contains the core implementation of TPTAF, including network modules, joint fusion-detection interfaces, inference and validation utilities, fusion metrics, detection metrics, and configuration templates.

## Citation

```bibtex
@article{du2026tptaf,
  title={TPTAF: Task-Prior Tripartite Attention for Infrared and Visible Image Fusion},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  doi={10.1109/TIP.2026.3709518}
}
```

## License

The original TPTAF code in this repository is released under the MIT License.
