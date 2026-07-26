# TPTAF

**TPTAF: Task-Prior Tripartite Attention for Infrared and Visible Image Fusion**  
Accepted by **IEEE Transactions on Image Processing (TIP)**, 2026  
[Paper](https://ieeexplore.ieee.org/document/11602761) · [DOI](https://doi.org/10.1109/TIP.2026.3709518) · [中文说明](README_zh.md)

## Introduction

TPTAF introduces detection semantics as task-prior guidance for representation-level infrared-visible interaction. A decoupled encoder organizes each modality into structural and discriminative spaces. Detection semantics extracted from hybrid infrared-visible features are transformed into prior guidance and integrated with the decoupled representations through the Tripartite Attention Module (TAM). During joint training, Uncertainty-Weighted Learning (UWL) balances three fusion losses and three detection losses.

## Overall framework

<p align="center">
  <img src="assets/tptaf_framework.jpg" alt="Overall framework of TPTAF" width="100%">
</p>

The framework contains two parallel feature-extraction pathways: unimodal feature decomposition and multimodal detection-semantic extraction. Their representations are coordinated by TAM for detection-prior-guided cross-modal interaction.

## Core components

- **Decoupled Encoder (DE):** constructs structural and discriminative feature spaces.
- **Phase Coherent Structure Block (PCSB):** preserves phase-related structural organization from low-frequency features.
- **Discriminative Detail Enhancement Block (DDEB):** enhances high-frequency detail responses and local contrast.
- **Detection Semantics Extractor:** forms detection-relevant semantics from hybrid infrared-visible features.
- **Detection Prior Generator (DPG):** predicts temperature, spatial-prior, and sparse-sampling parameters for TAM.
- **Tripartite Attention Module (TAM):** coordinates structural constraints, discriminative details, and detection-prior guidance.
- **Uncertainty-Weighted Learning (UWL):** balances `L_sime`, `L_deco`, `L_grad`, `L_box`, `L_cls`, and `L_dfl`.

## Released code

This repository contains the core TPTAF implementation:

- PCSB, DDEB, GLA, DPG, TAM, and the reconstruction decoder;
- detector-input preprocessing and the external detector interface;
- the joint fusion-detection data flow;
- fusion losses and six-term UWL;
- paired-data and tiled-inference utilities.

## Code map

| Method element | Code |
|---|---|
| TPTAF network and DE | `src/tptaf/model.py` |
| PCSB, DDEB, GLA, DPG, and TAM | `src/tptaf/modules.py` |
| Detection preprocessing and detector interface | `src/tptaf/detector.py` |
| Joint fusion-detection interface | `src/tptaf/joint.py` |
| Fusion losses and UWL | `src/tptaf/losses.py` |
| Paired-data helpers | `src/tptaf/data.py` |
| Tiled inference | `src/tptaf/inference.py` |
| Conceptual data flow | `docs/PSEUDOCODE.md` |

## Installation

```bash
git clone https://github.com/Duxeno/TPTAF.git
cd TPTAF
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Minimal forward example

```python
import torch
from src.tptaf.model import TPTAF

model = TPTAF()
infrared = torch.rand(1, 1, 256, 256)
visible = torch.rand(1, 1, 256, 256)
detection_semantics = torch.rand(1, 64, 32, 32)

output = model(infrared, visible, detection_semantics)
print(output.fused.shape)
```

## Paper evaluation setting

The paper trains TPTAF on M3FD and evaluates fusion quality on M3FD, RoadScene, AVMS, and MSRS. YOLOv8s and SegFormer-B1 are used for downstream object detection and semantic segmentation evaluation, respectively. The detection semantics used by TPTAF are derived from detection supervision; semantic segmentation is an evaluation setting.

## Repository structure

```text
TPTAF/
├── assets/tptaf_framework.jpg
├── data/README.md
├── docs/PSEUDOCODE.md
├── src/tptaf/
│   ├── __init__.py
│   ├── data.py
│   ├── detector.py
│   ├── inference.py
│   ├── joint.py
│   ├── losses.py
│   ├── model.py
│   └── modules.py
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

The original TPTAF code in this repository is released under the [MIT License](LICENSE). See `LICENSE_NOTICE.md` for third-party resources.
