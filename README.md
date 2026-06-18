# Task-Prior Transfer and Alignment Fusion (TPTAF)

TPTAF is a **detection‑prior‑guided fusion framework** that aims to bridge modalities such as visible cameras, infrared cameras, LiDAR and radar. Unlike conventional multi‑task or reconstruction‑driven approaches, TPTAF extracts detection semantics as a *task prior* and uses this prior to guide representation‑level interaction between modalities. In other words, the method does not rely on broad multi‑task adaptability; it focuses on detection semantics to build stronger cross‑modal correspondences and fused representations.

## Key Features

* **Detection‑Prior Extraction:** TPTAF derives a task prior from detection supervision. The task prior captures target‑related semantics and guides the fusion process. This distinguishes TPTAF from output‑level or loss‑level task‑driven methods.
* **Representation‑Level Interaction:** Instead of simply combining outputs, TPTAF uses the detection prior to align and integrate feature representations across modalities (e.g., image–infrared, camera–LiDAR, camera–radar), strengthening both structural and discriminative cues.
* **Robust Fusion:** Experiments on multiple datasets—**M3FD**, **RoadScene**, **AVMS** and **MSRS**—show that TPTAF maintains stable fusion quality across different data distributions and improves the utility of fused images for downstream object detection and semantic segmentation tasks.
* **Tolerance to Label Noise:** The structural and discriminative representation paths in TPTAF help maintain fusion quality even when category supervision is corrupted. However, box‑level and presence‑level annotation imperfections may still introduce uncertainty, highlighting the need for accurate detection annotations.

## Repository Status and Release Plan

This repository is created to support the reproducibility statement of our paper. We have provided the project skeleton and will progressively open source the following materials after the paper is accepted, as promised in the manuscript:

* **Implementation framework** – core network architectures, data loaders and training scripts.
* **Pre‑trained weights** – checkpoints for reproducing our results.
* **Model configurations** – YAML/JSON files defining hyper‑parameters, backbone settings and training schedules.
* **Usage instructions** – step‑by‑step guides to prepare datasets, train the models and evaluate on benchmark tasks.

The repository currently contains placeholders for these components. Please check back after acceptance for the full release. Our goal is to help future researchers reproduce the main experimental results and compare with TPTAF under consistent settings.

## Repository Structure

```
TPTAF/
├── ├── docs/        # Documentation and release plan
├── configs/     # Configuration files (to be released)
├── models/      # Model weights (to be released)
├── data/        # Dataset scripts and links (to be released)
├── src/         # Implementation code (to be released)
└── README.md    # Project overview and instructions
)
* `docs/release_plan.md` - provides a roadmap of the progressive releases and instructions on how to cite this work.
* `configs/` - will contain example configuration files for training and evaluation.
* `models/` - will host trained model weights once released.
* `src/` - will include the high-level implementation modules and scripts.

## Getting Started (To Be Released)

### Environment

The implementation will be based on **Python ≥3.8** and **PyTorch**. We will provide a `requirements.txt` specifying the exact dependencies and versions once the code is released. The training scripts will be tested on both single‑GPU and multi‑GPU setups.

### Dataset Preparation

We evaluate TPTAF on the following public datasets:

* **M3FD** – infrared–visible fusion dataset
* **RoadScene** – autonomous driving dataset with thermal and visible channels
* **AVMS** – aerial vision multi‑spectral dataset
* **MSRS** – multi‑spectral road scene dataset

Scripts to download and prepare these datasets will be provided in the `data/` directory. You may need to register or request access on the original dataset websites.

### Training and Evaluation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Duxeno/TPTAF.git
   cd TPTAF
   ```
2. **Set up the environment** by installing the required packages (instructions will be provided).
3. **Prepare the datasets** using the provided scripts.
4. **Train the model** on your desired modality pair (e.g., VIS‑IR, VIS‑LiDAR, VIS‑Radar) using the configuration files.
5. **Evaluate the fusion results** and, if desired, perform downstream object detection or semantic segmentation tasks using the fused images.

Detailed command‑line examples and configuration explanations will be added after acceptance.

## Citation

If you use TPTAF in your research, please cite our paper. The full BibTeX entry will be provided once the paper has been accepted and assigned a DOI.

## License

The code and models will be released under an open‑source license (to be determined) after acceptance. Until then, this repository serves as an informational placeholder.

## Contact

For questions regarding the method or potential collaboration opportunities, please create an issue in this repository or contact the corresponding author listed in the paper.
