# TPTAF

**TPTAF: Task-Prior Tripartite Attention for Infrared and Visible Image Fusion**  
已被 **IEEE Transactions on Image Processing (TIP)** 接收，2026  
[论文](https://ieeexplore.ieee.org/document/11602761) · [DOI](https://doi.org/10.1109/TIP.2026.3709518) · [English](README.md)

## 论文简介

红外图像能够在复杂环境和弱光条件下提供稳定的热目标信息，可见光图像则保留丰富的纹理与场景结构。TPTAF 将检测语义作为任务先验，引导表示层面的跨模态特征交互。解耦编码器（DE）首先将每个模态组织到结构空间和判别空间；随后由红外—可见光联合特征提取检测相关语义，并进一步生成任务先验；三元注意力模块（TAM）在跨模态交互中协调结构约束、判别细节和检测先验信息。联合训练过程中，不确定性加权学习（UWL）用于平衡融合目标与检测目标。

## 整体框架

<p align="center">
  <img src="assets/tptaf_framework.svg" alt="TPTAF整体框架" width="100%">
</p>

## 核心组成

- **DE（Decoupled Encoder）**：构建结构表示与判别表示。
- **PCSB**：增强相位一致的结构信息。
- **DDEB**：强化判别性高频细节。
- **Detection Semantics Extractor**：提取与目标检测相关的语义信息。
- **DPG（Detection Prior Generator）**：将检测语义转换为任务先验。
- **TAM（Tripartite Attention Module）**：融合结构特征、判别特征与检测先验。
- **Decoder**：重建最终融合图像。
- **UWL（Uncertainty-Weighted Learning）**：平衡三项融合损失和三项检测损失。

## 发布内容

本仓库包含 TPTAF 的核心代码实现，包括：

- DE、PCSB、DDEB、DPG、TAM、Decoder 和 UWL 等网络模块；
- 融合—检测联合接口；
- 检测侧预处理与外部检测器接口定义；
- 配对数据读取、推理、验证、融合指标与检测指标；
- 配置模板与伪代码说明。

## 论文模块与代码对应

| 论文模块 | 公开代码 |
|---|---|
| TPTAF 主网络 | `src/tptaf/model.py::TPTAF` |
| 解耦编码器（DE） | `src/tptaf/model.py::DecoupledEncoder` |
| 相位一致结构块（PCSB） | `src/tptaf/modules.py::PhaseCoherentStructureBlock` |
| 判别细节增强块（DDEB） | `src/tptaf/modules.py::DiscriminativeDetailEnhancementBlock` |
| 检测输入预处理 | `src/tptaf/detector.py::DetectionInputPreprocessor` |
| 外部检测器接口 | `src/tptaf/detector.py::DetectionSemanticsProvider` |
| 检测先验生成器（DPG） | `src/tptaf/modules.py::DetectionPriorGenerator` |
| 三元注意力模块（TAM） | `src/tptaf/modules.py::TripartiteAttention` |
| 融合—检测联合接口 | `src/tptaf/joint.py::JointTPTAF` |
| 融合损失与六项 UWL | `src/tptaf/losses.py` |
| 验证流程 | `src/tptaf/validation.py`、`tools/val.py`、`val.py` |
| 融合指标 | `src/tptaf/metrics.py` |
| 检测指标 | `src/tptaf/detection_metrics.py` |

## 验证

仓库提供完整的协议化验证流程，包括图像配对、融合指标、检测指标以及结果汇总导出。

示例：

```bash
python tools/val.py \
  --fused-dir /path/to/fused \
  --data-root /path/to/dataset \
  --split val \
  --output-dir outputs/tptaf_validation
```

## 论文评估范围

论文在 **M3FD、RoadScene、AVMS 和 MSRS** 上评估融合质量，并采用 **YOLOv8s** 与 **SegFormer-B1** 分别进行目标检测和语义分割评估。

## 引用

```bibtex
@article{du2026tptaf,
  author  = {Xuyang Du and Xiwen Yao and Ankang Zang and Gong Cheng},
  title   = {TPTAF: Task-Prior Tripartite Attention for Infrared and Visible Image Fusion},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026},
  doi     = {10.1109/TIP.2026.3709518}
}
```

## 许可证

本仓库中的 TPTAF 原创代码采用 [MIT License](LICENSE)。第三方依赖、数据集、检测器实现及预训练权重仍遵循其各自许可证。
