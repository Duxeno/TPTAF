# TPTAF

**TPTAF: Task-Prior Tripartite Attention for Infrared and Visible Image Fusion**  
Xuyang Du, Xiwen Yao, Ankang Zang, and Gong Cheng  
**IEEE Transactions on Image Processing**, 2026  
[论文](https://ieeexplore.ieee.org/document/11602761) · [DOI](https://doi.org/10.1109/TIP.2026.3709518) · [English](README.md)

## 论文简介

红外图像能够在弱光和复杂环境中提供稳定的热目标信息，可见光图像则保留丰富的纹理与场景结构。已有融合方法往往只通过最终任务预测或整体损失引入任务监督，对中间表示交互的控制有限，因此与检测相关的弱目标线索仍可能在跨模态聚合过程中被稀释。

TPTAF 将**检测语义转换为任务先验，并用于表示层面的跨模态交互**。解耦编码器（DE）将每个模态组织到结构空间和判别空间；由红外—可见光联合特征提取的检测语义进一步生成任务先验；三元注意力模块（TAM）协调结构约束、判别细节与检测先验，使目标相关证据能够直接调节跨模态特征选择。联合训练阶段，基于不确定性的加权学习（UWL）自适应平衡融合与检测目标。

## 整体框架

<p align="center">
  <img src="assets/tptaf_framework.svg" alt="TPTAF整体框架" width="100%">
</p>

TPTAF 包含两条并行的特征提取路径：

1. **单模态特征分解：** DE 通过 PCSB 和 DDEB 构建结构空间与判别空间。
2. **多模态检测语义提取：** 对红外—可见光浅层特征进行联合编码，获得与检测相关的语义表示。

三类表示由 TAM 完成任务先验引导的交互，随后重建融合图像。UWL 在联合优化中平衡三项融合损失和三项目标检测损失。

> 上图是根据论文 Fig. 1 为代码仓库重新绘制的说明图；论文正式框架图请以原文为准。

## 核心特点

- **检测先验引导的交互：** 将检测语义转换为任务先验，直接引导中间层跨模态特征选择。
- **结构—判别特征组织：** 结构表示约束跨模态布局一致性，判别表示保留模态互补细节。
- **不确定性加权联合学习：** 自适应平衡六项融合与检测损失，降低对人工固定任务权重的依赖。

## 发布范围

本仓库公开 **TPTAF 的核心方法实现**和面向论文评估协议的完整验证流程，但不属于能够复现论文全部数值的端到端复现包。

已公开：

- DE、PCSB、DDEB、DPG、TAM 与重建网络；
- TPTAF 自有的检测预处理及外部检测器接口；
- 六项 UWL 的实际代码；
- 配对数据读取、推理、融合指标、检测指标与验证结果导出；
- 配置模板和伪代码说明，并区分论文确认参数与尚未恢复的实现细节。

未公开：

- 数据集与标注；
- 最终训练权重；
- 第三方检测器源代码与权重；
- 完整原始训练环境和尚未恢复的结构超参数。

因此，本仓库可用于查看、测试和扩展核心实现，但在缺少原始数据、权重、检测环境和最终实验设置的情况下，不承诺端到端精确复现论文表格。

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

## 安装

```bash
git clone https://github.com/Duxeno/TPTAF.git
cd TPTAF
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell 使用：

```powershell
.venv\Scripts\Activate.ps1
```

## 核心网络检查

论文采用的第三方检测器实现未在本仓库重新分发。核心融合网络可直接接收外部检测器提供的语义特征：

```python
import torch

from src.tptaf.model import TPTAF

model = TPTAF(
    channels=64,
    detection_channels=64,
    num_heads=4,
    num_points=8,
    window_size=8,
)

infrared = torch.rand(1, 1, 256, 256)
visible = torch.rand(1, 1, 256, 256)
detection_semantics = torch.rand(1, 64, 32, 32)

output = model(infrared, visible, detection_semantics)
print(output.fused.shape)
```

以上数值仅用于演示代码调用，并不表示所有结构设置均与最终实验完全一致。具体状态见 `docs/PARAMETER_STATUS.md` 与 `configs/tptaf_template.yaml`。

## 验证代码

公开的验证流程用于评估**已经生成的融合图像**，包括：

- 根据文件名严格配对红外图像、可见光图像和融合图像；
- SSIM、VIF、QAB/F、SCD、CC 和 FMIdct；
- 通过程序化检测运行器计算逐类 AP 与 IoU 0.50–0.95 范围的 mAP；
- 通过损失运行器统计六项 UWL 损失及其有效权重；
- 输出逐图 CSV、逐类 CSV 与 JSON 汇总。

融合指标验证示例：

```bash
python tools/val.py \
  --fused-dir /path/to/fused \
  --data-root /path/to/dataset \
  --split val \
  --output-dir outputs/tptaf_validation
```

数据目录规范见 `data/README.md`。当第三方指标实现、图像预处理或检测器设置与原始实验不同时，结果末位数值可能与论文表格存在差异。

## 论文评估范围

论文在 **M3FD、RoadScene、AVMS 和 MSRS** 上评估融合质量，并采用 **YOLOv8s** 和 **SegFormer-B1** 分别进行目标检测与语义分割评估。TPTAF 训练中的任务先验来自检测监督；语义分割仅用于下游评估，不参与任务先验构建。

## 参数说明

`configs/tptaf_template.yaml` 记录论文明确给出的参数，并将尚未可靠恢复的结构参数保留为 `null`。详细说明见：

- `docs/PARAMETER_STATUS.md`
- `docs/PSEUDOCODE.md`

这样可以避免把不确定的实现选择错误地表述为最终实验设置。

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

## 许可证与第三方代码

本仓库中的 TPTAF 原创代码采用 [MIT License](LICENSE)。第三方依赖、数据集、检测器实现及预训练权重仍遵循其各自许可证，详见 `LICENSE_NOTICE.md`。
