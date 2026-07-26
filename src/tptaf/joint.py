"""Joint data flow between TPTAF and an external detection-semantic provider."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from torch import Tensor, nn
from .detector import DetectionSemanticsProvider, crop_semantic_feature, validate_detection_losses
from .model import TPTAF, TPTAFOutput

@dataclass
class JointOutput:
    fusion: TPTAFOutput
    detection_losses: dict[str, Tensor] | None
    full_detection_semantics: Tensor
    detector_predictions: object | None = None

class JointTPTAF(nn.Module):
    def __init__(self, fusion: TPTAF, detector: DetectionSemanticsProvider) -> None:
        super().__init__(); self.fusion = fusion; self.detector = detector
    def forward(self, infrared_full: Tensor, visible_full: Tensor, infrared_patch: Tensor, visible_patch: Tensor, crop_boxes: Tensor, labels: Mapping[str, Tensor] | None = None) -> JointOutput:
        d = self.detector(infrared_full, visible_full, labels)
        semantics = crop_semantic_feature(d.semantics, crop_boxes, infrared_full.shape[-2:])
        f = self.fusion(infrared_patch, visible_patch, semantics)
        return JointOutput(fusion=f, detection_losses=validate_detection_losses(d.losses), full_detection_semantics=d.semantics, detector_predictions=d.predictions)
