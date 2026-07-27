"""Joint fusion-detection data flow for TPTAF."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from .detector import (
    DetectionSemanticsProvider,
    DetectorOutput,
    crop_semantic_feature,
    validate_detection_losses,
)
from .model import TPTAF, TPTAFOutput


@dataclass(frozen=True)
class JointOutput:
    """Outputs shared by fusion reconstruction and detector supervision."""

    fusion: TPTAFOutput
    detection_semantics: Tensor
    detection_losses: dict[str, Tensor] | None
    detector_predictions: Any | None = None
    detector_input: Tensor | None = None
    detector_auxiliary: Mapping[str, Any] | None = None


class JointTPTAF(nn.Module):
    """Coordinate full-image detection semantics with fusion reconstruction."""

    def __init__(
        self,
        fusion: TPTAF,
        detector: DetectionSemanticsProvider,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.detector = detector

    def forward(
        self,
        infrared_full: Tensor,
        visible_full: Tensor,
        *,
        infrared_fusion: Tensor | None = None,
        visible_fusion: Tensor | None = None,
        crop_boxes: Tensor | None = None,
        targets: Mapping[str, Tensor] | None = None,
    ) -> JointOutput:
        """Run the paper-level parallel data flow.

        Detection semantics are extracted from full-resolution paired inputs.
        Fusion can operate on the same images or on aligned patches. When
        patches are supplied, ``crop_boxes`` aligns the full-image semantic map
        to each patch before TAM.
        """
        detector_output: DetectorOutput = self.detector(
            infrared_full,
            visible_full,
            targets,
        )

        if infrared_fusion is None and visible_fusion is None:
            infrared_fusion = infrared_full
            visible_fusion = visible_full
            semantics = detector_output.semantics
        elif infrared_fusion is None or visible_fusion is None:
            raise ValueError("infrared_fusion and visible_fusion must be supplied together")
        else:
            if crop_boxes is None:
                raise ValueError("crop_boxes are required for patch-level fusion")
            if infrared_fusion.shape != visible_fusion.shape:
                raise ValueError("fusion patches must have identical shapes")
            semantics = crop_semantic_feature(
                detector_output.semantics,
                crop_boxes,
                infrared_full.shape[-2:],
                output_size=infrared_fusion.shape[-2:],
            )

        fusion_output = self.fusion(
            infrared_fusion,
            visible_fusion,
            semantics,
        )
        return JointOutput(
            fusion=fusion_output,
            detection_semantics=detector_output.semantics,
            detection_losses=validate_detection_losses(detector_output.losses),
            detector_predictions=detector_output.predictions,
            detector_input=detector_output.detector_input,
            detector_auxiliary=detector_output.auxiliary,
        )
