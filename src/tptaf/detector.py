"""Detection-semantic contract used by TPTAF.

The fusion core depends on detection semantics, predictions, and the three YOLO
loss terms, while the detector-specific backbone, neck, assignment strategy,
and feature-tap selection remain encapsulated behind this interface.  This
keeps the released method code focused on the representation-level interaction
introduced by TPTAF.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modules import (
    ChannelSpatialGating,
    DepthwiseSeparableConv,
    GlobalLocalAttention,
)


DETECTION_LOSS_NAMES = ("box", "cls", "dfl")


@dataclass(frozen=True)
class DetectorOutput:
    """Information supplied by the detection-semantic pathway.

    Attributes:
        semantics: Detection-relevant feature map ``F_det`` consumed by TAM.
        losses: Optional ``L_box``, ``L_cls``, and ``L_dfl`` terms.
        predictions: Detector-native predictions for downstream processing.
        detector_input: Hybrid image passed to the detector, when exposed.
        auxiliary: Optional detector-native intermediate information.
    """

    semantics: Tensor
    losses: Mapping[str, Tensor] | None = None
    predictions: Any | None = None
    detector_input: Tensor | None = None
    auxiliary: Mapping[str, Any] | None = None


class ModalitySemanticStem(nn.Module):
    """Extract modality-specific responses before hybrid encoding."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.embedding = DepthwiseSeparableConv(1, channels)
        self.selection = ChannelSpatialGating(channels)

    def forward(self, image: Tensor) -> Tensor:
        feature = self.embedding(image)
        return self.selection(feature, residual=True)


class DetectionInputPreprocessor(nn.Module):
    """Construct the hybrid infrared-visible detector input.

    The module exposes the public multimodal preprocessing logic shown in the
    framework: modality-specific encoding, hybrid feature construction, GLA,
    and projection to the detector input space.
    """

    def __init__(
        self,
        channels: int = 32,
        output_channels: int = 3,
        num_heads: int = 4,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.infrared_stem = ModalitySemanticStem(channels)
        self.visible_stem = ModalitySemanticStem(channels)
        self.hybrid_encoding = DepthwiseSeparableConv(
            2 * channels,
            2 * channels,
        )
        self.global_local_interaction = GlobalLocalAttention(
            2 * channels,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.input_projection = nn.Conv2d(
            2 * channels,
            output_channels,
            kernel_size=1,
        )

    def forward(self, infrared: Tensor, visible: Tensor) -> Tensor:
        if infrared.shape != visible.shape:
            raise ValueError("infrared and visible inputs must have identical shapes")
        if infrared.ndim != 4 or infrared.shape[1] != 1:
            raise ValueError("detector preprocessing expects [B, 1, H, W] inputs")

        infrared_feature = self.infrared_stem(infrared)
        visible_feature = self.visible_stem(visible)
        hybrid_feature = self.hybrid_encoding(
            torch.cat((infrared_feature, visible_feature), dim=1)
        )
        hybrid_feature = self.global_local_interaction(hybrid_feature)
        return torch.sigmoid(self.input_projection(hybrid_feature))


class DetectionSemanticsProvider(nn.Module, ABC):
    """Public integration boundary between a detector and TPTAF.

    A concrete provider performs the detector-specific operations and returns
    ``DetectorOutput``.  The semantic tensor must preserve the batch dimension
    and provide the channel count selected when constructing ``TPTAF``.
    """

    semantic_channels: int

    @abstractmethod
    def forward(
        self,
        infrared: Tensor,
        visible: Tensor,
        targets: Mapping[str, Tensor] | None = None,
    ) -> DetectorOutput:
        """Extract ``F_det`` and, during training, detection loss terms."""
        raise NotImplementedError

    def postprocess_predictions(
        self,
        predictions: Any,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
    ) -> list[Tensor]:
        """Convert detector-native predictions into final detections."""
        raise NotImplementedError


def validate_detection_losses(
    losses: Mapping[str, Tensor] | None,
) -> dict[str, Tensor] | None:
    """Return the three detection losses in the order used by UWL."""
    if losses is None:
        return None

    missing = [name for name in DETECTION_LOSS_NAMES if name not in losses]
    if missing:
        raise KeyError(f"missing detection losses: {missing}")

    validated: dict[str, Tensor] = {}
    for name in DETECTION_LOSS_NAMES:
        value = losses[name]
        if not isinstance(value, Tensor) or value.numel() != 1:
            raise ValueError(f"detection loss '{name}' must be a scalar tensor")
        validated[name] = value
    return validated


def crop_semantic_feature(
    semantics: Tensor,
    crop_boxes: Tensor,
    source_size: tuple[int, int],
    *,
    output_size: tuple[int, int] | None = None,
) -> Tensor:
    """Align full-image detection semantics with fusion patches.

    ``crop_boxes`` uses pixel coordinates ``[top, left, height, width]`` in the
    source-image coordinate system.  One box is supplied for each semantic-map
    batch element.  Cropped semantics are resized to ``output_size`` so that
    every fusion patch receives an aligned task-prior feature map.
    """
    if semantics.ndim != 4:
        raise ValueError("semantics must have shape [B, C, H, W]")
    if crop_boxes.ndim != 2 or crop_boxes.shape[1] != 4:
        raise ValueError("crop_boxes must have shape [B, 4]")
    if crop_boxes.shape[0] != semantics.shape[0]:
        raise ValueError("crop_boxes and semantics must share the batch size")

    source_height, source_width = source_size
    if source_height <= 0 or source_width <= 0:
        raise ValueError("source_size must contain positive dimensions")

    feature_height, feature_width = semantics.shape[-2:]
    aligned: list[Tensor] = []
    for index, box in enumerate(crop_boxes.detach().cpu().tolist()):
        top, left, height, width = box
        if height <= 0 or width <= 0:
            raise ValueError("crop height and width must be positive")

        y0 = round(top * feature_height / source_height)
        x0 = round(left * feature_width / source_width)
        y1 = round((top + height) * feature_height / source_height)
        x1 = round((left + width) * feature_width / source_width)

        y0 = min(max(y0, 0), feature_height - 1)
        x0 = min(max(x0, 0), feature_width - 1)
        y1 = min(max(y1, y0 + 1), feature_height)
        x1 = min(max(x1, x0 + 1), feature_width)
        aligned.append(semantics[index : index + 1, :, y0:y1, x0:x1])

    if output_size is None:
        output_height = max(feature.shape[-2] for feature in aligned)
        output_width = max(feature.shape[-1] for feature in aligned)
        output_size = (output_height, output_width)

    return torch.cat(
        [
            F.interpolate(
                feature,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            for feature in aligned
        ],
        dim=0,
    )
