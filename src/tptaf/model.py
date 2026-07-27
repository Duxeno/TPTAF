"""Paper-oriented TPTAF fusion model.

The model keeps the three stages of the method explicit:

1. modality-wise structural/discriminative decomposition;
2. detection-prior-guided tripartite interaction;
3. discriminative image reconstruction.

The detector branch is connected through the semantic interface defined in
``tptaf.detector`` and is therefore independent of the fusion core.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modules import (
    AbsolutePositionEmbedding,
    ChannelSpatialGating,
    DepthwiseSeparableConv,
    DiscriminativeDetailEnhancementBlock,
    FrequencySubbands,
    HaarWaveletDecomposition,
    PhaseCoherentStructureBlock,
    TripartiteAttention,
    TripartiteAttentionOutput,
)


@dataclass(frozen=True)
class EncoderOutput:
    """Structural and discriminative representations of one modality."""

    embedded: Tensor
    subbands: FrequencySubbands
    structural: Tensor
    discriminative: Tensor


@dataclass(frozen=True)
class TPTAFOutput:
    """Fusion result together with the method-level intermediate states."""

    fused: Tensor
    infrared: EncoderOutput
    visible: EncoderOutput
    interaction: TripartiteAttentionOutput

    @property
    def structural_ir(self) -> Tensor:
        return self.infrared.structural

    @property
    def structural_vi(self) -> Tensor:
        return self.visible.structural

    @property
    def discriminative_ir(self) -> Tensor:
        return self.infrared.discriminative

    @property
    def discriminative_vi(self) -> Tensor:
        return self.visible.discriminative


class DecoupledEncoder(nn.Module):
    """Construct structural and discriminative spaces for one modality.

    The embedding and positional term correspond to Eqs. (1)--(2).  The
    wavelet decomposition, PCSB, and DDEB correspond to Eqs. (3)--(11).
    """

    def __init__(self, in_channels: int = 1, channels: int = 64) -> None:
        super().__init__()
        self.embedding = DepthwiseSeparableConv(in_channels, channels)
        self.position = AbsolutePositionEmbedding(channels)
        self.wavelet = HaarWaveletDecomposition()
        self.structural_projection = PhaseCoherentStructureBlock(channels)
        self.discriminative_projection = (
            DiscriminativeDetailEnhancementBlock(channels)
        )

    def forward(self, image: Tensor) -> EncoderOutput:
        embedded = self.position(self.embedding(image))
        subbands = self.wavelet(embedded)
        output_size = embedded.shape[-2:]

        structural = self.structural_projection(
            subbands.low_low,
            output_size,
        )
        discriminative = self.discriminative_projection(
            subbands.low_high,
            subbands.high_low,
            subbands.high_high,
            output_size,
        )
        return EncoderOutput(
            embedded=embedded,
            subbands=subbands,
            structural=structural,
            discriminative=discriminative,
        )


class DiscriminativeReconstructionBlock(nn.Module):
    """Refine a stable reconstruction carrier with selected detail evidence.

    The block mirrors the encoder roles at reconstruction time: the base path
    maintains spatial organization, while the detail path selectively restores
    local contrast and target-related evidence selected by TAM.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.base_refinement = DepthwiseSeparableConv(channels, channels)
        self.detail_selection = ChannelSpatialGating(channels)
        self.detail_refinement = DepthwiseSeparableConv(channels, channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(channels, channels),
        )

    def forward(self, base: Tensor, detail: Tensor) -> Tensor:
        stable_base = self.base_refinement(base)
        selected_detail = self.detail_selection(detail, residual=False)
        selected_detail = self.detail_refinement(selected_detail)
        update = self.fusion(torch.cat((stable_base, selected_detail), dim=1))
        return base + update


class ReconstructionDecoder(nn.Module):
    """Reconstruct the fused luminance from TAM representations.

    The decoder consumes the two semantic roles produced by TAM instead of a
    single undifferentiated feature map.  Its depth is configurable because the
    paper-level contribution lies in the representation organization and prior
    interaction rather than in a fixed decoder depth.
    """

    def __init__(
        self,
        channels: int = 64,
        out_channels: int = 1,
        num_stages: int = 3,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError("num_stages must be positive")

        self.input_fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [DiscriminativeReconstructionBlock(channels) for _ in range(num_stages)]
        )
        hidden_channels = max(channels // 2, out_channels)
        self.image_head = nn.Sequential(
            DepthwiseSeparableConv(channels, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, interaction: TripartiteAttentionOutput) -> Tensor:
        base = interaction.reconstruction_base
        detail = interaction.task_aware_detail
        feature = self.input_fusion(torch.cat((base, detail), dim=1))
        for stage in self.stages:
            feature = stage(feature, detail)
        return torch.sigmoid(self.image_head(feature))


class TPTAF(nn.Module):
    """Task-Prior Tripartite Attention for infrared-visible fusion."""

    def __init__(
        self,
        channels: int = 64,
        detection_channels: int = 64,
        num_heads: int = 4,
        num_points: int = 8,
        window_size: int = 8,
        prior_strength: float = 1.0,
        decoder_stages: int = 3,
    ) -> None:
        super().__init__()
        self.encoder_ir = DecoupledEncoder(1, channels)
        self.encoder_vi = DecoupledEncoder(1, channels)
        self.tripartite_attention = TripartiteAttention(
            channels=channels,
            detection_channels=detection_channels,
            num_heads=num_heads,
            num_points=num_points,
            window_size=window_size,
            prior_strength=prior_strength,
        )
        self.decoder = ReconstructionDecoder(
            channels=channels,
            out_channels=1,
            num_stages=decoder_stages,
        )

    @staticmethod
    def _validate_inputs(
        infrared: Tensor,
        visible: Tensor,
        detection_semantics: Tensor,
    ) -> None:
        if infrared.ndim != 4 or visible.ndim != 4:
            raise ValueError("source images must be four-dimensional tensors")
        if infrared.shape != visible.shape:
            raise ValueError("infrared and visible tensors must have identical shapes")
        if infrared.shape[1] != 1:
            raise ValueError("TPTAF expects grayscale source images [B, 1, H, W]")
        if detection_semantics.ndim != 4:
            raise ValueError("detection semantics must be [B, C_det, H_det, W_det]")
        if detection_semantics.shape[0] != infrared.shape[0]:
            raise ValueError("source images and detection semantics must share batch size")

    def forward(
        self,
        infrared: Tensor,
        visible: Tensor,
        detection_semantics: Tensor,
    ) -> TPTAFOutput:
        self._validate_inputs(infrared, visible, detection_semantics)

        infrared_features = self.encoder_ir(infrared)
        visible_features = self.encoder_vi(visible)
        interaction = self.tripartite_attention(
            infrared_features.structural,
            infrared_features.discriminative,
            visible_features.structural,
            visible_features.discriminative,
            detection_semantics,
        )
        fused = self.decoder(interaction)
        if fused.shape[-2:] != infrared.shape[-2:]:
            fused = F.interpolate(
                fused,
                size=infrared.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return TPTAFOutput(
            fused=fused,
            infrared=infrared_features,
            visible=visible_features,
            interaction=interaction,
        )

    def load_checkpoint(
        self,
        path: str,
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get(
            "fusion_model",
            checkpoint.get("model", checkpoint),
        )
        incompatible = self.load_state_dict(state_dict, strict=strict)
        return {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
