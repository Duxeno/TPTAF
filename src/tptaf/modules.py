"""Core method modules for TPTAF.

The implementation follows the representation flow described in the paper:

1. frequency-aware structural/discriminative decomposition;
2. global-local modeling for modality features;
3. detection-prior generation;
4. sparse tripartite attention for task-aware feature selection.

Detector-specific internals are intentionally separated from this file.  TPTAF
only requires the detection-semantic feature map defined by the public contract
in :mod:`tptaf.detector`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FrequencySubbands:
    """Wavelet subbands of an encoded modality feature."""

    low_low: Tensor
    low_high: Tensor
    high_low: Tensor
    high_high: Tensor


@dataclass(frozen=True)
class DetectionPrior:
    """Detection-conditioned guidance generated from ``K_det``.

    Attributes:
        temperature: Positive per-head temperature ``T_d``.
        spatial_distribution: Window-normalized spatial prior ``P_d``.
        reference_points: Normalized reference locations ``R``.
        offsets: Local sampling offsets ``Delta_m``.
        sampling_positions: Continuous positions ``p_m = R + Delta_m``.
    """

    temperature: Tensor
    spatial_distribution: Tensor
    reference_points: Tensor
    offsets: Tensor
    sampling_positions: Tensor


@dataclass(frozen=True)
class TripartiteAttentionOutput:
    """Outputs used by the reconstruction stage and for method inspection."""

    reconstruction_base: Tensor
    task_aware_detail: Tensor
    query: Tensor
    discriminative_key: Tensor
    detection_key: Tensor
    prior: DetectionPrior
    attention_weights: Tensor


class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial filtering followed by channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, feature: Tensor) -> Tensor:
        return self.block(feature)


def _absolute_position_encoding(
    channels: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create the two-dimensional absolute positional encoding in Eq. (2)."""
    if channels % 4 != 0:
        raise ValueError("channels must be divisible by four for 2-D encoding")

    quarter = channels // 4
    frequency = 1.0 / (
        10000
        ** (
            torch.arange(quarter, device=device, dtype=torch.float32)
            / max(quarter - 1, 1)
        )
    )
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None]

    y_phase = y * frequency[None, :]
    x_phase = x * frequency[None, :]
    y_encoding = torch.cat((y_phase.sin(), y_phase.cos()), dim=1)
    x_encoding = torch.cat((x_phase.sin(), x_phase.cos()), dim=1)

    y_encoding = y_encoding.T[:, :, None].expand(-1, -1, width)
    x_encoding = x_encoding.T[:, None, :].expand(-1, height, -1)
    return torch.cat((y_encoding, x_encoding), dim=0)[None].to(dtype=dtype)


class AbsolutePositionEmbedding(nn.Module):
    """Learnably scale the absolute position term in Eq. (2)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, feature: Tensor) -> Tensor:
        position = _absolute_position_encoding(
            self.channels,
            feature.shape[-2],
            feature.shape[-1],
            device=feature.device,
            dtype=feature.dtype,
        )
        return feature + self.scale * position


class HaarWaveletDecomposition(nn.Module):
    """Two-dimensional Haar DWT used for Eq. (3)."""

    def __init__(self) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ) / 2.0
        self.register_buffer("filters", filters[:, None], persistent=False)

    def forward(self, feature: Tensor) -> FrequencySubbands:
        batch, channels, height, width = feature.shape
        if height % 2 or width % 2:
            feature = F.pad(
                feature,
                (0, width % 2, 0, height % 2),
                mode="replicate",
            )

        transformed = F.conv2d(
            feature,
            self.filters.repeat(channels, 1, 1, 1).to(feature.dtype),
            stride=2,
            groups=channels,
        )
        transformed = transformed.reshape(
            batch,
            channels,
            4,
            feature.shape[-2] // 2,
            feature.shape[-1] // 2,
        )
        return FrequencySubbands(
            low_low=transformed[:, :, 0],
            low_high=transformed[:, :, 1],
            high_low=transformed[:, :, 2],
            high_high=transformed[:, :, 3],
        )


class SqueezeExcitation(nn.Module):
    """Channel-response recalibration used by PCSB and DDEB."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return feature * self.gate(feature)


class ChannelSpatialGating(nn.Module):
    """Channel-spatial gating mechanism of Eqs. (9)--(11)."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.spatial_projection = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False,
        )

    def forward(self, feature: Tensor, *, residual: bool = True) -> Tensor:
        channel_descriptor = F.adaptive_avg_pool2d(feature, 1)
        channel_gate = torch.sigmoid(self.channel_mlp(channel_descriptor))

        spatial_descriptor = torch.cat(
            (
                feature.mean(dim=1, keepdim=True),
                feature.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = torch.sigmoid(self.spatial_projection(spatial_descriptor))
        composite_gate = channel_gate * spatial_gate
        if residual:
            return feature * (1.0 + composite_gate)
        return feature * composite_gate


class PhaseCoherentStructureBlock(nn.Module):
    """Construct the phase-coherent structural space (Eqs. (4)--(6))."""

    def __init__(
        self,
        channels: int,
        mask_kernel: int = 3,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        self.amplitude_mask = nn.Conv2d(
            channels,
            channels,
            kernel_size=mask_kernel,
            padding=mask_kernel // 2,
            groups=channels,
        )
        self.residual_strength = nn.Parameter(torch.zeros(()))
        self.spatial_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.channel_recalibration = SqueezeExcitation(channels, reduction)

    def forward(self, low_frequency: Tensor, output_size: tuple[int, int]) -> Tensor:
        spectrum = torch.fft.fft2(low_frequency, norm="ortho")
        amplitude = spectrum.abs()
        phase = torch.angle(spectrum)

        mask = torch.sigmoid(self.amplitude_mask(amplitude))
        reweighted_amplitude = amplitude * (
            1.0 + self.residual_strength * mask
        )
        phase_preserved_spectrum = torch.polar(reweighted_amplitude, phase)
        structural = torch.fft.ifft2(
            phase_preserved_spectrum,
            norm="ortho",
        ).real

        structural = F.interpolate(
            structural,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        structural = F.silu(self.spatial_projection(structural))
        return self.channel_recalibration(structural)


class DiverseBranchBlock(nn.Module):
    """Parallel local-detail branches used by DDEB in Eq. (8)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.spatial_branch = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )
        self.point_branch = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.context_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride=1, padding=padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.separable_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, feature: Tensor) -> Tensor:
        enhanced = (
            self.spatial_branch(feature)
            + self.point_branch(feature)
            + self.context_branch(feature)
            + self.separable_branch(feature)
        )
        return F.silu(self.normalization(enhanced))


class DiscriminativeDetailEnhancementBlock(nn.Module):
    """Construct the discriminative detail space (Eqs. (7)--(11))."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.local_enhancement = DiverseBranchBlock(3 * channels, channels)
        self.channel_recalibration = SqueezeExcitation(channels, reduction)
        self.significance_selection = ChannelSpatialGating(channels, reduction)

    def forward(
        self,
        low_high: Tensor,
        high_low: Tensor,
        high_high: Tensor,
        output_size: tuple[int, int],
    ) -> Tensor:
        restored_subbands = [
            F.interpolate(
                subband,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            for subband in (low_high, high_low, high_high)
        ]
        detail = torch.cat(restored_subbands, dim=1)
        detail = self.local_enhancement(detail)
        detail = self.channel_recalibration(detail)
        return self.significance_selection(detail, residual=True)


def _partition_windows(feature: Tensor, window_size: int) -> Tensor:
    batch, height, width, channels = feature.shape
    return (
        feature.view(
            batch,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size * window_size, channels)
    )


def _reverse_windows(
    windows: Tensor,
    window_size: int,
    batch: int,
    height: int,
    width: int,
) -> Tensor:
    channels = windows.shape[-1]
    return (
        windows.view(
            batch,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, channels)
    )


class WindowSelfAttention(nn.Module):
    """Self-attention inside local windows, with optional window shift."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if not 0 <= shift_size < window_size:
            raise ValueError("shift_size must be in [0, window_size)")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = self.head_dim ** -0.5

        self.normalization = nn.LayerNorm(channels)
        self.qkv_projection = nn.Linear(channels, 3 * channels)
        self.output_projection = nn.Linear(channels, channels)

    def _attention_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor | None:
        if self.shift_size == 0:
            return None

        mask = torch.zeros((1, height, width, 1), device=device)
        window = self.window_size
        shift = self.shift_size
        region_id = 0
        for y_slice in (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        ):
            for x_slice in (
                slice(0, -window),
                slice(-window, -shift),
                slice(-shift, None),
            ):
                mask[:, y_slice, x_slice, :] = region_id
                region_id += 1

        mask_windows = _partition_windows(mask, window).squeeze(-1)
        pairwise_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        return pairwise_mask.masked_fill(pairwise_mask != 0, -100.0).masked_fill(
            pairwise_mask == 0,
            0.0,
        )

    def forward(self, feature: Tensor) -> Tensor:
        batch, channels, height, width = feature.shape
        window = self.window_size
        pad_height = (window - height % window) % window
        pad_width = (window - width % window) % window

        padded = F.pad(feature, (0, pad_width, 0, pad_height))
        padded = padded.permute(0, 2, 3, 1)
        padded_height, padded_width = padded.shape[1:3]

        if self.shift_size:
            padded = torch.roll(
                padded,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )

        windows = _partition_windows(padded, window)
        normalized = self.normalization(windows)
        qkv = self.qkv_projection(normalized)
        qkv = qkv.reshape(
            qkv.shape[0],
            qkv.shape[1],
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        mask = self._attention_mask(padded_height, padded_width, feature.device)
        if mask is not None:
            windows_per_image = mask.shape[0]
            attention = attention.view(
                batch,
                windows_per_image,
                self.num_heads,
                window * window,
                window * window,
            )
            attention = attention + mask[None, :, None]
            attention = attention.view(
                -1,
                self.num_heads,
                window * window,
                window * window,
            )

        attention = attention.softmax(dim=-1)
        attended = attention @ value
        attended = attended.transpose(1, 2).reshape(
            normalized.shape[0],
            normalized.shape[1],
            channels,
        )
        attended = self.output_projection(attended) + windows
        attended = _reverse_windows(
            attended,
            window,
            batch,
            padded_height,
            padded_width,
        )

        if self.shift_size:
            attended = torch.roll(
                attended,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        attended = attended[:, :height, :width]
        return attended.permute(0, 3, 1, 2).contiguous()


class GlobalLocalAttention(nn.Module):
    """GLA with local-window and shifted-window information exchange.

    The first attention stage models local responses.  The shifted stage
    exchanges information across neighboring windows, while the depthwise
    branch preserves texture locality.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.local_window = WindowSelfAttention(
            channels,
            num_heads,
            window_size,
            shift_size=0,
        )
        self.sliding_window = WindowSelfAttention(
            channels,
            num_heads,
            window_size,
            shift_size=window_size // 2,
        )
        self.texture_branch = DepthwiseSeparableConv(channels, channels)
        self.fusion = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, feature: Tensor) -> Tensor:
        contextual = self.sliding_window(self.local_window(feature))
        local_texture = self.texture_branch(feature)
        update = self.fusion(torch.cat((contextual, local_texture), dim=1))
        return feature + F.silu(update)


def _window_softmax(logits: Tensor, window_size: int) -> Tensor:
    """Normalize support logits independently inside each attention window."""
    batch, channels, height, width = logits.shape
    if channels != 1:
        raise ValueError("spatial-prior logits must have one channel")

    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    padded = F.pad(logits, (0, pad_width, 0, pad_height))
    padded_height, padded_width = padded.shape[-2:]

    windows = padded.view(
        batch,
        1,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
    ).permute(0, 2, 4, 1, 3, 5)
    windows = windows.reshape(-1, window_size * window_size)
    windows = windows.softmax(dim=-1)
    windows = windows.view(
        batch,
        padded_height // window_size,
        padded_width // window_size,
        1,
        window_size,
        window_size,
    ).permute(0, 3, 1, 4, 2, 5)
    normalized = windows.reshape(batch, 1, padded_height, padded_width)
    return normalized[:, :, :height, :width]


class DetectionPriorGenerator(nn.Module):
    """Convert ``K_det`` into the guidance terms of Eqs. (18)--(21)."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_points: int = 8,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.window_size = window_size

        self.temperature_mapping = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, num_heads, kernel_size=1),
        )
        self.support_mapping = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.reference_mapping = nn.Conv2d(channels, 2, kernel_size=1)
        self.offset_mapping = nn.Conv2d(
            channels,
            2 * num_points,
            kernel_size=3,
            padding=1,
        )

    def forward(self, detection_key: Tensor) -> DetectionPrior:
        batch, _, height, width = detection_key.shape
        temperature = F.softplus(self.temperature_mapping(detection_key)) + 1e-4

        support_logits = self.support_mapping(detection_key)
        spatial_distribution = _window_softmax(
            support_logits,
            self.window_size,
        )

        reference_points = torch.sigmoid(self.reference_mapping(detection_key))
        offsets = 0.25 * torch.tanh(self.offset_mapping(detection_key))
        offsets = offsets.view(batch, self.num_points, 2, height, width)
        sampling_positions = (
            reference_points[:, None] + offsets
        ).clamp_(0.0, 1.0)

        return DetectionPrior(
            temperature=temperature,
            spatial_distribution=spatial_distribution,
            reference_points=reference_points,
            offsets=offsets,
            sampling_positions=sampling_positions,
        )


def sample_at_positions(feature: Tensor, positions: Tensor) -> Tensor:
    """Bilinearly sample ``M`` candidates for every spatial query."""
    batch, channels, height, width = feature.shape
    num_points = positions.shape[1]
    grid = positions.permute(0, 3, 4, 1, 2)
    grid = grid.reshape(batch, height, width * num_points, 2)
    grid = 2.0 * grid - 1.0

    sampled = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.view(batch, channels, height, width, num_points)
    return sampled.permute(0, 1, 4, 2, 3)


class TripartiteAttention(nn.Module):
    """Task-prior tripartite attention of Eqs. (15)--(23).

    Structural and discriminative features are first unified within each
    modality by GLA.  The fused-state query and value are formed from both
    modalities, while detail selection uses ``K_dis`` and detection semantics
    are converted into guidance through ``K_det`` and DPG.
    """

    def __init__(
        self,
        channels: int,
        detection_channels: int,
        num_heads: int = 4,
        num_points: int = 8,
        window_size: int = 8,
        prior_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.num_points = num_points
        self.prior_strength = prior_strength

        self.infrared_unification = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            GlobalLocalAttention(channels, num_heads, window_size),
        )
        self.visible_unification = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            GlobalLocalAttention(channels, num_heads, window_size),
        )
        self.query_projection = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.value_projection = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.discriminative_key_projection = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=1,
        )
        self.detection_key_projection = nn.Conv2d(
            detection_channels,
            channels,
            kernel_size=1,
        )
        self.prior_generator = DetectionPriorGenerator(
            channels,
            num_heads,
            num_points,
            window_size,
        )
        self.detail_projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        structural_ir: Tensor,
        discriminative_ir: Tensor,
        structural_vi: Tensor,
        discriminative_vi: Tensor,
        detection_semantics: Tensor,
    ) -> TripartiteAttentionOutput:
        target_size = structural_ir.shape[-2:]
        detection_semantics = F.interpolate(
            detection_semantics,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        infrared_feature = self.infrared_unification(
            torch.cat((structural_ir, discriminative_ir), dim=1)
        )
        visible_feature = self.visible_unification(
            torch.cat((structural_vi, discriminative_vi), dim=1)
        )
        cross_modal_feature = torch.cat(
            (infrared_feature, visible_feature),
            dim=1,
        )

        query = self.query_projection(cross_modal_feature)
        value = self.value_projection(cross_modal_feature)
        discriminative_key = self.discriminative_key_projection(
            torch.cat((discriminative_ir, discriminative_vi), dim=1)
        )
        detection_key = self.detection_key_projection(detection_semantics)
        prior = self.prior_generator(detection_key)

        sampled_keys = sample_at_positions(
            discriminative_key,
            prior.sampling_positions,
        )
        sampled_values = sample_at_positions(value, prior.sampling_positions)
        sampled_prior = sample_at_positions(
            prior.spatial_distribution,
            prior.sampling_positions,
        ).squeeze(1)

        batch, _, height, width = query.shape
        query_heads = query.view(
            batch,
            self.num_heads,
            self.head_dim,
            height,
            width,
        )
        key_heads = sampled_keys.view(
            batch,
            self.num_heads,
            self.head_dim,
            self.num_points,
            height,
            width,
        )
        value_heads = sampled_values.view(
            batch,
            self.num_heads,
            self.head_dim,
            self.num_points,
            height,
            width,
        )

        compatibility = (query_heads.unsqueeze(3) * key_heads).sum(dim=2)
        denominator = math.sqrt(self.head_dim) * prior.temperature.unsqueeze(2)
        compatibility = compatibility / denominator.clamp_min(1e-8)
        prior_bias = self.prior_strength * torch.log(
            sampled_prior[:, None].clamp_min(1e-8)
        )
        attention_logits = compatibility + prior_bias
        attention_weights = attention_logits.softmax(dim=2)

        attended_detail = (
            attention_weights.unsqueeze(2) * value_heads
        ).sum(dim=3)
        attended_detail = attended_detail.reshape(
            batch,
            self.channels,
            height,
            width,
        )
        task_aware_detail = self.detail_projection(attended_detail)

        # This feature is the stable reconstruction carrier paired with the
        # task-aware detail branch by the decoder.
        reconstruction_base = 0.5 * (infrared_feature + visible_feature)

        return TripartiteAttentionOutput(
            reconstruction_base=reconstruction_base,
            task_aware_detail=task_aware_detail,
            query=query,
            discriminative_key=discriminative_key,
            detection_key=detection_key,
            prior=prior,
            attention_weights=attention_weights,
        )


# Backward-compatible aliases retained for existing imports.
PositionalEmbedding = AbsolutePositionEmbedding
HaarDWT = HaarWaveletDecomposition
ChannelSpatialGate = ChannelSpatialGating
