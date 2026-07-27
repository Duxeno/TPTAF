"""Training objectives for TPTAF.

The objective follows the paper formulation:

L = [L_sime, L_deco, L_grad, L_box, L_cls, L_dfl]

and UWL learns one log-variance parameter for each heterogeneous objective.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import TPTAFOutput


class SimilarityLoss(nn.Module):
    """Pixel fidelity and SSIM similarity term ``L_sime``."""

    def __init__(self, c1: float = 1e-4, c2: float = 9e-4) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:
        window = 11
        padding = window // 2
        mean_x = F.avg_pool2d(x, window, 1, padding)
        mean_y = F.avg_pool2d(y, window, 1, padding)
        var_x = F.avg_pool2d(x * x, window, 1, padding) - mean_x.square()
        var_y = F.avg_pool2d(y * y, window, 1, padding) - mean_y.square()
        cov = F.avg_pool2d(x * y, window, 1, padding) - mean_x * mean_y

        numerator = (2 * mean_x * mean_y + self.c1) * (2 * cov + self.c2)
        denominator = (
            (mean_x.square() + mean_y.square() + self.c1)
            * (var_x + var_y + self.c2)
        )
        return (numerator / denominator.clamp_min(1e-12)).mean()

    def forward(
        self,
        fused: Tensor,
        infrared: Tensor,
        visible: Tensor,
    ) -> Tensor:
        loss = 0.0
        for source in (infrared, visible):
            loss = loss + 1.0 - self._ssim(source, fused)
            loss = loss + F.mse_loss(fused, source)
        return loss


class DecouplingLoss(nn.Module):
    """Structural/discriminative feature organization constraint ``L_deco``."""

    def __init__(self, c3: float = 0.01) -> None:
        super().__init__()
        self.c3 = c3

    def _correlation(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.flatten(2)
        y = y.flatten(2)
        x = x - x.mean(dim=2, keepdim=True)
        y = y - y.mean(dim=2, keepdim=True)
        numerator = (x * y).mean(dim=2)
        denominator = (
            x.square().mean(dim=2).sqrt()
            * y.square().mean(dim=2).sqrt()
            + self.c3
        )
        return (numerator / denominator).mean()

    def forward(self, output: TPTAFOutput) -> Tensor:
        detail_correlation = self._correlation(
            output.discriminative_ir,
            output.discriminative_vi,
        )
        phase_correlation = self._correlation(
            output.structural_ir,
            output.structural_vi,
        )
        return detail_correlation.square() / (1.0 + phase_correlation)


class GradientLoss(nn.Module):
    """Sobel gradient preservation term ``L_grad``."""

    def __init__(self, infrared_weight: float = 0.5) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        self.register_buffer("kx", kernel[None, None], persistent=False)
        self.register_buffer("ky", kernel.T[None, None], persistent=False)
        self.infrared_weight = infrared_weight

    def _gradient(self, image: Tensor) -> Tensor:
        return torch.cat(
            (
                F.conv2d(image, self.kx.to(image.dtype), padding=1),
                F.conv2d(image, self.ky.to(image.dtype), padding=1),
            ),
            dim=1,
        )

    def forward(
        self,
        fused: Tensor,
        infrared: Tensor,
        visible: Tensor,
    ) -> Tensor:
        target = (
            self.infrared_weight * self._gradient(infrared)
            + (1.0 - self.infrared_weight) * self._gradient(visible)
        )
        return F.l1_loss(self._gradient(fused), target)


class UncertaintyWeightedLoss(nn.Module):
    """Learnable weighting in Eq. (28)."""

    names = ("sime", "deco", "grad", "box", "cls", "dfl")

    def __init__(
        self,
        clamp_min: float = -4.6,
        clamp_max: float = 4.6,
    ) -> None:
        super().__init__()
        self.log_variances = nn.Parameter(torch.zeros(6))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, losses: Sequence[Tensor]) -> Tensor:
        total = 0.0
        for index, loss in enumerate(losses):
            total = total + (
                torch.exp(-self.log_variances[index]) * loss
                + 0.5 * self.log_variances[index]
            )
        return total

    @torch.no_grad()
    def clamp_(self) -> None:
        self.log_variances.clamp_(self.clamp_min, self.clamp_max)


class TPTAFLoss(nn.Module):
    """Complete fusion-detection objective used by joint training."""

    def __init__(self) -> None:
        super().__init__()
        self.similarity = SimilarityLoss()
        self.decoupling = DecouplingLoss()
        self.gradient = GradientLoss()
        self.uwl = UncertaintyWeightedLoss()

    def forward(
        self,
        output: TPTAFOutput,
        infrared: Tensor,
        visible: Tensor,
        detection_losses: Mapping[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        components = {
            "sime": self.similarity(output.fused, infrared, visible),
            "deco": self.decoupling(output),
            "grad": self.gradient(output.fused, infrared, visible),
            "box": detection_losses["box"],
            "cls": detection_losses["cls"],
            "dfl": detection_losses["dfl"],
        }
        return (
            self.uwl([components[name] for name in self.uwl.names]),
            components,
        )
