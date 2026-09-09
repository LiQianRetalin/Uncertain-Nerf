"""Semantic consensus mask and historical residual evidence for PURI-GS-RU."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StaticResponsibilityHead(nn.Module):
    """Fixed point-wise 384 -> 16 -> 1 static inlier predictor."""

    def __init__(self, feature_dim: int = 384, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError("mask-head features must be [B,C,H,W]")
        tokens = features.permute(0, 2, 3, 1)
        probabilities = self.mlp(tokens)
        return probabilities.permute(0, 3, 1, 2)

    def reference_weight_regularizer(self) -> Tensor:
        """Preserve the reference ``.data`` behavior (value-only, no gradient)."""

        first = self.mlp[0]
        second = self.mlp[2]
        return 0.5 * torch.max(torch.abs(first.weight.data)) * torch.max(
            torch.abs(second.weight.data)
        )


@dataclass(frozen=True)
class ResidualHistogramConfig:
    bins: int = 10_000
    momentum: float = 0.95
    lower_quantile: float = 0.60
    upper_quantile: float = 0.80


class ResidualHistogram:
    """CPU historical histogram matching the fixed reference accumulation rule."""

    def __init__(self, config: ResidualHistogramConfig = ResidualHistogramConfig()):
        if config.bins <= 0:
            raise ValueError("histogram bins must be positive")
        if not 0 <= config.momentum < 1:
            raise ValueError("histogram momentum must be in [0,1)")
        if not 0 <= config.lower_quantile < config.upper_quantile <= 1:
            raise ValueError("residual quantiles must satisfy 0 <= lower < upper <= 1")
        self.config = config
        self.histogram = torch.zeros(config.bins, dtype=torch.float32)

    @torch.no_grad()
    def update(self, residual: Tensor) -> tuple[float, float]:
        values = residual.detach().float().cpu().clamp(0, 1).reshape(-1)
        current = torch.histc(values, bins=self.config.bins, min=0.0, max=1.0)
        # The reference implementation uses 0.95 * history + current, not
        # 0.95 * history + 0.05 * current.  Keep that exact behavior.
        self.histogram.mul_(self.config.momentum).add_(current)
        return self.thresholds()

    def thresholds(self) -> tuple[float, float]:
        total = self.histogram.sum()
        if total <= 0:
            return 0.0, 1.0
        cumulative = self.histogram.cumsum(0)
        edges = torch.linspace(0, 1, self.config.bins + 1)

        def quantile(value: float) -> float:
            index = int(torch.searchsorted(cumulative, total * value).item())
            return float(edges[min(index, self.config.bins)].item())

        return quantile(self.config.lower_quantile), quantile(
            self.config.upper_quantile
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "histogram": self.histogram.clone(),
            "bins": self.config.bins,
            "momentum": self.config.momentum,
            "lower_quantile": self.config.lower_quantile,
            "upper_quantile": self.config.upper_quantile,
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the exact accumulated histogram after validating its contract."""
        required = {
            "histogram", "bins", "momentum", "lower_quantile", "upper_quantile"
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("residual histogram state has an invalid schema")
        saved_config = ResidualHistogramConfig(
            bins=state["bins"],
            momentum=state["momentum"],
            lower_quantile=state["lower_quantile"],
            upper_quantile=state["upper_quantile"],
        )
        if saved_config != self.config:
            raise ValueError("residual histogram configuration differs from runtime")
        histogram = state["histogram"]
        if (
            not isinstance(histogram, Tensor)
            or histogram.shape != self.histogram.shape
            or histogram.dtype != self.histogram.dtype
            or not torch.isfinite(histogram).all()
            or torch.any(histogram < 0)
        ):
            raise ValueError("residual histogram tensor is invalid")
        self.histogram.copy_(histogram.to(device=self.histogram.device))


def residual_interval_mask(residual: Tensor, threshold: float) -> Tensor:
    """Fixed 3x3 rule: center inlier OR a strict majority of inlier neighbors."""

    if residual.ndim == 3:
        residual = residual.unsqueeze(1)
    if residual.ndim != 4 or residual.shape[1] != 1:
        raise ValueError("residual must be [B,1,H,W] or [B,H,W]")
    inlier = (residual < threshold).to(residual.dtype)
    kernel = residual.new_full((1, 1, 3, 3), 1.0 / 9.0)
    majority = (F.conv2d(inlier, kernel, padding=1) > 0.5).to(residual.dtype)
    return ((majority + inlier) > 1e-3).to(residual.dtype)


def cosine_static_target(gt_features: Tensor, render_features: Tensor) -> Tensor:
    if gt_features.shape != render_features.shape:
        raise ValueError("GT and render feature grids must have identical shapes")
    gt = F.normalize(gt_features, dim=1)
    rendered = F.normalize(render_features, dim=1)
    cosine = (gt * rendered).sum(dim=1, keepdim=True)
    return torch.clamp(2.0 * cosine - 1.0, 0.0, 1.0).detach()


def hard_static_mask(
    probability: Tensor, *, threshold: float = 0.25, kernel_size: int = 7
) -> Tensor:
    if kernel_size != 7 or threshold != 0.25:
        raise ValueError("PURI-GS-RU fixes threshold=0.25 and kernel_size=7")
    hard = (probability > threshold).to(probability.dtype)
    return -F.max_pool2d(-hard, kernel_size=kernel_size, stride=1, padding=3)


def masked_photo_loss(
    render: Tensor,
    target: Tensor,
    safe_mask: Tensor,
    *,
    fused_ssim_fn,
    ssim_lambda: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fixed unnormalized masked L1 and masked DSSIM."""

    if render.shape != target.shape or render.ndim != 4 or render.shape[-1] != 3:
        raise ValueError("render and target must be matching [B,H,W,3] tensors")
    mask_bhwc = safe_mask.permute(0, 2, 3, 1)
    l1 = (mask_bhwc * torch.abs(render - target)).mean()
    masked_render = mask_bhwc * render
    masked_target = mask_bhwc * target
    dssim = 1.0 - fused_ssim_fn(
        masked_render.permute(0, 3, 1, 2),
        masked_target.permute(0, 3, 1, 2),
        padding="valid",
    )
    return (1.0 - ssim_lambda) * l1 + ssim_lambda * dssim, l1, dssim


def semantic_mask_loss(
    predicted_grid: Tensor,
    cosine_target_grid: Tensor,
    residual_low: Tensor,
    residual_high: Tensor,
    *,
    image_size: tuple[int, int],
    step: int,
    head: StaticResponsibilityHead,
    cosine_weight: float = 0.5,
    residual_weight: float = 0.5,
    static_prior_weight: float = 2.0,
    static_prior_decay: float = 2000.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    predicted = F.interpolate(
        predicted_grid, size=image_size, mode="bilinear", align_corners=False
    )
    cosine_target = F.interpolate(
        cosine_target_grid, size=image_size, mode="bilinear", align_corners=False
    )
    cosine_loss = torch.abs(predicted - cosine_target).mean()
    residual_loss = (
        F.relu(predicted - residual_high) + F.relu(residual_low - predicted)
    ).mean()
    static_prior = (
        static_prior_weight
        * math.exp(-float(step) / static_prior_decay)
        * (1.0 - predicted).mean()
    )
    weight_regularizer = head.reference_weight_regularizer()
    total = (
        cosine_weight * cosine_loss
        + residual_weight * residual_loss
        + static_prior
        + weight_regularizer
    )
    return total, {
        "mask_probability": predicted,
        "feature_similarity": cosine_target,
        "cosine_loss": cosine_loss,
        "residual_loss": residual_loss,
        "static_prior": static_prior,
        "weight_regularizer": weight_regularizer,
    }
