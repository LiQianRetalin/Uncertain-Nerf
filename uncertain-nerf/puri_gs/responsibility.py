"""Detached, one-sided static responsibility for PURI-GS A1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ResponsibilityConfig:
    """Single source of truth for A1 responsibility hyperparameters."""

    enabled: bool = True
    start_step: int = 3000
    threshold: float = 1.5
    min_weight: float = 0.2
    pool_size: int = 3
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 0 < self.min_weight <= 1:
            raise ValueError("min_weight must be in (0, 1]")
        if self.pool_size <= 0 or self.pool_size % 2 == 0:
            raise ValueError("pool_size must be a positive odd integer")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


def _validate_rgb(rendered: Tensor, target: Tensor) -> None:
    if rendered.shape != target.shape:
        raise ValueError(
            f"rendered and target shapes differ: {rendered.shape} vs {target.shape}"
        )
    if rendered.ndim != 4 or rendered.shape[-1] != 3:
        raise ValueError("RGB tensors must have shape [B, H, W, 3]")
    if not rendered.is_floating_point() or not target.is_floating_point():
        raise TypeError("RGB tensors must be floating point")


def _validate_mask(mask: Tensor | None, residual: Tensor) -> Tensor | None:
    if mask is None:
        return None
    if mask.shape != residual.shape:
        raise ValueError(
            f"valid_mask must have shape {residual.shape}, got {mask.shape}"
        )
    mask = mask.to(device=residual.device, dtype=torch.bool)
    if not mask.flatten(1).any(dim=1).all():
        raise ValueError("each image must contain at least one valid pixel")
    return mask


@torch.no_grad()
def compute_responsibility(
    residual: Tensor,
    config: ResponsibilityConfig,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute a detached responsibility map from per-pixel RGB L1 residual.

    Args:
        residual: Tensor shaped ``[B, H, W]``.
        config: Validated A1 configuration.
        valid_mask: Optional boolean tensor with the same shape as ``residual``.
    """

    if residual.ndim != 3 or not residual.is_floating_point():
        raise ValueError("residual must be a floating tensor shaped [B, H, W]")
    residual = residual.detach()
    valid_mask = _validate_mask(valid_mask, residual)

    responsibilities: list[Tensor] = []
    for image_index in range(residual.shape[0]):
        image = residual[image_index]
        valid = (
            torch.ones_like(image, dtype=torch.bool)
            if valid_mask is None
            else valid_mask[image_index]
        )
        values = image[valid]
        median = values.median()
        mad = (values - median).abs().median()
        scale = 1.4826 * mad + config.epsilon
        positive_z = torch.clamp_min((image - median) / scale, 0.0)
        weight = torch.where(
            positive_z <= config.threshold,
            torch.ones_like(positive_z),
            config.threshold / (positive_z + config.epsilon),
        )
        weight = weight.clamp(config.min_weight, 1.0)
        weight = torch.where(valid, weight, torch.ones_like(weight))
        responsibilities.append(weight)

    responsibility = torch.stack(responsibilities, dim=0)
    if config.pool_size > 1:
        padding = config.pool_size // 2
        if valid_mask is None:
            responsibility = F.avg_pool2d(
                responsibility.unsqueeze(1),
                kernel_size=config.pool_size,
                stride=1,
                padding=padding,
            ).squeeze(1)
        else:
            valid_float = valid_mask.to(dtype=responsibility.dtype)
            numerator = F.avg_pool2d(
                (responsibility * valid_float).unsqueeze(1),
                kernel_size=config.pool_size,
                stride=1,
                padding=padding,
            ).squeeze(1)
            denominator = F.avg_pool2d(
                valid_float.unsqueeze(1),
                kernel_size=config.pool_size,
                stride=1,
                padding=padding,
            ).squeeze(1)
            responsibility = numerator / denominator.clamp_min(config.epsilon)
            responsibility = torch.where(
                valid_mask, responsibility, torch.ones_like(responsibility)
            )

    return responsibility.clamp(config.min_weight, 1.0).detach()


def responsibility_weighted_l1(
    rendered: Tensor,
    target: Tensor,
    config: ResponsibilityConfig,
    step: int,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Return baseline L1 or A1 responsibility-weighted L1.

    The disabled and warm-up paths call ``torch.nn.functional.l1_loss``
    directly so B0/B1 retain the original gsplat loss exactly.
    """

    _validate_rgb(rendered, target)
    if step < 0:
        raise ValueError("step must be non-negative")
    if not config.enabled or step < config.start_step:
        return F.l1_loss(rendered, target), None

    pixel_l1 = (rendered - target).abs().mean(dim=-1)
    valid_mask = _validate_mask(valid_mask, pixel_l1)
    responsibility = compute_responsibility(
        pixel_l1.detach(), config=config, valid_mask=valid_mask
    )
    if valid_mask is None:
        numerator = (responsibility * pixel_l1).sum()
        denominator = responsibility.sum()
    else:
        valid_float = valid_mask.to(dtype=pixel_l1.dtype)
        numerator = (responsibility * pixel_l1 * valid_float).sum()
        denominator = (responsibility * valid_float).sum()
    loss = numerator / (denominator + config.epsilon)
    return loss, responsibility
