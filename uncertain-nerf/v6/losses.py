import torch

from v5.losses import DepthScaleAligner
from v5.robust import mad_scale, pseudo_huber


def uncertainty_variance(uncertainty, variance_min=1.0e-4, variance_max=0.05):
    return float(variance_min) + (float(variance_max) - float(variance_min)) * uncertainty


def heteroscedastic_color_nll(
    predicted_rgb, target_rgb, ray_uncertainty, variance_min=1.0e-4, variance_max=0.05
):
    squared_error = (predicted_rgb - target_rgb).square().mean(dim=-1)
    variance = uncertainty_variance(ray_uncertainty, variance_min, variance_max)
    return (0.5 * squared_error / variance + 0.5 * variance.log()).mean()


def geometry_loss(
    predicted_depth,
    student_uncertainty,
    prior_depth,
    mask,
    scale,
    gamma,
    huber_multiplier,
):
    if not mask.any():
        zero = predicted_depth.new_zeros(())
        return zero, {"delta": zero, "count": 0}
    residual = predicted_depth[mask] - float(scale) * prior_depth[mask]
    robust_scale = mad_scale(residual.detach())
    delta = max(float(huber_multiplier), 1.0e-3) * robust_scale
    # Stop-gradient is essential: uncertainty cannot increase to escape geometry.
    confidence = torch.exp(-float(gamma) * student_uncertainty[mask].detach())
    loss = (confidence * pseudo_huber(residual, delta)).mean()
    return loss, {"delta": delta.detach(), "count": int(mask.sum())}


__all__ = [
    "DepthScaleAligner",
    "geometry_loss",
    "heteroscedastic_color_nll",
    "uncertainty_variance",
]
