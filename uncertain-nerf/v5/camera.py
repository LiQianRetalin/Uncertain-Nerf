"""Differentiable camera distortion helpers used by training and rendering."""

import torch


def _broadcast_parameter(value, target):
    parameter = torch.as_tensor(value, dtype=target.dtype, device=target.device)
    if parameter.ndim > target.ndim:
        raise ValueError("Camera parameter has more dimensions than its target")
    while parameter.ndim < target.ndim:
        parameter = parameter.unsqueeze(-1)
    return parameter


def distort_simple_radial(normalized_xy, radial_k):
    """Apply COLMAP SIMPLE_RADIAL distortion to normalized image coordinates."""
    if normalized_xy.shape[-1] != 2:
        raise ValueError("normalized_xy must have a final dimension of size 2")
    radius_squared = normalized_xy.square().sum(dim=-1)
    radial_k = _broadcast_parameter(radial_k, radius_squared)
    scale = 1.0 + radial_k * radius_squared
    return normalized_xy * scale.unsqueeze(-1)


def undistort_simple_radial(distorted_xy, radial_k, iterations=10):
    """Invert COLMAP SIMPLE_RADIAL distortion with differentiable Newton steps."""
    if distorted_xy.shape[-1] != 2:
        raise ValueError("distorted_xy must have a final dimension of size 2")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    distorted_radius = torch.linalg.vector_norm(distorted_xy, dim=-1)
    radial_k = _broadcast_parameter(radial_k, distorted_radius)
    radius = distorted_radius
    for _ in range(iterations):
        radius_squared = radius.square()
        residual = radius * (1.0 + radial_k * radius_squared) - distorted_radius
        derivative = 1.0 + 3.0 * radial_k * radius_squared
        derivative = torch.where(
            derivative.abs() >= 1.0e-8,
            derivative,
            torch.where(derivative >= 0, 1.0e-8, -1.0e-8),
        )
        candidate = radius - residual / derivative
        radius = torch.where(
            torch.isfinite(candidate) & (candidate >= 0.0), candidate, radius
        )
    scale = torch.where(
        distorted_radius > 1.0e-12,
        radius / distorted_radius.clamp_min(1.0e-12),
        torch.ones_like(distorted_radius),
    )
    return distorted_xy * scale.unsqueeze(-1)
