import math

import torch

from v5.camera import distort_simple_radial
from v5.robust import mad_scale, tukey_biweight


def ray_termination_uncertainty(fine_result):
    weights = fine_result["weights"].detach()
    acc = fine_result["acc"].detach()
    probabilities = weights / (acc[:, None] + 1.0e-8)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum(dim=-1)
    entropy = entropy / max(math.log(weights.shape[-1]), 1.0)
    return (acc * entropy + (1.0 - acc)).clamp(0.0, 1.0)


def _bilinear_pixels(images, image_ids, u, v):
    """Sample four pixels per query without materializing BxV complete images."""
    height, width = images.shape[1:3]
    x0 = u.floor().long().clamp(0, width - 1)
    y0 = v.floor().long().clamp(0, height - 1)
    x1 = (x0 + 1).clamp_max(width - 1)
    y1 = (y0 + 1).clamp_max(height - 1)
    wx = (u - x0.to(u.dtype))[..., None]
    wy = (v - y0.to(v.dtype))[..., None]
    c00 = images[image_ids, y0, x0, :3]
    c01 = images[image_ids, y0, x1, :3]
    c10 = images[image_ids, y1, x0, :3]
    c11 = images[image_ids, y1, x1, :3]
    top = c00 * (1.0 - wx) + c01 * wx
    bottom = c10 * (1.0 - wx) + c11 * wx
    return top * (1.0 - wy) + bottom * wy


def multiview_surface_uncertainty(
    surface_points, reference_rgb, reference_ids, scene, tukey_multiplier
):
    source_ids = scene.source_views[reference_ids]
    source_valid = source_ids >= 0
    safe_ids = source_ids.clamp_min(0)
    source_poses = scene.poses[safe_ids]
    source_intrinsics = scene.intrinsics[safe_ids]
    source_radial_k = scene.radial_distortion[safe_ids]
    relative = surface_points[:, None, :] - source_poses[:, :, :3, 3]
    camera = torch.einsum("bvji,bvj->bvi", source_poses[:, :, :3, :3], relative)
    z = camera[..., 2]
    safe_z = torch.where(z < -1.0e-6, z, -torch.ones_like(z))
    normalized_xy = torch.stack(
        [-camera[..., 0] / safe_z, camera[..., 1] / safe_z], dim=-1
    )
    distorted_xy = distort_simple_radial(normalized_xy, source_radial_k)
    u = source_intrinsics[..., 0, 2] + source_intrinsics[..., 0, 0] * distorted_xy[..., 0]
    v = source_intrinsics[..., 1, 2] + source_intrinsics[..., 1, 1] * distorted_xy[..., 1]
    height, width = scene.images.shape[1:3]
    valid = (
        source_valid
        & (z < -1.0e-6)
        & (u >= 0)
        & (u <= width - 1)
        & (v >= 0)
        & (v <= height - 1)
    )
    sampled = _bilinear_pixels(scene.images, safe_ids, u, v)
    source_mean = scene.image_mean[safe_ids]
    source_std = scene.image_std[safe_ids]
    reference_mean = scene.image_mean[reference_ids][:, None, :]
    reference_std = scene.image_std[reference_ids][:, None, :]
    sampled_normalized = (sampled - source_mean) / source_std
    reference_normalized = (reference_rgb[:, None, :] - reference_mean) / reference_std
    residual = (sampled_normalized - reference_normalized).abs().mean(dim=-1)
    scale = mad_scale(residual.detach(), mask=valid)
    cutoff = max(float(tukey_multiplier), 1.0e-3) * scale
    robust = tukey_biweight(residual, cutoff) / (cutoff.square() / 6.0 + 1.0e-8)
    robust = torch.where(valid, robust, torch.zeros_like(robust))
    counts = valid.sum(dim=-1)
    uncertainty = robust.sum(dim=-1) / counts.clamp_min(1)
    return uncertainty.clamp(0.0, 1.0), counts > 0


class DecoupledTeacher:
    """Cheap, stopped-gradient teacher using surface disagreement and ray entropy."""

    def __init__(self, photometric_weight=0.7):
        self.photometric_weight = float(photometric_weight)

    def __call__(
        self, fine_result, reference_rgb, reference_ids, scene, tukey_multiplier
    ):
        with torch.no_grad():
            photometric, has_sources = multiview_surface_uncertainty(
                fine_result["surface_points"].detach(),
                reference_rgb.detach(),
                reference_ids,
                scene,
                tukey_multiplier,
            )
            termination = ray_termination_uncertainty(fine_result)
            mix = self.photometric_weight
            target = mix * photometric + (1.0 - mix) * termination
            target = torch.where(has_sources, target, termination)
        return target.clamp(0.0, 1.0), {
            "photometric": photometric,
            "termination": termination,
            "has_sources": has_sources,
        }


def neighborhood_uncertainty_regularization(
    field, surface_points, epsilon=1.0e-3
):
    if surface_points.numel() == 0:
        return surface_points.new_zeros(())
    direction = torch.randn_like(surface_points)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-8)
    _, center = field.density_uncertainty(surface_points.detach())
    _, neighbor = field.density_uncertainty((surface_points + epsilon * direction).detach())
    return ((neighbor - center) / max(float(epsilon), 1.0e-8)).square().mean()


__all__ = [
    "DecoupledTeacher",
    "multiview_surface_uncertainty",
    "neighborhood_uncertainty_regularization",
    "ray_termination_uncertainty",
]
