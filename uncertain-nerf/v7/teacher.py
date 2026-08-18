import math

import torch

from v5.camera import distort_simple_radial
from v5.data import pixels_to_rays
from v5.robust import mad_scale, tukey_biweight

from .data import pixel_ray_radii
from .rendering import render_rays


def ray_termination_uncertainty(fine_result):
    weights, acc = fine_result["weights"].detach(), fine_result["acc"].detach()
    probabilities = weights / (acc[:, None] + 1.0e-8)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum(dim=-1)
    entropy = entropy / max(math.log(weights.shape[-1]), 1.0)
    return (acc * entropy + (1.0 - acc)).clamp(0.0, 1.0)


def _bilinear_pixels(images, image_ids, u, v):
    height, width = images.shape[1:3]
    x0, y0 = u.floor().long().clamp(0, width - 1), v.floor().long().clamp(0, height - 1)
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    wx, wy = (u - x0.to(u.dtype))[..., None], (v - y0.to(v.dtype))[..., None]
    top = images[image_ids, y0, x0, :3] * (1.0 - wx) + images[image_ids, y0, x1, :3] * wx
    bottom = images[image_ids, y1, x0, :3] * (1.0 - wx) + images[image_ids, y1, x1, :3] * wx
    return top * (1.0 - wy) + bottom * wy


def _source_visibility(field, scene, safe_ids, u, v, projected_depth, valid,
                       background, near, far, n_samples, chunk, relative_tolerance,
                       absolute_tolerance, min_acc, occupancy_grid):
    flat_ids, flat_u, flat_v = safe_ids.reshape(-1), u.reshape(-1), v.reshape(-1)
    rays_o, rays_d = pixels_to_rays(scene.poses, scene.intrinsics, flat_ids, flat_v, flat_u,
                                    radial_distortion=scene.radial_distortion)
    radii = pixel_ray_radii(scene.intrinsics, flat_ids)
    rendered = render_rays(field, rays_o, rays_d, near, far,
                           background.expand(len(flat_ids), -1), ray_radii=radii,
                           n_samples=n_samples, n_importance=0, perturb=False, chunk=chunk,
                           occupancy_grid=occupancy_grid)["fine"]
    predicted_depth = rendered["depth"].reshape_as(projected_depth)
    predicted_acc = rendered["acc"].reshape_as(projected_depth)
    tolerance = torch.maximum(projected_depth.abs() * float(relative_tolerance),
                              projected_depth.new_tensor(float(absolute_tolerance)))
    consistent = (predicted_depth - projected_depth).abs() <= tolerance
    return valid & (predicted_acc >= float(min_acc)) & consistent


def multiview_surface_uncertainty(surface_points, reference_rgb, reference_ids, scene,
                                  tukey_multiplier, field, background, near, far,
                                  visibility_samples=32, visibility_relative_tolerance=0.05,
                                  visibility_absolute_tolerance=0.01,
                                  visibility_min_acc=0.1, chunk=65536,
                                  occupancy_grid=None):
    source_ids = scene.source_views[reference_ids]
    source_valid, safe_ids = source_ids >= 0, source_ids.clamp_min(0)
    poses = scene.poses[safe_ids]
    relative = surface_points[:, None] - poses[:, :, :3, 3]
    camera = torch.einsum("bvji,bvj->bvi", poses[:, :, :3, :3], relative)
    z = camera[..., 2]
    safe_z = torch.where(z < -1.0e-6, z, -torch.ones_like(z))
    normalized_xy = torch.stack([-camera[..., 0] / safe_z, camera[..., 1] / safe_z], -1)
    distorted = distort_simple_radial(normalized_xy, scene.radial_distortion[safe_ids])
    intrinsics = scene.intrinsics[safe_ids]
    u = intrinsics[..., 0, 2] + intrinsics[..., 0, 0] * distorted[..., 0]
    v = intrinsics[..., 1, 2] + intrinsics[..., 1, 1] * distorted[..., 1]
    height, width = scene.images.shape[1:3]
    valid = source_valid & (z < -1.0e-6) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    valid = _source_visibility(
        field, scene, safe_ids, u, v, -z, valid, background, near, far,
        visibility_samples, chunk, visibility_relative_tolerance,
        visibility_absolute_tolerance, visibility_min_acc, occupancy_grid)
    sampled = _bilinear_pixels(scene.images, safe_ids, u, v)
    sampled_normalized = (sampled - scene.image_mean[safe_ids]) / scene.image_std[safe_ids]
    reference_normalized = ((reference_rgb[:, None] - scene.image_mean[reference_ids][:, None])
                            / scene.image_std[reference_ids][:, None])
    residual = (sampled_normalized - reference_normalized).abs().mean(dim=-1)
    scale = mad_scale(residual.detach(), mask=valid)
    cutoff = max(float(tukey_multiplier), 1.0e-3) * scale
    robust = tukey_biweight(residual, cutoff) / (cutoff.square() / 6.0 + 1.0e-8)
    robust = torch.where(valid, robust, torch.zeros_like(robust))
    counts = valid.sum(dim=-1)
    return (robust.sum(dim=-1) / counts.clamp_min(1)).clamp(0.0, 1.0), counts > 0


class VisibilityAwareTeacher:
    def __init__(self, photometric_weight=0.7):
        self.photometric_weight = float(photometric_weight)

    def __call__(self, fine_result, reference_rgb, reference_ids, scene,
                 tukey_multiplier, field, background, near, far, **visibility_kwargs):
        with torch.no_grad():
            photometric, has_sources = multiview_surface_uncertainty(
                fine_result["surface_points"].detach(), reference_rgb.detach(), reference_ids,
                scene, tukey_multiplier, field, background, near, far, **visibility_kwargs)
            termination = ray_termination_uncertainty(fine_result)
            target = self.photometric_weight * photometric + (
                1.0 - self.photometric_weight) * termination
            target = torch.where(has_sources, target, termination)
        return target.clamp(0.0, 1.0), {"photometric": photometric,
            "termination": termination, "has_sources": has_sources}


def neighborhood_uncertainty_regularization(field, surface_points, epsilon=1.0e-3):
    if surface_points.numel() == 0:
        return surface_points.new_zeros(())
    direction = torch.nn.functional.normalize(torch.randn_like(surface_points), dim=-1)
    center = field.uncertainty(surface_points.detach())
    neighbor = field.uncertainty((surface_points + float(epsilon) * direction).detach())
    return ((neighbor - center) / max(float(epsilon), 1.0e-8)).square().mean()


__all__ = ["VisibilityAwareTeacher", "multiview_surface_uncertainty",
           "neighborhood_uncertainty_regularization", "ray_termination_uncertainty"]
