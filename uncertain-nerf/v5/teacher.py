import torch
from torch import nn
from torch.nn import functional as F

from .camera import distort_simple_radial
from .rendering import render_fixed_samples
from .robust import mad_scale, tukey_biweight


class TeacherEstimator(nn.Module):
    def __init__(self, mc_samples=4, kappa=2.0, variance_beta=1.0e-3):
        super().__init__()
        self.mc_samples = int(mc_samples)
        self.kappa = float(kappa)
        self.variance_beta = float(variance_beta)
        self.mix_logit = nn.Parameter(torch.zeros(()))

    def mc_uncertainty(self, field, fine_result, rays_d, background, query_chunk):
        depths = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                result = render_fixed_samples(
                    field,
                    fine_result["points"],
                    fine_result["z_vals"],
                    rays_d,
                    background,
                    chunk=query_chunk,
                )
                depths.append(result["depth"])
        depths = torch.stack(depths, dim=0)
        variance = depths.var(dim=0, unbiased=False)
        return variance / (variance + self.variance_beta)

    @staticmethod
    def multiview_uncertainty(
        points,
        weights,
        reference_rgb,
        reference_ids,
        scene,
        tukey_multiplier,
    ):
        source_ids = scene.source_views[reference_ids]
        source_valid = source_ids >= 0
        safe_ids = source_ids.clamp_min(0)
        source_images = scene.images[safe_ids][..., :3]
        source_poses = scene.poses[safe_ids]
        source_intrinsics = scene.intrinsics[safe_ids]
        source_radial_k = scene.radial_distortion[safe_ids]
        batch, n_views = safe_ids.shape
        n_points = points.shape[1]
        relative = points[:, None, :, :] - source_poses[:, :, None, :3, 3]
        camera = torch.einsum(
            "bvji,bvmj->bvmi", source_poses[:, :, :3, :3], relative
        )
        z = camera[..., 2]
        fx = source_intrinsics[..., 0, 0, None]
        fy = source_intrinsics[..., 1, 1, None]
        cx = source_intrinsics[..., 0, 2, None]
        cy = source_intrinsics[..., 1, 2, None]
        safe_z = torch.where(z < -1.0e-6, z, -torch.ones_like(z))
        normalized_xy = torch.stack(
            [-camera[..., 0] / safe_z, camera[..., 1] / safe_z], dim=-1
        )
        distorted_xy = distort_simple_radial(normalized_xy, source_radial_k)
        u = cx + fx * distorted_xy[..., 0]
        v = cy + fy * distorted_xy[..., 1]
        height, width = scene.images.shape[1:3]
        valid = (
            source_valid[..., None]
            & (z < -1.0e-6)
            & (u >= 0)
            & (u <= width - 1)
            & (v >= 0)
            & (v <= height - 1)
        )
        grid_x = 2.0 * u / max(width - 1, 1) - 1.0
        grid_y = 2.0 * v / max(height - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(batch * n_views, n_points, 1, 2)
        image_batch = source_images.permute(0, 1, 4, 2, 3).reshape(batch * n_views, 3, height, width)
        sampled = F.grid_sample(
            image_batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(batch, n_views, 3, n_points).permute(0, 1, 3, 2)
        residual = (sampled - reference_rgb[:, None, None, :]).abs().mean(dim=-1)
        scale = mad_scale(residual.detach(), mask=valid)
        cutoff = max(float(tukey_multiplier), 1.0e-3) * scale
        robust = tukey_biweight(residual, cutoff)
        robust = torch.where(valid, robust, torch.zeros_like(robust))
        counts = valid.sum(dim=1)
        point_error = robust.sum(dim=1) / counts.clamp_min(1)
        point_error = point_error / (cutoff.square() / 6.0 + 1.0e-8)
        ray_valid = counts.gt(0).any(dim=-1)
        ray_error = (weights * point_error).sum(dim=-1) / (weights.sum(dim=-1) + 1.0e-8)
        return ray_error.clamp(0.0, 1.0), ray_valid

    def forward(
        self,
        field,
        fine_result,
        rays_d,
        background,
        reference_rgb,
        reference_ids,
        scene,
        tukey_multiplier,
        query_chunk,
    ):
        dropout_component = self.mc_uncertainty(
            field, fine_result, rays_d, background, query_chunk
        ).detach()
        with torch.no_grad():
            photometric_component, has_sources = self.multiview_uncertainty(
                fine_result["points"].detach(),
                fine_result["weights"].detach(),
                reference_rgb.detach(),
                reference_ids,
                scene,
                tukey_multiplier,
            )
        mix = torch.sigmoid(self.mix_logit)
        fused = mix * dropout_component + (1.0 - mix) * photometric_component
        fused = torch.where(has_sources, fused, dropout_component)
        return torch.sigmoid(self.kappa * fused), {
            "dropout": dropout_component,
            "photometric": photometric_component,
            "has_sources": has_sources,
            "mix": mix,
        }


def uncertainty_regularization(field, fine_result, ray_indices, eta_amplitude=1.0e-2, eta_spatial=1.0e-3):
    ray_uncertainty = fine_result["uncertainty"]
    amplitude = ray_uncertainty.square().mean()
    if ray_indices.numel() == 0:
        return eta_amplitude * amplitude, {"amplitude": amplitude, "spatial": amplitude.new_zeros(())}
    points = fine_result["points"][ray_indices].detach().requires_grad_(True)
    _, uncertainty = field.density_uncertainty(points)
    gradients = torch.autograd.grad(
        uncertainty.sum(), points, create_graph=True, retain_graph=True
    )[0]
    weights = fine_result["weights"][ray_indices].detach()
    spatial = (weights * gradients.square().sum(dim=-1)).sum() / (weights.sum() + 1.0e-8)
    return eta_amplitude * amplitude + eta_spatial * spatial, {
        "amplitude": amplitude,
        "spatial": spatial,
    }
