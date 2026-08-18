import torch
from torch.nn import functional as F


def ray_aabb_intersection(rays_o, rays_d, aabb, near=None, far=None):
    sign = torch.where(rays_d >= 0, torch.ones_like(rays_d), -torch.ones_like(rays_d))
    safe_d = torch.where(rays_d.abs() > 1.0e-9, rays_d, sign * 1.0e-9)
    t0, t1 = (aabb[0] - rays_o) / safe_d, (aabb[1] - rays_o) / safe_d
    t_min = torch.minimum(t0, t1).amax(dim=-1).clamp_min(0.0)
    t_max = torch.maximum(t0, t1).amin(dim=-1)
    if near is not None:
        t_min = torch.maximum(t_min, torch.as_tensor(near, device=t_min.device,
                                                      dtype=t_min.dtype))
    if far is not None:
        t_max = torch.minimum(t_max, torch.as_tensor(far, device=t_max.device,
                                                      dtype=t_max.dtype))
    return t_min, t_max, t_max > t_min + 1.0e-6


def _interpolate_depth(near, far, steps, sampling_space):
    if sampling_space == "linear":
        return near[:, None] * (1.0 - steps) + far[:, None] * steps
    if sampling_space == "disparity":
        safe_near = near.clamp_min(1.0e-4)
        safe_far = far.clamp_min(safe_near + 1.0e-4)
        return 1.0 / ((1.0 - steps) / safe_near[:, None] + steps / safe_far[:, None])
    raise ValueError("sampling_space must be 'linear' or 'disparity'")


def sample_stratified(near, far, n_samples, perturb, sampling_space="disparity"):
    edges_t = torch.linspace(0.0, 1.0, n_samples + 1, device=near.device,
                             dtype=near.dtype)
    edges = _interpolate_depth(near, far, edges_t, sampling_space)
    lower, upper = edges[:, :-1], edges[:, 1:]
    t = torch.rand_like(lower) if perturb else torch.full_like(lower, 0.5)
    return lower + (upper - lower) * t


def sample_pdf(bins, weights, n_samples, deterministic=False):
    weights = weights + 1.0e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, dim=-1)], dim=-1)
    if deterministic:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device,
                           dtype=bins.dtype).expand(cdf.shape[0], -1)
    else:
        u = torch.rand(cdf.shape[0], n_samples, device=bins.device, dtype=bins.dtype)
    indices = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below, above = (indices - 1).clamp_min(0), indices.clamp_max(cdf.shape[-1] - 1)
    gather = torch.stack([below, above], dim=-1)
    cdf_g = torch.gather(cdf[:, None].expand(-1, n_samples, -1), 2, gather)
    bins_g = torch.gather(bins[:, None].expand(-1, n_samples, -1), 2, gather)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1.0e-5, torch.ones_like(denom), denom)
    return bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0])


def finite_intervals(z_vals, near, far):
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    edges = torch.cat([near[:, None], mids, far[:, None]], dim=-1)
    return (edges[..., 1:] - edges[..., :-1]).clamp_min(0.0)


def volume_integrate(raw, z_vals, rays_d, background, near, far,
                     density_activation=None):
    rgb = torch.sigmoid(raw[..., :3])
    sigma = (F.softplus(raw[..., 3] - 1.0) if density_activation is None
             else density_activation(raw[..., 3]))
    point_uncertainty = torch.sigmoid(raw[..., 4])
    ray_norm = torch.linalg.vector_norm(rays_d, dim=-1, keepdim=True)
    distances = finite_intervals(z_vals, near, far) * ray_norm
    metric_z = z_vals * ray_norm
    alpha = 1.0 - torch.exp(-sigma * distances)
    transmittance = torch.cumprod(torch.cat([
        torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1.0e-10], dim=-1), dim=-1)[..., :-1]
    weights = transmittance * alpha
    acc = weights.sum(dim=-1)
    terminal = (1.0 - acc).clamp(0.0, 1.0)
    rgb_map = (weights[..., None] * rgb).sum(dim=-2) + terminal[..., None] * background
    normalized_weights = weights / (acc[..., None] + 1.0e-8)
    depth = (normalized_weights * z_vals).sum(dim=-1)
    depth = torch.where(acc > 1.0e-6, depth, torch.zeros_like(depth))
    # The second V7 hard boundary: calibration gradients cannot reach density weights.
    uncertainty_weights = normalized_weights.detach()
    uncertainty = (uncertainty_weights * point_uncertainty).sum(dim=-1)
    return {"rgb": rgb_map, "depth": depth, "acc": acc, "uncertainty": uncertainty,
            "weights": weights, "point_uncertainty": point_uncertainty,
            "terminal_transmittance": terminal, "alpha": alpha, "deltas": distances,
            "metric_z": metric_z, "surface_points": None}


def query_field(field, points, rays_d, appearance=None, footprint=None, chunk=65536,
                occupancy_grid=None):
    directions = rays_d[:, None, :].expand_as(points)
    if appearance is None:
        appearance = points.new_zeros(points.shape[0], field.appearance_dim)
    expanded_appearance = appearance[:, None].expand(
        *points.shape[:-1], appearance.shape[-1])
    if footprint is None:
        footprint = points.new_zeros(points.shape[:-1])
    flat_points = points.reshape(-1, 3)
    flat_directions = directions.reshape(-1, 3)
    flat_appearance = expanded_appearance.reshape(len(flat_points), appearance.shape[-1])
    flat_footprint = footprint.reshape(-1)
    active = (occupancy_grid.occupied(flat_points) if occupancy_grid is not None
              else torch.ones(len(flat_points), dtype=torch.bool, device=points.device))
    # Empty samples have practically zero density and need no HashGrid/MLP evaluation.
    output = points.new_zeros(len(flat_points), 5)
    output[:, 3] = -100.0
    active_indices = torch.where(active)[0]
    for start in range(0, len(active_indices), int(chunk)):
        indices = active_indices[start:start + int(chunk)]
        values = field(flat_points[indices], flat_directions[indices],
                       flat_appearance[indices], flat_footprint[indices])
        # AMP may return fp16/bf16 while the sparse accumulation buffer follows
        # the fp32 ray coordinates. Index assignment requires an exact dtype match.
        output[indices] = values.to(output.dtype)
    return output.reshape(*points.shape[:-1], 5)


def render_fixed_samples(field, points, z_vals, rays_d, background, near, far,
                         appearance=None, ray_radii=None, chunk=65536,
                         occupancy_grid=None):
    if ray_radii is None:
        ray_radii = z_vals.new_zeros(z_vals.shape[0])
    footprint = z_vals * ray_radii[:, None]
    raw = query_field(field, points, rays_d, appearance, footprint, chunk, occupancy_grid)
    result = volume_integrate(raw, z_vals, rays_d, background, near, far,
                              density_activation=field.activate_density)
    acc = result["acc"]
    result["surface_points"] = (result["weights"][..., None] * points).sum(dim=-2) / (
        acc[..., None] + 1.0e-8)
    result.update({"raw": raw, "points": points, "z_vals": z_vals,
                   "footprint": footprint})
    return result


def distortion_loss(result):
    weights, z_vals = result["weights"], result["metric_z"]
    prefix_w = torch.cumsum(weights, dim=-1) - weights
    prefix_wz = torch.cumsum(weights * z_vals, dim=-1) - weights * z_vals
    inter = 2.0 * (weights * (z_vals * prefix_w - prefix_wz)).sum(dim=-1)
    intra = (weights.square() * result["deltas"] / 3.0).sum(dim=-1)
    return (inter + intra).mean()


def render_rays(field, rays_o, rays_d, near, far, background, appearance=None,
                ray_radii=None, n_samples=64, n_importance=64, perturb=True,
                chunk=65536, sampling_space="disparity", occupancy_grid=None):
    near_box, far_box, valid = ray_aabb_intersection(rays_o, rays_d, field.aabb, near, far)
    safe_far = torch.where(valid, far_box, near_box + 1.0e-4)
    z_coarse = sample_stratified(near_box, safe_far, n_samples, perturb, sampling_space)
    points = rays_o[:, None] + rays_d[:, None] * z_coarse[..., None]
    coarse = render_fixed_samples(field, points, z_coarse, rays_d, background, near_box,
                                  safe_far, appearance, ray_radii, chunk, occupancy_grid)
    if n_importance > 0:
        mids = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
        z_fine = sample_pdf(mids, coarse["weights"][..., 1:-1].detach(), n_importance,
                            deterministic=not perturb)
        z_all = torch.sort(torch.cat([z_coarse, z_fine.detach()], dim=-1), dim=-1).values
        points = rays_o[:, None] + rays_d[:, None] * z_all[..., None]
        fine = render_fixed_samples(field, points, z_all, rays_d, background, near_box,
                                    safe_far, appearance, ray_radii, chunk, occupancy_grid)
    else:
        fine = coarse
    invalid = ~valid
    if invalid.any():
        for result in (coarse, fine):
            result["rgb"] = torch.where(invalid[:, None], background, result["rgb"])
            for key in ("depth", "acc"):
                result[key] = torch.where(invalid, torch.zeros_like(result[key]), result[key])
            result["uncertainty"] = torch.where(
                invalid, torch.ones_like(result["uncertainty"]), result["uncertainty"])
            result["weights"] = torch.where(
                invalid[:, None], torch.zeros_like(result["weights"]), result["weights"])
    return {"coarse": coarse, "fine": fine, "valid": valid}


__all__ = ["distortion_loss", "finite_intervals", "render_fixed_samples", "render_rays",
           "sample_pdf", "sample_stratified", "volume_integrate"]
