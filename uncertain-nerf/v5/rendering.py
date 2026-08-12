import torch
from torch.nn import functional as F


def ray_aabb_intersection(rays_o, rays_d, aabb, near=None, far=None):
    inv_d = torch.where(rays_d.abs() > 1.0e-9, 1.0 / rays_d, torch.full_like(rays_d, 1.0e9))
    t0 = (aabb[0] - rays_o) * inv_d
    t1 = (aabb[1] - rays_o) * inv_d
    t_min = torch.minimum(t0, t1).amax(dim=-1)
    t_max = torch.maximum(t0, t1).amin(dim=-1)
    if near is not None:
        t_min = torch.maximum(t_min, torch.as_tensor(near, device=t_min.device, dtype=t_min.dtype))
    if far is not None:
        t_max = torch.minimum(t_max, torch.as_tensor(far, device=t_max.device, dtype=t_max.dtype))
    valid = t_max > t_min.clamp_min(0.0)
    return t_min.clamp_min(0.0), t_max, valid


def sample_stratified(near, far, n_samples, perturb):
    steps = torch.linspace(0.0, 1.0, n_samples, device=near.device, dtype=near.dtype)
    z_vals = near[:, None] * (1.0 - steps) + far[:, None] * steps
    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
    return z_vals


def sample_pdf(bins, weights, n_samples, deterministic=False):
    weights = weights + 1.0e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    if deterministic:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device, dtype=bins.dtype)
        u = u.expand(cdf.shape[0], n_samples)
    else:
        u = torch.rand(cdf.shape[0], n_samples, device=bins.device, dtype=bins.dtype)
    indices = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = (indices - 1).clamp_min(0)
    above = indices.clamp_max(cdf.shape[-1] - 1)
    gather = torch.stack([below, above], dim=-1)
    cdf_g = torch.gather(cdf[:, None, :].expand(-1, n_samples, -1), 2, gather)
    bins_g = torch.gather(bins[:, None, :].expand(-1, n_samples, -1), 2, gather)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1.0e-5, torch.ones_like(denom), denom)
    return bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])


def reliability(uncertainty, strength=4.0):
    return 1.0 / (1.0 + strength * uncertainty.square())


def volume_integrate(raw, z_vals, rays_d, background, reliability_strength=4.0):
    rgb = torch.sigmoid(raw[..., :3])
    sigma = F.relu(raw[..., 3])
    uncertainty = torch.sigmoid(raw[..., 4])
    distances = z_vals[..., 1:] - z_vals[..., :-1]
    distances = torch.cat([distances, torch.full_like(distances[..., :1], 1.0e10)], dim=-1)
    distances = distances * torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    effective_density = reliability(uncertainty, reliability_strength) * sigma
    alpha = 1.0 - torch.exp(-effective_density * distances)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1.0e-10], dim=-1),
        dim=-1,
    )[..., :-1]
    weights = transmittance * alpha
    terminal = torch.prod(1.0 - alpha + 1.0e-10, dim=-1)
    rgb_map = (weights[..., None] * rgb).sum(dim=-2) + terminal[..., None] * background
    depth_map = (weights * z_vals).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    uncertainty_map = (weights * uncertainty).sum(dim=-1) / (acc_map + 1.0e-8)
    return {
        "rgb": rgb_map,
        "depth": depth_map,
        "acc": acc_map,
        "uncertainty": uncertainty_map,
        "weights": weights,
        "point_uncertainty": uncertainty,
        "terminal_transmittance": terminal,
    }


def query_field(field, points, rays_d, chunk=65536):
    directions = rays_d[:, None, :].expand_as(points)
    flat_points = points.reshape(-1, 3)
    flat_directions = directions.reshape(-1, 3)
    outputs = []
    for start in range(0, flat_points.shape[0], chunk):
        outputs.append(field(flat_points[start : start + chunk], flat_directions[start : start + chunk]))
    return torch.cat(outputs, dim=0).reshape(*points.shape[:-1], 5)


def render_fixed_samples(field, points, z_vals, rays_d, background, chunk=65536, strength=4.0):
    raw = query_field(field, points, rays_d, chunk)
    result = volume_integrate(raw, z_vals, rays_d, background, strength)
    result.update({"raw": raw, "points": points, "z_vals": z_vals})
    return result


def render_rays(
    coarse,
    fine,
    rays_o,
    rays_d,
    near,
    far,
    background,
    n_samples=64,
    n_importance=64,
    perturb=True,
    chunk=65536,
    reliability_strength=4.0,
):
    near_box, far_box, valid = ray_aabb_intersection(rays_o, rays_d, coarse.aabb, near, far)
    safe_far = torch.where(valid, far_box, near_box + 1.0e-4)
    z_coarse = sample_stratified(near_box, safe_far, n_samples, perturb)
    points_coarse = rays_o[:, None, :] + rays_d[:, None, :] * z_coarse[..., None]
    coarse_result = render_fixed_samples(
        coarse, points_coarse, z_coarse, rays_d, background, chunk, reliability_strength
    )
    if fine is None or n_importance <= 0:
        fine_result = coarse_result
    else:
        mids = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
        z_fine = sample_pdf(
            mids,
            coarse_result["weights"][..., 1:-1].detach(),
            n_importance,
            deterministic=not perturb,
        )
        z_all = torch.sort(torch.cat([z_coarse, z_fine.detach()], dim=-1), dim=-1).values
        points_fine = rays_o[:, None, :] + rays_d[:, None, :] * z_all[..., None]
        fine_result = render_fixed_samples(
            fine, points_fine, z_all, rays_d, background, chunk, reliability_strength
        )
    invalid = ~valid
    if invalid.any():
        for result in (coarse_result, fine_result):
            result["rgb"] = torch.where(invalid[:, None], background, result["rgb"])
            for key in ("depth", "acc", "uncertainty"):
                result[key] = torch.where(invalid, torch.zeros_like(result[key]), result[key])
    return {"coarse": coarse_result, "fine": fine_result, "valid": valid}
