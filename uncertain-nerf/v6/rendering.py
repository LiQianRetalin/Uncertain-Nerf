import torch
from torch.nn import functional as F


def ray_aabb_intersection(rays_o, rays_d, aabb, near=None, far=None):
    sign = torch.where(rays_d >= 0, torch.ones_like(rays_d), -torch.ones_like(rays_d))
    safe_d = torch.where(rays_d.abs() > 1.0e-9, rays_d, sign * 1.0e-9)
    t0 = (aabb[0] - rays_o) / safe_d
    t1 = (aabb[1] - rays_o) / safe_d
    t_min = torch.minimum(t0, t1).amax(dim=-1).clamp_min(0.0)
    t_max = torch.maximum(t0, t1).amin(dim=-1)
    if near is not None:
        t_min = torch.maximum(t_min, torch.as_tensor(near, device=t_min.device, dtype=t_min.dtype))
    if far is not None:
        t_max = torch.minimum(t_max, torch.as_tensor(far, device=t_max.device, dtype=t_max.dtype))
    valid = t_max > t_min + 1.0e-6
    return t_min, t_max, valid


def _interpolate_depth(near, far, steps, sampling_space):
    if sampling_space == "linear":
        return near[:, None] * (1.0 - steps) + far[:, None] * steps
    if sampling_space == "disparity":
        safe_near = near.clamp_min(1.0e-4)
        safe_far = far.clamp_min(safe_near + 1.0e-4)
        return 1.0 / (
            (1.0 - steps) / safe_near[:, None] + steps / safe_far[:, None]
        )
    raise ValueError("sampling_space must be 'linear' or 'disparity'")


def sample_stratified(near, far, n_samples, perturb, sampling_space="disparity"):
    edges_t = torch.linspace(0.0, 1.0, n_samples + 1, device=near.device, dtype=near.dtype)
    edges = _interpolate_depth(near, far, edges_t, sampling_space)
    lower, upper = edges[:, :-1], edges[:, 1:]
    if perturb:
        t = torch.rand_like(lower)
    else:
        t = torch.full_like(lower, 0.5)
    return lower + (upper - lower) * t


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


def finite_intervals(z_vals, near, far):
    """Voronoi intervals bounded by the actual ray near/far planes."""
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    edges = torch.cat([near[:, None], mids, far[:, None]], dim=-1)
    return (edges[..., 1:] - edges[..., :-1]).clamp_min(0.0)


def volume_integrate(raw, z_vals, rays_d, background, near, far, density_activation=None):
    rgb = torch.sigmoid(raw[..., :3])
    if density_activation is None:
        sigma = F.softplus(raw[..., 3] - 1.0)
    else:
        sigma = density_activation(raw[..., 3])
    point_uncertainty = torch.sigmoid(raw[..., 4])
    ray_norm = torch.linalg.vector_norm(rays_d, dim=-1, keepdim=True)
    distances = finite_intervals(z_vals, near, far) * ray_norm
    metric_z = z_vals * ray_norm
    alpha = 1.0 - torch.exp(-sigma * distances)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1.0e-10], dim=-1),
        dim=-1,
    )[..., :-1]
    weights = transmittance * alpha
    acc = weights.sum(dim=-1)
    terminal = (1.0 - acc).clamp(0.0, 1.0)
    rgb_map = (weights[..., None] * rgb).sum(dim=-2) + terminal[..., None] * background
    normalized_weights = weights / (acc[..., None] + 1.0e-8)
    depth = (normalized_weights * z_vals).sum(dim=-1)
    depth = torch.where(acc > 1.0e-6, depth, torch.zeros_like(depth))
    uncertainty = (normalized_weights * point_uncertainty).sum(dim=-1)
    surface_points = None
    return {
        "rgb": rgb_map,
        "depth": depth,
        "acc": acc,
        "uncertainty": uncertainty,
        "weights": weights,
        "point_uncertainty": point_uncertainty,
        "terminal_transmittance": terminal,
        "alpha": alpha,
        "deltas": distances,
        "metric_z": metric_z,
        "surface_points": surface_points,
    }


def query_field(field, points, rays_d, appearance=None, chunk=65536):
    directions = rays_d[:, None, :].expand_as(points)
    if appearance is None:
        appearance = points.new_zeros(points.shape[0], field.appearance_dim)
    expanded_appearance = appearance[:, None, :].expand(*points.shape[:-1], appearance.shape[-1])
    flat_points = points.reshape(-1, 3)
    flat_directions = directions.reshape(-1, 3)
    flat_appearance = expanded_appearance.reshape(-1, appearance.shape[-1])
    outputs = []
    for start in range(0, flat_points.shape[0], chunk):
        end = start + chunk
        outputs.append(
            field(flat_points[start:end], flat_directions[start:end], flat_appearance[start:end])
        )
    return torch.cat(outputs, dim=0).reshape(*points.shape[:-1], 5)


def render_fixed_samples(
    field, points, z_vals, rays_d, background, near, far, appearance=None, chunk=65536
):
    raw = query_field(field, points, rays_d, appearance=appearance, chunk=chunk)
    result = volume_integrate(
        raw, z_vals, rays_d, background, near, far, density_activation=field.activate_density
    )
    acc = result["acc"]
    result["surface_points"] = (
        result["weights"][..., None] * points
    ).sum(dim=-2) / (acc[..., None] + 1.0e-8)
    result.update({"raw": raw, "points": points, "z_vals": z_vals})
    return result


def distortion_loss(result):
    """O(N) form of mip-NeRF 360's inter- and intra-sample distortion."""
    weights = result["weights"]
    z_vals = result["metric_z"]
    prefix_w = torch.cumsum(weights, dim=-1) - weights
    prefix_wz = torch.cumsum(weights * z_vals, dim=-1) - weights * z_vals
    inter = 2.0 * (weights * (z_vals * prefix_w - prefix_wz)).sum(dim=-1)
    intra = (weights.square() * result["deltas"] / 3.0).sum(dim=-1)
    return (inter + intra).mean()


def render_rays(
    field,
    rays_o,
    rays_d,
    near,
    far,
    background,
    appearance=None,
    n_samples=64,
    n_importance=64,
    perturb=True,
    chunk=65536,
    sampling_space="disparity",
):
    near_box, far_box, valid = ray_aabb_intersection(rays_o, rays_d, field.aabb, near, far)
    safe_far = torch.where(valid, far_box, near_box + 1.0e-4)
    z_coarse = sample_stratified(
        near_box, safe_far, n_samples, perturb, sampling_space=sampling_space
    )
    points_coarse = rays_o[:, None, :] + rays_d[:, None, :] * z_coarse[..., None]
    coarse = render_fixed_samples(
        field, points_coarse, z_coarse, rays_d, background, near_box, safe_far,
        appearance=appearance, chunk=chunk,
    )
    if n_importance > 0:
        mids = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
        z_fine = sample_pdf(
            mids, coarse["weights"][..., 1:-1].detach(), n_importance,
            deterministic=not perturb,
        )
        z_all = torch.sort(torch.cat([z_coarse, z_fine.detach()], dim=-1), dim=-1).values
        points_fine = rays_o[:, None, :] + rays_d[:, None, :] * z_all[..., None]
        fine = render_fixed_samples(
            field, points_fine, z_all, rays_d, background, near_box, safe_far,
            appearance=appearance, chunk=chunk,
        )
    else:
        fine = coarse
    invalid = ~valid
    if invalid.any():
        for result in (coarse, fine):
            result["rgb"] = torch.where(invalid[:, None], background, result["rgb"])
            for key in ("depth", "acc"):
                result[key] = torch.where(invalid, torch.zeros_like(result[key]), result[key])
            result["uncertainty"] = torch.where(
                invalid, torch.ones_like(result["uncertainty"]), result["uncertainty"]
            )
            result["weights"] = torch.where(
                invalid[:, None], torch.zeros_like(result["weights"]), result["weights"]
            )
    return {"coarse": coarse, "fine": fine, "valid": valid}


__all__ = [
    "distortion_loss",
    "finite_intervals",
    "render_fixed_samples",
    "render_rays",
    "sample_pdf",
    "sample_stratified",
    "volume_integrate",
]
