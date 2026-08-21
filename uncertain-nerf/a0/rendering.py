"""Original NeRF ray construction, NDC transform and hierarchical renderer."""

import torch
from torch.nn import functional as F


def get_rays(height, width, focal, pose):
    device, dtype = pose.device, pose.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing="ij")
    directions = torch.stack([
        (xs - 0.5 * width) / focal,
        -(ys - 0.5 * height) / focal,
        -torch.ones_like(xs)], dim=-1)
    rays_d = directions @ pose[:3, :3].transpose(0, 1)
    rays_o = pose[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


def pixels_to_rays(poses, image_ids, ys, xs, height, width, focal):
    dtype = poses.dtype
    directions = torch.stack([
        (xs.to(dtype) - 0.5 * width) / focal,
        -(ys.to(dtype) - 0.5 * height) / focal,
        -torch.ones_like(xs, dtype=dtype)], dim=-1)
    rotations = poses[image_ids, :3, :3]
    rays_d = torch.bmm(rotations, directions[..., None]).squeeze(-1)
    rays_o = poses[image_ids, :3, 3]
    return rays_o, rays_d


def ndc_rays(height, width, focal, near, rays_o, rays_d):
    epsilon = torch.finfo(rays_d.dtype).eps
    safe_z = torch.where(
        rays_d[..., 2].abs() > epsilon, rays_d[..., 2],
        torch.full_like(rays_d[..., 2], -epsilon))
    distance = -(float(near) + rays_o[..., 2]) / safe_z
    shifted_o = rays_o + distance[..., None] * rays_d
    oz = torch.where(
        shifted_o[..., 2].abs() > epsilon, shifted_o[..., 2],
        torch.full_like(shifted_o[..., 2], -epsilon))
    scale_x = -1.0 / (width / (2.0 * focal))
    scale_y = -1.0 / (height / (2.0 * focal))
    origin = torch.stack([
        scale_x * shifted_o[..., 0] / oz,
        scale_y * shifted_o[..., 1] / oz,
        1.0 + 2.0 * float(near) / oz], dim=-1)
    direction = torch.stack([
        scale_x * (rays_d[..., 0] / safe_z - shifted_o[..., 0] / oz),
        scale_y * (rays_d[..., 1] / safe_z - shifted_o[..., 1] / oz),
        -2.0 * float(near) / oz], dim=-1)
    return origin, direction


def sample_pdf(bins, weights, count, deterministic=False):
    weights = weights + 1.0e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[..., :1]),
                     torch.cumsum(pdf, dim=-1)], dim=-1)
    if deterministic:
        samples = torch.linspace(
            0.0, 1.0, int(count), device=bins.device, dtype=bins.dtype)
        samples = samples.expand(*cdf.shape[:-1], int(count))
    else:
        samples = torch.rand(
            *cdf.shape[:-1], int(count), device=bins.device, dtype=bins.dtype)
    indices = torch.searchsorted(cdf.contiguous(), samples.contiguous(), right=True)
    below = (indices - 1).clamp_min(0)
    above = indices.clamp_max(cdf.shape[-1] - 1)
    gather = torch.stack([below, above], dim=-1)
    expanded_cdf = cdf.unsqueeze(-2).expand(*gather.shape[:-1], cdf.shape[-1])
    expanded_bins = bins.unsqueeze(-2).expand(*gather.shape[:-1], bins.shape[-1])
    cdf_values = torch.gather(expanded_cdf, -1, gather)
    bin_values = torch.gather(expanded_bins, -1, gather)
    denominator = cdf_values[..., 1] - cdf_values[..., 0]
    denominator = torch.where(
        denominator < 1.0e-5, torch.ones_like(denominator), denominator)
    fraction = (samples - cdf_values[..., 0]) / denominator
    return bin_values[..., 0] + fraction * (
        bin_values[..., 1] - bin_values[..., 0])


def raw_to_outputs(raw, z_vals, rays_d, raw_noise_std=0.0,
                   white_background=False):
    distances = z_vals[..., 1:] - z_vals[..., :-1]
    distances = torch.cat([
        distances, torch.full_like(distances[..., :1], 1.0e10)], dim=-1)
    distances = distances * torch.linalg.vector_norm(
        rays_d[..., None, :], dim=-1)
    rgb = torch.sigmoid(raw[..., :3])
    noise = (torch.randn_like(raw[..., 3]) * float(raw_noise_std)
             if raw_noise_std > 0 else 0.0)
    alpha = 1.0 - torch.exp(-F.relu(raw[..., 3] + noise) * distances)
    transmittance = torch.cumprod(torch.cat([
        torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1.0e-10], dim=-1),
        dim=-1)[..., :-1]
    weights = alpha * transmittance
    acc = weights.sum(dim=-1)
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    if white_background:
        rgb_map = rgb_map + (1.0 - acc[..., None])
    depth = (weights * z_vals).sum(dim=-1)
    disparity = 1.0 / torch.maximum(
        depth / acc.clamp_min(1.0e-10), torch.full_like(depth, 1.0e-10))
    return {
        "rgb": rgb_map, "depth": depth, "disparity": disparity,
        "acc": acc, "weights": weights, "alpha": alpha,
        "terminal_transmittance": (1.0 - acc).clamp(0.0, 1.0),
    }


def _sample_coarse(near, far, ray_count, count, perturb, device, dtype):
    positions = torch.linspace(0.0, 1.0, int(count), device=device, dtype=dtype)
    z_vals = near * (1.0 - positions) + far * positions
    z_vals = z_vals.expand(ray_count, int(count))
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
    return z_vals


def render_rays(model, rays_o, rays_d, height, width, focal,
                n_samples=64, n_importance=64, perturb=True,
                raw_noise_std=0.0, white_background=False,
                netchunk=65536, use_ndc=True):
    viewdirs = F.normalize(rays_d, dim=-1)
    render_o, render_d = rays_o, rays_d
    near, far = 0.0, 1.0
    if use_ndc:
        render_o, render_d = ndc_rays(
            height, width, focal, 1.0, render_o, render_d)
    else:
        raise ValueError("The fern A0 recovery baseline requires LLFF NDC rays")

    z_coarse = _sample_coarse(
        near, far, len(rays_o), n_samples, perturb,
        rays_o.device, rays_o.dtype)
    points = render_o[:, None] + render_d[:, None] * z_coarse[..., None]
    raw_coarse = model.query(
        model.coarse, points, viewdirs, chunk=netchunk)
    coarse = raw_to_outputs(
        raw_coarse, z_coarse, render_d, raw_noise_std, white_background)
    coarse.update({"raw": raw_coarse, "z_vals": z_coarse})

    if int(n_importance) > 0:
        mids = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
        z_fine = sample_pdf(
            mids, coarse["weights"][..., 1:-1].detach(), n_importance,
            deterministic=not perturb).detach()
        z_all = torch.sort(torch.cat([z_coarse, z_fine], dim=-1), dim=-1).values
        points = render_o[:, None] + render_d[:, None] * z_all[..., None]
        raw_fine = model.query(model.fine, points, viewdirs, chunk=netchunk)
        fine = raw_to_outputs(
            raw_fine, z_all, render_d, raw_noise_std, white_background)
        fine.update({"raw": raw_fine, "z_vals": z_all})
    else:
        fine = coarse
    return {"coarse": coarse, "fine": fine}


__all__ = [
    "get_rays", "ndc_rays", "pixels_to_rays", "raw_to_outputs",
    "render_rays", "sample_pdf",
]
