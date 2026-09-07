"""Prospective alpha/support marginals and clone-slot topology for RU-PART."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import Tensor


RHO0 = 0.5
N_MAX = 2_312_002
ALPHA_CUTOFF = 1.0 / 255.0
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GrowMasks:
    clone: Tensor
    split: Tensor
    average_grad2d: Tensor


@dataclass
class SparseFootprint:
    candidate_id: int
    y0: int
    y1: int
    x0: int
    x1: int
    gaussian: Tensor
    initial_harm: float = 0.0
    initial_benefit: float = 0.0
    alpha_mass: float = 0.0

    def overlaps(self, other: "SparseFootprint") -> bool:
        return self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1


@dataclass(frozen=True)
class Replacement:
    birth_id: int
    clone_id: int
    benefit: float
    harm: float
    clone_benefit: float
    clone_harm: float
    alpha_mass: float


def compute_default_grow_masks(params: Mapping[str, Tensor], state: Mapping[str, Any], strategy: Any, step: int) -> GrowMasks:
    """Pure reproduction of gsplat 1.5.3 ``DefaultStrategy._grow_gs`` proposals."""

    count = state["count"]
    grad2d = state["grad2d"]
    if not isinstance(count, Tensor) or not isinstance(grad2d, Tensor):
        raise ValueError("strategy statistics are not initialized")
    grads = grad2d / count.clamp_min(1)
    is_grad_high = grads > float(strategy.grow_grad2d)
    is_small = torch.exp(params["scales"]).max(dim=-1).values <= float(strategy.grow_scale3d) * float(state["scene_scale"])
    clone = is_grad_high & is_small
    split = is_grad_high & ~is_small
    if step < int(strategy.refine_scale2d_stop_iter):
        split |= state["radii"] > float(strategy.grow_scale2d)
    return GrowMasks(clone=clone, split=split, average_grad2d=grads)


def alpha_increment(alpha: Tensor, opacity: float | Tensor, gaussian: Tensor) -> Tensor:
    return (1.0 - alpha) * torch.as_tensor(opacity, dtype=alpha.dtype, device=alpha.device) * gaussian


def support_increment(support: Tensor, gaussian: Tensor, rho0: float = RHO0) -> Tensor:
    return (1.0 - support) * float(rho0) * gaussian


def benefit_harm(support: Tensor, static_evidence: Tensor, footprint: SparseFootprint, *, fixed_harm: bool = True) -> tuple[Tensor, Tensor]:
    region_f = support[footprint.y0:footprint.y1, footprint.x0:footprint.x1]
    region_s = static_evidence[footprint.y0:footprint.y1, footprint.x0:footprint.x1]
    delta = support_increment(region_f, footprint.gaussian)
    normalizer = float(support.numel())
    benefit = (region_s * delta).sum() / normalizer
    if fixed_harm:
        harm = delta.new_tensor(footprint.initial_harm)
    else:
        harm = ((1.0 - region_s) * delta).sum() / normalizer
    return benefit, harm


def apply_support(support: Tensor, footprint: SparseFootprint) -> None:
    region = support[footprint.y0:footprint.y1, footprint.x0:footprint.x1]
    region.add_(support_increment(region, footprint.gaussian))


def sparse_footprints_from_projection(
    candidate_ids: Tensor,
    means2d: Tensor,
    conics: Tensor,
    radii: Tensor,
    *,
    height: int = 36,
    width: int = 36,
    truncation_opacity: float = RHO0,
) -> list[SparseFootprint]:
    """Evaluate anisotropic EWA footprints only inside gsplat's projected bounds."""

    if means2d.ndim != 2 or means2d.shape[-1] != 2 or conics.shape != (len(means2d), 3):
        raise ValueError("projection tensors must be [N,2], [N,3]")
    if radii.shape not in {(len(means2d),), (len(means2d), 2)}:
        raise ValueError("radii must be [N] or [N,2]")
    if radii.ndim == 2:
        radii_xy = radii
    else:
        radii_xy = radii[:, None].expand(-1, 2)
    output: list[SparseFootprint] = []
    for index in range(len(means2d)):
        rx, ry = (int(v) for v in radii_xy[index].tolist())
        if rx <= 0 or ry <= 0 or not torch.isfinite(means2d[index]).all() or not torch.isfinite(conics[index]).all():
            continue
        mx, my = (float(v) for v in means2d[index].tolist())
        x0, x1 = max(0, math.floor(mx - rx)), min(width, math.ceil(mx + rx) + 1)
        y0, y1 = max(0, math.floor(my - ry)), min(height, math.ceil(my + ry) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys = torch.arange(y0, y1, device=means2d.device, dtype=means2d.dtype) + 0.5
        xs = torch.arange(x0, x1, device=means2d.device, dtype=means2d.dtype) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx, dy = xx - means2d[index, 0], yy - means2d[index, 1]
        a, b, c = conics[index]
        sigma = 0.5 * (a * dx.square() + c * dy.square()) + b * dx * dy
        raw = torch.exp(-sigma).clamp(0.0, 1.0)
        # gsplat ignores per-Gaussian alpha below 1/255 and caps alpha at 0.999.
        alpha = (float(truncation_opacity) * raw).clamp_max(0.999)
        gaussian = torch.where(alpha >= ALPHA_CUTOFF, alpha / float(truncation_opacity), torch.zeros_like(raw))
        if torch.any(gaussian > 0):
            output.append(SparseFootprint(int(candidate_ids[index]), y0, y1, x0, x1, gaussian))
    return output


def initialize_footprint_scores(
    footprints: Sequence[SparseFootprint],
    support: Tensor,
    alpha: Tensor,
    static_evidence: Tensor,
    *,
    initial_opacity: float,
) -> None:
    for footprint in footprints:
        benefit, harm = benefit_harm(support, static_evidence, footprint, fixed_harm=False)
        footprint.initial_benefit = float(benefit)
        footprint.initial_harm = float(harm)
        region_a = alpha[footprint.y0:footprint.y1, footprint.x0:footprint.x1]
        planned = float(initial_opacity) * footprint.gaussian
        planned = torch.where(planned >= ALPHA_CUTOFF, planned, torch.zeros_like(planned))
        footprint.alpha_mass = float(((1.0 - region_a) * planned).sum() / alpha.numel())


def exhaustive_acceptance_order(footprints: Sequence[SparseFootprint], support: Tensor, static_evidence: Tensor, limit: int) -> list[int]:
    current = support.clone()
    remaining = {item.candidate_id: item for item in footprints}
    accepted: list[int] = []
    while remaining and len(accepted) < limit:
        scored = []
        for candidate_id, item in remaining.items():
            benefit, harm = benefit_harm(current, static_evidence, item)
            utility = float(benefit - harm)
            scored.append((-utility, candidate_id, float(benefit), float(harm)))
        _, candidate_id, benefit, harm = min(scored)
        if benefit <= 0 or benefit <= harm:
            break
        item = remaining.pop(candidate_id)
        accepted.append(candidate_id)
        apply_support(current, item)
    return accepted


def celf_acceptance_order(footprints: Sequence[SparseFootprint], support: Tensor, static_evidence: Tensor, limit: int) -> list[int]:
    """CELF selection with deterministic candidate-id tie breaking."""

    current = support.clone()
    items = {item.candidate_id: item for item in footprints}
    heap: list[tuple[float, int, int, float, float]] = []
    for item in footprints:
        benefit, harm = benefit_harm(current, static_evidence, item)
        heapq.heappush(heap, (-float(benefit - harm), item.candidate_id, 0, float(benefit), float(harm)))
    accepted: list[int] = []
    revision = 0
    while heap and len(accepted) < limit:
        _, candidate_id, evaluated_revision, benefit, harm = heapq.heappop(heap)
        item = items[candidate_id]
        if evaluated_revision != revision:
            b, h = benefit_harm(current, static_evidence, item)
            heapq.heappush(heap, (-float(b - h), candidate_id, revision, float(b), float(h)))
            continue
        if benefit <= 0 or benefit <= harm:
            break
        accepted.append(candidate_id)
        apply_support(current, item)
        revision += 1
    return accepted


def pair_births_with_clones(
    birth_footprints: Sequence[SparseFootprint],
    clone_footprints: Sequence[SparseFootprint],
    support: Tensor,
    static_evidence: Tensor,
    *,
    limit: int,
    clone_tiebreak: Mapping[int, float] | None = None,
) -> tuple[list[Replacement], Tensor]:
    """CELF-select births and replace only Pareto-dominated clone slots."""

    current = support.clone()
    births = {item.candidate_id: item for item in birth_footprints}
    clones = {item.candidate_id: item for item in clone_footprints}
    clone_scores = {
        item.candidate_id: tuple(float(v) for v in benefit_harm(current, static_evidence, item))
        for item in clone_footprints
    }
    heap: list[tuple[float, int, int, float, float]] = []
    for item in birth_footprints:
        benefit, harm = benefit_harm(current, static_evidence, item)
        heapq.heappush(heap, (-float(benefit - harm), item.candidate_id, 0, float(benefit), float(harm)))
    replacements: list[Replacement] = []
    revision = 0
    while heap and clones and len(replacements) < limit:
        _, birth_id, evaluated_revision, birth_b, birth_h = heapq.heappop(heap)
        birth = births.get(birth_id)
        if birth is None:
            continue
        if evaluated_revision != revision:
            benefit, harm = benefit_harm(current, static_evidence, birth)
            heapq.heappush(heap, (-float(benefit - harm), birth_id, revision, float(benefit), float(harm)))
            continue
        if birth_b <= 0 or birth_b <= birth_h or birth.alpha_mass <= 0:
            births.pop(birth_id)
            continue
        eligible = []
        for clone_id, clone in clones.items():
            clone_b, clone_h = clone_scores[clone_id]
            if birth_b > float(clone_b) and birth_h <= float(clone_h):
                tie = clone_tiebreak.get(clone_id, 0.0) if clone_tiebreak else 0.0
                eligible.append((float(clone_b), float(tie), clone.candidate_id, float(clone_h)))
        if not eligible:
            births.pop(birth_id)
            continue
        clone_b, _, clone_id, clone_h = min(eligible)
        replacements.append(Replacement(birth_id, clone_id, birth_b, birth_h, clone_b, clone_h, birth.alpha_mass))
        apply_support(current, birth)
        births.pop(birth_id)
        clones.pop(clone_id)
        clone_scores.pop(clone_id)
        for remaining_id, clone in clones.items():
            if birth.overlaps(clone):
                clone_scores[remaining_id] = tuple(
                    float(v) for v in benefit_harm(current, static_evidence, clone)
                )
        revision += 1
    return replacements, current


def project_gaussians(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    viewmat: Tensor,
    K: Tensor,
    *,
    width: int = 36,
    height: int = 36,
    opacity_for_radius: float = RHO0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Use gsplat's exact 3D->2D covariance convention without rasterizing pixels."""

    from gsplat.cuda._wrapper import fully_fused_projection

    opacities = means.new_full((len(means),), float(opacity_for_radius))
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, viewmat[None], K[None], width, height,
        eps2d=0.3, packed=False, near_plane=0.01, far_plane=1.0e10,
        radius_clip=0.0, sparse_grad=False, calc_compensations=False,
        camera_model="pinhole", opacities=opacities,
    )
    return radii[0], means2d[0], depths[0], conics[0]


def projection_jacobian(world_xyz: np.ndarray, camera_to_world: np.ndarray, K_grid: np.ndarray) -> np.ndarray:
    world_to_cam = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    R, t = world_to_cam[:3, :3], world_to_cam[:3, 3]
    camera = R @ np.asarray(world_xyz, dtype=np.float64) + t
    x, y, z = camera
    if z <= 0 or not np.isfinite(camera).all():
        raise ValueError("candidate is behind a track camera")
    fx, fy = float(K_grid[0, 0]), float(K_grid[1, 1])
    local = np.array([[fx / z, 0.0, -fx * x / (z * z)], [0.0, fy / z, -fy * y / (z * z)]])
    return local @ R


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Stable rotation-matrix to gsplat ``wxyz`` quaternion conversion."""

    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        q = np.array([0.25 * scale, (m[2, 1] - m[1, 2]) / scale,
                      (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / scale, 0.25 * scale,
                      (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale])
    elif m[1, 1] > m[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale,
                      0.25 * scale, (m[1, 2] + m[2, 1]) / scale])
    else:
        scale = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale,
                      (m[1, 2] + m[2, 1]) / scale, 0.25 * scale])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def initialize_birth_geometry(
    world_xyz: np.ndarray,
    camera_to_worlds: Sequence[np.ndarray],
    grid_intrinsics: Sequence[np.ndarray],
    *,
    target_radius_patches: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-information covariance scaled to median 0.5-patch 1-sigma radius."""

    jacobians = [projection_jacobian(world_xyz, c2w, K) for c2w, K in zip(camera_to_worlds, grid_intrinsics)]
    information = sum(J.T @ J for J in jacobians)
    if np.linalg.matrix_rank(information) < 3:
        raise ValueError("birth inverse-information matrix is rank deficient")
    covariance = np.linalg.inv(information)
    radii = []
    for J in jacobians:
        projected = J @ covariance @ J.T
        radii.append(math.sqrt(max(float(np.linalg.eigvalsh(projected).max()), 0.0)))
    median_radius = float(np.median(radii))
    if not np.isfinite(median_radius) or median_radius <= 0:
        raise ValueError("birth projected covariance is invalid")
    covariance *= (target_radius_patches / median_radius) ** 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0) or not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise ValueError("birth covariance eigendecomposition is invalid")
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    scales = np.sqrt(eigenvalues)
    quaternion = rotation_matrix_to_wxyz(eigenvectors)
    return np.log(scales).astype(np.float32), quaternion.astype(np.float32)


def rgb_to_sh0(rgb: Tensor) -> Tensor:
    return (rgb - 0.5) / SH_C0


@torch.no_grad()
def append_birth_rows(
    params: MutableMapping[str, Tensor],
    optimizers: Mapping[str, torch.optim.Optimizer],
    state: MutableMapping[str, Any],
    rows: Mapping[str, Tensor],
) -> None:
    """Append births with zero optimizer and strategy state, matching gsplat child semantics."""

    count = len(rows["means"])
    if count == 0:
        return
    old_count = len(params["means"])
    for name in list(params):
        old = params[name]
        addition = rows[name].to(device=old.device, dtype=old.dtype)
        if addition.shape[1:] != old.shape[1:]:
            raise ValueError(f"birth row shape mismatch for {name}")
        new = torch.nn.Parameter(torch.cat((old, addition)), requires_grad=old.requires_grad)
        params[name] = new
        if name not in optimizers:
            if old.requires_grad:
                raise ValueError(f"trainable parameter {name} has no optimizer")
            continue
        optimizer = optimizers[name]
        for group in optimizer.param_groups:
            if len(group["params"]) != 1 or group["params"][0] is not old:
                raise ValueError("RU-PART supports one parameter per optimizer group")
            old_state = optimizer.state.pop(old, {})
            for key, value in list(old_state.items()):
                if key != "step" and isinstance(value, Tensor):
                    old_state[key] = torch.cat((value, value.new_zeros((count, *value.shape[1:]))))
            group["params"] = [new]
            optimizer.state[new] = old_state
    for key, value in list(state.items()):
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == old_count:
            state[key] = torch.cat((value, value.new_zeros((count, *value.shape[1:]))))


def validate_parameter_state_shapes(params: Mapping[str, Tensor], optimizers: Mapping[str, torch.optim.Optimizer], state: Mapping[str, Any]) -> None:
    count = len(params["means"])
    if any(len(value) != count for value in params.values()):
        raise RuntimeError("Gaussian parameter first dimensions diverged")
    for name, optimizer in optimizers.items():
        parameter = params[name]
        for value in optimizer.state[parameter].values():
            if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] not in (1, count):
                raise RuntimeError(f"optimizer state first dimension diverged for {name}")
    for key, value in state.items():
        if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] != count:
            raise RuntimeError(f"strategy state first dimension diverged for {key}")
