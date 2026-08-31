"""Cross-view transient responsibility (CVTR) primitives.

CVTR masks are built once from a frozen B1 checkpoint.  Training only loads the
resulting binary PNGs and applies a constant 0.2/1.0 responsibility map; no
geometry or rasterization from this module runs in the training loop.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


@dataclass(frozen=True)
class CVTRConfig:
    residual_sample_stride: int = 8
    alpha_min: float = 0.5
    residual_scale_multiplier: float = 3.0
    epsilon: float = 1e-6
    neighbor_count: int = 3
    depth_relative_tolerance: float = 0.05
    minimum_valid_neighbors: int = 2
    static_support_threshold: float = 0.5
    patch_size: int = 3
    patch_min_support: float = 5.0 / 9.0
    max_transient_area: float = 0.15
    transient_weight: float = 0.2
    minimum_neff_ratio: float = 0.90

    def __post_init__(self) -> None:
        if self.residual_sample_stride <= 0:
            raise ValueError("residual_sample_stride must be positive")
        if not 0.0 <= self.alpha_min <= 1.0:
            raise ValueError("alpha_min must be in [0, 1]")
        if self.residual_scale_multiplier <= 0.0:
            raise ValueError("residual_scale_multiplier must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.neighbor_count <= 0:
            raise ValueError("neighbor_count must be positive")
        if not 0.0 < self.depth_relative_tolerance <= 1.0:
            raise ValueError("depth_relative_tolerance must be in (0, 1]")
        if not 1 <= self.minimum_valid_neighbors <= self.neighbor_count:
            raise ValueError(
                "minimum_valid_neighbors must be between 1 and neighbor_count"
            )
        if not 0.0 <= self.static_support_threshold <= 1.0:
            raise ValueError("static_support_threshold must be in [0, 1]")
        if self.patch_size != 3:
            raise ValueError("Phase 3 fixes patch_size to 3")
        if not math.isclose(self.patch_min_support, 5.0 / 9.0, abs_tol=1e-12):
            raise ValueError("Phase 3 fixes patch_min_support to 5/9")
        if not 0.0 < self.max_transient_area <= 1.0:
            raise ValueError("max_transient_area must be in (0, 1]")
        if not 0.0 < self.transient_weight <= 1.0:
            raise ValueError("transient_weight must be in (0, 1]")
        if not 0.0 < self.minimum_neff_ratio <= 1.0:
            raise ValueError("minimum_neff_ratio must be in (0, 1]")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ViewEvidence:
    """Frozen per-view evidence needed by cross-view reprojection."""

    image_name: str
    residual: Tensor
    depth: Tensor
    alpha: Tensor
    K: Tensor
    camtoworld: Tensor

    def __post_init__(self) -> None:
        shape = self.residual.shape
        if self.residual.ndim != 2:
            raise ValueError("residual must have shape [H, W]")
        if self.depth.shape != shape or self.alpha.shape != shape:
            raise ValueError("residual, depth, and alpha shapes must match")
        if self.K.shape != (3, 3):
            raise ValueError("K must have shape [3, 3]")
        if self.camtoworld.shape != (4, 4):
            raise ValueError("camtoworld must have shape [4, 4]")


@dataclass(frozen=True)
class CrossViewResult:
    candidate_mask: Tensor
    raw_transient_mask: Tensor
    valid_neighbor_counts: Tensor

    @property
    def candidate_ratio(self) -> float:
        return float(self.candidate_mask.float().mean().item())

    def valid_neighbor_ratio(self, minimum_valid_neighbors: int) -> float:
        candidate_count = int(self.candidate_mask.sum().item())
        if candidate_count == 0:
            return 1.0
        supported = (
            self.valid_neighbor_counts[self.candidate_mask]
            >= minimum_valid_neighbors
        )
        return float(supported.float().mean().item())


CVTR_STAGE_NAMES = (
    "S0_residual_threshold",
    "S1_current_alpha",
    "S2_reprojectable_2plus",
    "S3_neighbor_alpha_2plus",
    "S4_depth_consistent_2plus",
    "S5_static_support",
    "S6_spatial_support",
    "S7_area_cap_final",
)


@dataclass(frozen=True)
class CVTRStageTrace:
    """Detached, read-only evidence from the frozen production CVTR pipeline."""

    stage_masks: tuple[Tensor, ...]
    production_candidate_mask: Tensor
    rgb_residual: Tensor
    current_alpha: Tensor
    inbounds_positive_neighbor_count: Tensor
    neighbor_alpha_valid_count: Tensor
    depth_consistent_neighbor_count: Tensor
    static_support_ratio: Tensor
    spatial_support_score: Tensor
    depth_consistency_error_per_neighbor: Tensor

    def __post_init__(self) -> None:
        if len(self.stage_masks) != len(CVTR_STAGE_NAMES):
            raise ValueError("CVTR trace must contain exactly eight stage masks")
        shape = self.rgb_residual.shape
        if self.rgb_residual.ndim != 2 or self.current_alpha.shape != shape:
            raise ValueError("trace residual and alpha must have matching [H, W] shapes")
        for mask in self.stage_masks:
            if mask.shape != shape or mask.dtype != torch.bool:
                raise ValueError("each CVTR stage mask must be boolean [H, W]")
        for counts in (
            self.inbounds_positive_neighbor_count,
            self.neighbor_alpha_valid_count,
            self.depth_consistent_neighbor_count,
        ):
            if counts.shape != shape:
                raise ValueError("CVTR neighbor-count maps must match the source image")
        if self.static_support_ratio.shape != shape:
            raise ValueError("static-support map must match the source image")
        if self.spatial_support_score.shape != shape:
            raise ValueError("spatial-support map must match the source image")
        if self.depth_consistency_error_per_neighbor.ndim != 3:
            raise ValueError("depth errors must have shape [neighbors, H, W]")
        if self.depth_consistency_error_per_neighbor.shape[1:] != shape:
            raise ValueError("depth-error maps must match the source image")

    @property
    def final_mask(self) -> Tensor:
        return self.stage_masks[-1]

    def named_stage_masks(self) -> dict[str, Tensor]:
        return dict(zip(CVTR_STAGE_NAMES, self.stage_masks))


def _validate_map_pair(residual: Tensor, alpha: Tensor) -> None:
    if residual.ndim != 2 or residual.shape != alpha.shape:
        raise ValueError("residual and alpha must have matching [H, W] shapes")
    if not residual.is_floating_point() or not alpha.is_floating_point():
        raise TypeError("residual and alpha must be floating point")


def scene_residual_threshold(
    residuals: Sequence[Tensor],
    alphas: Sequence[Tensor],
    config: CVTRConfig = CVTRConfig(),
) -> tuple[float, float, float, int]:
    """Return the single scene-level ``(tau, median, scale, sample_count)``."""

    if len(residuals) != len(alphas) or not residuals:
        raise ValueError("residuals and alphas must be non-empty and equally sized")
    samples: list[Tensor] = []
    stride = config.residual_sample_stride
    for residual, alpha in zip(residuals, alphas):
        _validate_map_pair(residual, alpha)
        sampled_residual = residual.detach()[::stride, ::stride]
        sampled_alpha = alpha.detach()[::stride, ::stride]
        valid = (
            torch.isfinite(sampled_residual)
            & torch.isfinite(sampled_alpha)
            & (sampled_alpha >= config.alpha_min)
        )
        if valid.any():
            samples.append(sampled_residual[valid].to(device="cpu", dtype=torch.float64))
    if not samples:
        raise RuntimeError("scene contains no finite residual samples with alpha >= 0.5")
    values = torch.cat(samples)
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad + config.epsilon
    threshold = median + config.residual_scale_multiplier * scale
    return (
        float(threshold.item()),
        float(median.item()),
        float(scale.item()),
        int(values.numel()),
    )


def select_camera_neighbors(
    camtoworlds: Tensor,
    image_names: Sequence[str],
    neighbor_count: int = 3,
) -> dict[str, list[str]]:
    """Choose Euclidean-nearest camera centers with filename tie-breaking."""

    if camtoworlds.ndim != 3 or camtoworlds.shape[1:] != (4, 4):
        raise ValueError("camtoworlds must have shape [N, 4, 4]")
    if len(image_names) != camtoworlds.shape[0]:
        raise ValueError("image_names and camtoworlds length mismatch")
    if len(set(image_names)) != len(image_names):
        raise ValueError("image_names must be unique")
    if neighbor_count <= 0 or len(image_names) <= neighbor_count:
        raise ValueError("scene must contain more images than neighbor_count")

    centers = camtoworlds.detach().to(device="cpu", dtype=torch.float64)[:, :3, 3]
    output: dict[str, list[str]] = {}
    for source_index, source_name in enumerate(image_names):
        ranked: list[tuple[float, str]] = []
        for target_index, target_name in enumerate(image_names):
            if source_index == target_index:
                continue
            distance = torch.linalg.vector_norm(
                centers[source_index] - centers[target_index]
            ).item()
            ranked.append((float(distance), target_name))
        ranked.sort(key=lambda item: (item[0], item[1]))
        output[source_name] = [name for _, name in ranked[:neighbor_count]]
    return output


def unproject_camera_z(
    pixels_xy: Tensor,
    camera_z: Tensor,
    K: Tensor,
    camtoworld: Tensor,
) -> Tensor:
    """Unproject image coordinates using gsplat expected camera-z depth."""

    if pixels_xy.ndim != 2 or pixels_xy.shape[-1] != 2:
        raise ValueError("pixels_xy must have shape [N, 2]")
    if camera_z.shape != pixels_xy.shape[:1]:
        raise ValueError("camera_z must have shape [N]")
    if K.shape != (3, 3) or camtoworld.shape != (4, 4):
        raise ValueError("K and camtoworld must have shapes [3, 3] and [4, 4]")
    dtype = camera_z.dtype
    device = camera_z.device
    pixels_xy = pixels_xy.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    camtoworld = camtoworld.to(device=device, dtype=dtype)
    homogeneous_pixels = torch.cat(
        [pixels_xy, torch.ones_like(pixels_xy[:, :1])], dim=-1
    )
    camera_points = (torch.linalg.inv(K) @ homogeneous_pixels.T).T
    camera_points = camera_points * camera_z[:, None]
    homogeneous_camera = torch.cat(
        [camera_points, torch.ones_like(camera_points[:, :1])], dim=-1
    )
    return (camtoworld @ homogeneous_camera.T).T[:, :3]


def project_camera_z(
    world_points: Tensor,
    K: Tensor,
    camtoworld: Tensor,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Project world points, returning image ``(x, y)`` and camera-z."""

    if world_points.ndim != 2 or world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape [N, 3]")
    if K.shape != (3, 3) or camtoworld.shape != (4, 4):
        raise ValueError("K and camtoworld must have shapes [3, 3] and [4, 4]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    dtype = world_points.dtype
    device = world_points.device
    K = K.to(device=device, dtype=dtype)
    camtoworld = camtoworld.to(device=device, dtype=dtype)
    homogeneous_world = torch.cat(
        [world_points, torch.ones_like(world_points[:, :1])], dim=-1
    )
    camera_points = (torch.linalg.inv(camtoworld) @ homogeneous_world.T).T[:, :3]
    projected = (K @ camera_points.T).T
    z = camera_points[:, 2]
    safe_z = torch.where(z.abs() > epsilon, z, torch.full_like(z, epsilon))
    return projected[:, :2] / safe_z[:, None], z


def bilinear_sample(image: Tensor, pixels_xy: Tensor) -> Tensor:
    """Bilinearly sample a ``[H,W]`` or ``[H,W,C]`` tensor at image coordinates."""

    if image.ndim not in (2, 3):
        raise ValueError("image must have shape [H, W] or [H, W, C]")
    if pixels_xy.ndim != 2 or pixels_xy.shape[-1] != 2:
        raise ValueError("pixels_xy must have shape [N, 2]")
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("bilinear sampling requires height and width >= 2")
    scalar = image.ndim == 2
    channels_first = (
        image.unsqueeze(-1) if scalar else image
    ).permute(2, 0, 1).unsqueeze(0)
    xy = pixels_xy.to(device=image.device, dtype=image.dtype)
    normalized = torch.empty_like(xy)
    normalized[:, 0] = 2.0 * xy[:, 0] / (width - 1) - 1.0
    normalized[:, 1] = 2.0 * xy[:, 1] / (height - 1) - 1.0
    grid = normalized.view(1, 1, -1, 2)
    sampled = F.grid_sample(
        channels_first,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, :, 0].T
    return sampled[:, 0] if scalar else sampled


@torch.inference_mode()
def compute_cvtr_stage_trace(
    source: ViewEvidence,
    neighbors: Sequence[ViewEvidence],
    scene_threshold: float,
    config: CVTRConfig = CVTRConfig(),
) -> CVTRStageTrace:
    """Trace the eight frozen CVTR stages without changing production semantics."""

    if len(neighbors) != config.neighbor_count:
        raise ValueError(f"exactly {config.neighbor_count} neighbors are required")
    device = source.residual.device
    dtype = source.depth.dtype
    s0 = torch.isfinite(source.residual) & (source.residual > scene_threshold)
    s1 = (
        s0
        & torch.isfinite(source.alpha)
        & (source.alpha >= config.alpha_min)
    )
    valid_source_depth = torch.isfinite(source.depth) & (source.depth > 0)
    production_candidate = (
        s1
        & valid_source_depth
    )
    finite_source = (
        torch.isfinite(source.residual)
        & torch.isfinite(source.depth)
        & torch.isfinite(source.alpha)
        & (source.depth > 0)
    )
    expected_production_candidate = (
        finite_source
        & (source.residual > scene_threshold)
        & (source.alpha >= config.alpha_min)
    )
    if not torch.equal(production_candidate, expected_production_candidate):
        raise RuntimeError("trace candidate diverged from the production CVTR gate")

    inbounds_counts = torch.zeros_like(source.residual, dtype=torch.int16)
    alpha_counts = torch.zeros_like(source.residual, dtype=torch.int16)
    depth_counts = torch.zeros_like(source.residual, dtype=torch.int16)
    static_counts_map = torch.zeros_like(source.residual, dtype=torch.int16)
    depth_errors = torch.full(
        (config.neighbor_count, *source.residual.shape),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    candidate_yx = production_candidate.nonzero(as_tuple=False)
    if candidate_yx.numel() == 0:
        zero_ratio = torch.zeros_like(source.residual, dtype=dtype)
        s2 = torch.zeros_like(s0)
        s3 = torch.zeros_like(s0)
        s4 = torch.zeros_like(s0)
        s5 = torch.zeros_like(s0)
        spatial_score = spatial_support_scores(s5, config)
        s6 = spatial_score >= config.patch_min_support
        s7 = apply_area_cap(s6, source.residual, config)
        return CVTRStageTrace(
            stage_masks=tuple(
                mask.detach() for mask in (s0, s1, s2, s3, s4, s5, s6, s7)
            ),
            production_candidate_mask=production_candidate.detach(),
            rgb_residual=source.residual.detach(),
            current_alpha=source.alpha.detach(),
            inbounds_positive_neighbor_count=inbounds_counts.detach(),
            neighbor_alpha_valid_count=alpha_counts.detach(),
            depth_consistent_neighbor_count=depth_counts.detach(),
            static_support_ratio=zero_ratio.detach(),
            spatial_support_score=spatial_score.detach(),
            depth_consistency_error_per_neighbor=depth_errors.detach(),
        )

    pixels_xy = candidate_yx[:, [1, 0]].to(device=device, dtype=dtype)
    source_depth = source.depth[production_candidate]
    world_points = unproject_camera_z(
        pixels_xy, source_depth, source.K, source.camtoworld
    )
    candidate_inbounds_counts = torch.zeros(
        len(pixels_xy), device=device, dtype=torch.int16
    )
    candidate_alpha_counts = torch.zeros(
        len(pixels_xy), device=device, dtype=torch.int16
    )
    candidate_depth_counts = torch.zeros(
        len(pixels_xy), device=device, dtype=torch.int16
    )
    candidate_static_counts = torch.zeros_like(candidate_depth_counts)

    for neighbor_index, neighbor in enumerate(neighbors):
        projected_xy, projected_z = project_camera_z(
            world_points,
            neighbor.K,
            neighbor.camtoworld,
            epsilon=config.epsilon,
        )
        height, width = neighbor.residual.shape
        in_bounds = (
            (projected_xy[:, 0] >= 0)
            & (projected_xy[:, 0] <= width - 1)
            & (projected_xy[:, 1] >= 0)
            & (projected_xy[:, 1] <= height - 1)
        )
        positive_depth = torch.isfinite(projected_z) & (projected_z > 0)
        sampled_alpha = bilinear_sample(
            neighbor.alpha.to(device=device, dtype=dtype), projected_xy
        )
        sampled_depth = bilinear_sample(
            neighbor.depth.to(device=device, dtype=dtype), projected_xy
        )
        sampled_residual = bilinear_sample(
            neighbor.residual.to(device=device, dtype=dtype), projected_xy
        )
        relative_depth_error = (sampled_depth - projected_z).abs() / (
            projected_z + config.epsilon
        )
        inbounds_positive = in_bounds & positive_depth
        alpha_valid = (
            inbounds_positive
            & torch.isfinite(sampled_alpha)
            & (sampled_alpha >= config.alpha_min)
        )
        depth_valid = (
            alpha_valid
            & torch.isfinite(sampled_depth)
            & torch.isfinite(sampled_residual)
            & (relative_depth_error <= config.depth_relative_tolerance)
        )
        candidate_inbounds_counts += inbounds_positive.to(torch.int16)
        candidate_alpha_counts += alpha_valid.to(torch.int16)
        candidate_depth_counts += depth_valid.to(torch.int16)
        candidate_static_counts += (
            depth_valid & (sampled_residual <= scene_threshold)
        ).to(torch.int16)
        depth_errors[neighbor_index, candidate_yx[:, 0], candidate_yx[:, 1]] = (
            relative_depth_error
        )

    inbounds_counts[candidate_yx[:, 0], candidate_yx[:, 1]] = (
        candidate_inbounds_counts
    )
    alpha_counts[candidate_yx[:, 0], candidate_yx[:, 1]] = candidate_alpha_counts
    depth_counts[candidate_yx[:, 0], candidate_yx[:, 1]] = candidate_depth_counts
    static_counts_map[candidate_yx[:, 0], candidate_yx[:, 1]] = (
        candidate_static_counts
    )

    s2 = production_candidate & (
        inbounds_counts >= config.minimum_valid_neighbors
    )
    s3 = s2 & (alpha_counts >= config.minimum_valid_neighbors)
    s4 = s3 & (depth_counts >= config.minimum_valid_neighbors)
    static_support = static_counts_map.to(dtype) / depth_counts.clamp_min(1).to(dtype)
    s5 = s4 & (static_support >= config.static_support_threshold)
    spatial_score = spatial_support_scores(s5, config)
    s6 = spatial_score >= config.patch_min_support
    s7 = apply_area_cap(s6, source.residual, config)
    return CVTRStageTrace(
        stage_masks=tuple(
            mask.detach() for mask in (s0, s1, s2, s3, s4, s5, s6, s7)
        ),
        production_candidate_mask=production_candidate.detach(),
        rgb_residual=source.residual.detach(),
        current_alpha=source.alpha.detach(),
        inbounds_positive_neighbor_count=inbounds_counts.detach(),
        neighbor_alpha_valid_count=alpha_counts.detach(),
        depth_consistent_neighbor_count=depth_counts.detach(),
        static_support_ratio=static_support.detach(),
        spatial_support_score=spatial_score.detach(),
        depth_consistency_error_per_neighbor=depth_errors.detach(),
    )


def compute_cross_view_transient(
    source: ViewEvidence,
    neighbors: Sequence[ViewEvidence],
    scene_threshold: float,
    config: CVTRConfig = CVTRConfig(),
) -> CrossViewResult:
    """Compute the production candidate and raw masks before spatial filtering."""

    trace = compute_cvtr_stage_trace(source, neighbors, scene_threshold, config)
    return CrossViewResult(
        candidate_mask=trace.production_candidate_mask,
        raw_transient_mask=trace.stage_masks[5],
        valid_neighbor_counts=trace.depth_consistent_neighbor_count,
    )


def spatial_support_scores(
    raw_transient_mask: Tensor,
    config: CVTRConfig = CVTRConfig(),
) -> Tensor:
    """Return the fixed 3x3 average-pooling support score."""

    if raw_transient_mask.ndim != 2:
        raise ValueError("raw_transient_mask must have shape [H, W]")
    return F.avg_pool2d(
        raw_transient_mask.to(dtype=torch.float32)[None, None],
        kernel_size=config.patch_size,
        stride=1,
        padding=config.patch_size // 2,
    )[0, 0]


def apply_area_cap(
    spatial_mask: Tensor,
    residual: Tensor,
    config: CVTRConfig = CVTRConfig(),
) -> Tensor:
    """Apply only the production top-residual area cap to a spatial mask."""

    if spatial_mask.ndim != 2 or spatial_mask.shape != residual.shape:
        raise ValueError("spatial_mask and residual must match [H, W]")
    mask = spatial_mask.to(torch.bool)
    maximum_count = math.floor(config.max_transient_area * mask.numel())
    selected_count = int(mask.sum().item())
    if selected_count > maximum_count:
        selected_indices = mask.flatten().nonzero(as_tuple=False).flatten()
        scores = residual.detach().flatten()[selected_indices]
        order = torch.argsort(scores, descending=True, stable=True)
        keep = selected_indices[order[:maximum_count]]
        capped = torch.zeros_like(mask.flatten())
        capped[keep] = True
        mask = capped.view_as(mask)
    return mask.detach()


def spatial_filter_and_cap(
    raw_transient_mask: Tensor,
    residual: Tensor,
    config: CVTRConfig = CVTRConfig(),
) -> Tensor:
    """Apply the fixed 3x3 support rule and top-residual 15% area cap."""

    if raw_transient_mask.ndim != 2 or raw_transient_mask.shape != residual.shape:
        raise ValueError("raw_transient_mask and residual must match [H, W]")
    support = spatial_support_scores(raw_transient_mask, config)
    mask = support >= config.patch_min_support
    return apply_area_cap(mask, residual, config)


def mask_to_responsibility(
    transient_mask: Tensor,
    transient_weight: float = 0.2,
) -> Tensor:
    if transient_mask.ndim not in (2, 3):
        raise ValueError("transient_mask must have shape [H, W] or [B, H, W]")
    if not 0.0 < transient_weight <= 1.0:
        raise ValueError("transient_weight must be in (0, 1]")
    responsibility = torch.where(
        transient_mask.to(torch.bool),
        torch.full_like(transient_mask, transient_weight, dtype=torch.float32),
        torch.ones_like(transient_mask, dtype=torch.float32),
    )
    return responsibility.detach()


def effective_sample_size_ratio(
    responsibility: Tensor,
    epsilon: float = 1e-6,
) -> float:
    if responsibility.numel() == 0 or not responsibility.is_floating_point():
        raise ValueError("responsibility must be a non-empty floating tensor")
    flat = responsibility.detach().to(torch.float64).flatten()
    neff = flat.sum().square() / (flat.square().sum() + epsilon)
    return float((neff / flat.numel()).item())


def cvtr_weighted_l1(
    rendered: Tensor,
    target: Tensor,
    *,
    enabled: bool,
    transient_mask: Tensor | None = None,
    transient_weight: float = 0.2,
) -> tuple[Tensor, Tensor | None]:
    """Return exact B1 L1 when disabled, otherwise full-pixel-mean CVTR L1."""

    if rendered.shape != target.shape:
        raise ValueError("rendered and target shapes differ")
    if rendered.ndim != 4 or rendered.shape[-1] != 3:
        raise ValueError("RGB tensors must have shape [B, H, W, 3]")
    if not enabled:
        return F.l1_loss(rendered, target), None
    if transient_mask is None or transient_mask.shape != rendered.shape[:-1]:
        raise ValueError("transient_mask must have shape [B, H, W]")
    pixel_l1 = (rendered - target).abs().mean(dim=-1)
    responsibility = mask_to_responsibility(transient_mask, transient_weight).to(
        device=pixel_l1.device, dtype=pixel_l1.dtype
    )
    loss = (responsibility * pixel_l1).mean()
    return loss, responsibility.detach()


def save_binary_mask(path: str | Path, transient_mask: Tensor | np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = (
        transient_mask.detach().to(device="cpu").numpy()
        if isinstance(transient_mask, Tensor)
        else np.asarray(transient_mask)
    )
    if array.ndim != 2:
        raise ValueError("transient_mask must have shape [H, W]")
    Image.fromarray(array.astype(bool).astype(np.uint8) * 255, mode="L").save(output)


def load_binary_mask(path: str | Path) -> Tensor:
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"CVTR mask does not exist: {mask_path}")
    with Image.open(mask_path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    values = set(np.unique(array).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"CVTR mask must contain only 0/255, got {sorted(values)}")
    return torch.from_numpy(array == 255)


class CVTRMaskStore:
    """Manifest-validated, filename-keyed fixed mask loader with a CPU cache."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load CVTR manifest {manifest_path}: {error}") from error
        images = self.manifest.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("CVTR manifest.images must be a non-empty list")
        self._paths: dict[str, Path] = {}
        for item in images:
            if not isinstance(item, dict):
                raise ValueError("each CVTR manifest image entry must be an object")
            name = item.get("image_name")
            relative_path = item.get("mask_path")
            if not isinstance(name, str) or not isinstance(relative_path, str):
                raise ValueError("manifest image_name and mask_path must be strings")
            if name in self._paths:
                raise ValueError(f"duplicate CVTR image_name: {name}")
            path = (self.root / relative_path).resolve()
            try:
                path.relative_to(self.root.resolve())
            except ValueError as error:
                raise ValueError(f"mask path escapes CVTR root: {relative_path}") from error
            if not path.is_file():
                raise FileNotFoundError(f"manifest CVTR mask is missing: {path}")
            self._paths[name] = path
        self._cache: dict[str, Tensor] = {}

    def load(
        self,
        image_name: str,
        *,
        expected_shape: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if image_name not in self._paths:
            raise KeyError(f"CVTR manifest has no mask for training image {image_name!r}")
        if image_name not in self._cache:
            self._cache[image_name] = load_binary_mask(self._paths[image_name])
        mask = self._cache[image_name]
        if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
            raise ValueError(
                f"CVTR mask shape for {image_name!r} is {tuple(mask.shape)}, "
                f"expected {tuple(expected_shape)}"
            )
        return mask.to(device=device) if device is not None else mask

    def load_batch(
        self,
        image_names: Sequence[str],
        *,
        expected_shape: Sequence[int],
        device: torch.device | str,
    ) -> Tensor:
        return torch.stack(
            [
                self.load(name, expected_shape=expected_shape, device=device)
                for name in image_names
            ]
        ).detach()


def binary_mask_metrics(predicted: Tensor, target: Tensor) -> dict[str, float]:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target mask shapes differ")
    predicted = predicted.to(torch.bool)
    target = target.to(torch.bool)
    true_positive = int((predicted & target).sum().item())
    false_positive = int((predicted & ~target).sum().item())
    false_negative = int((~predicted & target).sum().item())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-15)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }
