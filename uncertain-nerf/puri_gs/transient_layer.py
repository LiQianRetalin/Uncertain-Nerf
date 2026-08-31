"""Fixed camera-plane Gaussian layer for the Phase 4A feasibility gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class TransientLayerSpec:
    grid_stride: int = 16
    plane_z: float = 1.0
    initial_opacity: float = 0.01
    sh_degree: int = 0
    scale_z_fraction: float = 0.01

    def validate(self) -> None:
        if self != TransientLayerSpec():
            raise ValueError("Phase 4A transient geometry and initialization are frozen")


def _require_camera_inputs(K: Tensor, camtoworld: Tensor) -> None:
    if K.shape != (3, 3) or camtoworld.shape != (4, 4):
        raise ValueError("K and camtoworld must have shapes [3,3] and [4,4]")
    if not bool(torch.isfinite(K).all()) or not bool(torch.isfinite(camtoworld).all()):
        raise ValueError("camera tensors must be finite")
    if float(K[0, 0]) <= 0.0 or float(K[1, 1]) <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    expected_bottom = camtoworld.new_tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(camtoworld[3], expected_bottom, atol=1e-6, rtol=0.0):
        raise ValueError("camtoworld must be a homogeneous camera-to-world matrix")


def make_grid_centers(
    width: int,
    height: int,
    stride: int = 16,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return row-major ``[u,v]`` centers from the frozen Phase 4A formula."""

    if width <= 0 or height <= 0 or stride <= 0:
        raise ValueError("image dimensions and grid stride must be positive")
    count_x = math.ceil(width / stride)
    count_y = math.ceil(height / stride)
    u = torch.arange(count_x, device=device, dtype=dtype) * stride + stride / 2
    v = torch.arange(count_y, device=device, dtype=dtype) * stride + stride / 2
    u = u.clamp_max(width - 1).unique(sorted=True)
    v = v.clamp_max(height - 1).unique(sorted=True)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    centers = torch.stack((grid_u.reshape(-1), grid_v.reshape(-1)), dim=-1)
    expected = count_x * count_y
    if centers.shape != (expected, 2):
        raise RuntimeError("edge de-duplication changed the frozen Gaussian count")
    return centers


def camera_plane_geometry(
    centers_uv: Tensor,
    K: Tensor,
    camtoworld: Tensor,
    *,
    stride: int = 16,
    plane_z: float = 1.0,
    scale_z_fraction: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Create world means and positive local xyz scales for the camera plane."""

    _require_camera_inputs(K, camtoworld)
    if centers_uv.ndim != 2 or centers_uv.shape[-1] != 2:
        raise ValueError("centers_uv must have shape [N,2]")
    if stride <= 0 or plane_z <= 0.0 or scale_z_fraction <= 0.0:
        raise ValueError("camera-plane geometry constants must be positive")
    K = K.to(device=centers_uv.device, dtype=centers_uv.dtype)
    camtoworld = camtoworld.to(device=centers_uv.device, dtype=centers_uv.dtype)
    u, v = centers_uv.unbind(dim=-1)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    camera = torch.stack(
        ((u - cx) * plane_z / fx, (v - cy) * plane_z / fy, torch.full_like(u, plane_z)),
        dim=-1,
    )
    means = camera @ camtoworld[:3, :3].T + camtoworld[:3, 3]
    scale_x = 0.5 * stride * plane_z / fx
    scale_y = 0.5 * stride * plane_z / fy
    scale_z = scale_z_fraction * torch.minimum(scale_x, scale_y)
    one_scale = torch.stack((scale_x, scale_y, scale_z))
    scales = one_scale.unsqueeze(0).expand(centers_uv.shape[0], -1).clone()
    return means, scales


def project_world_to_pixels(means: Tensor, K: Tensor, camtoworld: Tensor) -> Tensor:
    """Project world points with the parser's camera-to-world convention."""

    _require_camera_inputs(K, camtoworld)
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape [N,3]")
    K = K.to(device=means.device, dtype=means.dtype)
    worldtocamera = torch.linalg.inv(camtoworld.to(device=means.device, dtype=means.dtype))
    camera = means @ worldtocamera[:3, :3].T + worldtocamera[:3, 3]
    projected = camera @ K.T
    if bool((projected[:, 2] <= 0).any()):
        raise ValueError("camera-plane points must project in front of the camera")
    return projected[:, :2] / projected[:, 2:3]


def rotation_matrix_to_quaternion_wxyz(rotation: Tensor) -> Tensor:
    """Convert a proper 3x3 rotation to gsplat's documented wxyz order."""

    if rotation.shape != (3, 3) or not bool(torch.isfinite(rotation).all()):
        raise ValueError("rotation must be one finite 3x3 matrix")
    identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
    if not torch.allclose(rotation.T @ rotation, identity, atol=1e-4, rtol=0.0):
        raise ValueError("camera rotation is not orthonormal")
    if not torch.allclose(torch.linalg.det(rotation), rotation.new_tensor(1.0), atol=1e-4, rtol=0.0):
        raise ValueError("camera rotation must have determinant +1")

    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    squared = torch.stack(
        (
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        )
    ).clamp_min(0.0)
    magnitudes = torch.sqrt(squared)
    candidates = torch.stack(
        (
            torch.stack((squared[0], m21 - m12, m02 - m20, m10 - m01)),
            torch.stack((m21 - m12, squared[1], m10 + m01, m02 + m20)),
            torch.stack((m02 - m20, m10 + m01, squared[2], m12 + m21)),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, squared[3])),
        )
    )
    candidates = candidates / (2.0 * magnitudes[:, None].clamp_min(1e-8))
    quaternion = candidates[int(torch.argmax(magnitudes).item())]
    quaternion = F.normalize(quaternion, dim=0)
    return torch.where(quaternion[0] < 0, -quaternion, quaternion)


def quaternion_wxyz_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    """Reference conversion used by the quaternion-order contract test."""

    if quaternion.shape != (4,):
        raise ValueError("quaternion must have shape [4]")
    w, x, y, z = F.normalize(quaternion, dim=0)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        )
    )


def bilinear_sample_rgb(image: Tensor, centers_uv: Tensor) -> Tensor:
    """Bilinearly sample an HWC RGB image at pixel-coordinate centers."""

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape [H,W,3]")
    if centers_uv.ndim != 2 or centers_uv.shape[-1] != 2:
        raise ValueError("centers_uv must have shape [N,2]")
    height, width = image.shape[:2]
    x = torch.zeros_like(centers_uv[:, 0]) if width == 1 else 2 * centers_uv[:, 0] / (width - 1) - 1
    y = torch.zeros_like(centers_uv[:, 1]) if height == 1 else 2 * centers_uv[:, 1] / (height - 1) - 1
    grid = torch.stack((x, y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, :, 0].T


def rgb_to_sh0(rgb: Tensor) -> Tensor:
    return (rgb - 0.5) / SH_C0


class TransientGaussianLayer(nn.Module):
    """One view-owned layer with frozen geometry and trainable color/opacity logits."""

    def __init__(
        self,
        target_rgb: Tensor,
        K: Tensor,
        camtoworld: Tensor,
        spec: TransientLayerSpec = TransientLayerSpec(),
    ) -> None:
        super().__init__()
        spec.validate()
        if target_rgb.ndim != 3 or target_rgb.shape[-1] != 3:
            raise ValueError("target_rgb must have shape [H,W,3]")
        if not bool(torch.isfinite(target_rgb).all()):
            raise ValueError("target_rgb must be finite")
        height, width = target_rgb.shape[:2]
        centers = make_grid_centers(
            width,
            height,
            spec.grid_stride,
            device=target_rgb.device,
            dtype=target_rgb.dtype,
        )
        means, positive_scales = camera_plane_geometry(
            centers,
            K,
            camtoworld,
            stride=spec.grid_stride,
            plane_z=spec.plane_z,
            scale_z_fraction=spec.scale_z_fraction,
        )
        quaternion = rotation_matrix_to_quaternion_wxyz(
            camtoworld[:3, :3].to(device=target_rgb.device, dtype=target_rgb.dtype)
        )
        quaternions = quaternion.unsqueeze(0).expand(centers.shape[0], -1).clone()
        initial_colors = bilinear_sample_rgb(target_rgb, centers).clamp(1e-4, 1.0 - 1e-4)
        initial_opacity = target_rgb.new_full((centers.shape[0],), spec.initial_opacity)

        self.register_buffer("centers_uv", centers)
        self.register_buffer("means", means)
        self.register_buffer("quaternions", quaternions)
        self.register_buffer("log_scales", positive_scales.log())
        self.color_logits = nn.Parameter(torch.logit(initial_colors))
        self.opacity_logits = nn.Parameter(torch.logit(initial_opacity))
        self.width = width
        self.height = height
        self.spec = spec

    @property
    def colors(self) -> Tensor:
        return torch.sigmoid(self.color_logits)

    @property
    def opacities(self) -> Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def scales(self) -> Tensor:
        return torch.exp(self.log_scales)

    def render(
        self,
        K: Tensor,
        camtoworld: Tensor,
        rasterization_fn: Callable[..., tuple[Tensor, Tensor, dict[str, Any]]] | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        if rasterization_fn is None:
            from gsplat.rendering import rasterization as rasterization_fn

        colors_sh0 = rgb_to_sh0(self.colors).unsqueeze(1)
        rendered, alpha, metadata = rasterization_fn(
            means=self.means,
            quats=self.quaternions,
            scales=self.scales,
            opacities=self.opacities,
            colors=colors_sh0,
            viewmats=torch.linalg.inv(camtoworld[None]),
            Ks=K[None],
            width=self.width,
            height=self.height,
            near_plane=0.01,
            far_plane=1e10,
            packed=False,
            backgrounds=self.means.new_zeros((1, 3)),
            render_mode="RGB",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
            distributed=False,
            camera_model="pinhole",
            sh_degree=0,
            with_ut=False,
            with_eval3d=False,
        )
        transient_premultiplied = rendered[0]
        transient_alpha = alpha[0]
        if transient_premultiplied.shape != (self.height, self.width, 3):
            raise RuntimeError("unexpected transient RGB rasterizer shape")
        if transient_alpha.shape != (self.height, self.width, 1):
            raise RuntimeError("unexpected transient alpha rasterizer shape")
        return transient_premultiplied, transient_alpha, metadata

    def serializable_state(self, image_name: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "image_name": image_name,
            "quaternion_order": "wxyz",
            "colors_are_sigmoid_logits": True,
            "opacities_are_sigmoid_logits": True,
            "means": self.means.detach().cpu(),
            "quaternions": self.quaternions.detach().cpu(),
            "log_scales": self.log_scales.detach().cpu(),
            "color_logits": self.color_logits.detach().cpu(),
            "opacity_logits": self.opacity_logits.detach().cpu(),
        }


def materialize_autograd_safe_constant(value: Tensor) -> Tensor:
    """Copy an inference tensor into ordinary detached tensor storage.

    Autograd may need to save a constant operand while differentiating another
    operand.  PyTorch deliberately forbids saving tensors created in
    ``inference_mode``, even when those tensors do not require gradients.
    """

    with torch.inference_mode(False):
        materialized = value.detach().clone()
    if torch.is_inference(materialized):
        raise RuntimeError("failed to materialize an autograd-safe constant")
    return materialized


class FrozenStaticGaussians(nn.Module):
    """Read-only B1 splats with no optimizer, strategy, densification, or pruning."""

    REQUIRED_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")

    def __init__(self, splats: dict[str, Tensor]) -> None:
        super().__init__()
        missing = set(self.REQUIRED_KEYS).difference(splats)
        if missing:
            raise ValueError(f"static checkpoint is missing tensors: {sorted(missing)}")
        self.gaussians = nn.ParameterDict(
            {
                name: nn.Parameter(splats[name].detach(), requires_grad=False)
                for name in self.REQUIRED_KEYS
            }
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def render(self, K: Tensor, camtoworld: Tensor, width: int, height: int) -> Tensor:
        from gsplat.rendering import rasterization

        with torch.inference_mode():
            rendered, _, _ = rasterization(
                means=self.gaussians["means"],
                quats=self.gaussians["quats"],
                scales=torch.exp(self.gaussians["scales"]),
                opacities=torch.sigmoid(self.gaussians["opacities"]),
                colors=torch.cat((self.gaussians["sh0"], self.gaussians["shN"]), dim=1),
                viewmats=torch.linalg.inv(camtoworld[None]),
                Ks=K[None],
                width=width,
                height=height,
                near_plane=0.01,
                far_plane=1e10,
                packed=False,
                render_mode="RGB",
                sparse_grad=False,
                absgrad=True,
                rasterize_mode="classic",
                distributed=False,
                camera_model="pinhole",
                sh_degree=3,
                with_ut=False,
                with_eval3d=False,
            )
            static_rgb = rendered[0].clamp(0.0, 1.0)
        return materialize_autograd_safe_constant(static_rgb)


def compose_premultiplied(
    static_rgb: Tensor, transient_premultiplied_rgb: Tensor, transient_alpha: Tensor
) -> Tensor:
    """Compose once: transient RGB is already premultiplied by rasterization."""

    if static_rgb.shape != transient_premultiplied_rgb.shape:
        raise ValueError("static and transient RGB shapes must match")
    if transient_alpha.shape != static_rgb.shape[:-1] + (1,):
        raise ValueError("transient alpha must have one final channel")
    if not all(bool(torch.isfinite(value).all()) for value in (static_rgb, transient_premultiplied_rgb, transient_alpha)):
        raise ValueError("compositing inputs must be finite")
    composite = transient_premultiplied_rgb + (1.0 - transient_alpha) * static_rgb
    if not bool(torch.isfinite(composite).all()):
        raise RuntimeError("compositing produced NaN or Inf")
    return composite


def static_transient_loss(
    composite: Tensor,
    target: Tensor,
    alpha: Tensor,
    ssim_fn: Callable[..., Tensor],
    *,
    ssim_lambda: float = 0.2,
    alpha_weight: float = 0.02,
) -> tuple[Tensor, dict[str, Tensor]]:
    if composite.shape != target.shape or alpha.shape != target.shape[:-1] + (1,):
        raise ValueError("loss tensors have incompatible shapes")
    l1 = F.l1_loss(composite, target)
    dssim = 1.0 - ssim_fn(
        composite.permute(2, 0, 1).unsqueeze(0),
        target.permute(2, 0, 1).unsqueeze(0),
        padding="valid",
    )
    photo = (1.0 - ssim_lambda) * l1 + ssim_lambda * dssim
    alpha_sparsity = alpha.mean()
    total = photo + alpha_weight * alpha_sparsity
    return total, {"l1": l1, "dssim": dssim, "photo": photo, "alpha": alpha_sparsity}


def projection_error_percentile(layer: TransientGaussianLayer, K: Tensor, camtoworld: Tensor) -> float:
    projected = project_world_to_pixels(layer.means, K, camtoworld)
    errors = torch.linalg.vector_norm(projected - layer.centers_uv, dim=-1)
    return float(torch.quantile(errors, 0.95).item())


def run_cpu_math_contract() -> dict[str, Any]:
    """Dependency-light geometry, composition, and directional optimization audit."""

    K = torch.tensor([[80.0, 0.0, 31.0], [0.0, 90.0, 23.0], [0.0, 0.0, 1.0]])
    angle = torch.tensor(math.pi / 3)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    camtoworld = torch.eye(4)
    camtoworld[:3, :3] = rotation
    camtoworld[:3, 3] = torch.tensor([0.2, -0.1, 0.4])
    target = torch.linspace(0.05, 0.95, 48 * 64 * 3).reshape(48, 64, 3)
    layer = TransientGaussianLayer(target, K, camtoworld)
    projection_p95 = projection_error_percentile(layer, K, camtoworld)
    quat_error = float(
        (quaternion_wxyz_to_rotation_matrix(layer.quaternions[0]) - rotation).abs().max().item()
    )

    static = torch.rand(5, 7, 3)
    alpha_zero = torch.zeros(5, 7, 1)
    alpha_one = torch.ones(5, 7, 1)
    red_pre = torch.zeros_like(static)
    red_pre[..., 0] = 1.0
    zero_result = compose_premultiplied(static, torch.zeros_like(static), alpha_zero)
    red_result = compose_premultiplied(static, red_pre, alpha_one)

    def direction_case(patched: bool) -> tuple[float, float, float, float, float]:
        base = torch.linspace(0.1, 0.9, 16 * 16 * 3).reshape(16, 16, 3)
        wanted = base.clone()
        mask = torch.zeros(16, 16, dtype=torch.bool)
        if patched:
            mask[4:12, 5:13] = True
            wanted[mask] = 1.0 - wanted[mask]
        colors = nn.Parameter(torch.logit(wanted.clamp(1e-4, 1 - 1e-4)))
        opacities = nn.Parameter(torch.logit(torch.full((16, 16, 1), 0.01)))
        optimizer = torch.optim.Adam(
            [{"params": [colors], "lr": 0.01}, {"params": [opacities], "lr": 0.05}],
            betas=(0.9, 0.99),
            eps=1e-8,
        )
        initial_alpha = float(torch.sigmoid(opacities).mean().item())
        initial_patch_error = float((base[mask] - wanted[mask]).abs().mean().item()) if patched else 0.0
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            alpha = torch.sigmoid(opacities)
            color = torch.sigmoid(colors)
            composite = alpha * color + (1 - alpha) * base
            loss = 0.8 * F.l1_loss(composite, wanted) + 0.02 * alpha.mean()
            loss.backward()
            optimizer.step()
        final_alpha_map = torch.sigmoid(opacities).detach()[..., 0]
        final_alpha = float(final_alpha_map.mean().item())
        final_composite = final_alpha_map[..., None] * torch.sigmoid(colors).detach() + (1 - final_alpha_map[..., None]) * base
        final_patch_error = float((final_composite[mask] - wanted[mask]).abs().mean().item()) if patched else 0.0
        separation = float(final_alpha_map[mask].mean().item() - final_alpha_map[~mask].mean().item()) if patched else 0.0
        return initial_alpha, final_alpha, initial_patch_error, final_patch_error, separation

    clean_initial, clean_final, _, _, _ = direction_case(False)
    patch_initial_alpha, patch_final_alpha, patch_initial_error, patch_final_error, patch_alpha_separation = direction_case(True)
    contract = {
        "grid_projection_p95_pixels": projection_p95,
        "quaternion_rotation_max_error": quat_error,
        "alpha_zero_returns_static": bool(torch.equal(zero_result, static)),
        "alpha_one_red_returns_red": bool(torch.equal(red_result, red_pre)),
        "composite_finite": bool(torch.isfinite(zero_result).all() and torch.isfinite(red_result).all()),
        "composite_in_range": bool(((zero_result >= 0) & (zero_result <= 1)).all() and ((red_result >= 0) & (red_result <= 1)).all()),
        "clean_initial_mean_alpha": clean_initial,
        "clean_final_mean_alpha": clean_final,
        "patch_initial_mean_alpha": patch_initial_alpha,
        "patch_final_mean_alpha": patch_final_alpha,
        "patch_initial_error": patch_initial_error,
        "patch_final_error": patch_final_error,
        "patch_inside_minus_outside_alpha": patch_alpha_separation,
    }
    contract["status"] = "PASS" if (
        projection_p95 <= 0.1
        and quat_error <= 1e-5
        and contract["alpha_zero_returns_static"]
        and contract["alpha_one_red_returns_red"]
        and contract["composite_finite"]
        and contract["composite_in_range"]
        and clean_final < clean_initial
        and patch_final_alpha > patch_initial_alpha
        and patch_alpha_separation > 0.0
        and patch_final_error < patch_initial_error
    ) else "FAIL"
    return contract
