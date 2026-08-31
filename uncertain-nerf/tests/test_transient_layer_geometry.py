import math

import torch

from puri_gs.transient_layer import (
    TransientGaussianLayer,
    bilinear_sample_rgb,
    camera_plane_geometry,
    make_grid_centers,
    project_world_to_pixels,
    quaternion_wxyz_to_rotation_matrix,
    rotation_matrix_to_quaternion_wxyz,
)


def test_fixed_grid_count_and_edge_centers():
    centers = make_grid_centers(width=778, height=519)
    assert centers.shape == (math.ceil(778 / 16) * math.ceil(519 / 16), 2)
    assert centers.unique(dim=0).shape == centers.shape
    assert torch.equal(centers[0], torch.tensor([8.0, 8.0]))
    assert float(centers[:, 0].max()) == 776.0
    assert float(centers[:, 1].max()) == 518.0


def test_camera_plane_projects_to_grid_with_subpixel_margin():
    dtype = torch.float64
    K = torch.tensor([[793.1324, 0.0, 389.25], [0.0, 795.0201, 259.8757], [0.0, 0.0, 1.0]], dtype=dtype)
    angle = torch.tensor(0.7, dtype=dtype)
    rotation = torch.tensor(
        [[torch.cos(angle), 0.0, torch.sin(angle)], [0.0, 1.0, 0.0], [-torch.sin(angle), 0.0, torch.cos(angle)]],
        dtype=dtype,
    )
    camtoworld = torch.eye(4, dtype=dtype)
    camtoworld[:3, :3] = rotation
    camtoworld[:3, 3] = torch.tensor([0.4, -0.2, 1.1], dtype=dtype)
    centers = make_grid_centers(778, 519, dtype=dtype)
    means, scales = camera_plane_geometry(centers, K, camtoworld)
    projected = project_world_to_pixels(means, K, camtoworld)
    errors = torch.linalg.vector_norm(projected - centers, dim=-1)
    assert float(torch.quantile(errors, 0.95)) <= 0.1
    expected = torch.tensor(
        [0.5 * 16 / K[0, 0], 0.5 * 16 / K[1, 1]], dtype=dtype
    )
    assert torch.allclose(scales[0, :2], expected)
    assert torch.allclose(scales[0, 2], 0.01 * expected.min())


def test_quaternion_is_wxyz_and_reconstructs_camera_rotation():
    angle = torch.tensor(math.pi / 2, dtype=torch.float64)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    quaternion = rotation_matrix_to_quaternion_wxyz(rotation)
    expected = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=torch.float64)
    assert torch.allclose(quaternion, expected, atol=1e-7, rtol=0.0)
    assert torch.allclose(quaternion_wxyz_to_rotation_matrix(quaternion), rotation, atol=1e-7, rtol=0.0)


def test_bilinear_initial_color_and_trainable_parameter_boundary():
    target = torch.linspace(0.0, 1.0, 32 * 48 * 3).reshape(32, 48, 3)
    K = torch.tensor([[60.0, 0.0, 23.5], [0.0, 60.0, 15.5], [0.0, 0.0, 1.0]])
    camtoworld = torch.eye(4)
    layer = TransientGaussianLayer(target, K, camtoworld)
    expected = bilinear_sample_rgb(target, layer.centers_uv).clamp(1e-4, 1 - 1e-4)
    assert torch.allclose(layer.colors, expected, atol=1e-6)
    assert set(dict(layer.named_parameters())) == {"color_logits", "opacity_logits"}
    assert not layer.means.requires_grad
    assert not layer.quaternions.requires_grad
    assert not layer.log_scales.requires_grad
    assert torch.allclose(layer.opacities, torch.full_like(layer.opacities, 0.01))
