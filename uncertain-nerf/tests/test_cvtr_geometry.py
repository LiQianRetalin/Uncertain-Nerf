from pathlib import Path
import importlib.util

import pytest
import torch

from puri_gs.cvtr import (
    CVTRConfig,
    ViewEvidence,
    compute_cross_view_transient,
    project_camera_z,
    select_camera_neighbors,
    unproject_camera_z,
)


def _view(name: str, residual_value: float, depth_value: float = 2.0) -> ViewEvidence:
    residual = torch.full((5, 5), residual_value)
    return ViewEvidence(
        image_name=name,
        residual=residual,
        depth=torch.full((5, 5), depth_value),
        alpha=torch.ones(5, 5),
        K=torch.tensor([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        camtoworld=torch.eye(4),
    )


def test_camera_neighbor_selection_is_deterministic_with_filename_ties():
    cameras = torch.eye(4).repeat(4, 1, 1)
    cameras[1, 0, 3] = 1.0
    cameras[2, 0, 3] = -1.0
    cameras[3, 0, 3] = 3.0
    names = ["source.png", "b.png", "a.png", "far.png"]
    neighbors = select_camera_neighbors(cameras, names, neighbor_count=3)
    assert neighbors["source.png"] == ["a.png", "b.png", "far.png"]
    assert neighbors == select_camera_neighbors(cameras, names, neighbor_count=3)


def test_projection_unprojection_camera_z_roundtrip():
    K = torch.tensor([[200.0, 0.0, 64.0], [0.0, 180.0, 48.0], [0.0, 0.0, 1.0]])
    camtoworld = torch.eye(4)
    camtoworld[:3, 3] = torch.tensor([0.4, -0.3, 1.2])
    pixels = torch.tensor([[12.25, 17.5], [80.0, 70.0]])
    depths = torch.tensor([2.0, 4.5])
    world = unproject_camera_z(pixels, depths, K, camtoworld)
    projected, projected_z = project_camera_z(world, K, camtoworld)
    assert torch.allclose(projected, pixels, atol=1e-5)
    assert torch.allclose(projected_z, depths, atol=1e-6)


def test_high_source_low_neighbor_residual_is_transient():
    source = _view("source", 0.0)
    source.residual[2, 2] = 1.0
    neighbors = [_view(str(index), 0.0) for index in range(3)]
    result = compute_cross_view_transient(source, neighbors, scene_threshold=0.5)
    assert result.candidate_mask[2, 2]
    assert result.raw_transient_mask[2, 2]
    assert result.valid_neighbor_counts[2, 2] == 3


def test_persistent_multiview_residual_is_not_transient():
    source = _view("source", 0.0)
    source.residual[2, 2] = 1.0
    neighbors = [_view(str(index), 1.0) for index in range(3)]
    result = compute_cross_view_transient(source, neighbors, scene_threshold=0.5)
    assert not result.raw_transient_mask[2, 2]


def test_no_valid_neighbors_and_depth_inconsistency_are_conservative():
    source = _view("source", 0.0)
    source.residual[2, 2] = 1.0
    inconsistent = [_view(str(index), 0.0, depth_value=2.2) for index in range(3)]
    result = compute_cross_view_transient(source, inconsistent, scene_threshold=0.5)
    assert result.valid_neighbor_counts[2, 2] == 0
    assert not result.raw_transient_mask.any()


def _cuda_architecture_is_supported() -> bool:
    if importlib.util.find_spec("gsplat") is None or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


@pytest.mark.skipif(
    not _cuda_architecture_is_supported(),
    reason="single-Gaussian gsplat depth audit requires a compatible CUDA wheel",
)
def test_single_gaussian_rgb_ed_is_expected_camera_z():
    from gsplat.rendering import rasterization

    device = torch.device("cuda")
    expected_z = 2.5
    renders, alpha, _ = rasterization(
        means=torch.tensor([[0.0, 0.0, expected_z]], device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        scales=torch.tensor([[0.15, 0.15, 0.15]], device=device),
        opacities=torch.tensor([0.99], device=device),
        colors=torch.tensor([[0.4, 0.5, 0.6]], device=device),
        viewmats=torch.eye(4, device=device)[None],
        Ks=torch.tensor(
            [[[80.0, 0.0, 16.0], [0.0, 80.0, 16.0], [0.0, 0.0, 1.0]]],
            device=device,
        ),
        width=33,
        height=33,
        packed=False,
        render_mode="RGB+ED",
    )
    valid = alpha[0, ..., 0] > 1e-3
    assert valid.any()
    assert torch.allclose(
        renders[0, ..., 3][valid],
        torch.full_like(renders[0, ..., 3][valid], expected_z),
        atol=1e-4,
    )


def test_cvtr_patch_adds_no_rasterization_to_training_or_inference():
    root = Path(__file__).resolve().parents[1]
    patch = (root / "patches" / "gsplat_v1.5.3_puri_gs_cvtr.patch").read_text(
        encoding="utf-8"
    )
    added_lines = [line[1:] for line in patch.splitlines() if line.startswith("+")]
    assert not any("rasterization(" in line for line in added_lines)
    assert "cvtr_weighted_l1" in patch
    assert "runner.train(init_step=10000)" in patch
    assert "optimizer=fresh strategy=fresh rng=fresh" in patch
