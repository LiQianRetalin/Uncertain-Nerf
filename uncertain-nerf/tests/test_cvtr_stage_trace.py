import numpy as np
import torch

from puri_gs.cvtr import (
    CVTR_STAGE_NAMES,
    CVTRConfig,
    ViewEvidence,
    bilinear_sample,
    compute_cross_view_transient,
    compute_cvtr_stage_trace,
    spatial_filter_and_cap,
    spatial_support_scores,
)
from tools.analyze_cvtr_failure_attribution import (
    bilinear_contract_audit,
    camera_inverse_error,
    same_view_roundtrip,
)


def _view(name: str, residual: torch.Tensor | None = None) -> ViewEvidence:
    residual = torch.zeros(9, 9) if residual is None else residual
    return ViewEvidence(
        image_name=name,
        residual=residual,
        depth=torch.full((9, 9), 2.0),
        alpha=torch.ones(9, 9),
        K=torch.tensor([[6.0, 0.0, 4.0], [0.0, 6.0, 4.0], [0.0, 0.0, 1.0]]),
        camtoworld=torch.eye(4),
    )


def _trace():
    residual = torch.zeros(9, 9)
    residual[2:7, 2:7] = 1.0
    source = _view("source", residual)
    neighbors = [_view(f"neighbor_{index}") for index in range(3)]
    return source, neighbors, compute_cvtr_stage_trace(source, neighbors, 0.5)


def test_trace_final_mask_equals_the_production_final_mask_pixelwise():
    source, neighbors, trace = _trace()
    production = compute_cross_view_transient(source, neighbors, 0.5)
    final = spatial_filter_and_cap(production.raw_transient_mask, source.residual)
    assert torch.equal(trace.final_mask, final)
    assert torch.equal(trace.stage_masks[5], production.raw_transient_mask)


def test_stage_masks_only_delete_before_spatial_support_and_after_area_cap():
    _, _, trace = _trace()
    for index in range(1, 6):
        assert not (trace.stage_masks[index] & ~trace.stage_masks[index - 1]).any()
    assert not (trace.stage_masks[7] & ~trace.stage_masks[6]).any()


def test_spatial_support_is_the_documented_non_monotonic_exception():
    raw = torch.zeros(5, 5, dtype=torch.bool)
    raw[1, 1:4] = True
    raw[2, 1] = True
    raw[2, 3] = True
    assert not raw[2, 2]
    support = spatial_support_scores(raw)
    spatial = support >= 5.0 / 9.0
    assert spatial[2, 2]


def test_trace_area_cap_never_exceeds_fifteen_percent():
    residual = torch.ones(20, 20)
    source = ViewEvidence(
        image_name="source",
        residual=residual,
        depth=torch.full((20, 20), 2.0),
        alpha=torch.ones(20, 20),
        K=torch.tensor([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]),
        camtoworld=torch.eye(4),
    )
    neighbor = ViewEvidence(
        image_name="neighbor",
        residual=torch.zeros(20, 20),
        depth=torch.full((20, 20), 2.0),
        alpha=torch.ones(20, 20),
        K=source.K,
        camtoworld=torch.eye(4),
    )
    trace = compute_cvtr_stage_trace(source, [neighbor, neighbor, neighbor], 0.5)
    assert trace.final_mask.sum().item() <= int(0.15 * trace.final_mask.numel())


def test_trace_arrays_are_detached_and_have_expected_shapes():
    _, _, trace = _trace()
    assert len(trace.stage_masks) == len(CVTR_STAGE_NAMES) == 8
    tensors = [
        *trace.stage_masks,
        trace.rgb_residual,
        trace.current_alpha,
        trace.inbounds_positive_neighbor_count,
        trace.neighbor_alpha_valid_count,
        trace.depth_consistent_neighbor_count,
        trace.static_support_ratio,
        trace.spatial_support_score,
        trace.depth_consistency_error_per_neighbor,
    ]
    assert all(not tensor.requires_grad for tensor in tensors)
    assert trace.depth_consistency_error_per_neighbor.shape == (3, 9, 9)


def test_trace_neighbor_counts_are_three_for_identity_views():
    _, _, trace = _trace()
    candidate = trace.production_candidate_mask
    assert torch.all(trace.inbounds_positive_neighbor_count[candidate] == 3)
    assert torch.all(trace.neighbor_alpha_valid_count[candidate] == 3)
    assert torch.all(trace.depth_consistent_neighbor_count[candidate] == 3)


def test_same_view_projection_roundtrip_is_subpixel_exact():
    view = _view("roundtrip")
    errors = same_view_roundtrip(view, grid_stride=2)
    assert errors.size > 0
    assert np.percentile(errors, 95) <= 1e-5


def test_camera_inverse_contract():
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.4, -0.2, 1.3])
    assert camera_inverse_error(pose) <= 1e-12


def test_bilinear_sampling_corners_center_subpixel_and_outside():
    audit = bilinear_contract_audit()
    assert audit["status"] == "PASS"
    image = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    values = bilinear_sample(
        image,
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [-2.0, -2.0]]),
    )
    assert torch.allclose(values, torch.tensor([0.0, 3.0, 1.5, 0.0]))


def test_trace_respects_the_frozen_configuration():
    assert CVTRConfig().to_dict() == {
        "residual_sample_stride": 8,
        "alpha_min": 0.5,
        "residual_scale_multiplier": 3.0,
        "epsilon": 1e-6,
        "neighbor_count": 3,
        "depth_relative_tolerance": 0.05,
        "minimum_valid_neighbors": 2,
        "static_support_threshold": 0.5,
        "patch_size": 3,
        "patch_min_support": 5.0 / 9.0,
        "max_transient_area": 0.15,
        "transient_weight": 0.2,
        "minimum_neff_ratio": 0.9,
    }
