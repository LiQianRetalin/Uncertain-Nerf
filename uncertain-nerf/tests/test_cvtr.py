import json
from pathlib import Path

import torch
import torch.nn.functional as F

from puri_gs.cvtr import (
    CVTRConfig,
    CVTRMaskStore,
    cvtr_weighted_l1,
    effective_sample_size_ratio,
    load_binary_mask,
    mask_to_responsibility,
    save_binary_mask,
    scene_residual_threshold,
    spatial_filter_and_cap,
)


def test_scene_level_mad_threshold_uses_one_fixed_sample_population():
    config = CVTRConfig(residual_sample_stride=1)
    residuals = [torch.tensor([[0.0, 1.0], [2.0, 100.0]])]
    alphas = [torch.tensor([[1.0, 1.0], [1.0, 0.0]])]
    threshold, median, scale, count = scene_residual_threshold(
        residuals, alphas, config
    )
    assert count == 3
    assert median == 1.0
    assert abs(scale - (1.4826 + 1e-6)) < 1e-9
    assert abs(threshold - (median + 3.0 * scale)) < 1e-9


def test_patch_support_requires_five_of_nine():
    raw = torch.zeros(5, 5, dtype=torch.bool)
    raw[1, 1:4] = True
    raw[2, 1:3] = True
    residual = torch.ones(5, 5)
    mask = spatial_filter_and_cap(raw, residual)
    assert mask[2, 2]
    raw[2, 2] = False
    mask = spatial_filter_and_cap(raw, residual)
    assert not mask[2, 2]


def test_area_cap_keeps_highest_residual_fifteen_percent():
    raw = torch.ones(10, 10, dtype=torch.bool)
    residual = torch.arange(100, dtype=torch.float32).view(10, 10)
    mask = spatial_filter_and_cap(raw, residual)
    assert int(mask.sum()) == 15
    # The 3x3 rule first removes four corners; the cap then ranks only the
    # surviving final-mask pixels, exactly as the Phase 3 protocol specifies.
    supported = spatial_filter_and_cap(
        raw, residual, CVTRConfig(max_transient_area=1.0)
    )
    kept_scores = residual[mask]
    dropped_scores = residual[supported & ~mask]
    assert kept_scores.min() >= dropped_scores.max()


def test_responsibility_values_gradient_and_neff_guard():
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask.flatten()[:15] = True
    q = mask_to_responsibility(mask)
    assert torch.allclose(q.unique(), torch.tensor([0.2, 1.0]))
    assert not q.requires_grad
    assert effective_sample_size_ratio(q) >= 0.90


def test_fixed_mask_save_load_and_manifest_store(tmp_path: Path):
    mask = torch.tensor([[False, True], [True, False]])
    mask_path = tmp_path / "masks" / "frame.jpg.png"
    save_binary_mask(mask_path, mask)
    assert torch.equal(load_binary_mask(mask_path), mask)
    manifest = {
        "images": [
            {"image_name": "frame.jpg", "mask_path": "masks/frame.jpg.png"}
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    store = CVTRMaskStore(tmp_path)
    loaded = store.load_batch(
        ["frame.jpg"], expected_shape=(2, 2), device="cpu"
    )
    assert loaded.shape == (1, 2, 2)
    assert torch.equal(loaded[0], mask)
    assert not loaded.requires_grad


def test_cvtr_disabled_is_exact_b1_and_enabled_uses_full_pixel_mean():
    rendered = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]], requires_grad=True
    )
    target = torch.ones_like(rendered)
    baseline, q = cvtr_weighted_l1(rendered, target, enabled=False)
    assert q is None
    assert torch.equal(baseline, F.l1_loss(rendered, target))
    mask = torch.tensor([[[True, False]]])
    weighted, q = cvtr_weighted_l1(
        rendered, target, enabled=True, transient_mask=mask
    )
    assert torch.isclose(weighted, torch.tensor(0.1))
    assert q is not None and not q.requires_grad
    weighted.backward()
    assert rendered.grad is not None


def test_cvtr_configuration_is_frozen():
    import pytest

    with pytest.raises(ValueError, match="patch_size"):
        CVTRConfig(patch_size=5)
    with pytest.raises(ValueError, match="5/9"):
        CVTRConfig(patch_min_support=0.5)
