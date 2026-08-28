from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from puri_gs.responsibility import (
    ResponsibilityConfig,
    compute_responsibility,
    responsibility_weighted_l1,
)


def test_only_high_side_outlier_is_downweighted_without_gradient():
    residual = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 9.0]]],
        requires_grad=True,
    )
    config = ResponsibilityConfig(pool_size=1)
    responsibility = compute_responsibility(residual, config)
    assert responsibility.requires_grad is False
    assert responsibility.min() >= config.min_weight
    assert responsibility.max() <= 1.0
    assert torch.all(responsibility[residual.detach() <= 0.1] == 1.0)
    assert responsibility[0, 2, 2] < 1.0


def test_extreme_residual_is_clamped_and_constant_image_is_finite():
    config = ResponsibilityConfig(pool_size=1)
    residual = torch.zeros(1, 2, 3)
    residual[0, 1, 2] = 1e9
    responsibility = compute_responsibility(residual, config)
    assert responsibility[0, 1, 2].item() == pytest.approx(config.min_weight)
    constant = compute_responsibility(torch.ones(1, 4, 4), config)
    assert torch.isfinite(constant).all()
    assert torch.all(constant == 1.0)


def test_pooling_preserves_shape_range_and_finiteness():
    torch.manual_seed(7)
    residual = torch.rand(2, 9, 11)
    config = ResponsibilityConfig(pool_size=3)
    responsibility = compute_responsibility(residual, config)
    assert responsibility.shape == residual.shape
    assert torch.isfinite(responsibility).all()
    assert responsibility.min() >= config.min_weight
    assert responsibility.max() <= 1.0


def test_disabled_and_warmup_paths_are_exact_baseline_l1():
    torch.manual_seed(8)
    rendered = torch.rand(1, 5, 7, 3, requires_grad=True)
    target = torch.rand_like(rendered)
    expected = F.l1_loss(rendered, target)
    disabled, disabled_map = responsibility_weighted_l1(
        rendered,
        target,
        ResponsibilityConfig(enabled=False),
        step=9999,
    )
    warmup, warmup_map = responsibility_weighted_l1(
        rendered,
        target,
        ResponsibilityConfig(enabled=True, start_step=3000),
        step=2999,
    )
    torch.testing.assert_close(disabled, expected, rtol=0, atol=0)
    torch.testing.assert_close(warmup, expected, rtol=0, atol=0)
    assert disabled_map is None
    assert warmup_map is None


def test_robust_loss_backward_has_finite_parameter_gradient():
    target = torch.zeros(1, 8, 8, 3)
    rendered = torch.nn.Parameter(torch.full_like(target, 0.25))
    rendered.data[:, 2:5, 2:5] = 1.0
    loss, responsibility = responsibility_weighted_l1(
        rendered,
        target,
        ResponsibilityConfig(start_step=0),
        step=0,
    )
    assert responsibility is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
    assert rendered.grad.abs().sum() > 0


def test_checkpoint_roundtrip_preserves_parameters_and_config(tmp_path):
    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    config = ResponsibilityConfig()
    checkpoint = tmp_path / "a1_smoke.pt"
    torch.save(
        {
            "step": 9,
            "parameter": parameter.detach(),
            "optimizer": optimizer.state_dict(),
            "responsibility": asdict(config),
        },
        checkpoint,
    )
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.testing.assert_close(loaded["parameter"], parameter.detach())
    assert loaded["step"] == 9
    assert loaded["responsibility"] == asdict(config)


def test_invalid_even_pool_size_stops_early():
    try:
        ResponsibilityConfig(pool_size=2)
    except ValueError as error:
        assert "odd" in str(error)
    else:
        raise AssertionError("even pooling kernel must be rejected")
