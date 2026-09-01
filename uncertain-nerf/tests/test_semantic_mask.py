import torch

from puri_gs.semantic_mask import (
    ResidualHistogram,
    StaticResponsibilityHead,
    cosine_static_target,
    hard_static_mask,
    residual_interval_mask,
)


def test_mask_head_shape_range_and_no_nan():
    head = StaticResponsibilityHead()
    output = head(torch.randn(2, 384, 16, 16))
    assert output.shape == (2, 1, 16, 16)
    assert torch.isfinite(output).all()
    assert torch.all((0 <= output) & (output <= 1))


def test_fixed_threshold_and_seven_by_seven_min_pool():
    probability = torch.ones(1, 1, 11, 11)
    probability[:, :, 5, 5] = 0.25
    safe = hard_static_mask(probability)
    assert safe[:, :, 2:9, 2:9].sum() == 0
    assert safe[:, :, 0, 0].item() == 1


def test_reference_three_by_three_majority_rule():
    residual = torch.ones(1, 1, 5, 5)
    residual[:, :, 1:4, 1:4] = 0.1
    mask = residual_interval_mask(residual, 0.5)
    assert mask[0, 0, 2, 2] == 1
    assert mask[0, 0, 0, 0] == 0


def test_histogram_quantiles_and_cosine_target():
    histogram = ResidualHistogram()
    lower, upper = histogram.update(torch.linspace(0, 1, 1001))
    assert 0.59 <= lower <= 0.61
    assert 0.79 <= upper <= 0.81
    feature = torch.randn(1, 384, 4, 4)
    target = cosine_static_target(feature, feature)
    torch.testing.assert_close(target, torch.ones_like(target))


def test_reference_weight_regularizer_is_value_only():
    head = StaticResponsibilityHead()
    regularizer = head.reference_weight_regularizer()
    assert regularizer.requires_grad is False

