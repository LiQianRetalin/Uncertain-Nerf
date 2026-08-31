import pytest
import torch

from puri_gs.transient_layer import compose_premultiplied


def test_premultiplied_compositing_contract():
    static = torch.rand(9, 11, 3)
    zero_alpha = torch.zeros(9, 11, 1)
    one_alpha = torch.ones(9, 11, 1)
    red = torch.zeros_like(static)
    red[..., 0] = 1.0
    assert torch.equal(compose_premultiplied(static, torch.zeros_like(static), zero_alpha), static)
    assert torch.equal(compose_premultiplied(static, red, one_alpha), red)


def test_composite_stays_finite_and_in_unit_interval():
    static = torch.rand(6, 7, 3)
    alpha = torch.rand(6, 7, 1)
    color = torch.rand(6, 7, 3)
    premultiplied = alpha * color
    composite = compose_premultiplied(static, premultiplied, alpha)
    assert torch.isfinite(composite).all()
    assert ((composite >= 0.0) & (composite <= 1.0)).all()


def test_compositing_rejects_nonfinite_and_wrong_alpha_shape():
    static = torch.zeros(2, 3, 3)
    with pytest.raises(ValueError, match="one final channel"):
        compose_premultiplied(static, static, torch.zeros(2, 3))
    bad = static.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compose_premultiplied(static, bad, torch.zeros(2, 3, 1))
