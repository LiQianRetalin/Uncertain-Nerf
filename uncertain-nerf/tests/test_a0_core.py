import torch

from a0.data import split_indices
from a0.model import OriginalNeRF, PositionalEncoder
from a0.rendering import ndc_rays, render_rays
from a0.trainer import original_learning_rate


def tiny_model():
    return OriginalNeRF(
        depth=6, width=16, fine_depth=6, fine_width=16,
        point_frequencies=2, view_frequencies=1)


def test_a0_uses_original_fern_split():
    train, val, test = split_indices(20, holdout=8, val_every=8)
    assert test.tolist() == [0, 8, 16]
    assert val.tolist() == [1, 10, 19]
    assert train.tolist() == [2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 17, 18]


def test_a0_model_has_coarse_fine_fields_and_no_uncertainty():
    model = tiny_model()
    names = [name for name, _ in model.named_parameters()]
    assert any(name.startswith("coarse.") for name in names)
    assert any(name.startswith("fine.") for name in names)
    assert not any("uncertainty" in name for name in names)
    points = torch.randn(3, 5, 3)
    directions = torch.nn.functional.normalize(torch.randn(3, 3), dim=-1)
    output = model.query(model.fine, points, directions, chunk=7)
    assert output.shape == (3, 5, 5)


def test_positional_encoder_matches_original_dimension():
    encoder = PositionalEncoder(3, frequencies=10)
    assert encoder.output_dim == 63
    encoded = encoder(torch.zeros(2, 3))
    assert encoded.shape == (2, 63)
    assert torch.isfinite(encoded).all()


def test_ndc_and_hierarchical_render_are_finite():
    model = tiny_model()
    origins = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    directions = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.1, -1.0]])
    ndc_o, ndc_d = ndc_rays(8, 8, 4.0, 1.0, origins, directions)
    assert torch.isfinite(ndc_o).all() and torch.isfinite(ndc_d).all()
    result = render_rays(
        model, origins, directions, 8, 8, 4.0,
        n_samples=4, n_importance=2, perturb=False, netchunk=16)
    assert result["fine"]["rgb"].shape == (2, 3)
    assert result["fine"]["weights"].shape == (2, 6)
    assert torch.isfinite(result["fine"]["rgb"]).all()


def test_original_learning_rate_schedule():
    assert original_learning_rate(0, 5.0e-4, 250) == 5.0e-4
    assert abs(original_learning_rate(250000, 5.0e-4, 250) - 5.0e-5) < 1.0e-12
