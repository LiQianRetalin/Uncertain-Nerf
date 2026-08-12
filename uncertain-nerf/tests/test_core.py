import torch

from v5.model import RadianceFieldV5
from v5.rendering import reliability, render_rays, volume_integrate
from v5.robust import LossCalibrator, mad_scale, pseudo_huber, tukey_biweight, warmup_cosine_lr


def tiny_field():
    return RadianceFieldV5(
        aabb=[[-1, -1, -1], [1, 1, 1]],
        hidden_dim=16,
        hidden_layers=2,
        dropout=0.0,
        hash_levels=3,
        hash_min_resolution=4,
        hash_max_resolution=16,
        hash_features=2,
        hash_log2_size=8,
        direction_frequencies=2,
    )


def test_hash_field_shapes_and_gradients():
    field = tiny_field()
    points = torch.rand(2, 7, 3) * 1.8 - 0.9
    directions = torch.randn_like(points)
    raw = field(points, directions)
    assert raw.shape == (2, 7, 5)
    raw.square().mean().backward()
    assert field.position_encoder.tables[-1].weight.grad is not None
    assert torch.isfinite(field.position_encoder.tables[-1].weight.grad).all()


def test_reliability_is_bounded_and_monotonic():
    uncertainty = torch.linspace(0, 1, 100)
    mapped = reliability(uncertainty, strength=4.0)
    assert torch.all(mapped[:-1] >= mapped[1:])
    assert mapped[0] == 1
    assert torch.isclose(mapped[-1], torch.tensor(0.2))


def test_volume_weights_and_terminal_are_conservative():
    raw = torch.zeros(3, 8, 5)
    raw[..., 3] = 2.0
    z_vals = torch.linspace(0.1, 1.0, 8).expand(3, -1)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1)
    result = volume_integrate(raw, z_vals, rays_d, torch.zeros(3, 3))
    total = result["weights"].sum(dim=-1) + result["terminal_transmittance"]
    assert torch.allclose(total, torch.ones_like(total), atol=1.0e-5)
    assert torch.all((result["uncertainty"] >= 0) & (result["uncertainty"] <= 1))


def test_coarse_fine_render_shapes():
    coarse, fine = tiny_field(), tiny_field()
    rays_o = torch.tensor([[0.0, 0.0, -2.0]]).expand(5, -1)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]]).expand(5, -1)
    result = render_rays(
        coarse, fine, rays_o, rays_d, 0.0, 5.0, torch.zeros(5, 3),
        n_samples=8, n_importance=4, perturb=False,
    )
    assert result["coarse"]["weights"].shape == (5, 8)
    assert result["fine"]["weights"].shape == (5, 12)
    assert result["fine"]["rgb"].shape == (5, 3)


def test_robust_functions_and_calibration():
    values = torch.tensor([0.0, 0.1, 0.2, 50.0])
    scale = mad_scale(values)
    assert 0 < scale < 1
    tukey = tukey_biweight(values, torch.tensor(1.0))
    assert torch.isclose(tukey[-1], torch.tensor(1.0 / 6.0))
    huber = pseudo_huber(values, torch.tensor(1.0))
    assert torch.isfinite(huber).all()
    calibrator = LossCalibrator(calibration_steps=2)
    losses = {"color": torch.tensor(1.0), "geo": torch.tensor(2.0), "u": torch.tensor(4.0), "reg": torch.tensor(5.0)}
    calibrator.update(1, losses, 0.2)
    weights = calibrator.update(2, losses, 0.2)
    assert weights["geo"] > 0 and weights["u"] > 0 and weights["reg"] > 0
    assert warmup_cosine_lr(1, 100, warmup=10) < warmup_cosine_lr(10, 100, warmup=10)
