import torch

from v6.model import RadianceFieldV6
from v6.rendering import render_rays, volume_integrate


def tiny_v6_field():
    return RadianceFieldV6(
        aabb=[[-1, -1, -1], [1, 1, 1]],
        hidden_dim=16,
        hidden_layers=2,
        hash_levels=3,
        hash_min_resolution=4,
        hash_max_resolution=16,
        hash_features=2,
        hash_log2_size=8,
        direction_frequencies=2,
        appearance_dim=4,
        hash_backend="torch",
    )


def test_uncertainty_does_not_change_opacity_or_rgb():
    raw_low = torch.zeros(2, 8, 5)
    raw_low[..., 3] = 1.0
    raw_low[..., 4] = -8.0
    raw_high = raw_low.clone()
    raw_high[..., 4] = 8.0
    z_vals = torch.linspace(0.1, 0.9, 8).expand(2, -1)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1)
    background = torch.zeros(2, 3)
    near, far = torch.full((2,), 0.0), torch.full((2,), 1.0)
    low = volume_integrate(raw_low, z_vals, rays_d, background, near, far)
    high = volume_integrate(raw_high, z_vals, rays_d, background, near, far)
    assert torch.allclose(low["acc"], high["acc"])
    assert torch.allclose(low["rgb"], high["rgb"])
    assert torch.all(low["uncertainty"] < high["uncertainty"])


def test_finite_far_interval_preserves_background_transmittance():
    raw = torch.zeros(1, 4, 5)
    raw[..., 3] = -20.0
    z_vals = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])
    result = volume_integrate(
        raw,
        z_vals,
        rays_d,
        torch.ones(1, 3),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )
    assert result["terminal_transmittance"].item() > 0.99
    assert torch.all(result["rgb"] > 0.99)


def test_shared_v6_field_coarse_fine_shapes_and_gradients():
    field = tiny_v6_field()
    rays_o = torch.tensor([[0.0, 0.0, -2.0]]).expand(3, -1)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1)
    result = render_rays(
        field,
        rays_o,
        rays_d,
        0.0,
        5.0,
        torch.zeros(3, 3),
        appearance=torch.zeros(3, 4),
        n_samples=8,
        n_importance=4,
        perturb=False,
    )
    assert result["coarse"]["weights"].shape == (3, 8)
    assert result["fine"]["weights"].shape == (3, 12)
    result["fine"]["rgb"].mean().backward()
    assert field.position_encoder.tables[-1].weight.grad is not None
