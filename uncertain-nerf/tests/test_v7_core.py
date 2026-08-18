import torch

from v7.losses import termination_distribution_loss, uncertainty_color_nll
from v7.model import RadianceFieldV7
from v7.occupancy import OccupancyGrid
from v7.rendering import render_rays, volume_integrate


def tiny_v7_field(mip_enabled=True):
    return RadianceFieldV7(
        aabb=[[-1, -1, -1], [1, 1, 1]], hidden_dim=16, hidden_layers=2,
        hash_levels=3, hash_min_resolution=4, hash_max_resolution=16,
        hash_features=2, hash_log2_size=8, direction_frequencies=2,
        appearance_dim=0, hash_backend="torch", mip_enabled=mip_enabled)


def test_uncertainty_loss_only_updates_uncertainty_head():
    field = tiny_v7_field()
    rays_o = torch.tensor([[0.0, 0.0, -2.0]]).expand(3, -1)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1)
    fine = render_rays(
        field, rays_o, rays_d, 0.0, 5.0, torch.zeros(3, 3),
        ray_radii=torch.full((3,), 0.01), n_samples=8, n_importance=4,
        perturb=False)["fine"]
    loss = uncertainty_color_nll(fine["rgb"], torch.full((3, 3), 0.2),
                                 fine["uncertainty"])
    loss.backward()
    changed = {name for name, parameter in field.named_parameters()
               if parameter.grad is not None and parameter.grad.abs().sum() > 0}
    assert changed
    assert all(name.startswith("uncertainty_head.") for name in changed)


def test_uncertainty_never_changes_rgb_density_or_opacity():
    raw_low = torch.zeros(2, 8, 5); raw_low[..., 3] = 1.0; raw_low[..., 4] = -8.0
    raw_high = raw_low.clone(); raw_high[..., 4] = 8.0
    z = torch.linspace(0.1, 0.9, 8).expand(2, -1)
    d = torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1)
    args = (z, d, torch.zeros(2, 3), torch.zeros(2), torch.ones(2))
    low, high = volume_integrate(raw_low, *args), volume_integrate(raw_high, *args)
    assert torch.allclose(low["rgb"], high["rgb"])
    assert torch.allclose(low["acc"], high["acc"])
    assert torch.all(low["uncertainty"] < high["uncertainty"])


def test_termination_distribution_prefers_mass_at_prior_depth():
    z = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    good = torch.tensor([[0.01, 0.97, 0.01, 0.0]], requires_grad=True)
    bad = torch.tensor([[0.97, 0.01, 0.01, 0.0]], requires_grad=True)
    terminal = torch.tensor([0.01])
    mask = torch.tensor([True]); prior = torch.tensor([2.0])
    good_loss, _ = termination_distribution_loss(good, terminal, z, prior, mask, 1.0)
    bad_loss, _ = termination_distribution_loss(bad, terminal, z, prior, mask, 1.0)
    assert good_loss < bad_loss
    good_loss.backward()
    assert good.grad is not None


def test_mip_footprint_suppresses_fine_hash_levels():
    field = tiny_v7_field()
    points = torch.zeros(2, 3)
    small = field.position_encoder(points, torch.zeros(2))
    large = field.position_encoder(points, torch.full((2,), 2.0))
    assert large[:, -2:].abs().max() < small[:, -2:].abs().max()


def test_active_occupancy_grid_keeps_sparse_query_gradients():
    field = tiny_v7_field()
    grid = OccupancyGrid(field.aabb, resolution=4, threshold=0.01, warmup_steps=0)
    grid.values.fill_(1.0); grid.active = True
    rays_o = torch.tensor([[0.0, 0.0, -2.0]])
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])
    fine = render_rays(
        field, rays_o, rays_d, 0.0, 5.0, torch.zeros(1, 3),
        ray_radii=torch.full((1,), 0.01), n_samples=8, n_importance=0,
        perturb=False, occupancy_grid=grid)["fine"]
    fine["rgb"].mean().backward()
    assert field.trunk[0].weight.grad is not None


def test_sparse_query_accepts_autocast_output_dtype():
    field = tiny_v7_field()
    rays_o = torch.tensor([[0.0, 0.0, -2.0]])
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        fine = render_rays(
            field, rays_o, rays_d, 0.0, 5.0, torch.zeros(1, 3),
            ray_radii=torch.full((1,), 0.01), n_samples=8, n_importance=0,
            perturb=False)["fine"]
    assert fine["rgb"].dtype == torch.float32
    assert torch.isfinite(fine["rgb"]).all()
