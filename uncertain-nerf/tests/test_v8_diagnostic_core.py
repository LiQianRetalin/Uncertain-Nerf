import torch

from v7.losses import uncertainty_color_nll
from v7.rendering import render_rays
from v8.diagnostic_model import DiagnosticRadianceField
from v8.diagnostic_trainer import assert_uq_gradient_isolation


def tiny_field(mode):
    return DiagnosticRadianceField(
        aabb=[[-1, -1, -1], [1, 1, 1]], mode=mode,
        hidden_dim=16, hidden_layers=2, hash_levels=3,
        hash_min_resolution=4, hash_max_resolution=16,
        hash_features=2, hash_log2_size=8, direction_frequencies=2,
        hash_backend="torch", mip_enabled=True)


def test_baseline_has_no_learnable_uncertainty_parameters():
    field = tiny_field("baseline")
    assert field.uncertainty_head is None
    assert not list(field.uncertainty_parameters())
    assert all("uncertainty" not in name for name, _ in field.named_parameters())


def test_matched_seed_gives_identical_reconstruction_initialization():
    torch.manual_seed(19)
    baseline = tiny_field("baseline")
    torch.manual_seed(19)
    v75 = tiny_field("v7_5")
    baseline_state = dict(baseline.reconstruction_named_parameters())
    v75_state = dict(v75.reconstruction_named_parameters())
    assert baseline_state.keys() == v75_state.keys()
    for name in baseline_state:
        assert torch.equal(baseline_state[name], v75_state[name]), name
    points = torch.rand(7, 3) * 1.5 - 0.75
    directions = torch.nn.functional.normalize(torch.randn(7, 3), dim=-1)
    assert torch.equal(baseline(points, directions)[..., :4],
                       v75(points, directions)[..., :4])


def test_uq_backward_has_no_reconstruction_gradients():
    field = tiny_field("v7_5")
    points = torch.rand(8, 3) * 1.5 - 0.75
    directions = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    raw = field(points, directions)
    predicted_rgb = torch.sigmoid(raw[..., :3])
    uncertainty = torch.sigmoid(raw[..., 4])
    loss_uq = uncertainty_color_nll(
        predicted_rgb, torch.full_like(predicted_rgb, 0.25), uncertainty)
    assert_uq_gradient_isolation(loss_uq, field.reconstruction_parameters())
    field.zero_grad(set_to_none=True)
    loss_uq.backward()
    reconstruction_gradients = [
        parameter.grad for parameter in field.reconstruction_parameters()]
    assert all(gradient is None or torch.count_nonzero(gradient) == 0
               for gradient in reconstruction_gradients)
    assert any(parameter.grad is not None
               and torch.count_nonzero(parameter.grad) > 0
               for parameter in field.uncertainty_parameters())


def test_uq_value_cannot_change_rgb_density_or_render_weights():
    field = tiny_field("v7_5")
    rays_o = torch.tensor([[0.0, 0.0, -2.0]]).expand(3, -1)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1)
    kwargs = dict(ray_radii=torch.full((3,), 0.01), n_samples=8,
                  n_importance=4, perturb=False)
    low = render_rays(
        field, rays_o, rays_d, 0.0, 5.0, torch.zeros(3, 3), **kwargs)["fine"]
    field.uncertainty_head[-1].bias.data.fill_(20.0)
    high = render_rays(
        field, rays_o, rays_d, 0.0, 5.0, torch.zeros(3, 3), **kwargs)["fine"]
    for key in ("rgb", "depth", "acc", "weights", "alpha"):
        assert torch.equal(low[key], high[key]), key
    assert torch.all(low["uncertainty"] < high["uncertainty"])
