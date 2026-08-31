import torch

from puri_gs.transient_layer import (
    SH_C0,
    FrozenStaticGaussians,
    TransientGaussianLayer,
    compose_premultiplied,
    static_transient_loss,
)


def _static_splats():
    return {
        "means": torch.zeros(2, 3),
        "scales": torch.zeros(2, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "opacities": torch.zeros(2),
        "sh0": torch.zeros(2, 1, 3),
        "shN": torch.zeros(2, 15, 3),
    }


def _fake_rasterization(**kwargs):
    height, width = kwargs["height"], kwargs["width"]
    rgb = kwargs["colors"][:, 0, :] * SH_C0 + 0.5
    color = rgb.mean(dim=0).view(1, 1, 1, 3).expand(1, height, width, 3)
    alpha = kwargs["opacities"].mean().view(1, 1, 1, 1).expand(1, height, width, 1)
    return alpha * color, alpha, {}


def _simple_ssim(predicted, target, padding):
    assert padding == "valid"
    return 1.0 - (predicted - target).square().mean()


def test_static_parameters_frozen_and_only_transient_logits_receive_gradients():
    static_module = FrozenStaticGaussians(_static_splats())
    assert all(not parameter.requires_grad for parameter in static_module.parameters())
    target = torch.rand(32, 32, 3)
    K = torch.tensor([[50.0, 0.0, 15.5], [0.0, 50.0, 15.5], [0.0, 0.0, 1.0]])
    camtoworld = torch.eye(4)
    layer = TransientGaussianLayer(target, K, camtoworld)
    transient_pre, alpha, _ = layer.render(K, camtoworld, _fake_rasterization)
    static_rgb = torch.rand_like(target).detach()
    composite = compose_premultiplied(static_rgb, transient_pre, alpha)
    loss, _ = static_transient_loss(composite, target, alpha, _simple_ssim)
    loss.backward()
    assert layer.color_logits.grad is not None
    assert torch.isfinite(layer.color_logits.grad).all()
    assert float(layer.color_logits.grad.abs().sum()) > 0.0
    assert layer.opacity_logits.grad is not None
    assert torch.isfinite(layer.opacity_logits.grad).all()
    assert float(layer.opacity_logits.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in static_module.parameters())
    assert layer.means.grad is None
    assert layer.quaternions.grad is None
    assert layer.log_scales.grad is None
