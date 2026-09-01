import torch
import torch.nn.functional as F
from torch import nn

from puri_gs.ru_training import prepare_render_for_dino
from puri_gs.semantic_mask import (
    StaticResponsibilityHead,
    cosine_static_target,
    hard_static_mask,
    masked_photo_loss,
    semantic_mask_loss,
)


def test_raw_gaussian_render_is_clamped_only_for_detached_dino_input():
    render = torch.tensor(
        [[[[-0.25, 0.5, 1.25], [0.1, 0.9, 1.0]]]],
        requires_grad=True,
    )
    original = render.detach().clone()

    prepared = prepare_render_for_dino(render)

    assert prepared.requires_grad is False
    assert prepared.min().item() == 0.0
    assert prepared.max().item() == 1.0
    assert torch.equal(render.detach(), original)


class FrozenFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, image):
        pooled = F.adaptive_avg_pool2d(image, (4, 4))
        return pooled.repeat(1, 128, 1, 1) * self.scale


def _fake_ssim(render, target, padding):
    assert padding == "valid"
    return 1.0 - torch.abs(render - target).mean()


def test_photo_and_mask_backward_are_bidirectionally_isolated():
    gaussian = nn.Parameter(torch.randn(1, 8, 8, 3))
    target = torch.rand(1, 8, 8, 3)
    head = StaticResponsibilityHead()
    dino = FrozenFeatureModel().eval().requires_grad_(False)
    gt_features = torch.randn(1, 384, 4, 4)

    predicted = head(gt_features)
    safe = hard_static_mask(
        F.interpolate(predicted.detach(), size=(8, 8), mode="bilinear", align_corners=False)
    )
    photo, _, _ = masked_photo_loss(
        gaussian.sigmoid(), target, safe, fused_ssim_fn=_fake_ssim
    )
    photo.backward()
    assert gaussian.grad is not None and torch.isfinite(gaussian.grad).all()
    assert all(parameter.grad is None for parameter in head.parameters())
    assert all(parameter.grad is None for parameter in dino.parameters())

    gaussian.grad = None
    render_features = dino(gaussian.detach().sigmoid().permute(0, 3, 1, 2))
    similarity = cosine_static_target(gt_features, render_features)
    predicted = head(gt_features)
    zeros = torch.zeros(1, 1, 8, 8)
    ones = torch.ones(1, 1, 8, 8)
    mask_loss, _ = semantic_mask_loss(
        predicted,
        similarity,
        zeros,
        ones,
        image_size=(8, 8),
        step=100,
        head=head,
    )
    mask_loss.backward()
    assert gaussian.grad is None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )
    assert all(parameter.grad is None for parameter in dino.parameters())
