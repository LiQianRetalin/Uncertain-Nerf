import json

import torch
from torch import nn

from puri_gs.dino_features import (
    COARSE_GRID,
    FEATURE_DIM,
    FEATURE_EXTRACTOR_VERSION,
    FINE_GRID,
    FeatureCache,
    extract_patch_grid,
    preprocess_rgb,
)


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward_features(self, images):
        grid = images.shape[-1] // 14
        pooled = torch.nn.functional.avg_pool2d(images, kernel_size=14, stride=14)
        tokens = pooled.repeat(1, FEATURE_DIM // 3, 1, 1)
        return {"x_norm_patchtokens": tokens.flatten(2).transpose(1, 2)}


def test_fixed_preprocess_and_feature_shapes_are_deterministic():
    model = FakeDINO().eval().requires_grad_(False)
    image = torch.linspace(0, 1, 31 * 47 * 3).reshape(1, 31, 47, 3)
    assert preprocess_rgb(image, 224).shape == (1, 3, 224, 224)
    coarse_a = extract_patch_grid(model, image, COARSE_GRID)
    coarse_b = extract_patch_grid(model, image, COARSE_GRID)
    fine = extract_patch_grid(model, image, FINE_GRID)
    assert coarse_a.shape == (1, FEATURE_DIM, 16, 16)
    assert fine.shape == (1, FEATURE_DIM, 36, 36)
    assert torch.equal(coarse_a, coarse_b)
    assert torch.isfinite(coarse_a).all() and torch.isfinite(fine).all()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0


def test_feature_cache_validates_mapping_sha_shapes_and_reuses_load(tmp_path):
    payload = {
        "coarse": torch.zeros(FEATURE_DIM, 16, 16),
        "fine": torch.ones(FEATURE_DIM, 36, 36),
    }
    torch.save(payload, tmp_path / "0000_frame.pt")
    manifest = {
        "scene": "unit",
        "model_name": "dinov2_vits14_reg",
        "model_weight_sha256": "abc",
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "images": [{"image_name": "frame.png", "file": "0000_frame.pt"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    cache = FeatureCache(tmp_path, expected_weight_sha256="abc")
    first = cache.load("frame.png", 16)
    second = cache.load("frame.png", 16)
    assert first.data_ptr() == second.data_ptr()
    assert cache.load("frame.png", 36).shape == (FEATURE_DIM, 36, 36)

