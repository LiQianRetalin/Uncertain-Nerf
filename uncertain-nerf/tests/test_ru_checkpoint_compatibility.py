import torch


def test_standard_checkpoint_contains_only_existing_gaussian_contract(tmp_path):
    path = tmp_path / "ckpt_29999_rank0.pt"
    splats = {
        "means": torch.zeros(2, 3),
        "scales": torch.zeros(2, 3),
        "quats": torch.zeros(2, 4),
        "opacities": torch.zeros(2),
        "sh0": torch.zeros(2, 1, 3),
        "shN": torch.zeros(2, 15, 3),
    }
    torch.save({"step": 29999, "splats": splats}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"step", "splats"}
    assert checkpoint["step"] == 29999
    assert "mask_head" not in checkpoint
    assert "dino" not in checkpoint
    assert "residual_histogram" not in checkpoint

