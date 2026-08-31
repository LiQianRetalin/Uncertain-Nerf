import argparse
from pathlib import Path

import torch

from puri_gs.config import load_experiment_config, trainer_method_args
from puri_gs.cvtr import cvtr_weighted_l1
from run_puri_gs import _build_command


ROOT = Path(__file__).resolve().parents[1]


def _config(name: str):
    return load_experiment_config(ROOT / "configs" / name)


def test_continuation_profiles_differ_only_by_fixed_cvtr_loss():
    control = _config("puri_gs_b1_continuation.yaml")
    cvtr = _config("puri_gs_cvtr.yaml")
    assert control["training"] == cvtr["training"]
    assert control["strategy"] == cvtr["strategy"]
    assert control["responsibility"] == cvtr["responsibility"]
    assert control["continuation"] == cvtr["continuation"] == {
        "enabled": True,
        "source_step": 9999,
        "target_step": 14999,
        "additional_steps": 5000,
    }
    assert not control["cvtr"]["enabled"]
    assert cvtr["cvtr"]["enabled"]
    assert {k: v for k, v in control["cvtr"].items() if k != "enabled"} == {
        k: v for k, v in cvtr["cvtr"].items() if k != "enabled"
    }
    assert "--cvtr_enabled" not in trainer_method_args(control)
    assert "--cvtr_enabled" in trainer_method_args(cvtr)


def test_launcher_builds_fixed_10k_to_15k_commands(tmp_path: Path):
    checkpoint = tmp_path / "ckpt_9999_rank0.pt"
    checkpoint.touch()
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    args = argparse.Namespace(
        data_factor=None,
        train_keyword=None,
        test_keyword=None,
        checkpoint=None,
        resume_checkpoint=checkpoint,
        cvtr_mask_dir=mask_dir,
        max_steps=1,
    )
    command = _build_command(
        args,
        _config("puri_gs_cvtr.yaml"),
        tmp_path / "gsplat",
        tmp_path / "data",
        tmp_path / "result",
    )
    assert command[command.index("--max_steps") + 1] == "15000"
    assert command[command.index("--save_steps") + 1] == "15000"
    assert command[command.index("--resume_ckpt") + 1] == str(checkpoint.resolve())
    assert command[command.index("--cvtr_mask_dir") + 1] == str(mask_dir.resolve())


def test_checkpoint_load_fresh_optimizer_forward_backward_and_save(tmp_path: Path):
    source_path = tmp_path / "source.pt"
    source = {
        "step": 9999,
        "splats": {"rgb": torch.nn.Parameter(torch.zeros(1, 2, 2, 3)).detach()},
    }
    torch.save(source, source_path)
    loaded = torch.load(source_path, map_location="cpu", weights_only=True)
    assert set(loaded) == {"step", "splats"}
    assert not set(loaded).intersection({"optimizer", "strategy", "rng"})
    parameter = torch.nn.Parameter(loaded["splats"]["rgb"].clone())
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    target = torch.ones_like(parameter)
    transient_mask = torch.zeros(1, 2, 2, dtype=torch.bool)
    transient_mask[:, 0, 0] = True
    loss, q = cvtr_weighted_l1(
        parameter,
        target,
        enabled=True,
        transient_mask=transient_mask,
    )
    assert q is not None and not q.requires_grad
    loss.backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    optimizer.step()
    target_path = tmp_path / "target.pt"
    torch.save({"step": 14999, "splats": {"rgb": parameter.detach()}}, target_path)
    restored = torch.load(target_path, map_location="cpu", weights_only=True)
    assert restored["step"] == 14999
    assert torch.isfinite(restored["splats"]["rgb"]).all()
