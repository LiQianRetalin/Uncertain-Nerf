"""Fixed-tensor formula and actual training integration checks, NOT replay tests."""
from pathlib import Path
import inspect
import json

import pytest
import torch

from puri_gs.ru_part_v3 import static_rescue_l1
from puri_gs.semantic_mask import masked_photo_loss
from puri_gs.config import load_experiment_config, trainer_method_args, validate_experiment_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher_cli(tmp_path, monkeypatch):
    """Use the wrapper's real CLI and launcher validation, without CUDA/assets."""
    import run_puri_gs as launcher
    from tools import ru_part_v3_screen as screen

    gsplat = tmp_path / "gsplat"
    (gsplat / "examples").mkdir(parents=True)
    (gsplat / "examples/simple_trainer.py").write_text("ru_v3_mode: str = ''\n")
    dino = tmp_path / "dinov2"
    dino.mkdir()
    (dino / "hubconf.py").touch()
    features = tmp_path / "features"
    features.mkdir()
    (features / "manifest.json").write_text("{}")
    weight = tmp_path / "weight.pth"
    weight.touch()
    cache = tmp_path / "tracks.pt"
    cache.touch()
    env = {
        "python": launcher.sys.executable, "gpu": 0,
        "gsplat": str(gsplat), "data": str(tmp_path / "garden"),
        "dino_repo": str(dino), "dino_weight": str(weight),
        "feature_cache": str(features), "track_cache": str(cache),
    }
    monkeypatch.setattr(screen, "output_root", lambda: tmp_path / "outputs")
    monkeypatch.setattr(launcher, "_git", lambda *args, **kwargs: "ru-part")
    monkeypatch.setattr(launcher, "_verify_gsplat", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "_verify_dataset", lambda *args, **kwargs: None)

    def invoke(stage="smoke-v3", extra=()):
        command = screen.command_for(stage, env)
        monkeypatch.setattr(launcher.sys, "argv", command[1:] + ["--dry-run", *extra])
        return launcher.main()

    return invoke


@pytest.mark.parametrize("stage", ["smoke-parent", "smoke-v3", "parent", "v3"])
def test_wrapper_training_cli_reaches_trainer_command(launcher_cli, capsys, stage, tmp_path):
    import shlex

    assert launcher_cli(stage) == 0
    printed = capsys.readouterr().out.splitlines()[-1]
    command = shlex.split(printed)
    assert command[0] == "CUDA_VISIBLE_DEVICES=0"
    assert command[command.index("--max_steps") + 1] == "30000"
    assert command[command.index("--ru_v3_stop_step") + 1] == (
        "599" if stage.startswith("smoke-") else "29999"
    )
    mode = "parent" if "parent" in stage else "v3"
    assert command[command.index("--ru_v3_mode") + 1] == mode
    assert ("--ru_v3_track_cache" in command) == (mode == "v3")
    if mode == "v3":
        assert command[command.index("--ru_v3_track_cache") + 1] == str(tmp_path / "tracks.pt")
    assert "--puri_gs_ru_part_enabled" not in command
    assert "--ru_part_track_cache" not in command
    assert not (tmp_path / "outputs").exists()  # dry-run neither trains nor creates results


@pytest.mark.parametrize("extra", [
    ["--replay-checkpoint", "replay.pt"],
    ["--non-scientific-smoke"],
    ["--diagnostic-stage", "SMOKE"],
    ["--diagnostic-stop-after-step", "50"],
    ["--diagnostic-roi-dir", "roi"],
    ["--diagnostic-roi-confirmed"],
])
def test_v3_cli_still_rejects_legacy_part_options(launcher_cli, extra):
    with pytest.raises(ValueError, match="V3 starts fresh|RU-PART-only"):
        launcher_cli(extra=extra)


def test_parent_cli_does_not_load_static_support(launcher_cli, tmp_path):
    with pytest.raises(ValueError, match="standard Parent does not load static support"):
        launcher_cli("smoke-parent", ["--track-cache", str(tmp_path / "tracks.pt")])


def test_v3_cli_still_requires_existing_static_cache(launcher_cli, tmp_path):
    with pytest.raises(FileNotFoundError, match="V3 static-track cache is missing"):
        launcher_cli(extra=["--track-cache", str(tmp_path / "missing.pt")])


def test_plain_ru_cli_still_rejects_track_cache(launcher_cli, tmp_path):
    config = load_experiment_config(ROOT / "configs/puri_gs_ru_part_v3_v3_garden30k.yaml")
    del config["v3_screening"]
    path = tmp_path / "plain_ru.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="RU-PART-only"):
        launcher_cli(extra=["--config", str(path)])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("step", [499, 500, 29999])
def test_exact_loss_and_rgb_gradient(dtype, step):
    pred = torch.tensor([[[[.3, -.2, .4], [.8, .1, -.7]],
                          [[-.1, .6, .9], [.2, -.5, .3]]]], dtype=dtype, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[[1., 0.], [0., 0.]]]], dtype=dtype, requires_grad=True)
    c = torch.tensor([[[[.7, 0.], [1., .25]]]], dtype=dtype, requires_grad=True)
    loss = .8 * static_rescue_l1(pred, target, mask, c, step=step)
    loss.backward()
    # Independent hand-specified four pixel weights, expanded over RGB.
    weights = pred.new_tensor([0., 0., 1., .25]).reshape(1, 2, 2, 1)
    enabled = float(step >= 500)
    expected_grad = .8 * enabled / 12 * weights * pred.detach().sign()
    expected_loss = .8 * enabled / 12 * (1.6 + .25 * 1.0)
    tolerance = 2e-7 if dtype == torch.float32 else 1e-14
    error = float((pred.grad - expected_grad).abs().max())
    print(f"dtype={dtype} step={step} max_rgb_gradient_error={error:.3g} loss_error={abs(loss.item()-expected_loss):.3g}")
    assert abs(loss.item() - expected_loss) <= tolerance
    torch.testing.assert_close(pred.grad, expected_grad, atol=tolerance, rtol=0)
    assert mask.grad is None and c.grad is None


@pytest.mark.parametrize("zero_case", ["C=0", "M=1"])
def test_zero_support_preserves_real_parent_expression(zero_case):
    pred = torch.linspace(.1, .9, 48).reshape(1, 4, 4, 3).requires_grad_()
    target = torch.zeros_like(pred)
    mask = torch.ones(1, 1, 4, 4) if zero_case == "M=1" else torch.zeros(1, 1, 4, 4)
    c = torch.ones_like(mask) if zero_case == "M=1" else torch.zeros_like(mask)
    calls = []
    def ssim_spy(a, b, **kwargs):
        calls.append((a.detach().clone(), b.detach().clone(), kwargs))
        return (a * b).mean()
    base, _, _ = masked_photo_loss(pred, target, mask, fused_ssim_fn=ssim_spy)
    added = .8 * static_rescue_l1(pred, target, mask, c, step=500)
    assert (base + added).item() == base.item()
    gradient, = torch.autograd.grad(added, pred)
    assert torch.count_nonzero(gradient) == 0
    assert len(calls) == 1 and calls[0][2] == {"padding": "valid"}
    assert torch.equal(calls[0][0], pred.detach().permute(0, 3, 1, 2) * mask)


def test_loss_has_no_alpha_argument():
    assert set(inspect.signature(static_rescue_l1).parameters) == {"render", "target", "final_mask", "support", "step"}


def test_fixed_profiles_and_disabled_paths():
    for mode in ("parent", "v3"):
        config = load_experiment_config(ROOT / f"configs/puri_gs_ru_part_v3_{mode}_garden30k.yaml")
        args = trainer_method_args(config)
        assert "--puri_gs_ru_part_enabled" not in args
        assert args[args.index("--ru_v3_mode") + 1] == mode
        assert config["bootstrap_switch_step"] == 20000 and config["total_steps"] == 30000
        config["gaussian_hard_cap"] = 2312002
        with pytest.raises(ValueError):
            validate_experiment_config(config)


def test_training_patch_uses_existing_parent_loss_and_backward():
    patch = (ROOT / "patches/gsplat_v1.5.3_ru_part_v3.patch").read_text()
    additions = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    assert "screening.add_loss(" in additions
    removed_backwards = sum(line == "-            loss.backward()" for line in patch.splitlines())
    added_backwards = sum(line == "+            loss.backward()" for line in patch.splitlines())
    assert added_backwards == removed_backwards  # instrumentation never adds a second backward
    assert "masked_photo_loss(" not in additions  # unchanged Parent implementation
    assert "max_steps = 600" not in additions
    assert "screening.stop" in additions


def test_shared_parameter_receives_added_gradient():
    parameter = torch.tensor(.4, requires_grad=True)
    render = parameter.expand(1, 2, 2, 3)
    loss = .8 * static_rescue_l1(render, torch.zeros_like(render), torch.zeros(1,1,2,2), torch.ones(1,1,2,2), step=500)
    loss.backward()
    assert parameter.grad.item() == pytest.approx(.8, abs=2e-7)


def test_report_never_promotes_unknown_or_failed_gates():
    from puri_gs.v3_report import classify
    yes = {"gate": True}
    assert classify(yes, yes, yes) == "PROMISING_SINGLE_SEED"
    assert classify(yes, yes, {"time": None}) == "COMPARISON_INCOMPLETE"
    assert classify(yes, yes, yes, complete=False) == "COMPARISON_INCOMPLETE"
    assert classify(yes, yes, yes, integrity=False) == "INVALID_RUN"
    assert classify({"PSNR": False}, yes, yes) == "QUALITY_RECOVERY_FAIL / NO_GO"
    assert classify(yes, {"LPIPS": False}, yes) == "PARENT_COMPARISON_FAIL"
    assert classify(yes, yes, {"count": False}) == "QUALITY_PASS_RESOURCE_FAIL"


def test_completion_rejects_wrong_step_or_corrupt_standard_checkpoint(tmp_path):
    from tools.ru_part_v3_screen import validate_checkpoint
    checkpoint = tmp_path / "checkpoint.pt"
    splats = {k: torch.ones(shape) for k,shape in {
        "means": (2,3), "scales": (2,3), "quats": (2,4), "opacities": (2,),
        "sh0": (2,1,3), "shN": (2,15,3),
    }.items()}
    torch.save({"step": 599, "splats": splats}, checkpoint)
    assert validate_checkpoint(checkpoint, 599) == 2
    with pytest.raises(ValueError):
        validate_checkpoint(checkpoint, 29999)
    splats["means"][0,0] = float("nan")
    torch.save({"step": 599, "splats": splats}, checkpoint)
    with pytest.raises(ValueError):
        validate_checkpoint(checkpoint, 599)
