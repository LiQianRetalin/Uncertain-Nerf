import math

import pytest
import torch

from run_nerf_v8 import config_parser, validate_args
from test_scene import create_scene
from v8.diagnostic_trainer import DiagnosticTrainer


def diagnostic_args(scene_dir, log_dir, mode, expname):
    args = config_parser().parse_args([
        "--mode", mode, "--expname", expname, "--basedir", str(log_dir),
        "--datadir", str(scene_dir), "--device", "cpu", "--factor", "2",
        "--llffhold", "2", "--val_every", "0", "--netdepth", "2",
        "--netwidth", "8", "--hash_levels", "2",
        "--hash_min_resolution", "4", "--hash_max_resolution", "8",
        "--hash_features", "2", "--hash_log2_size", "6",
        "--multires_views", "1", "--N_rand", "8", "--N_samples", "4",
        "--N_importance", "2", "--N_iters", "2", "--chunk", "64",
        "--netchunk", "128", "--teacher_ray_ratio", "0.5",
        "--teacher_interval", "1", "--teacher_views", "2",
        "--prior_ray_ratio", "0.25", "--edge_ray_ratio", "0.25",
        "--depth_scale_warmup", "0", "--depth_scale_interval", "1",
        "--uq_isolation_check_interval", "1", "--i_eval", "0",
        "--i_weights", "2", "--i_diagnostics", "0", "--i_print", "1"])
    validate_args(args)
    return args


@pytest.mark.parametrize("step", [1, 10000])
def test_baseline_and_v75_one_step_reconstruction_match(tmp_path, step):
    scene_dir = tmp_path / "scene-v8"
    create_scene(scene_dir)
    baseline_args = diagnostic_args(
        scene_dir, tmp_path / "logs-v8", "baseline", f"baseline-{step}")
    baseline_args.N_iters = max(2, step)
    baseline = DiagnosticTrainer(baseline_args)
    baseline_stats = baseline.training_step(step)
    baseline_rng_state = torch.random.get_rng_state().clone()
    baseline_parameters = {
        name: parameter.detach().clone()
        for name, parameter in baseline.field.reconstruction_named_parameters()}
    baseline_background = baseline.background_logit.detach().clone()

    v75_args = diagnostic_args(
        scene_dir, tmp_path / "logs-v8", "v7_5", f"v7_5-{step}")
    v75_args.N_iters = max(2, step)
    v75 = DiagnosticTrainer(v75_args)
    v75_stats = v75.training_step(step)
    assert torch.equal(baseline_rng_state, torch.random.get_rng_state())
    assert baseline_stats["loss_reconstruction"] == v75_stats["loss_reconstruction"]
    assert baseline_stats["opacity_mean"] == v75_stats["opacity_mean"]
    for name, parameter in v75.field.reconstruction_named_parameters():
        assert torch.equal(baseline_parameters[name], parameter), name
    assert torch.equal(baseline_background, v75.background_logit)
    assert v75_stats["gradient_norm_uncertainty"] > 0
    assert baseline_stats["gradient_norm_uncertainty"] == 0
    assert all(math.isfinite(v75_stats[key]) for key in (
        "loss", "loss_reconstruction", "loss_uq", "psnr",
        "gradient_norm_density", "gradient_norm_color",
        "gradient_norm_uncertainty"))


def test_v8_diagnostic_checkpoint_resume_and_artifacts(tmp_path):
    scene_dir = tmp_path / "scene-v8-checkpoint"
    create_scene(scene_dir)
    args = diagnostic_args(
        scene_dir, tmp_path / "logs-v8", "v7_5", "resume-v7_5")
    trainer = DiagnosticTrainer(args)
    trainer.training_step(1)
    trainer.save_checkpoint(1)
    output = trainer.save_diagnostics(1)
    assert (tmp_path / "logs-v8" / "resume-v7_5" /
            "diagnostics_000001" / "diagnostic_summary.json").exists()
    assert output.endswith("diagnostics_000001")
    resumed = DiagnosticTrainer(diagnostic_args(
        scene_dir, tmp_path / "logs-v8", "v7_5", "resume-v7_5"))
    assert resumed.start_step == 1
    state = resumed.checkpoint_state(1)
    assert state["version"] == "puri-nerf-v8-diagnostic-1"
    assert state["mode"] == "v7_5"
