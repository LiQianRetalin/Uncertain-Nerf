import json
import math

from run_nerf_a0 import config_parser, validate_args
from test_scene import create_scene
from a0.trainer import A0Trainer, CHECKPOINT_VERSION


def a0_args(scene_dir, log_dir, expname):
    args = config_parser().parse_args([
        "--expname", expname, "--basedir", str(log_dir),
        "--datadir", str(scene_dir), "--device", "cpu",
        "--factor", "2", "--llffhold", "2", "--val_every", "0",
        "--netdepth", "2", "--netwidth", "16",
        "--netdepth_fine", "2", "--netwidth_fine", "16",
        "--multires", "2", "--multires_views", "1",
        "--use_viewdirs", "--N_rand", "8", "--N_samples", "4",
        "--N_importance", "2", "--N_iters", "2", "--chunk", "16",
        "--netchunk", "32", "--raw_noise_std", "0",
        "--i_print", "1", "--i_eval", "0", "--i_weights", "2"])
    validate_args(args)
    return args


def test_a0_training_step_checkpoint_resume_and_render(tmp_path):
    scene_dir = tmp_path / "scene-a0"
    create_scene(scene_dir)
    log_dir = tmp_path / "logs-a0"
    trainer = A0Trainer(a0_args(scene_dir, log_dir, "a0-smoke"))
    stats = trainer.training_step(1)
    assert all(math.isfinite(stats[key]) for key in (
        "loss", "loss_rgb_fine", "loss_rgb_coarse", "psnr",
        "opacity_mean", "depth_mean", "gradient_norm_density",
        "gradient_norm_color"))
    assert stats["gradient_norm_uncertainty"] == 0
    checkpoint = trainer.save_checkpoint(1)
    resumed = A0Trainer(a0_args(scene_dir, log_dir, "a0-smoke"))
    assert resumed.start_step == 1
    assert resumed.checkpoint_state(1)["version"] == CHECKPOINT_VERSION

    render_args = a0_args(scene_dir, log_dir, "a0-review")
    render_args.render_only = True
    render_args.render_split = "test"
    render_args.ft_path = checkpoint
    renderer = A0Trainer(render_args)
    output = renderer.render()
    metrics_dir = log_dir / "a0-review" / "render_test_000001"
    assert output == str(metrics_dir)
    assert (metrics_dir / "rgb_000.png").exists()
    assert (metrics_dir / "gt_rgb_000.png").exists()
    assert (metrics_dir / "acc_000.npy").exists()
    assert (metrics_dir / "uncertainty_000.npy").exists()
    with open(log_dir / "a0-review" / "render_args.json", encoding="utf-8") as handle:
        assert json.load(handle)["render_only"] is True
