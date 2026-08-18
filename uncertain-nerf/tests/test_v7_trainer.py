import math

from run_nerf_v7 import config_parser, validate_args
from test_scene import create_scene
from v7.trainer import TrainerV7


def v7_smoke_args(scene_dir, log_dir):
    args = config_parser().parse_args([
        "--expname", "smoke-v7", "--basedir", str(log_dir), "--datadir", str(scene_dir),
        "--device", "cpu", "--factor", "2", "--llffhold", "2", "--val_every", "0",
        "--netdepth", "2", "--netwidth", "8", "--hash_levels", "2",
        "--hash_min_resolution", "4", "--hash_max_resolution", "8",
        "--hash_features", "2", "--hash_log2_size", "6", "--multires_views", "1",
        "--N_rand", "8", "--N_samples", "4", "--N_importance", "2", "--N_iters", "2",
        "--chunk", "64", "--netchunk", "128", "--teacher_ray_ratio", "0.5",
        "--teacher_interval", "1", "--teacher_views", "2", "--prior_ray_ratio", "0.25",
        "--edge_ray_ratio", "0.25", "--uncertainty_ray_ratio", "0.25",
        "--depth_scale_warmup", "0", "--depth_scale_interval", "1",
        "--i_eval", "0", "--i_weights", "2", "--i_print", "1"])
    validate_args(args)
    return args


def test_v7_cpu_training_checkpoint_and_render(tmp_path):
    scene_dir = tmp_path / "scene-v7"; create_scene(scene_dir)
    args = v7_smoke_args(scene_dir, tmp_path / "logs-v7")
    trainer = TrainerV7(args)
    assert trainer.field.appearance_dim == 0
    stats = trainer.training_step(1)
    assert stats["teacher_count"] == 0
    assert all(math.isfinite(stats[key]) for key in
               ("loss", "color", "nll", "geo", "teacher", "spatial", "psnr"))
    trainer.save_checkpoint(1)
    frame = trainer._render_pose(trainer.scene.poses[0], trainer.scene.intrinsics[0])
    assert frame["rgb"].shape == (8, 8, 3)
    assert frame["depth"].shape == frame["uncertainty"].shape == (8, 8)
    resumed = TrainerV7(v7_smoke_args(scene_dir, tmp_path / "logs-v7"))
    assert resumed.start_step == 1
    assert resumed.checkpoint_state(1)["version"] == "uncertain-nerf-v7-1"
