import math

import numpy as np

from run_nerf import config_parser
from test_scene import create_scene
from v5.trainer import Trainer


def smoke_args(scene_dir, log_dir):
    return config_parser().parse_args(
        [
            "--expname", "smoke",
            "--basedir", str(log_dir),
            "--datadir", str(scene_dir),
            "--device", "cpu",
            "--factor", "2",
            "--llffhold", "2",
            "--netdepth", "2",
            "--netwidth", "8",
            "--hash_levels", "2",
            "--hash_min_resolution", "4",
            "--hash_max_resolution", "8",
            "--hash_features", "2",
            "--hash_log2_size", "6",
            "--multires_views", "1",
            "--N_rand", "4",
            "--N_samples", "4",
            "--N_importance", "2",
            "--N_iters", "2",
            "--chunk", "64",
            "--netchunk", "128",
            "--teacher_ray_ratio", "0.5",
            "--teacher_views", "2",
            "--mc_samples", "2",
            "--prior_ray_ratio", "0.5",
            "--loss_calibration_steps", "1",
            "--robust_anneal_steps", "2",
            "--depth_scale_warmup", "0",
            "--depth_scale_interval", "1",
            "--i_weights", "2",
            "--i_print", "1",
        ]
    )


def test_cpu_training_checkpoint_and_render_forward(tmp_path):
    scene_dir = tmp_path / "scene"
    create_scene(scene_dir)
    args = smoke_args(scene_dir, tmp_path / "logs")
    trainer = Trainer(args)
    stats = trainer.training_step(1)
    assert all(math.isfinite(stats[key]) for key in ("loss", "color", "geo", "u", "reg", "psnr"))
    stats_second = trainer.training_step(2)
    assert all(math.isfinite(stats_second[key]) for key in ("loss", "color", "geo", "u", "reg", "psnr"))
    trainer.save_checkpoint(2)
    frame = trainer._render_pose(trainer.scene.poses[0], trainer.scene.intrinsics[0])
    assert frame["rgb"].shape == (8, 8, 3)
    assert frame["depth"].shape == (8, 8)
    assert frame["uncertainty"].shape == (8, 8)
    resumed_args = smoke_args(scene_dir, tmp_path / "logs")
    resumed = Trainer(resumed_args)
    assert resumed.start_step == 2
    assert math.isclose(resumed.depth_aligner.scale, trainer.depth_aligner.scale)
    pose_file = tmp_path / "one_pose.npy"
    np.save(pose_file, resumed.scene.poses[:1].numpy())
    resumed.args.render_only = True
    resumed.args.render_poses = str(pose_file)
    output_dir = resumed.render()
    for name in (
        "rgb_000.png", "depth_000.npy", "depth_000.png",
        "uncertainty_000.npy", "uncertainty_000.png",
        "rgb.mp4", "depth.mp4", "uncertainty.mp4",
    ):
        assert (tmp_path / "logs" / "smoke" / f"render_custom_{resumed.start_step:06d}" / name).exists()


def test_cpu_training_step_with_simple_radial_camera(tmp_path):
    scene_dir = tmp_path / "radial_scene"
    create_scene(scene_dir, camera_model="SIMPLE_RADIAL", radial_k=-0.08)
    args = smoke_args(scene_dir, tmp_path / "radial_logs")
    trainer = Trainer(args)
    stats = trainer.training_step(1)
    assert all(
        math.isfinite(stats[key])
        for key in ("loss", "color", "geo", "u", "reg", "psnr")
    )
