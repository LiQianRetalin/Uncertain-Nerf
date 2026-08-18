"""Training and rendering entry point for Uncertain-NeRF V7."""

import configargparse

from v7.trainer import TrainerV7


def config_parser():
    p = configargparse.ArgumentParser()
    p.add_argument("--config", is_config_file=True)
    p.add_argument("--expname", required=True); p.add_argument("--basedir", default="./logs-v7")
    p.add_argument("--datadir", required=True); p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--factor", type=int, default=2)
    p.add_argument("--llffhold", type=int, default=8); p.add_argument("--val_every", type=int, default=8)
    p.add_argument("--aabb_scale", type=float, default=1.25)
    p.add_argument("--sampling_space", choices=["linear", "disparity"], default="disparity")
    p.add_argument("--netdepth", type=int, default=2); p.add_argument("--netwidth", type=int, default=64)
    p.add_argument("--density_bias", type=float, default=-1.0)
    p.add_argument("--appearance_dim", type=int, default=0)
    p.add_argument("--hash_backend", choices=["torch", "tcnn"], default="torch")
    p.add_argument("--hash_levels", type=int, default=16); p.add_argument("--hash_min_resolution", type=int, default=16)
    p.add_argument("--hash_max_resolution", type=int, default=2048); p.add_argument("--hash_features", type=int, default=2)
    p.add_argument("--hash_log2_size", type=int, default=19); p.add_argument("--multires_views", type=int, default=4)
    p.add_argument("--disable_mip", action="store_true")
    p.add_argument("--N_rand", type=int, default=1024); p.add_argument("--N_samples", type=int, default=64)
    p.add_argument("--N_importance", type=int, default=64); p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--netchunk", type=int, default=65536); p.add_argument("--N_iters", type=int, default=100000)
    p.add_argument("--teacher_start_step", type=int, default=10000)
    p.add_argument("--teacher_ray_ratio", type=float, default=0.1); p.add_argument("--teacher_interval", type=int, default=4)
    p.add_argument("--teacher_views", type=int, default=4); p.add_argument("--teacher_photometric_weight", type=float, default=0.7)
    p.add_argument("--teacher_visibility_samples", type=int, default=32)
    p.add_argument("--teacher_visibility_relative_tolerance", type=float, default=0.05)
    p.add_argument("--teacher_visibility_absolute_tolerance", type=float, default=0.01)
    p.add_argument("--teacher_visibility_min_acc", type=float, default=0.1)
    p.add_argument("--spatial_epsilon", type=float, default=0.005)
    p.add_argument("--tukey_start", type=float, default=6.0); p.add_argument("--tukey_end", type=float, default=4.685)
    p.add_argument("--robust_anneal_steps", type=int, default=20000)
    p.add_argument("--prior_ray_ratio", type=float, default=0.25)
    p.add_argument("--edge_ray_ratio", type=float, default=0.25)
    p.add_argument("--uncertainty_ray_ratio", type=float, default=0.25)
    p.add_argument("--uncertainty_sampling_ema", type=float, default=0.1)
    p.add_argument("--sampling_probability_floor", type=float, default=0.05)
    p.add_argument("--depth_scale_warmup", type=int, default=1000); p.add_argument("--depth_scale_interval", type=int, default=500)
    p.add_argument("--depth_scale_ema", type=float, default=0.1)
    p.add_argument("--depth_distribution_relative_sigma", type=float, default=0.01)
    p.add_argument("--depth_distribution_interval_sigma", type=float, default=1.0)
    p.add_argument("--variance_min", type=float, default=1e-4); p.add_argument("--variance_max", type=float, default=0.05)
    p.add_argument("--coarse_loss_weight", type=float, default=0.1); p.add_argument("--lambda_nll", type=float, default=0.05)
    p.add_argument("--lambda_geo", type=float, default=0.1); p.add_argument("--lambda_teacher", type=float, default=0.1)
    p.add_argument("--lambda_spatial", type=float, default=0.0); p.add_argument("--lambda_distortion", type=float, default=1e-3)
    p.add_argument("--use_occupancy_grid", action="store_true")
    p.add_argument("--occupancy_resolution", type=int, default=128); p.add_argument("--occupancy_threshold", type=float, default=0.01)
    p.add_argument("--occupancy_decay", type=float, default=0.95); p.add_argument("--occupancy_warmup", type=int, default=10000)
    p.add_argument("--occupancy_update_interval", type=int, default=16)
    p.add_argument("--occupancy_update_samples", type=int, default=262144)
    p.add_argument("--lrate", type=float, default=1e-2); p.add_argument("--lrate_background", type=float, default=1e-3)
    p.add_argument("--lrate_min", type=float, default=1e-4); p.add_argument("--lr_warmup", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0); p.add_argument("--amp", action="store_true")
    p.add_argument("--no_reload", action="store_true"); p.add_argument("--ft_path", default=None)
    p.add_argument("--i_print", type=int, default=100); p.add_argument("--i_eval", type=int, default=5000)
    p.add_argument("--eval_max_views", type=int, default=0); p.add_argument("--i_weights", type=int, default=10000)
    p.add_argument("--target_psnr", type=float, default=25.0); p.add_argument("--render_only", action="store_true")
    p.add_argument("--render_split", choices=["path", "test"], default="path"); p.add_argument("--render_poses", default=None)
    return p


def validate_args(args):
    if args.appearance_dim != 0:
        raise ValueError("V7 static LLFF requires appearance_dim=0")
    for name in ("teacher_ray_ratio", "prior_ray_ratio", "edge_ray_ratio", "uncertainty_ray_ratio"):
        if not 0 <= getattr(args, name) <= 1: raise ValueError(f"{name} must be in [0, 1]")
    if args.prior_ray_ratio + args.edge_ray_ratio + args.uncertainty_ray_ratio > 1:
        raise ValueError("prior/edge/uncertainty ray ratios must sum to at most 1")
    if args.teacher_start_step < 10000:
        raise ValueError("V7 teacher_start_step must be at least 10000")
    if args.lambda_spatial != 0:
        raise ValueError("V7 correction phase requires lambda_spatial=0")
    if args.variance_min <= 0 or args.variance_max <= args.variance_min:
        raise ValueError("Require 0 < variance_min < variance_max")
    if args.factor < 1: raise ValueError("factor must be positive")


def main():
    args = config_parser().parse_args(); validate_args(args); TrainerV7(args).train()


if __name__ == "__main__": main()
