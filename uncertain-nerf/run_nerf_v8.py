"""Entry point for the staged PURI-NeRF V8 implementation."""

import configargparse

from v8.diagnostic_trainer import DiagnosticTrainer


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True)
    parser.add_argument("--mode", choices=["baseline", "v7_5"], required=True)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--basedir", default="./logs-v8")
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--val_every", type=int, default=8)
    parser.add_argument("--aabb_scale", type=float, default=1.25)
    parser.add_argument("--sampling_space", choices=["linear", "disparity"],
                        default="disparity")
    parser.add_argument("--netdepth", type=int, default=2)
    parser.add_argument("--netwidth", type=int, default=64)
    parser.add_argument("--density_bias", type=float, default=-1.0)
    parser.add_argument("--hash_backend", choices=["torch", "tcnn"],
                        default="torch")
    parser.add_argument("--hash_levels", type=int, default=16)
    parser.add_argument("--hash_min_resolution", type=int, default=16)
    parser.add_argument("--hash_max_resolution", type=int, default=2048)
    parser.add_argument("--hash_features", type=int, default=2)
    parser.add_argument("--hash_log2_size", type=int, default=19)
    parser.add_argument("--multires_views", type=int, default=4)
    parser.add_argument("--disable_mip", action="store_true")
    parser.add_argument("--N_rand", type=int, default=1024)
    parser.add_argument("--N_samples", type=int, default=64)
    parser.add_argument("--N_importance", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--netchunk", type=int, default=65536)
    parser.add_argument("--N_iters", type=int, default=100000)

    # Both modes use this exact fixed sampler.  No UQ-derived probability exists.
    parser.add_argument("--prior_ray_ratio", type=float, default=0.25)
    parser.add_argument("--edge_ray_ratio", type=float, default=0.25)
    parser.add_argument("--sampling_probability_floor", type=float, default=0.05)

    parser.add_argument("--teacher_start_step", type=int, default=10000)
    parser.add_argument("--teacher_ray_ratio", type=float, default=0.1)
    parser.add_argument("--teacher_interval", type=int, default=4)
    parser.add_argument("--teacher_views", type=int, default=4)
    parser.add_argument("--teacher_photometric_weight", type=float, default=0.7)
    parser.add_argument("--teacher_visibility_samples", type=int, default=32)
    parser.add_argument("--teacher_visibility_relative_tolerance", type=float,
                        default=0.05)
    parser.add_argument("--teacher_visibility_absolute_tolerance", type=float,
                        default=0.01)
    parser.add_argument("--teacher_visibility_min_acc", type=float, default=0.1)
    parser.add_argument("--tukey_start", type=float, default=6.0)
    parser.add_argument("--tukey_end", type=float, default=4.685)
    parser.add_argument("--robust_anneal_steps", type=int, default=20000)
    parser.add_argument("--variance_min", type=float, default=1.0e-4)
    parser.add_argument("--variance_max", type=float, default=0.05)
    parser.add_argument("--lambda_uq_nll", type=float, default=0.05)
    parser.add_argument("--lambda_uq_teacher", type=float, default=0.1)
    parser.add_argument("--uq_isolation_check_interval", type=int, default=100)

    parser.add_argument("--depth_scale_warmup", type=int, default=1000)
    parser.add_argument("--depth_scale_interval", type=int, default=500)
    parser.add_argument("--depth_scale_ema", type=float, default=0.1)
    parser.add_argument("--depth_distribution_relative_sigma", type=float,
                        default=0.01)
    parser.add_argument("--depth_distribution_interval_sigma", type=float,
                        default=1.0)
    parser.add_argument("--coarse_loss_weight", type=float, default=0.1)
    parser.add_argument("--lambda_geo", type=float, default=0.1)
    parser.add_argument("--lambda_distortion", type=float, default=1.0e-3)
    parser.add_argument("--lrate", type=float, default=1.0e-2)
    parser.add_argument("--lrate_background", type=float, default=1.0e-3)
    parser.add_argument("--lrate_uq", type=float, default=1.0e-3)
    parser.add_argument("--lrate_min", type=float, default=1.0e-4)
    parser.add_argument("--lrate_uq_min", type=float, default=1.0e-5)
    parser.add_argument("--lr_warmup", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_reload", action="store_true")
    parser.add_argument("--ft_path", default=None)
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_eval", type=int, default=5000)
    parser.add_argument("--eval_max_views", type=int, default=0)
    parser.add_argument("--i_weights", type=int, default=10000)
    parser.add_argument("--i_diagnostics", type=int, default=10000)
    parser.add_argument("--target_psnr", type=float, default=25.0)
    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--render_split",
                        choices=["path", "train", "val", "test"],
                        default="path")
    parser.add_argument("--render_poses", default=None)
    return parser


def validate_args(args):
    for name in ("teacher_ray_ratio", "prior_ray_ratio", "edge_ray_ratio"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.prior_ray_ratio + args.edge_ray_ratio > 1:
        raise ValueError("prior_ray_ratio + edge_ray_ratio must be at most 1")
    if args.mode == "v7_5" and args.teacher_start_step < 10000:
        raise ValueError("V7.5 teacher_start_step must be at least 10000")
    if args.variance_min <= 0 or args.variance_max <= args.variance_min:
        raise ValueError("Require 0 < variance_min < variance_max")
    if args.uq_isolation_check_interval < 1:
        raise ValueError("uq_isolation_check_interval must be positive")
    if args.factor < 1 or args.N_rand < 1 or args.N_samples < 2:
        raise ValueError("factor, N_rand and N_samples must be positive")


def main():
    args = config_parser().parse_args()
    validate_args(args)
    DiagnosticTrainer(args).train()


if __name__ == "__main__":
    main()
