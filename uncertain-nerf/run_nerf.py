"""Training and inference entry point for Uncertain-NeRF V5."""

import configargparse

from v5.trainer import Trainer


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--basedir", default="./logs")
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--dataset_type", default="llff_colmap", choices=["llff_colmap"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--no_ndc", action="store_true", default=True)

    parser.add_argument("--netdepth", type=int, default=2)
    parser.add_argument("--netwidth", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hash_levels", type=int, default=16)
    parser.add_argument("--hash_min_resolution", type=int, default=16)
    parser.add_argument("--hash_max_resolution", type=int, default=2048)
    parser.add_argument("--hash_features", type=int, default=2)
    parser.add_argument("--hash_log2_size", type=int, default=19)
    parser.add_argument("--multires_views", type=int, default=4)

    parser.add_argument("--N_rand", type=int, default=1024)
    parser.add_argument("--N_samples", type=int, default=64)
    parser.add_argument("--N_importance", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--netchunk", type=int, default=65536)
    parser.add_argument("--N_iters", type=int, default=200000)

    parser.add_argument("--reliability_strength", type=float, default=4.0)
    parser.add_argument("--teacher_ray_ratio", type=float, default=0.2)
    parser.add_argument("--teacher_views", type=int, default=8)
    parser.add_argument("--mc_samples", type=int, default=4)
    parser.add_argument("--teacher_kappa", type=float, default=2.0)
    parser.add_argument("--variance_beta", type=float, default=1.0e-3)
    parser.add_argument("--prior_ray_ratio", type=float, default=0.3)
    parser.add_argument("--geo_gamma", type=float, default=2.3)
    parser.add_argument("--reg_amplitude", type=float, default=1.0e-2)
    parser.add_argument("--reg_spatial", type=float, default=1.0e-3)

    parser.add_argument("--tukey_start", type=float, default=6.0)
    parser.add_argument("--tukey_end", type=float, default=4.685)
    parser.add_argument("--huber_start", type=float, default=3.0)
    parser.add_argument("--huber_end", type=float, default=1.345)
    parser.add_argument("--robust_anneal_steps", type=int, default=20000)
    parser.add_argument("--depth_scale_warmup", type=int, default=1000)
    parser.add_argument("--depth_scale_interval", type=int, default=500)
    parser.add_argument("--depth_scale_ema", type=float, default=0.1)

    parser.add_argument("--loss_calibration_steps", type=int, default=500)
    parser.add_argument("--loss_ema_rho", type=float, default=0.02)
    parser.add_argument("--alpha_geo", type=float, default=0.5)
    parser.add_argument("--alpha_u", type=float, default=0.5)
    parser.add_argument("--alpha_reg", type=float, default=0.1)

    parser.add_argument("--lrate", type=float, default=1.0e-2)
    parser.add_argument("--lrate_min", type=float, default=1.0e-4)
    parser.add_argument("--lr_warmup", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--no_reload", action="store_true")
    parser.add_argument("--ft_path", default=None)
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_weights", type=int, default=10000)

    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--render_split", choices=["path", "test"], default="path")
    parser.add_argument("--render_poses", default=None)
    return parser


def main():
    args = config_parser().parse_args()
    if not args.no_ndc:
        raise ValueError("Uncertain-NeRF V5 requires no_ndc=True for COLMAP geometry")
    Trainer(args).train()


if __name__ == "__main__":
    main()
