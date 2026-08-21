"""Entry point for the genuine original-NeRF A0 recovery gate."""

import configargparse

from a0.trainer import A0Trainer


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--basedir", default="./logs-a0")
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--val_every", type=int, default=8)

    parser.add_argument("--netdepth", type=int, default=8)
    parser.add_argument("--netwidth", type=int, default=256)
    parser.add_argument("--netdepth_fine", type=int, default=8)
    parser.add_argument("--netwidth_fine", type=int, default=256)
    parser.add_argument("--multires", type=int, default=10)
    parser.add_argument("--multires_views", type=int, default=4)
    parser.add_argument("--use_viewdirs", action="store_true")
    parser.add_argument("--no_ndc", action="store_true")

    parser.add_argument("--N_rand", type=int, default=1024)
    parser.add_argument("--N_samples", type=int, default=64)
    parser.add_argument("--N_importance", type=int, default=64)
    parser.add_argument("--N_iters", type=int, default=200000)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--netchunk", type=int, default=65536)
    parser.add_argument("--raw_noise_std", type=float, default=1.0)
    parser.add_argument("--white_bkgd", action="store_true")
    parser.add_argument("--coarse_loss_weight", type=float, default=1.0)

    parser.add_argument("--lrate", type=float, default=5.0e-4)
    parser.add_argument("--lrate_decay", type=float, default=250.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_reload", action="store_true")
    parser.add_argument("--ft_path")
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_eval", type=int, default=5000)
    parser.add_argument("--eval_max_views", type=int, default=0)
    parser.add_argument("--i_weights", type=int, default=10000)
    parser.add_argument("--target_psnr", type=float, default=25.0)
    parser.add_argument("--render_only", action="store_true")
    parser.add_argument(
        "--render_split", choices=["path", "train", "val", "test"],
        default="path")
    return parser


def validate_args(args):
    for name in (
            "factor", "N_rand", "N_samples", "N_iters", "chunk", "netchunk"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.N_importance < 0:
        raise ValueError("N_importance cannot be negative")
    if not args.use_viewdirs:
        raise ValueError("Original fern A0 requires --use_viewdirs")
    if args.no_ndc:
        raise ValueError("Original forward-facing LLFF A0 requires NDC")
    if args.coarse_loss_weight != 1.0:
        raise ValueError("Original A0 requires coarse_loss_weight=1.0")


def main():
    args = config_parser().parse_args()
    validate_args(args)
    A0Trainer(args).train()


if __name__ == "__main__":
    main()
