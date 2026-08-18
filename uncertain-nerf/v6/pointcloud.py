"""Back-project V6 test depth maps to a world-space NumPy point cloud."""

import argparse
import glob
import os
import re

import numpy as np
import torch

from .data import load_scene, pixels_to_rays


def _number(path):
    match = re.search(r"(\d+)(?=\.npy$)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def export(args):
    device = torch.device(args.device)
    scene = load_scene(
        args.datadir,
        factor=args.factor,
        holdout=args.llffhold,
        n_source_views=1,
        device=device,
        aabb_scale=args.aabb_scale,
        val_every=args.val_every,
    )
    depth_paths = sorted(glob.glob(os.path.join(args.render_dir, "depth_*.npy")), key=_number)
    acc_paths = sorted(glob.glob(os.path.join(args.render_dir, "acc_*.npy")), key=_number)
    if len(depth_paths) != len(scene.test_indices) or len(acc_paths) != len(depth_paths):
        raise ValueError("Expected one depth and acc map per LLFF test view")
    height, width, _ = scene.hwf
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    points = []
    for output_index, image_index in enumerate(scene.test_indices):
        count = height * width
        image_ids = torch.full((count,), int(image_index), dtype=torch.long, device=device)
        rays_o, rays_d = pixels_to_rays(
            scene.poses,
            scene.intrinsics,
            image_ids,
            ys.reshape(-1),
            xs.reshape(-1),
            radial_distortion=scene.radial_distortion,
        )
        depth = torch.from_numpy(np.load(depth_paths[output_index])).to(device).reshape(-1)
        acc = torch.from_numpy(np.load(acc_paths[output_index])).to(device).reshape(-1)
        valid = torch.isfinite(depth) & (depth > 0) & (acc >= args.opacity_threshold)
        points.append((rays_o[valid] + rays_d[valid] * depth[valid, None]).cpu().numpy())
    merged = np.concatenate(points, axis=0)
    if args.max_points > 0 and len(merged) > args.max_points:
        rng = np.random.default_rng(args.seed)
        merged = merged[rng.choice(len(merged), args.max_points, replace=False)]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, merged.astype(np.float32))
    print(f"Saved {len(merged)} points to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--val_every", type=int, default=8)
    parser.add_argument("--aabb_scale", type=float, default=1.25)
    parser.add_argument("--opacity_threshold", type=float, default=0.5)
    parser.add_argument("--max_points", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    export(parser.parse_args())
