#!/usr/bin/env python3
"""One-pass GPU verification for the pinned robot 3DGS environment."""

from __future__ import annotations

import argparse
from importlib.metadata import version

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpu", default="NVIDIA L20")
    args = parser.parse_args()

    import cv2
    import gsplat
    import imageio
    import nerfview
    import pycolmap
    import splines
    import torchmetrics
    import tyro
    import viser
    from fused_ssim import fused_ssim
    from gsplat.rendering import rasterization
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    if torch.__version__.split("+")[0] != "2.4.0":
        raise RuntimeError(f"Expected torch 2.4.0, got {torch.__version__}")
    if torch.version.cuda != "12.1":
        raise RuntimeError(f"Expected torch CUDA 12.1, got {torch.version.cuda}")
    if not gsplat.__version__.startswith("1.5.3"):
        raise RuntimeError(f"Expected gsplat 1.5.3, got {gsplat.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_name = torch.cuda.get_device_name(0)
    if args.expected_gpu not in device_name:
        raise RuntimeError(f"Expected {args.expected_gpu}, got {device_name}")
    if not hasattr(pycolmap, "SceneManager"):
        raise RuntimeError("Pinned pycolmap does not expose SceneManager")

    device = torch.device("cuda")
    means = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.2, 0.2, 0.2]], device=device)
    opacities = torch.tensor([0.8], device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    viewmats = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor(
        [[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    rendered, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=32,
        height=32,
        sh_degree=None,
    )
    torch.cuda.synchronize()
    if rendered.shape != (1, 32, 32, 3) or alpha.shape != (1, 32, 32, 1):
        raise RuntimeError("Unexpected gsplat rasterization output shape")
    if not torch.isfinite(rendered).all() or float(alpha.max()) <= 0.0:
        raise RuntimeError("gsplat rasterization returned invalid values")

    sample = torch.rand((1, 3, 64, 64), device=device)
    if not torch.isfinite(fused_ssim(sample, sample)):
        raise RuntimeError("fused SSIM verification failed")
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    lpips_value = lpips(sample, sample)
    if not torch.isfinite(lpips_value):
        raise RuntimeError("LPIPS verification failed")

    print("decision=PASS")
    print(f"gpu={device_name}")
    print(f"torch={torch.__version__}")
    print(f"gsplat={gsplat.__version__}")
    print(f"torchmetrics={torchmetrics.__version__}")
    print(f"opencv={cv2.__version__}")
    print(f"imageio={imageio.__version__}")
    print(f"tyro={version('tyro')}")
    print(f"viser={version('viser')}")
    print(f"nerfview={version('nerfview')}")
    print(f"splines={version('splines')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
