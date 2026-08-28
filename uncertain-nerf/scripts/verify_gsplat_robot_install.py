#!/usr/bin/env python3
"""One-pass GPU verification for the pinned robot 3DGS environment."""

from __future__ import annotations

import argparse
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch


def _has_compatible_architecture(
    capability: tuple[int, int], compiled_architectures: list[str]
) -> bool:
    """CUDA cubins are forward-compatible within one compute-capability major."""

    for architecture in compiled_architectures:
        if not architecture.startswith("sm_"):
            continue
        digits = architecture.removeprefix("sm_")
        if len(digits) < 2 or not digits.isdigit():
            continue
        compiled = (int(digits[:-1]), int(digits[-1]))
        if compiled[0] == capability[0] and compiled[1] <= capability[1]:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpu", default="NVIDIA L20")
    args = parser.parse_args()

    if torch.__version__.split("+")[0] != "2.4.0":
        raise RuntimeError(f"Expected torch 2.4.0, got {torch.__version__}")
    if torch.version.cuda != "12.1":
        raise RuntimeError(f"Expected torch CUDA 12.1, got {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_name = torch.cuda.get_device_name(0)
    if args.expected_gpu not in device_name:
        raise RuntimeError(f"Expected {args.expected_gpu}, got {device_name}")
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    supported_architectures = torch.cuda.get_arch_list()
    if not _has_compatible_architecture(capability, supported_architectures):
        raise RuntimeError(
            f"GPU architecture {architecture} is not supported by this PyTorch; "
            f"compiled architectures: {supported_architectures}"
        )

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

    if not gsplat.__version__.startswith("1.5.3"):
        raise RuntimeError(f"Expected gsplat 1.5.3, got {gsplat.__version__}")
    if not hasattr(pycolmap, "SceneManager"):
        raise RuntimeError("Pinned pycolmap does not expose SceneManager")

    device = torch.device("cuda")
    parameters = {
        "means": torch.tensor(
            [[0.0, 0.0, 3.0]], device=device, requires_grad=True
        ),
        "quats": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True
        ),
        "scales": torch.tensor(
            [[0.2, 0.2, 0.2]], device=device, requires_grad=True
        ),
        "opacities": torch.tensor([0.8], device=device, requires_grad=True),
        "colors": torch.tensor(
            [[1.0, 0.0, 0.0]], device=device, requires_grad=True
        ),
    }
    viewmats = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor(
        [[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    rendered, alpha, _ = rasterization(
        means=parameters["means"],
        quats=parameters["quats"],
        scales=parameters["scales"],
        opacities=parameters["opacities"],
        colors=parameters["colors"],
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
    (rendered.square().mean() + alpha.mean()).backward()
    for name, parameter in parameters.items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"gsplat backward returned invalid {name} gradients")

    with tempfile.TemporaryDirectory(prefix="gsplat153-checkpoint-") as directory:
        checkpoint = Path(directory) / "smoke.pt"
        torch.save({name: value.detach() for name, value in parameters.items()}, checkpoint)
        restored = torch.load(checkpoint, map_location=device, weights_only=True)
        for name, parameter in parameters.items():
            if not torch.equal(restored[name], parameter.detach()):
                raise RuntimeError(f"checkpoint roundtrip failed for {name}")

    sample = torch.rand((1, 3, 64, 64), device=device)
    if not torch.isfinite(fused_ssim(sample, sample)):
        raise RuntimeError("fused SSIM verification failed")
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    lpips_value = lpips(sample, sample)
    if not torch.isfinite(lpips_value):
        raise RuntimeError("LPIPS verification failed")

    print("decision=PASS")
    print(f"gpu={device_name}")
    print(f"gpu_architecture={architecture}")
    print(f"torch={torch.__version__}")
    print(f"gsplat={gsplat.__version__}")
    print(f"torchmetrics={torchmetrics.__version__}")
    print(f"opencv={cv2.__version__}")
    print(f"imageio={imageio.__version__}")
    print(f"tyro={version('tyro')}")
    print(f"viser={version('viser')}")
    print(f"nerfview={version('nerfview')}")
    print(f"splines={version('splines')}")
    print("rasterization_forward_backward=PASS")
    print("checkpoint_roundtrip=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
