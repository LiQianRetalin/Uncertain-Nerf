#!/usr/bin/env python3
"""10-step real-image A1 loss/backward/checkpoint/metric smoke test."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from puri_gs.config import load_experiment_config
from puri_gs.responsibility import ResponsibilityConfig, responsibility_weighted_l1


def _save_map(path: Path, value: torch.Tensor) -> None:
    array = value.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    Image.fromarray(array).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if not 10 <= args.steps <= 100:
        raise ValueError("smoke steps must be in [10, 100]")

    profile = load_experiment_config(args.config)
    responsibility_fields = dict(profile["responsibility"])
    responsibility_fields["start_step"] = min(3, args.steps - 1)
    config = ResponsibilityConfig(**responsibility_fields)
    torch.manual_seed(profile["seed"])

    target_image = Image.open(args.image).convert("RGB")
    target_image.thumbnail((96, 96))
    target = torch.from_numpy(np.asarray(target_image, dtype=np.float32) / 255.0)
    target = target.unsqueeze(0)
    rendered = torch.nn.Parameter((target + 0.15 * torch.randn_like(target)).clamp(0, 1))
    with torch.no_grad():
        height, width = rendered.shape[1:3]
        rendered[:, height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 1.0
    optimizer = torch.optim.Adam([rendered], lr=0.03)

    losses: list[float] = []
    responsibility = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, responsibility = responsibility_weighted_l1(
            rendered, target, config=config, step=step
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        if rendered.grad is None or not torch.isfinite(rendered.grad).all():
            raise RuntimeError(f"missing or non-finite gradient at step {step}")
        optimizer.step()
        losses.append(float(loss.detach()))

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint.pt"
    torch.save(
        {
            "step": args.steps - 1,
            "rendered": rendered.detach(),
            "optimizer": optimizer.state_dict(),
            "responsibility": asdict(config),
        },
        checkpoint,
    )
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.testing.assert_close(loaded["rendered"], rendered.detach().cpu())

    with torch.no_grad():
        mse = torch.mean((rendered.clamp(0, 1) - target) ** 2).item()
        psnr = -10.0 * math.log10(max(mse, 1e-12))
        residual = (rendered.clamp(0, 1) - target).abs().mean(dim=-1)[0]
    if responsibility is None:
        raise RuntimeError("A1 responsibility did not activate during smoke test")
    _save_map(args.output / "responsibility_map.png", responsibility[0])
    _save_map(args.output / "residual_map.png", residual)
    Image.fromarray(
        rendered[0].detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    ).save(args.output / "rgb_render.png")

    metrics = {
        "protocol": "puri-gs-a1-real-image-loss-smoke-1",
        "steps": args.steps,
        "input_image": str(args.image.resolve()),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "psnr": psnr,
        "responsibility_min": float(responsibility.min()),
        "responsibility_max": float(responsibility.max()),
        "responsibility_requires_grad": responsibility.requires_grad,
        "checkpoint_roundtrip": True,
        "scope": "loss/gradient/checkpoint/metric only; no gsplat rasterizer",
    }
    (args.output / "smoke_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
