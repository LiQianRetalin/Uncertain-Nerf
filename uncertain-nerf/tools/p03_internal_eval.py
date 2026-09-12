#!/usr/bin/env python3
"""Evaluate or time one frozen internal gsplat checkpoint for P03."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("timing", "quality", "memory"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--expected-step", type=int, default=29999)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("invalid timing repetitions")
    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    examples_dir = gsplat_dir / "examples"
    code_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root))
    sys.path.insert(0, str(examples_dir))
    from simple_trainer import Config, Runner
    from gsplat.strategy import DefaultStrategy

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        disable_viewer=True,
        disable_video=True,
        data_dir=str(args.data_dir.expanduser().resolve()),
        data_factor=4,
        dataset_format="ontogo-corner",
        result_dir=str(output),
        test_every=8,
        val_every=0,
        eval_split="test",
        train_keyword="clutter",
        test_keyword="extra",
        sh_degree=3,
        ssim_lambda=0.2,
        tb_every=0,
        strategy=DefaultStrategy(absgrad=True, grow_grad2d=0.0006),
    )
    runner = Runner(0, 0, 1, cfg)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=runner.device, weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"splats", "step"}:
        raise RuntimeError("internal checkpoint must contain only splats and step")
    if int(checkpoint["step"]) != args.expected_step:
        raise RuntimeError(
            f"P03 RU checkpoint must be step {args.expected_step}, got {checkpoint['step']}"
        )
    for key in runner.splats:
        runner.splats[key].data = checkpoint["splats"][key].to(runner.device)
    del checkpoint

    prepared = []
    for item_index in range(len(runner.valset)):
        data = runner.valset[item_index]
        parser_index = int(runner.valset.indices[item_index])
        name = str(runner.parser.image_names[parser_index])
        pixels = data["image"]
        height, width = pixels.shape[:2]
        prepared.append(
            {
                "index": item_index,
                "name": name,
                "camtoworld": data["camtoworld"].unsqueeze(0).to(runner.device),
                "K": data["K"].unsqueeze(0).to(runner.device),
                "height": int(height),
                "width": int(width),
                "mask": (
                    data["mask"].unsqueeze(0).to(runner.device)
                    if "mask" in data
                    else None
                ),
            }
        )
    if len(prepared) != 20:
        raise RuntimeError(f"Corner test loader returned {len(prepared)} views")

    def render_one(item: dict[str, object]) -> torch.Tensor:
        colors, _, _ = runner.rasterize_splats(
            camtoworlds=item["camtoworld"],
            Ks=item["K"],
            width=item["width"],
            height=item["height"],
            sh_degree=cfg.sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            masks=item["mask"],
        )
        return colors.clamp(0.0, 1.0)

    metadata = {
        "schema": "puri-gs-p03-internal-eval-v1",
        "mode": args.mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": args.expected_step,
        "gaussian_count": int(len(runner.splats["means"])),
        "test_names": [item["name"] for item in prepared],
        "image_size": [prepared[0]["width"], prepared[0]["height"]],
        "precision": str(runner.splats["means"].dtype),
    }
    with torch.no_grad():
        if args.mode == "timing":
            for index in range(args.warmup):
                render_one(prepared[index % len(prepared)])
            torch.cuda.synchronize()
            rows = []
            repeats = []
            for repeat in range(1, args.repeats + 1):
                total = 0.0
                for item in prepared:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    render_one(item)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    total += seconds
                    rows.append(
                        {
                            "repeat": repeat,
                            "image_index": item["index"],
                            "image_name": item["name"],
                            "seconds": seconds,
                        }
                    )
                repeats.append(
                    {
                        "repeat": repeat,
                        "frames": len(prepared),
                        "seconds": total,
                        "fps": len(prepared) / total,
                    }
                )
            with (output / "per_image_latency.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            metadata.update(
                {
                    "warmup": args.warmup,
                    "repeats": repeats,
                    "boundary": "per-view CUDA sync; native rasterize_splats plus clamp; cameras prepared; no metrics or image write",
                }
            )
        elif args.mode == "quality":
            prediction_dir = output / "float_predictions"
            prediction_dir.mkdir()
            for item in prepared:
                prediction = render_one(item)[0].cpu().numpy().astype(np.float32)
                np.save(
                    prediction_dir / Path(str(item["name"])).with_suffix(".npy").name,
                    prediction,
                    allow_pickle=False,
                )
            metadata["prediction_count"] = len(prepared)
        else:
            torch.cuda.reset_peak_memory_stats()
            for item in prepared:
                render_one(item)
            torch.cuda.synchronize()
            metadata.update(
                {
                    "boundary": "checkpoint loaded; peak reset immediately before one complete native inference pass",
                    "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
    (output / "result.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"P03_INTERNAL_{args.mode.upper()}=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
