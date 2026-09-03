#!/usr/bin/env python3
"""Audit one prepared Patio-High scene without training or modifying it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.ontogo import (  # noqa: E402
    OnTheGoPatioHighParser,
    validate_prepared_patio_high,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--verify-image-hashes", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples_dir = args.gsplat_dir.expanduser().resolve() / "examples"
    if not (examples_dir / "datasets" / "normalize.py").is_file():
        raise FileNotFoundError(f"pinned gsplat normalization code is missing: {examples_dir}")
    sys.path.insert(0, str(examples_dir))
    contract = validate_prepared_patio_high(
        args.data_dir,
        factor=4,
        verify_image_hashes=args.verify_image_hashes,
    )
    parser = OnTheGoPatioHighParser(
        str(args.data_dir.expanduser().resolve()), factor=4, normalize=True
    )
    names = parser.image_names
    counts = {
        "clutter": sum(name.startswith("clutter_") for name in names),
        "extra": sum(name.startswith("extra_") for name in names),
        "excluded": sum(name.startswith("excluded_") for name in names),
    }
    if counts != {"clutter": 221, "extra": 45, "excluded": 1}:
        raise RuntimeError(f"prepared Patio-High parser split is invalid: {counts}")
    camera_determinants = np.linalg.det(parser.camtoworlds[:, :3, :3])
    if not np.allclose(camera_determinants, 1.0, atol=1e-5):
        raise RuntimeError("prepared Patio-High camera rotations are invalid")
    protocol = contract["protocol"]
    result = {
        "decision": "PATIO_HIGH_PROTOCOL_PASS",
        "schema": protocol["schema"],
        "dataset_format": protocol["dataset_format"],
        "data_dir": str(contract["root"]),
        "data_factor": 4,
        "frame_count": len(names),
        "train_count": counts["clutter"],
        "test_count": counts["extra"],
        "excluded_count": counts["excluded"],
        "initialization_algorithm": protocol["initialization"]["algorithm"],
        "initial_point_count": len(parser.points),
        "initial_points_sha256": protocol["initial_points_sha256"],
        "initial_colors_sha256": protocol["initial_colors_sha256"],
        "image_hashes_verified": bool(args.verify_image_hashes),
        "undistorted_image_size": list(parser.imsize_dict[1]),
        "undistorted_K": parser.Ks_dict[1].tolist(),
        "normalized_scene_scale": parser.scene_scale,
        "normalization_transform": parser.transform.tolist(),
    }
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        if output.exists():
            raise RuntimeError(f"output already exists; refusing overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for name in (
        "decision",
        "frame_count",
        "train_count",
        "test_count",
        "excluded_count",
        "initialization_algorithm",
        "initial_point_count",
        "initial_points_sha256",
        "image_hashes_verified",
        "undistorted_image_size",
        "normalized_scene_scale",
    ):
        print(f"{name.upper()}={result[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
