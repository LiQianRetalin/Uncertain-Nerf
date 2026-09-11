#!/usr/bin/env python3
"""Read-only checkpoint structure audit; compatible with Python 3.7+."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_info(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True,
    }


def inspect_checkpoint(path, family):
    value = torch.load(str(path), map_location="cpu")
    result = {"family": family, "python_type": type(value).__name__}
    if family == "robustsplat":
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise RuntimeError("RobustSplat checkpoint must be (capture, iteration)")
        capture, step = value
        if not isinstance(capture, (tuple, list)) or len(capture) != 12:
            raise RuntimeError("RobustSplat capture tuple length mismatch")
        xyz = capture[1]
        if not torch.is_tensor(xyz) or xyz.ndim != 2 or xyz.shape[1] != 3:
            raise RuntimeError("RobustSplat xyz tensor mismatch")
        result.update({
            "model_format": "RobustSplat GaussianModel.capture tuple + iteration",
            "step": int(step),
            "gaussian_count": int(xyz.shape[0]),
            "primary_tensor": tensor_info(xyz),
            "top_level_length": 2,
            "capture_length": 12,
        })
    else:
        if not isinstance(value, dict) or "splats" not in value or "step" not in value:
            raise RuntimeError("gsplat checkpoint must contain splats and step")
        splats = value["splats"]
        key = "means3d" if "means3d" in splats else "means"
        means = splats[key]
        if not torch.is_tensor(means) or means.ndim != 2 or means.shape[1] != 3:
            raise RuntimeError("gsplat means tensor mismatch")
        result.update({
            "model_format": "gsplat dict with splats + step",
            "step": int(value["step"]),
            "gaussian_count": int(means.shape[0]),
            "primary_tensor_key": key,
            "primary_tensor": tensor_info(means),
            "splat_keys": sorted(splats.keys()),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--family", choices=("internal", "sls-mlp", "robustsplat"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.checkpoint.expanduser().resolve()
    result = inspect_checkpoint(path, args.family)
    output = {
        "schema": "puri-gs-p02a-checkpoint-audit-v1",
        "status": "PASS",
        "run_id": args.run_id,
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_bytes": int(path.stat().st_size),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("P02A_CHECKPOINT_AUDIT_PASS={}".format(args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
