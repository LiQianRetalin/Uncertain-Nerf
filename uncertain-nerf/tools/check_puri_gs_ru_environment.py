#!/usr/bin/env python3
"""Read-only server audit for the fixed PURI-GS-RU runtime and DINO assets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.dino_features import (  # noqa: E402
    COARSE_GRID,
    FINE_GRID,
    FeatureCache,
    extract_patch_grid,
    load_frozen_dinov2,
)


EXPECTED_GSPLAT_COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--dino-repo-dir", type=Path, required=True)
    parser.add_argument("--dino-weight-path", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import gsplat

    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    runtime_path = Path(gsplat.__file__).resolve()
    version = importlib.metadata.version("gsplat")
    if version.split("+")[0] != "1.5.3":
        raise RuntimeError(f"runtime gsplat must be 1.5.3, got {version}")
    if _git_commit(gsplat_dir) != EXPECTED_GSPLAT_COMMIT:
        raise RuntimeError("gsplat source checkout is not the pinned v1.5.3 commit")
    if runtime_path.is_relative_to(gsplat_dir):
        raise RuntimeError("gsplat source checkout shadows the pinned installed wheel")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    model, dino_metadata = load_frozen_dinov2(
        args.dino_repo_dir, args.dino_weight_path, device=args.device
    )
    dummy = torch.zeros(1, 32, 48, 3, device=args.device)
    coarse = extract_patch_grid(model, dummy, COARSE_GRID)
    fine = extract_patch_grid(model, dummy, FINE_GRID)
    result = {
        "status": "PURI_GS_RU_ENVIRONMENT_READY",
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "gsplat_version": version,
        "gsplat_runtime_path": str(runtime_path),
        "gsplat_source_path": str(gsplat_dir),
        "gsplat_source_commit": EXPECTED_GSPLAT_COMMIT,
        "dino": dino_metadata,
        "coarse_shape": list(coarse.shape),
        "fine_shape": list(fine.shape),
        "features_finite": bool(torch.isfinite(coarse).all() and torch.isfinite(fine).all()),
    }
    if args.feature_cache_dir is not None:
        cache = FeatureCache(
            args.feature_cache_dir,
            expected_weight_sha256=dino_metadata["weight_sha256"],
        )
        result["feature_cache"] = {
            "path": str(cache.directory),
            "scene": cache.scene,
            "images": len(cache.records),
        }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise RuntimeError(f"refusing to overwrite environment audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

