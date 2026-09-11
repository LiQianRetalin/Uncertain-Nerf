#!/usr/bin/env python3
"""Record the exact P02 feature-model cache identities without downloading anything."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SD_MODEL_ID = "sd2-community/stable-diffusion-2-1"
SD_REVISION = "bb2154823665391b4fb29b0b9cf82a198964ee05"
DINO_SOURCE_REVISION = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
DINO_WEIGHT_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--robust-source", type=Path, required=True)
    parser.add_argument("--spotless-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_root.resolve()
    homes = [work / "hf-home", Path.home() / ".cache/huggingface"]
    snapshots = []
    for home in homes:
        candidate = home / "hub/models--sd2-community--stable-diffusion-2-1/snapshots" / SD_REVISION
        if candidate.is_dir():
            snapshots.append(candidate)
    if not snapshots:
        raise RuntimeError("fixed SD snapshot was not found in the existing caches")
    snapshot = snapshots[0]
    required = [
        "model_index.json", "text_encoder/pytorch_model.bin",
        "unet/diffusion_pytorch_model.bin", "vae/diffusion_pytorch_model.bin",
    ]
    sd_files = {name: record(snapshot / name) for name in required if (snapshot / name).is_file()}
    if set(sd_files) != set(required):
        raise RuntimeError("fixed SD snapshot is incomplete")

    dino_candidates = []
    for checkpoint_root in (
        work / "torch-home/hub/checkpoints",
        Path.home() / ".cache/torch/hub/checkpoints",
    ):
        dino_candidates.extend(checkpoint_root.glob("dinov2_vits14_reg4_pretrain.pth"))
    if not dino_candidates:
        raise RuntimeError("the existing DINO weight was not found")
    dino_weight = record(dino_candidates[0])
    if dino_weight["sha256"] != DINO_WEIGHT_SHA256:
        raise RuntimeError("DINO weight hash mismatch")
    hub_roots = list((work / "torch-home/hub").glob("facebookresearch_dinov2_*"))
    hub_roots += list((Path.home() / ".cache/torch/hub").glob("facebookresearch_dinov2_*"))
    matching_hubs = [path for path in hub_roots if DINO_SOURCE_REVISION[:7] in path.name and path.is_dir()]

    output = {
        "schema": "puri-gs-p02a-weight-provenance-v1", "status": "PASS",
        "stable_diffusion": {
            "model_id": SD_MODEL_ID, "revision": SD_REVISION,
            "snapshot_path": str(snapshot.resolve()), "alternate_matching_snapshot_paths": [str(path.resolve()) for path in snapshots[1:]], "files": sd_files,
            "equivalence_status": "SOURCE_FIXED_ACTUAL_BYTES_HASHED; equivalence to a differently named upstream repository is not proven without counterpart hashes",
            "feature_parameters": {"input_resize": [800, 800], "prompt": "", "ensemble_size": 4, "timestep": 261, "up_feature_indices": [1], "seed": 42},
            "spotless_notebook_sha256": sha256_file(args.spotless_source.resolve() / "examples/datasets/sd_feature_extraction.ipynb"),
            "extractor_sha256": sha256_file(args.code_root.resolve() / "tools/p02_extract_sls_features.py"),
        },
        "dinov2": {
            "architecture": "dinov2_vits14_reg", "source_revision": DINO_SOURCE_REVISION,
            "weight": dino_weight, "hub_source_candidates": [str(path.resolve()) for path in matching_hubs],
            "robust_mask_utils_sha256": sha256_file(args.robust_source.resolve() / "utils/mask_utils.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("P02A_WEIGHT_PROVENANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
