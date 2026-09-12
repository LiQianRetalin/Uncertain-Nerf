#!/usr/bin/env python3
"""Independently validate every frozen Corner RU feature payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.feature_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("scene") != "corner"
        or manifest.get("dataset_format") != "ontogo-corner"
        or manifest.get("train_keyword") != "clutter"
        or manifest.get("test_keyword") != "extra"
        or len(manifest.get("images", [])) != 101
    ):
        raise RuntimeError("RU Corner feature manifest identity mismatch")
    records = []
    seen = set()
    for row in manifest["images"]:
        if row["image_name"] in seen or not row["image_name"].startswith("clutter_"):
            raise RuntimeError("RU feature image mapping is duplicated or non-training")
        seen.add(row["image_name"])
        path = root / row["file"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        coarse, fine = payload.get("coarse"), payload.get("fine")
        if (
            not torch.is_tensor(coarse)
            or not torch.is_tensor(fine)
            or tuple(coarse.shape) != (384, 16, 16)
            or tuple(fine.shape) != (384, 36, 36)
            or coarse.dtype != torch.float32
            or fine.dtype != torch.float32
            or not torch.isfinite(coarse).all()
            or not torch.isfinite(fine).all()
        ):
            raise RuntimeError(f"invalid RU feature payload: {path}")
        records.append(
            {
                "image_name": row["image_name"],
                "file": row["file"],
                "sha256": sha256_file(path),
                "coarse_shape": list(coarse.shape),
                "fine_shape": list(fine.shape),
                "dtype": str(coarse.dtype),
            }
        )
    result = {
        "schema": "puri-gs-p03-ru-feature-validation-v1",
        "status": "PASS",
        "generation_manifest_sha256": sha256_file(manifest_path),
        "feature_count": len(records),
        "train_only": True,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("P03_RU_FEATURE_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
