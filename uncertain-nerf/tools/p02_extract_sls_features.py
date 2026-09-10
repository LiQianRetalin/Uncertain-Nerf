#!/usr/bin/env python3
"""Extract SpotLessSplats SD features for frozen training images only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image


SPOTLESS_COMMIT = "0caae3cc45bb1fddf86bd47e4a521888f5c49889"
NOTEBOOK_SHA256 = "d868fbd29ba36b8fb836f6b1b15c87ae53d06c9c38b8bc4fb37124bb8d12975b"
MODEL_ID = "stabilityai/stable-diffusion-2-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spotless-source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        help="Feature output directory (defaults to DATA_DIR/SD).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-status", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_official_featurizer(source: Path):
    notebook_path = source / "examples" / "datasets" / "sd_feature_extraction.ipynb"
    if sha256_file(notebook_path) != NOTEBOOK_SHA256:
        raise RuntimeError("SpotLessSplats feature notebook SHA-256 mismatch")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "class SDFeaturizer" in "".join(cell["source"])
    ]
    if len(cells) != 1:
        raise RuntimeError("cannot identify the pinned SDFeaturizer notebook cell")
    # The plotting import is unused by feature extraction and is not in official requirements.
    source_code = cells[0].replace("import matplotlib.pyplot as plt\n", "")
    namespace: dict[str, object] = {}
    exec(compile(source_code, str(notebook_path), "exec"), namespace)
    return namespace["SDFeaturizer"]


def _validate_feature(path: Path) -> dict[str, object]:
    feature = np.load(path, allow_pickle=False)
    if feature.shape != (1280, 50, 50):
        raise RuntimeError(f"unexpected SLS feature shape {feature.shape}: {path}")
    if not np.isfinite(feature).all():
        raise RuntimeError(f"non-finite SLS feature: {path}")
    return {
        "name": path.name,
        "shape": list(feature.shape),
        "dtype": str(feature.dtype),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    source = args.spotless_source.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True, encoding="utf-8"
    ).strip()
    if commit != SPOTLESS_COMMIT:
        raise RuntimeError(f"SpotLessSplats commit mismatch: {commit}")
    protocol_path = data_dir / "common_input_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    train_records = [
        record for record in protocol["images"] if "clutter" in record["name"].casefold()
    ]
    if len(train_records) != protocol["train_count"]:
        raise RuntimeError("frozen train-image count mismatch")
    feature_dir = (
        args.feature_dir.expanduser().resolve()
        if args.feature_dir
        else data_dir / "SD"
    )
    feature_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    records: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for record in train_records:
        path = feature_dir / Path(record["name"]).with_suffix(".npy")
        if path.exists():
            records.append(_validate_feature(path))
        else:
            missing.append(record)

    if args.validate_only and missing:
        raise RuntimeError(f"missing {len(missing)} SLS feature files")
    if missing:
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        SDFeaturizer = _load_official_featurizer(source)
        featurizer = SDFeaturizer(sd_id=MODEL_ID)
        for index, record in enumerate(missing, 1):
            image_path = data_dir / "images" / record["name"]
            if sha256_file(image_path) != record["sha256"]:
                raise RuntimeError(f"training image hash mismatch: {record['name']}")
            image = Image.open(image_path).convert("RGB").resize((800, 800))
            image_tensor = (torch.tensor(np.array(image)) / 255.0 - 0.5) * 2
            image_tensor = image_tensor.permute(2, 0, 1)
            features, _ = featurizer.forward(
                image_tensor,
                prompt="",
                ensemble_size=4,
                t=261,
                up_ft_index=[1],
            )
            feature_path = feature_dir / Path(record["name"]).with_suffix(".npy")
            temporary = feature_path.with_suffix(".npy.building")
            with temporary.open("wb") as stream:
                np.save(stream, features.detach().cpu().numpy())
            temporary.replace(feature_path)
            records.append(_validate_feature(feature_path))
            print(f"P02_SLS_FEATURES={index}/{len(missing)}", flush=True)
        torch.cuda.empty_cache()
        gc.collect()

    by_name = {record["name"]: record for record in records}
    ordered = [by_name[Path(record["name"]).with_suffix(".npy").name] for record in train_records]
    output = {
        "schema": "puri-gs-p02-sls-feature-cache-v1",
        "status": "VALID",
        "source_commit": commit,
        "source_notebook_sha256": NOTEBOOK_SHA256,
        "protocol_sha256": sha256_file(protocol_path),
        "model_id": MODEL_ID,
        "seed": args.seed,
        "train_only": True,
        "feature_dir": str(feature_dir),
        "parameters": {
            "input_resize": [800, 800],
            "prompt": "",
            "ensemble_size": 4,
            "timestep": 261,
            "up_feature_indices": [1],
        },
        "feature_count": len(ordered),
        "incremental_seconds": time.time() - started,
        "files": ordered,
    }
    args.output_status.parent.mkdir(parents=True, exist_ok=True)
    args.output_status.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("P02_SLS_FEATURE_CACHE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
