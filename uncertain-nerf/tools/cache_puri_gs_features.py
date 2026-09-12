#!/usr/bin/env python3
"""Precompute the single fixed two-scale DINOv2 cache for PURI-GS-RU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.dino_features import (  # noqa: E402
    COARSE_GRID,
    FEATURE_EXTRACTOR_VERSION,
    FINE_GRID,
    extract_patch_grid,
    load_frozen_dinov2,
)


EXPECTED_TRAIN_COUNTS = {
    "android": 122,
    "room": 272,
    "garden": 161,
    "patio_high": 221,
    "corner": 101,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=tuple(EXPECTED_TRAIN_COUNTS))
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument(
        "--dataset-format",
        choices=("colmap", "ontogo-patio-high", "ontogo-corner"),
        default="colmap",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dino-repo-dir", type=Path, required=True)
    parser.add_argument("--dino-weight-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-keyword")
    parser.add_argument("--test-keyword")
    return parser.parse_args()


def _load_dataset_classes(gsplat_dir: Path, dataset_format: str):
    examples_dir = gsplat_dir.resolve() / "examples"
    if not (examples_dir / "datasets" / "colmap.py").is_file():
        raise FileNotFoundError(f"gsplat COLMAP loader is missing: {examples_dir}")
    sys.path.insert(0, str(examples_dir))
    from datasets.colmap import Dataset, Parser

    if dataset_format == "ontogo-patio-high":
        from puri_gs.ontogo import OnTheGoPatioHighParser

        Parser = OnTheGoPatioHighParser
    elif dataset_format == "ontogo-corner":
        from puri_gs.ontogo_corner import OnTheGoCornerParser

        Parser = OnTheGoCornerParser

    return Parser, Dataset


def main() -> int:
    args = parse_args()
    if (args.train_keyword is None) != (args.test_keyword is None):
        raise ValueError("train-keyword and test-keyword must be provided together")
    if args.scene in {"android", "patio_high", "corner"} and (
        args.train_keyword != "clutter" or args.test_keyword != "extra"
    ):
        raise ValueError(
            f"{args.scene} cache requires --train-keyword clutter --test-keyword extra"
        )
    if args.scene in {"room", "garden"} and args.train_keyword is not None:
        raise ValueError(
            f"{args.scene} uses the fixed every-eighth test split, not keywords"
        )
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is not available for DINOv2 feature extraction")

    output_dir = args.output_dir.expanduser().resolve()
    building_dir = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists; refusing overwrite: {output_dir}")
    if building_dir.exists():
        raise RuntimeError(
            f"incomplete prior cache exists; inspect it before removal: {building_dir}"
        )
    building_dir.parent.mkdir(parents=True, exist_ok=True)
    building_dir.mkdir()

    expected_format = {
        "patio_high": "ontogo-patio-high",
        "corner": "ontogo-corner",
    }.get(args.scene, "colmap")
    if args.dataset_format != expected_format:
        raise ValueError(
            f"{args.scene} must use --dataset-format {expected_format}"
        )

    Parser, Dataset = _load_dataset_classes(args.gsplat_dir, args.dataset_format)
    parser = Parser(
        data_dir=str(args.data_dir.expanduser().resolve()),
        factor=args.data_factor,
        normalize=True,
        test_every=8,
    )
    trainset = Dataset(
        parser,
        split="train",
        val_every=0,
        train_keyword=args.train_keyword,
        test_keyword=args.test_keyword,
    )
    expected_count = EXPECTED_TRAIN_COUNTS[args.scene]
    if len(trainset) != expected_count:
        raise RuntimeError(
            f"{args.scene} split expected {expected_count} training images, got {len(trainset)}"
        )

    model, model_metadata = load_frozen_dinov2(
        args.dino_repo_dir, args.dino_weight_path, device=args.device
    )
    records: list[dict] = []
    image_mapping: dict[str, str] = {}
    extraction_started = perf_counter()
    for item in range(len(trainset)):
        data = trainset[item]
        parser_index = int(trainset.indices[item])
        image_name = str(parser.image_names[parser_index])
        pixels = data["image"].unsqueeze(0).to(args.device) / 255.0
        coarse = extract_patch_grid(model, pixels, COARSE_GRID)[0].float().cpu()
        fine = extract_patch_grid(model, pixels, FINE_GRID)[0].float().cpu()
        file_name = f"{item:04d}_{Path(image_name).stem}.pt"
        torch.save({"coarse": coarse, "fine": fine}, building_dir / file_name)
        height, width = data["image"].shape[:2]
        record = {
            "scene": args.scene,
            "image_name": image_name,
            "parser_index": parser_index,
            "dataset_item": item,
            "file": file_name,
            "original_height": int(height),
            "original_width": int(width),
            "coarse_shape": list(coarse.shape),
            "fine_shape": list(fine.shape),
        }
        records.append(record)
        image_mapping[image_name] = file_name
        print(f"[{item + 1}/{len(trainset)}] cached {image_name}", flush=True)
    extraction_seconds = perf_counter() - extraction_started

    manifest = {
        "schema_version": 1,
        "scene": args.scene,
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "data_factor": args.data_factor,
        "dataset_format": args.dataset_format,
        "train_keyword": args.train_keyword,
        "test_keyword": args.test_keyword,
        "model_name": model_metadata["model_name"],
        "model_weight_sha256": model_metadata["weight_sha256"],
        "dino_repository_commit": model_metadata["repository_commit"],
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "feature_extraction_seconds": extraction_seconds,
        "feature_extraction_mean_seconds": extraction_seconds / len(records),
        "image_mapping": image_mapping,
        "images": records,
    }
    (building_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    building_dir.rename(output_dir)
    print(
        f"FEATURE_CACHE_READY scene={args.scene} images={len(records)} path={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
