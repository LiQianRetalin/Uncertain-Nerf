#!/usr/bin/env python3
"""Prepare the immutable factor-4 Corner input for P03.

The sparse points use the same fixed-pose, training-only SIFT triangulation
algorithm and constants that were frozen for Patio-High in P01.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.ontogo import (  # noqa: E402
    COLORS_FILE,
    EXPECTED_FACTOR,
    POINTS_FILE,
    PROTOCOL_FILE,
    _prepared_name,
    _read_json,
    sha256_file,
)
from puri_gs.ontogo_corner import (  # noqa: E402
    DATASET_FORMAT,
    EXPECTED_UNASSIGNED,
    PROTOCOL_SCHEMA,
    SCENE,
    _validate_split,
    _validate_transforms,
    validate_prepared_corner,
)
from tools.prepare_ontogo_patio_high import (  # noqa: E402
    LOWE_RATIO,
    MAX_INITIAL_POINTS,
    MAX_POINTS_PER_PAIR,
    MAX_RADIUS_IN_CAMERA_EXTENTS,
    MAX_REPROJECTION_ERROR_PX,
    MIN_INITIAL_POINTS,
    MIN_PARALLAX_DEGREES,
    PAIR_GAPS,
    SIFT_FEATURES,
    VOXELS_PER_CAMERA_EXTENT,
    _triangulate_initialization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _resize_images(
    source: Path,
    building: Path,
    frames: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
) -> list[dict[str, Any]]:
    image_dir = building / f"images_{EXPECTED_FACTOR}"
    image_dir.mkdir()
    train, test = set(train_indices), set(test_indices)
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        source_name = Path(frame["file_path"]).name
        source_path = source / "images" / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Corner source image is missing: {source_path}")
        output_name = _prepared_name(index, source_name, train, test)
        output_path = image_dir / output_name
        with Image.open(source_path) as image:
            rgb = image.convert("RGB")
            expected_size = (
                int(round(rgb.width / EXPECTED_FACTOR)),
                int(round(rgb.height / EXPECTED_FACTOR)),
            )
            resized = rgb.resize(expected_size, Image.Resampling.BICUBIC)
            resized.save(output_path, format="PNG", optimize=False)
        records.append(
            {
                "frame_index": index,
                "source_name": source_name,
                "source_sha256": sha256_file(source_path),
                "source_width": rgb.width,
                "source_height": rgb.height,
                "file": output_name,
                "width": expected_size[0],
                "height": expected_size[1],
                "sha256": sha256_file(output_path),
            }
        )
        print(f"RESIZED={index + 1}/{len(frames)} {output_name}", flush=True)
    return records


def main() -> int:
    args = parse_args()
    source = args.source_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    building = output.with_name(output.name + ".building")
    if output.exists():
        raise RuntimeError(f"output already exists; refusing overwrite: {output}")
    if building.exists():
        raise RuntimeError(f"incomplete prior output exists; inspect it first: {building}")
    if source.name.casefold() != SCENE:
        raise ValueError("this protocol accepts only the Corner source directory")

    transforms_path = source / "transforms.json"
    split_path = source / "split.json"
    transforms = _read_json(transforms_path)
    frames = _validate_transforms(transforms)
    split = _read_json(split_path)
    train_indices, test_indices = _validate_split(split, len(frames))

    building.parent.mkdir(parents=True, exist_ok=True)
    building.mkdir()
    shutil.copy2(transforms_path, building / "transforms.json")
    shutil.copy2(split_path, building / "split.json")
    image_records = _resize_images(
        source, building, frames, train_indices, test_indices
    )
    points, colors, initialization = _triangulate_initialization(
        building, frames, train_indices, image_records, transforms
    )
    np.save(building / POINTS_FILE, points, allow_pickle=False)
    np.save(building / COLORS_FILE, colors, allow_pickle=False)
    initialization["reused_protocol_source"] = "tools/prepare_ontogo_patio_high.py"
    initialization["frozen_constants"] = {
        "sift_features": SIFT_FEATURES,
        "pair_gaps": list(PAIR_GAPS),
        "lowe_ratio": LOWE_RATIO,
        "max_reprojection_error_px": MAX_REPROJECTION_ERROR_PX,
        "min_parallax_degrees": MIN_PARALLAX_DEGREES,
        "max_radius_in_camera_extents": MAX_RADIUS_IN_CAMERA_EXTENTS,
        "max_points_per_pair": MAX_POINTS_PER_PAIR,
        "max_initial_points": MAX_INITIAL_POINTS,
        "min_initial_points": MIN_INITIAL_POINTS,
        "voxels_per_camera_extent": VOXELS_PER_CAMERA_EXTENT,
    }
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "dataset_format": DATASET_FORMAT,
        "scene": SCENE,
        "data_factor": EXPECTED_FACTOR,
        "frame_count": len(frames),
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "unassigned_frame_indices": EXPECTED_UNASSIGNED,
        "source_directory": str(source),
        "transforms_sha256": sha256_file(building / "transforms.json"),
        "split_sha256": sha256_file(building / "split.json"),
        "initialization": initialization,
        "initial_point_count": len(points),
        "initial_points_sha256": sha256_file(building / POINTS_FILE),
        "initial_colors_sha256": sha256_file(building / COLORS_FILE),
        "images": image_records,
    }
    (building / PROTOCOL_FILE).write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    validate_prepared_corner(building, verify_image_hashes=True)
    building.rename(output)
    print(f"CORNER_PREPARED={output}")
    print(f"CORNER_TRAIN_IMAGES={len(train_indices)}")
    print(f"CORNER_TEST_IMAGES={len(test_indices)}")
    print(f"CORNER_INITIAL_POINTS={len(points)}")
    print(f"CORNER_INITIAL_POINTS_SHA256={protocol['initial_points_sha256']}")
    print("CORNER_PROTOCOL=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
