"""Read-only checks for the single LLFF dataset used by the first 3DGS screen."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np


def _image_files(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _unique_stems(paths: list[Path]) -> tuple[set[str], bool]:
    stems = [path.stem.casefold() for path in paths]
    return set(stems), len(stems) == len(set(stems))


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    if header[12:16] != b"IHDR":
        raise ValueError("PNG has no leading IHDR chunk")
    return struct.unpack(">II", header[16:24])


def _colmap_record_count(path: Path) -> int:
    with path.open("rb") as handle:
        prefix = handle.read(8)
    if len(prefix) != 8:
        raise ValueError("COLMAP binary is shorter than its record-count header")
    return int(struct.unpack("<Q", prefix)[0])


def audit_fern_dataset(dataset: str | Path, expected_images: int = 20) -> dict[str, Any]:
    """Audit Fern without modifying it or requiring pycolmap/COLMAP."""

    root = Path(dataset).expanduser().resolve()
    failures: list[str] = []

    original_images = _image_files(root / "images", {".jpg", ".jpeg", ".png"})
    half_images = _image_files(root / "images_2", {".png"})
    quarter_images = _image_files(root / "images_4", {".png"})
    original_stems, originals_unique = _unique_stems(original_images)
    half_stems, halves_unique = _unique_stems(half_images)
    _, quarters_unique = _unique_stems(quarter_images)

    if not root.is_dir():
        failures.append("dataset root does not exist")
    if len(original_images) != expected_images:
        failures.append(
            f"images count is {len(original_images)}, expected {expected_images}"
        )
    if len(half_images) != expected_images:
        failures.append(f"images_2 count is {len(half_images)}, expected {expected_images}")
    if len(quarter_images) != expected_images:
        failures.append(
            f"images_4 count is {len(quarter_images)}, expected {expected_images}"
        )
    if not originals_unique or not halves_unique or not quarters_unique:
        failures.append("duplicate image stems make frame matching ambiguous")
    if original_stems != half_stems:
        failures.append("images and images_2 do not contain the same frame stems")
    # gsplat's pinned COLMAP parser maps downsampled files to COLMAP image names
    # by sorted position, so images_4 may validly use image000.png-style names.

    dimensions: set[tuple[int, int]] = set()
    for image in half_images:
        try:
            dimensions.add(_png_dimensions(image))
        except (OSError, ValueError) as error:
            failures.append(f"cannot read {image.name}: {error}")
    if len(dimensions) > 1:
        failures.append("images_2 contains inconsistent image dimensions")

    quarter_dimensions: set[tuple[int, int]] = set()
    for image in quarter_images:
        try:
            quarter_dimensions.add(_png_dimensions(image))
        except (OSError, ValueError) as error:
            failures.append(f"cannot read images_4/{image.name}: {error}")
    if len(quarter_dimensions) > 1:
        failures.append("images_4 contains inconsistent image dimensions")
    if quarter_dimensions and quarter_dimensions != {(1008, 756)}:
        failures.append(
            "images_4 dimensions must be 1008x756 for the Fern reproduction protocol"
        )

    sparse = root / "sparse" / "0"
    colmap_counts: dict[str, int | None] = {}
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        path = sparse / filename
        try:
            colmap_counts[filename] = _colmap_record_count(path)
        except (OSError, ValueError) as error:
            colmap_counts[filename] = None
            failures.append(f"cannot read sparse/0/{filename}: {error}")

    if colmap_counts.get("cameras.bin") == 0:
        failures.append("COLMAP model contains no camera")
    if colmap_counts.get("images.bin") != expected_images:
        failures.append(
            "COLMAP registered-image count is "
            f"{colmap_counts.get('images.bin')}, expected {expected_images}"
        )
    if colmap_counts.get("points3D.bin") == 0:
        failures.append("COLMAP model contains no sparse point")

    poses_path = root / "poses_bounds.npy"
    poses_shape: list[int] | None = None
    try:
        poses = np.load(poses_path, allow_pickle=False)
        poses_shape = list(poses.shape)
        if poses.shape != (expected_images, 17):
            failures.append(
                f"poses_bounds.npy shape is {tuple(poses.shape)}, "
                f"expected ({expected_images}, 17)"
            )
        elif not np.isfinite(poses).all():
            failures.append("poses_bounds.npy contains a non-finite value")
        else:
            for index, (near, far) in enumerate(poses[:, -2:]):
                if not (math.isfinite(float(near)) and 0 < near < far):
                    failures.append(f"invalid near/far bounds at frame {index}")
                    break
    except (OSError, ValueError) as error:
        failures.append(f"cannot read poses_bounds.npy: {error}")

    return {
        "protocol": "puri-v8-fern-data-audit-2",
        "dataset": str(root),
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "expected_images": expected_images,
        "counts": {
            "images": len(original_images),
            "images_2": len(half_images),
            "images_4": len(quarter_images),
            "colmap_cameras": colmap_counts.get("cameras.bin"),
            "colmap_images": colmap_counts.get("images.bin"),
            "colmap_points3D": colmap_counts.get("points3D.bin"),
        },
        "images_2_dimensions": [list(value) for value in sorted(dimensions)],
        "images_4_dimensions": [
            list(value) for value in sorted(quarter_dimensions)
        ],
        "poses_bounds_shape": poses_shape,
        "read_only": True,
    }
