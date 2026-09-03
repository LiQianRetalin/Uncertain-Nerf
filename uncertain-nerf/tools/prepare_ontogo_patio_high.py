#!/usr/bin/env python3
"""Prepare one immutable Patio-High input for paired PURI-GS B1/RU runs."""

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
    DATASET_FORMAT,
    EXPECTED_FACTOR,
    POINTS_FILE,
    PROTOCOL_FILE,
    PROTOCOL_SCHEMA,
    _prepared_name,
    _read_json,
    _validate_split,
    _validate_transforms,
    sha256_file,
    validate_prepared_patio_high,
)


SIFT_FEATURES = 2048
PAIR_GAPS = (1, 5)
LOWE_RATIO = 0.75
MAX_REPROJECTION_ERROR_PX = 2.0
MIN_PARALLAX_DEGREES = 1.0
MAX_RADIUS_IN_CAMERA_EXTENTS = 4.0
MAX_POINTS_PER_PAIR = 512
MAX_INITIAL_POINTS = 100_000
MIN_INITIAL_POINTS = 10_000
VOXELS_PER_CAMERA_EXTENT = 750.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _camera_contract(transforms: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    k = np.array(
        [
            [float(transforms["fl_x"]), 0.0, float(transforms["cx"])],
            [0.0, float(transforms["fl_y"]), float(transforms["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    k[:2, :] /= EXPECTED_FACTOR
    distortion = np.array(
        [
            float(transforms.get("k1", 0.0)),
            float(transforms.get("k2", 0.0)),
            float(transforms.get("p1", 0.0)),
            float(transforms.get("p2", 0.0)),
        ],
        dtype=np.float64,
    )
    return k, distortion


def _opencv_camtoworlds(frames: list[dict[str, Any]]) -> np.ndarray:
    opencv_from_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.stack(
        [
            np.asarray(frame["transform_matrix"], dtype=np.float64)
            @ opencv_from_opengl
            for frame in frames
        ]
    )


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
            raise FileNotFoundError(f"Patio-High source image is missing: {source_path}")
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
                "file": output_name,
                "width": expected_size[0],
                "height": expected_size[1],
                "sha256": sha256_file(output_path),
            }
        )
        print(f"RESIZED={index + 1}/267 {output_name}", flush=True)
    return records


def _mutual_ratio_matches(cv2: Any, descriptors_a: np.ndarray, descriptors_b: np.ndarray):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    backward = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    reverse: dict[int, int] = {}
    for matches in backward:
        if len(matches) == 2 and matches[0].distance < LOWE_RATIO * matches[1].distance:
            reverse[matches[0].queryIdx] = matches[0].trainIdx
    return [
        matches[0]
        for matches in forward
        if len(matches) == 2
        and matches[0].distance < LOWE_RATIO * matches[1].distance
        and reverse.get(matches[0].trainIdx) == matches[0].queryIdx
    ]


def _triangulate_initialization(
    building: Path,
    frames: list[dict[str, Any]],
    train_indices: list[int],
    image_records: list[dict[str, Any]],
    transforms: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required only for the one-time Patio-High sparse triangulation"
        ) from error

    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    k, distortion = _camera_contract(transforms)
    camtoworlds = _opencv_camtoworlds(frames)
    worldtocameras = np.linalg.inv(camtoworlds)
    camera_centers = camtoworlds[:, :3, 3]
    scene_center = camera_centers[train_indices].mean(axis=0)
    scene_scale = float(
        np.max(np.linalg.norm(camera_centers[train_indices] - scene_center, axis=1))
    )
    if not np.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("Patio-High training-camera extent is invalid")

    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    features: dict[int, tuple[list[Any], np.ndarray, np.ndarray]] = {}
    for position, frame_index in enumerate(train_indices):
        image_path = building / f"images_{EXPECTED_FACTOR}" / image_records[frame_index]["file"]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"OpenCV cannot decode prepared image: {image_path}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 16:
            raise RuntimeError(f"too few SIFT features in Patio-High frame {frame_index}")
        features[frame_index] = (keypoints, descriptors, bgr[:, :, ::-1])
        print(
            f"FEATURES={position + 1}/{len(train_indices)} frame={frame_index} count={len(keypoints)}",
            flush=True,
        )

    candidates: list[tuple[int, int, float, np.ndarray, np.ndarray]] = []
    pair_count = 0
    for gap in PAIR_GAPS:
        for position in range(len(train_indices) - gap):
            index_a = train_indices[position]
            index_b = train_indices[position + gap]
            keypoints_a, descriptors_a, rgb_a = features[index_a]
            keypoints_b, descriptors_b, rgb_b = features[index_b]
            matches = _mutual_ratio_matches(cv2, descriptors_a, descriptors_b)
            pair_index = pair_count
            pair_count += 1
            if len(matches) < 8:
                continue

            distorted_a = np.asarray(
                [keypoints_a[match.queryIdx].pt for match in matches], dtype=np.float64
            )
            distorted_b = np.asarray(
                [keypoints_b[match.trainIdx].pt for match in matches], dtype=np.float64
            )
            points_a = cv2.undistortPoints(
                distorted_a.reshape(-1, 1, 2), k, distortion, P=k
            ).reshape(-1, 2)
            points_b = cv2.undistortPoints(
                distorted_b.reshape(-1, 1, 2), k, distortion, P=k
            ).reshape(-1, 2)
            projection_a = k @ worldtocameras[index_a][:3]
            projection_b = k @ worldtocameras[index_b][:3]
            homogeneous = cv2.triangulatePoints(
                projection_a, projection_b, points_a.T, points_b.T
            ).T
            valid_w = np.abs(homogeneous[:, 3]) > 1e-10
            xyz = np.zeros((len(homogeneous), 3), dtype=np.float64)
            xyz[valid_w] = homogeneous[valid_w, :3] / homogeneous[valid_w, 3:4]
            xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)

            camera_a = (worldtocameras[index_a][:3] @ xyz_h.T).T
            camera_b = (worldtocameras[index_b][:3] @ xyz_h.T).T
            projected_a = (projection_a @ xyz_h.T).T
            projected_b = (projection_b @ xyz_h.T).T
            safe_depth = (
                valid_w
                & (camera_a[:, 2] > 1e-6)
                & (camera_b[:, 2] > 1e-6)
                & (np.abs(projected_a[:, 2]) > 1e-10)
                & (np.abs(projected_b[:, 2]) > 1e-10)
            )
            reprojection_a = np.full(len(xyz), np.inf)
            reprojection_b = np.full(len(xyz), np.inf)
            reprojection_a[safe_depth] = np.linalg.norm(
                projected_a[safe_depth, :2] / projected_a[safe_depth, 2:3]
                - points_a[safe_depth],
                axis=1,
            )
            reprojection_b[safe_depth] = np.linalg.norm(
                projected_b[safe_depth, :2] / projected_b[safe_depth, 2:3]
                - points_b[safe_depth],
                axis=1,
            )
            error = np.maximum(reprojection_a, reprojection_b)
            ray_a = xyz - camera_centers[index_a]
            ray_b = xyz - camera_centers[index_b]
            norms = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
            cosine = np.ones(len(xyz))
            nonzero = norms > 1e-12
            cosine[nonzero] = np.sum(ray_a[nonzero] * ray_b[nonzero], axis=1) / norms[nonzero]
            parallax = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            radius = np.linalg.norm(xyz - scene_center, axis=1)
            selector = (
                safe_depth
                & np.isfinite(xyz).all(axis=1)
                & (error <= MAX_REPROJECTION_ERROR_PX)
                & (parallax >= MIN_PARALLAX_DEGREES)
                & (radius <= MAX_RADIUS_IN_CAMERA_EXTENTS * scene_scale)
            )
            selected = np.flatnonzero(selector)
            if len(selected) == 0:
                continue
            selected = selected[np.argsort(error[selected], kind="stable")][
                :MAX_POINTS_PER_PAIR
            ]
            for rank, match_index in enumerate(selected):
                xa, ya = np.rint(distorted_a[match_index]).astype(int)
                xb, yb = np.rint(distorted_b[match_index]).astype(int)
                xa = int(np.clip(xa, 0, rgb_a.shape[1] - 1))
                ya = int(np.clip(ya, 0, rgb_a.shape[0] - 1))
                xb = int(np.clip(xb, 0, rgb_b.shape[1] - 1))
                yb = int(np.clip(yb, 0, rgb_b.shape[0] - 1))
                color = np.rint(
                    (rgb_a[ya, xa].astype(np.float32) + rgb_b[yb, xb].astype(np.float32))
                    / 2.0
                ).astype(np.uint8)
                candidates.append(
                    (rank, pair_index, float(error[match_index]), xyz[match_index], color)
                )
            print(
                f"PAIR={pair_index + 1}/{sum(len(train_indices) - g for g in PAIR_GAPS)} "
                f"frames={index_a},{index_b} accepted={len(selected)}",
                flush=True,
            )

    if not candidates:
        raise RuntimeError("Patio-High triangulation produced no sparse points")
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    voxel_size = scene_scale / VOXELS_PER_CAMERA_EXTENT
    occupied: set[tuple[int, int, int]] = set()
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for _, _, _, point, color in candidates:
        voxel = tuple(np.floor((point - scene_center) / voxel_size).astype(np.int64))
        if voxel in occupied:
            continue
        occupied.add(voxel)
        points.append(point)
        colors.append(color)
        if len(points) == MAX_INITIAL_POINTS:
            break
    if len(points) < MIN_INITIAL_POINTS:
        raise RuntimeError(
            f"Patio-High triangulation produced only {len(points)} unique points; "
            f"at least {MIN_INITIAL_POINTS} are required"
        )
    point_array = np.asarray(points, dtype=np.float32)
    color_array = np.asarray(colors, dtype=np.uint8)
    metadata = {
        "algorithm": "fixed-pose-sift-pair-triangulation-v1",
        "opencv_version": cv2.__version__,
        "seed": 42,
        "training_frames_only": True,
        "sift_features": SIFT_FEATURES,
        "pair_gaps": list(PAIR_GAPS),
        "pair_count": pair_count,
        "lowe_ratio": LOWE_RATIO,
        "max_reprojection_error_px": MAX_REPROJECTION_ERROR_PX,
        "min_parallax_degrees": MIN_PARALLAX_DEGREES,
        "max_radius_in_camera_extents": MAX_RADIUS_IN_CAMERA_EXTENTS,
        "max_points_per_pair": MAX_POINTS_PER_PAIR,
        "voxel_size": voxel_size,
        "candidate_count": len(candidates),
        "point_count": len(point_array),
    }
    return point_array, color_array, metadata


def main() -> int:
    args = parse_args()
    source = args.source_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    building = output.with_name(output.name + ".building")
    if output.exists():
        raise RuntimeError(f"output already exists; refusing overwrite: {output}")
    if building.exists():
        raise RuntimeError(f"incomplete prior output exists; inspect it first: {building}")
    transforms_path = source / "transforms.json"
    split_path = source / "split.json"
    transforms = _read_json(transforms_path)
    frames = _validate_transforms(transforms)
    split = _read_json(split_path)
    train_indices, test_indices = _validate_split(split, len(frames))
    if source.name.casefold() != "patio_high":
        raise ValueError("this protocol accepts only the Patio-High source directory")

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
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "dataset_format": DATASET_FORMAT,
        "scene": "patio_high",
        "data_factor": EXPECTED_FACTOR,
        "frame_count": len(frames),
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "unassigned_frame_indices": [266],
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
    validate_prepared_patio_high(building, verify_image_hashes=True)
    building.rename(output)
    print(f"PATIO_HIGH_PREPARED={output}")
    print(f"PATIO_HIGH_TRAIN_IMAGES={len(train_indices)}")
    print(f"PATIO_HIGH_TEST_IMAGES={len(test_indices)}")
    print(f"PATIO_HIGH_INITIAL_POINTS={len(points)}")
    print(f"PATIO_HIGH_INITIAL_POINTS_SHA256={protocol['initial_points_sha256']}")
    print("PATIO_HIGH_PROTOCOL=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
