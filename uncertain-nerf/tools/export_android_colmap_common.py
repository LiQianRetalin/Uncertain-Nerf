#!/usr/bin/env python3
"""Export Android factor4 as an undistorted PINHOLE common input."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.export_ontogo_colmap_common import (
    _ordered_hash,
    _qvec_to_rotation,
    _read_c_string,
)
from puri_gs.ontogo import sha256_file

OUTPUT_SCHEMA = "puri-gs-android-colmap-common-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_camera(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    with path.open("rb") as stream:
        count, camera_id, model_id, width, height = struct.unpack("<QiiQQ", stream.read(32))
        params = np.array(struct.unpack("<8d", stream.read(64)))
        if stream.read(1):
            raise RuntimeError("source cameras.bin has unexpected trailing bytes")
    if (count, camera_id, model_id) != (1, 1, 4):
        raise RuntimeError("Android source must contain one OPENCV camera with id 1")
    K = np.array([[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]])
    return K, params[4:], width, height


def _read_images(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            values = struct.unpack("<i7di", stream.read(64))
            name = _read_c_string(stream)
            point_count = struct.unpack("<Q", stream.read(8))[0]
            stream.seek(24 * point_count, 1)
            records.append(
                {
                    "source_id": values[0],
                    "qvec": np.array(values[1:5]),
                    "translation": np.array(values[5:8]),
                    "camera_id": values[8],
                    "source_name": name,
                    "name": str(Path(name).with_suffix(".png")),
                }
            )
        if stream.read(1):
            raise RuntimeError("source images.bin has unexpected trailing bytes")
    return sorted(records, key=lambda record: str(record["source_name"]))


def _read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    with path.open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            values = struct.unpack("<Q3d3BdQ", stream.read(51))
            xyz.append(values[1:4])
            rgb.append(values[4:7])
            stream.seek(8 * values[8], 1)
        if stream.read(1):
            raise RuntimeError("source points3D.bin has unexpected trailing bytes")
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def _write_model(
    sparse: Path,
    K: np.ndarray,
    width: int,
    height: int,
    images: list[dict[str, object]],
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    with (sparse / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQ4d", 1, 1, 1, width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    with (sparse / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(images)))
        for image_id, record in enumerate(images, 1):
            stream.write(
                struct.pack(
                    "<i7di",
                    image_id,
                    *record["qvec"],
                    *record["translation"],
                    1,
                )
            )
            stream.write(str(record["name"]).encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", 0))
    with (sparse / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, (point, color) in enumerate(zip(points, colors), 1):
            stream.write(struct.pack("<Q3d3BdQ", point_id, *point, *color, 0.0, 0))


def _validate(
    output: Path,
    K: np.ndarray,
    width: int,
    height: int,
    images: list[dict[str, object]],
    image_manifest: list[dict[str, object]],
    points: np.ndarray,
    colors: np.ndarray,
) -> dict[str, object]:
    sparse = output / "sparse" / "0"
    with (sparse / "cameras.bin").open("rb") as stream:
        camera = struct.unpack("<QiiQQ4d", stream.read(64))
    expected_camera = (1, 1, 1, width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    if not np.allclose(camera, expected_camera, atol=1e-12):
        raise RuntimeError("exported Android camera mismatch")

    max_pose_error = 0.0
    max_projection_error = 0.0
    names: list[str] = []
    with (sparse / "images.bin").open("rb") as stream:
        if struct.unpack("<Q", stream.read(8))[0] != len(images):
            raise RuntimeError("exported Android image count mismatch")
        for index, expected in enumerate(images, 1):
            values = struct.unpack("<i7di", stream.read(64))
            name = _read_c_string(stream)
            point_count = struct.unpack("<Q", stream.read(8))[0]
            if (values[0], values[8], name, point_count) != (index, 1, expected["name"], 0):
                raise RuntimeError(f"exported Android image record mismatch at {index}")
            qvec = np.array(values[1:5])
            translation = np.array(values[5:8])
            expected_rotation = _qvec_to_rotation(expected["qvec"])
            loaded_rotation = _qvec_to_rotation(qvec)
            max_pose_error = max(
                max_pose_error,
                float(np.max(np.abs(loaded_rotation - expected_rotation))),
                float(np.max(np.abs(translation - expected["translation"]))),
            )
            expected_projection = K @ np.column_stack((expected_rotation, expected["translation"]))
            loaded_projection = K @ np.column_stack((loaded_rotation, translation))
            max_projection_error = max(
                max_projection_error,
                float(np.max(np.abs(loaded_projection - expected_projection))),
            )
            names.append(name)

    with (sparse / "points3D.bin").open("rb") as stream:
        if struct.unpack("<Q", stream.read(8))[0] != len(points):
            raise RuntimeError("exported Android point count mismatch")
        for index, (expected_xyz, expected_rgb) in enumerate(zip(points, colors), 1):
            values = struct.unpack("<Q3d3BdQ", stream.read(51))
            if values[0] != index or values[8] != 0:
                raise RuntimeError(f"exported Android point record mismatch at {index}")
            if not np.allclose(values[1:4], expected_xyz, atol=1e-12) or not np.array_equal(values[4:7], expected_rgb):
                raise RuntimeError(f"exported Android point value mismatch at {index}")

    for image_record in image_manifest:
        image_path = output / "images" / str(image_record["name"])
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise RuntimeError(f"exported Android image size mismatch: {image_path}")
        if sha256_file(image_path) != image_record["sha256"]:
            raise RuntimeError(f"exported Android image hash mismatch: {image_path}")

    if max_pose_error > 1e-12 or max_projection_error > 1e-9:
        raise RuntimeError("exported Android pose/projection round-trip mismatch")
    return {
        "schema": "puri-gs-android-colmap-common-validation-v1",
        "decision": "ANDROID_COMMON_COLMAP_VALID",
        "frame_count": len(images),
        "initial_point_count": len(points),
        "ordered_image_name_sha256": _ordered_hash(names),
        "camera_intrinsics_verified": True,
        "poses_verified": True,
        "points_verified": True,
        "image_dimensions_verified": True,
        "image_hashes_verified": True,
        "projection_matrices_verified": True,
        "max_pose_abs_error": max_pose_error,
        "max_projection_matrix_abs_error": max_projection_error,
        "cameras_bin_sha256": sha256_file(sparse / "cameras.bin"),
        "images_bin_sha256": sha256_file(sparse / "images.bin"),
        "points3D_bin_sha256": sha256_file(sparse / "points3D.bin"),
    }


def export(source: Path, output: Path) -> dict[str, object]:
    cv2.setNumThreads(1)
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; refusing overwrite: {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"temporary output already exists: {building}")
    sparse_source = source / "sparse" / "0"
    source_K, distortion, source_width, source_height = _read_camera(sparse_source / "cameras.bin")
    images = _read_images(sparse_source / "images.bin")
    points, colors = _read_points(sparse_source / "points3D.bin")
    source_images = source / "images_4"
    with Image.open(source_images / str(images[0]["source_name"])) as first:
        stored_width, stored_height = first.size
    K = source_K.copy()
    K[:2, :] /= 4
    K[0, :] *= stored_width / (source_width // 4)
    K[1, :] *= stored_height / (source_height // 4)
    if np.any(distortion != 0):
        undistorted_K, roi = cv2.getOptimalNewCameraMatrix(
            K, distortion, (stored_width, stored_height), 0
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, distortion, None, undistorted_K, (stored_width, stored_height), cv2.CV_32FC1
        )
    else:
        undistorted_K, roi = K, (0, 0, stored_width, stored_height)
        mapx = mapy = None
    x, y, width, height = (int(value) for value in roi)
    undistorted_K = undistorted_K.copy()
    undistorted_K[0, 2] -= x
    undistorted_K[1, 2] -= y

    image_dir = building / "images"
    sparse = building / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse.mkdir(parents=True)
    image_manifest: list[dict[str, object]] = []
    for index, record in enumerate(images, 1):
        image = cv2.imread(str(source_images / str(record["source_name"])), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV cannot decode Android image: {record['source_name']}")
        if mapx is not None and mapy is not None:
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        image = image[y : y + height, x : x + width]
        output_image = image_dir / str(record["name"])
        if not cv2.imwrite(str(output_image), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"OpenCV cannot write Android common image: {output_image}")
        image_manifest.append(
            {
                "source_name": record["source_name"],
                "name": record["name"],
                "sha256": sha256_file(output_image),
                "split": str(record["source_name"])[1:].split("0", 1)[0] if str(record["source_name"])[0].isdigit() else "",
            }
        )
        if index == 1 or index % 20 == 0 or index == len(images):
            print(f"EXPORTED_ANDROID_IMAGES={index}/{len(images)}", flush=True)

    _write_model(sparse, undistorted_K, width, height, images, points, colors)
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "purpose": "shared input for RobustSplat and SpotLessSplats SLS-mlp",
        "conversion_only": True,
        "sfm_reestimated": False,
        "physical_data_factor": 4,
        "loader_data_factor": 1,
        "source_dir": str(source),
        "source_cameras_sha256": sha256_file(sparse_source / "cameras.bin"),
        "source_images_sha256": sha256_file(sparse_source / "images.bin"),
        "source_points3D_sha256": sha256_file(sparse_source / "points3D.bin"),
        "frame_count": len(images),
        "train_keyword": "clutter",
        "test_keyword": "extra",
        "train_count": sum("clutter" in str(record["source_name"]).casefold() for record in images),
        "test_count": sum("extra" in str(record["source_name"]).casefold() for record in images),
        "excluded_clean_count": sum("clean" in str(record["source_name"]).casefold() for record in images),
        "image_width": width,
        "image_height": height,
        "camera_model": "PINHOLE",
        "K": undistorted_K.tolist(),
        "initial_point_count": len(points),
        "images": image_manifest,
    }
    (building / "common_input_protocol.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    building.rename(output)
    validation = _validate(
        output, undistorted_K, width, height, images, image_manifest, points, colors
    )
    validation["common_input_protocol_sha256"] = sha256_file(output / "common_input_protocol.json")
    (output / "common_input_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"COMMON_INPUT_PROTOCOL={output / 'common_input_protocol.json'}")
    print("ANDROID_COMMON_COLMAP=PASS")
    return manifest


def main() -> int:
    args = parse_args()
    export(args.source_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
