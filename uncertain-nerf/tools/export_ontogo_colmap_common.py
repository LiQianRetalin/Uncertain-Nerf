#!/usr/bin/env python3
"""Export frozen Patio-High poses and pixels as a common COLMAP input.

This is a format conversion only.  It does not run feature matching, bundle
adjustment, triangulation, or any other SfM estimation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.ontogo import (  # noqa: E402
    PROTOCOL_FILE,
    OnTheGoPatioHighParser,
    sha256_file,
    validate_prepared_patio_high,
)

OUTPUT_SCHEMA = "puri-gs-patio-high-colmap-common-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def _write_cameras(path: Path, K: np.ndarray, width: int, height: int) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack("<QiiQQ", 1, 1, 1, width, height))
        stream.write(struct.pack("<4d", K[0, 0], K[1, 1], K[0, 2], K[1, 2]))


def _rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    # Davenport's matrix below follows the passive-rotation convention; COLMAP's
    # qvec2rotmat convention requires the transpose at this boundary.
    rotation = rotation.T
    matrix = np.array(
        [
            [
                rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
                rotation[1, 0] + rotation[0, 1],
                rotation[2, 0] + rotation[0, 2],
                rotation[1, 2] - rotation[2, 1],
            ],
            [
                rotation[1, 0] + rotation[0, 1],
                rotation[1, 1] - rotation[0, 0] - rotation[2, 2],
                rotation[2, 1] + rotation[1, 2],
                rotation[2, 0] - rotation[0, 2],
            ],
            [
                rotation[2, 0] + rotation[0, 2],
                rotation[2, 1] + rotation[1, 2],
                rotation[2, 2] - rotation[0, 0] - rotation[1, 1],
                rotation[0, 1] - rotation[1, 0],
            ],
            [
                rotation[1, 2] - rotation[2, 1],
                rotation[2, 0] - rotation[0, 2],
                rotation[0, 1] - rotation[1, 0],
                rotation[0, 0] + rotation[1, 1] + rotation[2, 2],
            ],
        ]
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    return qvec if qvec[0] >= 0 else -qvec


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _write_images(path: Path, camtoworlds: np.ndarray, names: list[str]) -> float:
    max_roundtrip_error = 0.0
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(names)))
        for image_id, (camtoworld, name) in enumerate(zip(camtoworlds, names), 1):
            worldtocamera = np.linalg.inv(camtoworld)
            qvec = _rotation_to_qvec(worldtocamera[:3, :3])
            translation = worldtocamera[:3, 3]
            reconstructed = _qvec_to_rotation(qvec)
            max_roundtrip_error = max(
                max_roundtrip_error,
                float(np.max(np.abs(reconstructed - worldtocamera[:3, :3]))),
            )
            stream.write(struct.pack("<i7di", image_id, *qvec, *translation, 1))
            stream.write(name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", 0))
    return max_roundtrip_error


def _write_points(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, (xyz, rgb) in enumerate(zip(points, colors), 1):
            stream.write(struct.pack("<Q3d3BdQ", point_id, *xyz, *rgb, 0.0, 0))


def _write_image(
    source: Path,
    output: Path,
    mapx: np.ndarray | None,
    mapy: np.ndarray | None,
    roi: tuple[int, int, int, int],
) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV cannot decode prepared image: {source}")
    if mapx is not None and mapy is not None:
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    x, y, width, height = roi
    image = image[y : y + height, x : x + width]
    if not cv2.imwrite(str(output), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"OpenCV cannot write common-input image: {output}")


def _ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _read_c_string(stream) -> str:
    value = bytearray()
    while True:
        byte = stream.read(1)
        if byte == b"\0":
            return value.decode("utf-8")
        if not byte:
            raise RuntimeError("unterminated COLMAP image name")
        value.extend(byte)


def validate_common_input(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = output / "common_input_protocol.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parser = OnTheGoPatioHighParser(str(source), normalize=False)
    sparse = output / "sparse" / "0"
    K = parser.Ks_dict[1]
    width, height = parser.imsize_dict[1]

    with (sparse / "cameras.bin").open("rb") as stream:
        camera = struct.unpack("<QiiQQ", stream.read(32))
        camera_params = np.array(struct.unpack("<4d", stream.read(32)))
        if stream.read(1):
            raise RuntimeError("cameras.bin has unexpected trailing bytes")
    expected_params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    if camera != (1, 1, 1, width, height) or not np.allclose(
        camera_params, expected_params, atol=1e-12
    ):
        raise RuntimeError("COLMAP camera does not match the frozen Patio camera")

    max_pose_error = 0.0
    max_projection_error = 0.0
    names: list[str] = []
    with (sparse / "images.bin").open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        if count != len(parser.image_names):
            raise RuntimeError("images.bin camera count mismatch")
        for index, expected_name in enumerate(parser.image_names, 1):
            values = struct.unpack("<i7di", stream.read(64))
            image_id, qvec, translation, camera_id = (
                values[0],
                np.array(values[1:5]),
                np.array(values[5:8]),
                values[8],
            )
            name = _read_c_string(stream)
            point_count = struct.unpack("<Q", stream.read(8))[0]
            if (image_id, camera_id, name, point_count) != (index, 1, expected_name, 0):
                raise RuntimeError(f"images.bin record mismatch at image {index}")
            names.append(name)
            worldtocamera = np.linalg.inv(parser.camtoworlds[index - 1])
            loaded_rotation = _qvec_to_rotation(qvec)
            pose_error = max(
                float(np.max(np.abs(loaded_rotation - worldtocamera[:3, :3]))),
                float(np.max(np.abs(translation - worldtocamera[:3, 3]))),
            )
            loaded_projection = K @ np.column_stack((loaded_rotation, translation))
            expected_projection = K @ worldtocamera[:3]
            max_pose_error = max(max_pose_error, pose_error)
            max_projection_error = max(
                max_projection_error,
                float(np.max(np.abs(loaded_projection - expected_projection))),
            )
        if stream.read(1):
            raise RuntimeError("images.bin has unexpected trailing bytes")

    with (sparse / "points3D.bin").open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        if count != len(parser.points):
            raise RuntimeError("points3D.bin point count mismatch")
        for index, (expected_xyz, expected_rgb) in enumerate(
            zip(parser.points, parser.points_rgb), 1
        ):
            values = struct.unpack("<Q3d3BdQ", stream.read(51))
            if values[0] != index or values[8] != 0:
                raise RuntimeError(f"points3D.bin record mismatch at point {index}")
            if not np.allclose(values[1:4], expected_xyz, atol=1e-12) or not np.array_equal(
                values[4:7], expected_rgb
            ):
                raise RuntimeError(f"points3D.bin value mismatch at point {index}")
        if stream.read(1):
            raise RuntimeError("points3D.bin has unexpected trailing bytes")

    image_records = manifest["images"]
    if len(image_records) != len(names):
        raise RuntimeError("common-input image manifest count mismatch")
    for index, (name, record) in enumerate(zip(names, image_records), 1):
        image_path = output / "images" / name
        if record["name"] != name or record["sha256"] != sha256_file(image_path):
            raise RuntimeError(f"common-input image hash mismatch at image {index}")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise RuntimeError(f"common-input image size mismatch at image {index}")
        if index == 1 or index % 50 == 0 or index == len(names):
            print(f"VALIDATED_IMAGES={index}/{len(names)}", flush=True)

    validation = {
        "schema": "puri-gs-patio-high-colmap-common-validation-v1",
        "decision": "PATIO_HIGH_COMMON_COLMAP_VALID",
        "common_input_protocol_sha256": sha256_file(manifest_path),
        "frame_count": len(names),
        "initial_point_count": len(parser.points),
        "ordered_image_name_sha256": _ordered_hash(names),
        "image_hashes_verified": True,
        "camera_intrinsics_verified": True,
        "poses_verified": True,
        "points_verified": True,
        "projection_matrices_verified": True,
        "max_pose_abs_error": max_pose_error,
        "max_projection_matrix_abs_error": max_projection_error,
        "cameras_bin_sha256": sha256_file(sparse / "cameras.bin"),
        "images_bin_sha256": sha256_file(sparse / "images.bin"),
        "points3D_bin_sha256": sha256_file(sparse / "points3D.bin"),
    }
    expected_hashes = {
        key: manifest[key]
        for key in ("ordered_image_name_sha256", "cameras_bin_sha256", "images_bin_sha256", "points3D_bin_sha256")
    }
    if any(validation[key] != value for key, value in expected_hashes.items()):
        raise RuntimeError("common-input manifest hash mismatch")
    return validation


def export(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; refusing overwrite: {output}")
    contract = validate_prepared_patio_high(source, verify_image_hashes=True)
    parser = OnTheGoPatioHighParser(str(source), normalize=False)
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"temporary output already exists: {building}")
    image_dir = building / "images"
    sparse_dir = building / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    camera_id = 1
    width, height = parser.imsize_dict[camera_id]
    roi = parser.roi_undist_dict.get(camera_id, (0, 0, width, height))
    mapx = parser.mapx_dict.get(camera_id)
    mapy = parser.mapy_dict.get(camera_id)
    image_records: list[dict[str, object]] = []
    for index, (source_image, name) in enumerate(
        zip(contract["image_paths"], parser.image_names), 1
    ):
        output_image = image_dir / name
        _write_image(source_image, output_image, mapx, mapy, roi)
        image_records.append(
            {
                "name": name,
                "sha256": sha256_file(output_image),
                "split": name.split("_", 1)[0],
            }
        )
        if index == 1 or index % 20 == 0 or index == len(parser.image_names):
            print(f"EXPORTED_IMAGES={index}/{len(parser.image_names)}", flush=True)

    with Image.open(image_dir / parser.image_names[0]) as image:
        if image.size != (width, height):
            raise RuntimeError("exported image dimensions do not match the frozen camera")
    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"
    points_path = sparse_dir / "points3D.bin"
    _write_cameras(cameras_path, parser.Ks_dict[camera_id], width, height)
    rotation_error = _write_images(images_path, parser.camtoworlds, parser.image_names)
    if rotation_error > 1e-12:
        raise RuntimeError(f"COLMAP rotation round-trip error is too large: {rotation_error}")
    _write_points(points_path, parser.points, parser.points_rgb)

    source_protocol = source / PROTOCOL_FILE
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "purpose": "shared input for RobustSplat and SpotLessSplats SLS-mlp",
        "conversion_only": True,
        "sfm_reestimated": False,
        "physical_data_factor": 4,
        "loader_data_factor": 1,
        "source_dir": str(source),
        "source_protocol_sha256": sha256_file(source_protocol),
        "source_transforms_sha256": contract["protocol"]["transforms_sha256"],
        "source_split_sha256": contract["protocol"]["split_sha256"],
        "source_initial_points_sha256": contract["protocol"]["initial_points_sha256"],
        "source_initial_colors_sha256": contract["protocol"]["initial_colors_sha256"],
        "frame_count": len(parser.image_names),
        "train_keyword": "clutter",
        "test_keyword": "extra",
        "train_count": 221,
        "test_count": 45,
        "excluded_count": 1,
        "image_width": width,
        "image_height": height,
        "camera_model": "PINHOLE",
        "K": parser.Ks_dict[camera_id].tolist(),
        "initial_point_count": len(parser.points),
        "opencv_c2w_to_colmap_w2c_verified": True,
        "max_rotation_roundtrip_abs_error": rotation_error,
        "ordered_image_name_sha256": _ordered_hash(parser.image_names),
        "cameras_bin_sha256": sha256_file(cameras_path),
        "images_bin_sha256": sha256_file(images_path),
        "points3D_bin_sha256": sha256_file(points_path),
        "images": image_records,
    }
    manifest_path = building / "common_input_protocol.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.copy2(source / "split.json", building / "split.json")
    building.rename(output)
    validation = validate_common_input(source, output)
    (output / "common_input_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"COMMON_INPUT_PROTOCOL={output / manifest_path.name}")
    print("PATIO_HIGH_COMMON_COLMAP=PASS")
    return manifest


def main() -> int:
    args = parse_args()
    if args.validate_existing:
        validation = validate_common_input(args.source_dir, args.output_dir)
        path = args.output_dir.expanduser().resolve() / "common_input_validation.json"
        path.write_text(
            json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"COMMON_INPUT_VALIDATION={path}")
        print("PATIO_HIGH_COMMON_COLMAP_VALIDATION=PASS")
    else:
        export(args.source_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
