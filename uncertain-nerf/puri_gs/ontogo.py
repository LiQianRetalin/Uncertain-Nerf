"""Strict Patio-High loader contract for the PURI-GS RU generalization audit.

The official NeRF On-the-go release stores camera-to-world matrices in
``transforms.json`` and its clutter/extra split in ``split.json``.  This module
loads one prepared Patio-High scene without adding a generic transforms loader or
fallback behaviour.  The preparation step supplies one deterministic sparse
initialization shared by the B1 and RU runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DATASET_FORMAT = "ontogo-patio-high"
PROTOCOL_SCHEMA = "puri-gs-ontogo-patio-high-v1"
PROTOCOL_FILE = "puri_gs_protocol.json"
POINTS_FILE = "initial_points.npy"
COLORS_FILE = "initial_colors.npy"
EXPECTED_FACTOR = 4
EXPECTED_FRAME_COUNT = 267
EXPECTED_TRAIN_COUNT = 221
EXPECTED_TEST_COUNT = 45
EXPECTED_UNASSIGNED = [266]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required Patio-High metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_split(split: Any, frame_count: int) -> tuple[list[int], list[int]]:
    if not isinstance(split, dict) or set(split) != {"clutter", "extra"}:
        raise ValueError("Patio-High split.json must contain only clutter and extra")
    train = split["clutter"]
    test = split["extra"]
    if not isinstance(train, list) or not isinstance(test, list):
        raise ValueError("Patio-High split entries must be index lists")
    if any(type(value) is not int for value in [*train, *test]):
        raise ValueError("Patio-High split indices must be integers")
    if len(train) != EXPECTED_TRAIN_COUNT or len(test) != EXPECTED_TEST_COUNT:
        raise ValueError(
            "Patio-High split must contain exactly 221 clutter and 45 extra frames"
        )
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("Patio-High split contains duplicate indices")
    if set(train).intersection(test):
        raise ValueError("Patio-High clutter and extra splits overlap")
    if any(value < 0 or value >= frame_count for value in [*train, *test]):
        raise ValueError("Patio-High split index is outside transforms.json")
    unassigned = sorted(set(range(frame_count)) - set(train) - set(test))
    if unassigned != EXPECTED_UNASSIGNED:
        raise ValueError(
            f"Patio-High must leave only frame 266 unassigned, got {unassigned}"
        )
    return train, test


def _validate_transforms(transforms: Any) -> list[dict[str, Any]]:
    if not isinstance(transforms, dict):
        raise ValueError("Patio-High transforms.json must be an object")
    frames = transforms.get("frames")
    if not isinstance(frames, list) or len(frames) != EXPECTED_FRAME_COUNT:
        raise ValueError("Patio-High transforms.json must contain exactly 267 frames")
    required = ("fl_x", "fl_y", "cx", "cy", "w", "h", "is_fisheye")
    if any(name not in transforms for name in required):
        raise ValueError("Patio-High transforms.json is missing camera intrinsics")
    if transforms["is_fisheye"] is not False:
        raise ValueError("Patio-High protocol supports only its perspective camera")
    numeric = {name: float(transforms[name]) for name in ("fl_x", "fl_y", "cx", "cy", "w", "h")}
    if any(not np.isfinite(value) for value in numeric.values()):
        raise ValueError("Patio-High camera intrinsics must be finite")
    if numeric["fl_x"] <= 0 or numeric["fl_y"] <= 0 or numeric["w"] <= 0 or numeric["h"] <= 0:
        raise ValueError("Patio-High focal lengths and image dimensions must be positive")
    if float(transforms.get("k3", 0.0)) != 0.0 or float(transforms.get("k4", 0.0)) != 0.0:
        raise ValueError("Patio-High protocol does not silently discard k3 or k4 distortion")

    seen: set[str] = set()
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Patio-High frame {index} is not an object")
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path.startswith("./images/"):
            raise ValueError(f"Patio-High frame {index} has an invalid image path")
        source_name = Path(file_path).name
        if source_name in seen:
            raise ValueError(f"Patio-High duplicate frame image: {source_name}")
        seen.add(source_name)
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"Patio-High frame {index} has an invalid transform")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(f"Patio-High frame {index} transform has an invalid last row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"Patio-High frame {index} rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError(f"Patio-High frame {index} rotation is not proper")
    return frames


def _prepared_name(index: int, source_name: str, train: set[int], test: set[int]) -> str:
    if index in train:
        prefix = "clutter"
    elif index in test:
        prefix = "extra"
    else:
        prefix = "excluded"
    return f"{prefix}_{index:04d}_{Path(source_name).stem}.png"


def validate_prepared_patio_high(
    data_dir: str | Path,
    *,
    factor: int = EXPECTED_FACTOR,
    load_points: bool = True,
    verify_image_hashes: bool = False,
) -> dict[str, Any]:
    """Validate and return the immutable prepared-scene contract."""

    root = Path(data_dir).expanduser().resolve()
    if factor != EXPECTED_FACTOR:
        raise ValueError("Patio-High protocol is frozen at data factor 4")
    transforms_path = root / "transforms.json"
    split_path = root / "split.json"
    protocol_path = root / PROTOCOL_FILE
    points_path = root / POINTS_FILE
    colors_path = root / COLORS_FILE
    image_dir = root / f"images_{factor}"
    for path in (transforms_path, split_path, protocol_path, points_path, colors_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"prepared Patio-High input is missing: {path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"prepared Patio-High image directory is missing: {image_dir}")

    transforms = _read_json(transforms_path)
    frames = _validate_transforms(transforms)
    split = _read_json(split_path)
    train, test = _validate_split(split, len(frames))
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("prepared Patio-High protocol schema is not supported")
    expected_fields = {
        "dataset_format": DATASET_FORMAT,
        "scene": "patio_high",
        "data_factor": EXPECTED_FACTOR,
        "frame_count": EXPECTED_FRAME_COUNT,
        "train_count": EXPECTED_TRAIN_COUNT,
        "test_count": EXPECTED_TEST_COUNT,
        "unassigned_frame_indices": EXPECTED_UNASSIGNED,
    }
    for name, expected in expected_fields.items():
        if protocol.get(name) != expected:
            raise ValueError(
                f"prepared Patio-High protocol field {name!r} must be {expected!r}"
            )
    if protocol.get("train_indices") != train or protocol.get("test_indices") != test:
        raise ValueError("prepared Patio-High protocol split indices do not match split.json")
    if protocol.get("transforms_sha256") != sha256_file(transforms_path):
        raise ValueError("prepared Patio-High transforms.json SHA-256 mismatch")
    if protocol.get("split_sha256") != sha256_file(split_path):
        raise ValueError("prepared Patio-High split.json SHA-256 mismatch")
    if protocol.get("initial_points_sha256") != sha256_file(points_path):
        raise ValueError("prepared Patio-High point initialization SHA-256 mismatch")
    if protocol.get("initial_colors_sha256") != sha256_file(colors_path):
        raise ValueError("prepared Patio-High color initialization SHA-256 mismatch")

    records = protocol.get("images")
    if not isinstance(records, list) or len(records) != EXPECTED_FRAME_COUNT:
        raise ValueError("prepared Patio-High protocol must list exactly 267 images")
    train_set, test_set = set(train), set(test)
    image_paths: list[Path] = []
    image_names: list[str] = []
    for index, (frame, record) in enumerate(zip(frames, records)):
        source_name = Path(frame["file_path"]).name
        expected_name = _prepared_name(index, source_name, train_set, test_set)
        if not isinstance(record, dict) or record.get("frame_index") != index:
            raise ValueError(f"prepared Patio-High image record {index} is invalid")
        if record.get("source_name") != source_name or record.get("file") != expected_name:
            raise ValueError(f"prepared Patio-High image record {index} does not match metadata")
        image_path = image_dir / expected_name
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise FileNotFoundError(f"prepared Patio-High image is missing: {image_path}")
        if verify_image_hashes and record.get("sha256") != sha256_file(image_path):
            raise ValueError(f"prepared Patio-High image SHA-256 mismatch: {expected_name}")
        if verify_image_hashes:
            with Image.open(image_path) as image:
                if [image.width, image.height] != [record.get("width"), record.get("height")]:
                    raise ValueError(f"prepared Patio-High image dimensions mismatch: {expected_name}")
        image_paths.append(image_path)
        image_names.append(expected_name)

    result: dict[str, Any] = {
        "root": root,
        "transforms": transforms,
        "frames": frames,
        "split": split,
        "train_indices": train,
        "test_indices": test,
        "protocol": protocol,
        "image_paths": image_paths,
        "image_names": image_names,
    }
    if load_points:
        points = np.load(points_path, allow_pickle=False)
        colors = np.load(colors_path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.float32:
            raise ValueError("Patio-High initial points must be float32 [N,3]")
        if colors.shape != points.shape or colors.dtype != np.uint8:
            raise ValueError("Patio-High initial colors must be uint8 [N,3]")
        if len(points) != protocol.get("initial_point_count") or len(points) < 10_000:
            raise ValueError("Patio-High sparse initialization count is invalid")
        if not np.isfinite(points).all():
            raise ValueError("Patio-High sparse initialization contains non-finite points")
        result["points"] = points
        result["colors"] = colors
    return result


class OnTheGoPatioHighParser:
    """Parser exposing the exact attribute contract used by gsplat's Dataset."""

    def __init__(
        self,
        data_dir: str,
        factor: int = EXPECTED_FACTOR,
        normalize: bool = False,
        test_every: int = 8,
        calibration_index: int = 0,
    ) -> None:
        del test_every  # The official split.json is mandatory for this protocol.
        contract = validate_prepared_patio_high(data_dir, factor=factor)
        if not 0 <= calibration_index < len(contract["image_paths"]):
            raise ValueError("calibration_index is outside the Patio-High image list")
        transforms = contract["transforms"]
        frames = contract["frames"]

        # Instant-NGP/Blender cameras use +Y up and -Z forward.  gsplat's COLMAP
        # path uses OpenCV cameras with +Y down and +Z forward.
        opencv_from_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
        camtoworlds = np.stack(
            [
                np.asarray(frame["transform_matrix"], dtype=np.float64)
                @ opencv_from_opengl
                for frame in frames
            ]
        )
        points = contract["points"].copy()
        points_rgb = contract["colors"].copy()

        if normalize:
            from datasets.normalize import (  # imported from the pinned gsplat examples
                align_principal_axes,
                similarity_from_cameras,
                transform_cameras,
                transform_points,
            )

            t1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(t1, camtoworlds)
            points = transform_points(t1, points)
            t2 = align_principal_axes(points)
            camtoworlds = transform_cameras(t2, camtoworlds)
            points = transform_points(t2, points)
            transform = t2 @ t1
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                t3 = np.diag([1.0, -1.0, -1.0, 1.0])
                camtoworlds = transform_cameras(t3, camtoworlds)
                points = transform_points(t3, points)
                transform = t3 @ transform
        else:
            transform = np.eye(4)

        camera_id = 1
        source_width = int(transforms["w"])
        source_height = int(transforms["h"])
        k = np.array(
            [
                [float(transforms["fl_x"]), 0.0, float(transforms["cx"])],
                [0.0, float(transforms["fl_y"]), float(transforms["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        k[:2, :] /= factor
        width, height = source_width // factor, source_height // factor
        with Image.open(contract["image_paths"][0]) as first_image:
            actual_width, actual_height = first_image.size
        scale_width = actual_width / width
        scale_height = actual_height / height
        k[0, :] *= scale_width
        k[1, :] *= scale_height
        width, height = actual_width, actual_height

        params = np.array(
            [
                float(transforms.get("k1", 0.0)),
                float(transforms.get("k2", 0.0)),
                float(transforms.get("p1", 0.0)),
                float(transforms.get("p2", 0.0)),
            ],
            dtype=np.float32,
        )
        self.mapx_dict: dict[int, np.ndarray] = {}
        self.mapy_dict: dict[int, np.ndarray] = {}
        self.roi_undist_dict: dict[int, tuple[int, int, int, int]] = {}
        self.mask_dict: dict[int, None] = {camera_id: None}
        if np.any(params != 0):
            import cv2

            k_undist, roi = cv2.getOptimalNewCameraMatrix(
                k, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                k, params, None, k_undist, (width, height), cv2.CV_32FC1
            )
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = tuple(int(value) for value in roi)
            k = k_undist
            width, height = int(roi[2]), int(roi[3])
        else:
            params = np.empty(0, dtype=np.float32)

        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        scene_scale = float(
            np.max(np.linalg.norm(camera_locations - scene_center, axis=1))
        )
        if not np.isfinite(scene_scale) or scene_scale <= 0:
            raise ValueError("Patio-High camera extent is invalid")

        self.data_dir = str(contract["root"])
        self.factor = factor
        self.normalize = normalize
        self.test_every = 0
        self.image_names = contract["image_names"]
        self.image_paths = [str(path) for path in contract["image_paths"]]
        self.camtoworlds = camtoworlds
        self.camera_ids = [camera_id] * len(frames)
        self.Ks_dict = {camera_id: k}
        self.params_dict = {camera_id: params}
        self.imsize_dict = {camera_id: (width, height)}
        self.points = points.astype(np.float32, copy=False)
        self.points_err = np.zeros(len(points), dtype=np.float32)
        self.points_rgb = points_rgb
        self.point_indices: dict[str, np.ndarray] = {}
        self.transform = transform
        self.bounds = np.array([0.01, 1.0])
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": False}
        self.scene_scale = scene_scale
