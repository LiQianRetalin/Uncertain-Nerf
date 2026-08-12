"""Minimal COLMAP sparse model reader for binary and text models."""

import os
import struct
from dataclasses import dataclass

import numpy as np


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_NAME_TO_PARAMS = {name: count for name, count in CAMERA_MODELS.values()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsic_matrix(self):
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            f, cx, cy = self.params[:3]
            fx = fy = f
        elif self.model == "PINHOLE":
            fx, fy, cx, cy = self.params[:4]
        else:
            raise ValueError(
                f"Camera model {self.model} is unsupported; expected PINHOLE, "
                "SIMPLE_PINHOLE, or SIMPLE_RADIAL"
            )
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def radial_distortion(self):
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return 0.0
        if self.model == "SIMPLE_RADIAL":
            return float(self.params[3])
        raise ValueError(
            f"Camera model {self.model} is unsupported; expected PINHOLE, "
            "SIMPLE_PINHOLE, or SIMPLE_RADIAL"
        )


@dataclass
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

    def rotation(self):
        w, x, y, z = self.qvec
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float64,
        )

    def center(self):
        return -self.rotation().T @ self.tvec


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


def _read_exact(handle, size):
    value = handle.read(size)
    if len(value) != size:
        raise EOFError("Unexpected end of COLMAP binary model")
    return value


def _unpack(handle, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(handle, size))


def _read_c_string(handle):
    chars = []
    while True:
        char = _read_exact(handle, 1)
        if char == b"\x00":
            return b"".join(chars).decode("utf-8")
        chars.append(char)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as handle:
        count = _unpack(handle, "<Q")[0]
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(handle, "<iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")
            model, n_params = CAMERA_MODELS[model_id]
            params = np.array(_unpack(handle, "<" + "d" * n_params), dtype=np.float64)
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as handle:
        count = _unpack(handle, "<Q")[0]
        for _ in range(count):
            fields = _unpack(handle, "<i7di")
            image_id = fields[0]
            qvec = np.array(fields[1:5], dtype=np.float64)
            tvec = np.array(fields[5:8], dtype=np.float64)
            camera_id = fields[8]
            name = _read_c_string(handle)
            n_points = _unpack(handle, "<Q")[0]
            raw = _unpack(handle, "<" + "ddq" * n_points) if n_points else ()
            xys = np.array(raw, dtype=object).reshape(-1, 3)[:, :2].astype(np.float64) if n_points else np.empty((0, 2))
            point_ids = np.array(raw, dtype=object).reshape(-1, 3)[:, 2].astype(np.int64) if n_points else np.empty((0,), dtype=np.int64)
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point_ids)
    return images


def read_points3d_binary(path):
    points = {}
    with open(path, "rb") as handle:
        count = _unpack(handle, "<Q")[0]
        for _ in range(count):
            point_id, x, y, z, _, _, _, error = _unpack(handle, "<QdddBBBd")
            track_len = _unpack(handle, "<Q")[0]
            track = _unpack(handle, "<" + "ii" * track_len) if track_len else ()
            track = np.asarray(track, dtype=np.int32).reshape(-1, 2)
            points[point_id] = Point3D(
                point_id,
                np.array([x, y, z], dtype=np.float64),
                float(error),
                track[:, 0].copy(),
                track[:, 1].copy(),
            )
    return points


def read_cameras_text(path):
    cameras = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            camera_id, model = int(fields[0]), fields[1]
            if model not in CAMERA_NAME_TO_PARAMS:
                raise ValueError(f"Unsupported COLMAP camera model {model}")
            cameras[camera_id] = Camera(
                camera_id, model, int(fields[2]), int(fields[3]),
                np.asarray(fields[4:], dtype=np.float64),
            )
    return cameras


def read_images_text(path):
    images = {}
    with open(path, "r", encoding="utf-8") as handle:
        lines = iter(handle)
        for line in lines:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            image_id = int(fields[0])
            observations = next(lines, "").split()
            triples = np.asarray(observations, dtype=object).reshape(-1, 3) if observations else np.empty((0, 3), dtype=object)
            images[image_id] = Image(
                image_id,
                np.asarray(fields[1:5], dtype=np.float64),
                np.asarray(fields[5:8], dtype=np.float64),
                int(fields[8]),
                " ".join(fields[9:]),
                triples[:, :2].astype(np.float64) if len(triples) else np.empty((0, 2)),
                triples[:, 2].astype(np.int64) if len(triples) else np.empty((0,), dtype=np.int64),
            )
    return images


def read_points3d_text(path):
    points = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            track = np.asarray(fields[8:], dtype=np.int64).reshape(-1, 2)
            point_id = int(fields[0])
            points[point_id] = Point3D(
                point_id,
                np.asarray(fields[1:4], dtype=np.float64),
                float(fields[7]),
                track[:, 0].astype(np.int32),
                track[:, 1].astype(np.int32),
            )
    return points


def read_model(model_dir):
    binary = all(os.path.exists(os.path.join(model_dir, name)) for name in (
        "cameras.bin", "images.bin", "points3D.bin"
    ))
    if binary:
        return (
            read_cameras_binary(os.path.join(model_dir, "cameras.bin")),
            read_images_binary(os.path.join(model_dir, "images.bin")),
            read_points3d_binary(os.path.join(model_dir, "points3D.bin")),
        )
    text = all(os.path.exists(os.path.join(model_dir, name)) for name in (
        "cameras.txt", "images.txt", "points3D.txt"
    ))
    if text:
        return (
            read_cameras_text(os.path.join(model_dir, "cameras.txt")),
            read_images_text(os.path.join(model_dir, "images.txt")),
            read_points3d_text(os.path.join(model_dir, "points3D.txt")),
        )
    raise FileNotFoundError(
        f"{model_dir} must contain cameras/images/points3D as either .bin or .txt"
    )
