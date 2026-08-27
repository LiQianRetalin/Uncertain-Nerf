"""Transport-neutral camera-pose contract for RGB SfM and FAST-LIVO2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_ALLOWED_SOURCES = {"rgb_sfm", "fast_livo2", "ground_truth"}


def _numbers(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of length {length}")
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class PosePacket:
    """One timestamped RGB camera pose consumed by the Gaussian mapper.

    ``world_from_camera`` is row-major SE(3).  The optional covariance is a
    row-major 6x6 tangent-space covariance and is kept for pose uncertainty.
    """

    timestamp_ns: int
    frame_id: str
    pose_source: str
    world_from_camera: tuple[float, ...]
    intrinsics: tuple[float, ...]
    image_width: int
    image_height: int
    pose_covariance: tuple[float, ...] | None = None


def parse_pose_packet(value: Mapping[str, Any]) -> PosePacket:
    source = str(value.get("pose_source", ""))
    if source not in _ALLOWED_SOURCES:
        raise ValueError(f"pose_source must be one of {sorted(_ALLOWED_SOURCES)}")
    timestamp_ns = int(value["timestamp_ns"])
    width, height = int(value["image_width"]), int(value["image_height"])
    if timestamp_ns < 0 or width <= 0 or height <= 0:
        raise ValueError("timestamp must be non-negative and image dimensions positive")
    frame_id = str(value["frame_id"])
    if not frame_id:
        raise ValueError("frame_id must not be empty")
    covariance_value = value.get("pose_covariance")
    covariance = (
        None
        if covariance_value is None
        else _numbers(covariance_value, 36, "pose_covariance")
    )
    return PosePacket(
        timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        pose_source=source,
        world_from_camera=_numbers(value["world_from_camera"], 16, "world_from_camera"),
        intrinsics=_numbers(value["intrinsics"], 9, "intrinsics"),
        image_width=width,
        image_height=height,
        pose_covariance=covariance,
    )
