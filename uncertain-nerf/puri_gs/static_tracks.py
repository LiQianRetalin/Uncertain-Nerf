"""Render-independent multi-view track construction for RU-PART.

The module is deliberately independent of the training renderer.  It consumes only
training-view camera metadata, frozen GT DINO grids, and (for accepted tracks) GT
RGB samples.  Test basenames are retained as hashes only and are rejected by every
payload loader.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


SCHEMA_VERSION = "puri-gs-ru-part-static-tracks-v1"


@dataclass(frozen=True)
class TrackBuildConfig:
    grid_size: int = 36
    pose_neighbors: int = 4
    epipolar_tolerance_patches: float = 1.0
    reprojection_tolerance_patches: float = 1.0
    minimum_distinct_cameras: int = 3

    def __post_init__(self) -> None:
        if self.grid_size != 36 or self.pose_neighbors != 4:
            raise ValueError("RU-PART fixes grid_size=36 and pose_neighbors=4")
        if self.epipolar_tolerance_patches != 1.0:
            raise ValueError("RU-PART fixes epipolar tolerance to one fine patch")
        if self.reprojection_tolerance_patches != 1.0:
            raise ValueError("RU-PART fixes reprojection tolerance to one fine patch")
        if self.minimum_distinct_cameras != 3:
            raise ValueError("RU-PART requires at least three distinct cameras")


@dataclass(frozen=True)
class CameraRecord:
    view_id: int
    basename: str
    camtoworld: np.ndarray
    K: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class TriangulationResult:
    world_xyz: np.ndarray
    reprojection_error_patch_units: np.ndarray
    rank: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_basenames(names: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(name) for name in names)) + "\n"
    return sha256_bytes(canonical.encode("utf-8"))


def patch_centers(grid_size: int = 36) -> np.ndarray:
    """Return row-major ``(x, y)`` fine-grid cell centres."""

    y, x = np.meshgrid(
        np.arange(grid_size, dtype=np.float64) + 0.5,
        np.arange(grid_size, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    return np.stack((x.reshape(-1), y.reshape(-1)), axis=-1)


def scaled_grid_intrinsics(camera: CameraRecord, grid_size: int = 36) -> np.ndarray:
    K = np.asarray(camera.K, dtype=np.float64).copy()
    K[0, :] *= float(grid_size) / float(camera.width)
    K[1, :] *= float(grid_size) / float(camera.height)
    return K


def world_to_camera(camera: CameraRecord) -> np.ndarray:
    return np.linalg.inv(np.asarray(camera.camtoworld, dtype=np.float64))


def projection_matrix(camera: CameraRecord, grid_size: int = 36) -> np.ndarray:
    return scaled_grid_intrinsics(camera, grid_size) @ world_to_camera(camera)[:3]


def fundamental_matrix(a: CameraRecord, b: CameraRecord, grid_size: int = 36) -> np.ndarray:
    """Compute the rank-two fundamental matrix mapping points in ``a`` to lines in ``b``."""

    Pa = projection_matrix(a, grid_size)
    Pb = projection_matrix(b, grid_size)
    center_a = np.asarray(a.camtoworld, dtype=np.float64)[:, 3]
    epipole_b = Pb @ center_a
    skew = np.array(
        [[0.0, -epipole_b[2], epipole_b[1]],
         [epipole_b[2], 0.0, -epipole_b[0]],
         [-epipole_b[1], epipole_b[0], 0.0]],
        dtype=np.float64,
    )
    F_ab = skew @ Pb @ np.linalg.pinv(Pa)
    norm = np.linalg.norm(F_ab)
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("degenerate camera pair has no finite fundamental matrix")
    return F_ab / norm


def _epipolar_distance(lines: Tensor, points_h: Tensor) -> Tensor:
    numerator = torch.abs(lines @ points_h.T)
    denominator = torch.linalg.vector_norm(lines[:, :2], dim=-1, keepdim=True)
    return numerator / denominator.clamp_min(torch.finfo(lines.dtype).eps)


def mutual_epipolar_matches(
    features_a: Tensor,
    features_b: Tensor,
    F_ab: np.ndarray | Tensor,
    *,
    tolerance: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return deterministic epipolar-constrained mutual nearest feature matches.

    Cosine is used only for ordering.  There is intentionally no dataset-specific
    cosine threshold: geometry, mutuality, track length, positive depth and final
    reprojection are the preregistered gates.
    """

    if features_a.ndim != 3 or features_a.shape != features_b.shape:
        raise ValueError("feature grids must be matching [C,H,W] tensors")
    channels, height, width = features_a.shape
    if (height, width) != (36, 36):
        raise ValueError("RU-PART matching requires the 36x36 fine grid")
    dtype = torch.float64
    device = features_a.device
    centres = torch.as_tensor(patch_centers(width), dtype=dtype, device=device)
    points_h = torch.cat((centres, torch.ones(len(centres), 1, dtype=dtype, device=device)), 1)
    fundamental = torch.as_tensor(F_ab, dtype=dtype, device=device)
    lines_b = points_h @ fundamental.T
    lines_a = points_h @ fundamental
    valid_ab = _epipolar_distance(lines_b, points_h) <= tolerance
    valid_ba = _epipolar_distance(lines_a, points_h) <= tolerance

    fa = F.normalize(features_a.reshape(channels, -1).T.float(), dim=-1)
    fb = F.normalize(features_b.reshape(channels, -1).T.float(), dim=-1)
    cosine = fa @ fb.T
    minus_inf = torch.tensor(float("-inf"), device=device, dtype=cosine.dtype)
    score_ab = torch.where(valid_ab, cosine, minus_inf)
    score_ba = torch.where(valid_ba, cosine.T, minus_inf)
    best_b = score_ab.argmax(dim=1)
    best_a = score_ba.argmax(dim=1)
    ids_a = torch.arange(len(best_b), device=device)
    finite = torch.isfinite(score_ab[ids_a, best_b])
    mutual = finite & (best_a[best_b] == ids_a)
    matched_a = ids_a[mutual]
    matched_b = best_b[mutual]
    return torch.stack((matched_a, matched_b), dim=1).cpu(), cosine[matched_a, matched_b].detach().cpu()


def nearest_pose_neighbors(cameras: Sequence[CameraRecord], count: int = 4) -> list[list[int]]:
    if count != 4:
        raise ValueError("RU-PART fixes four pose neighbours")
    if len(cameras) <= count:
        raise ValueError("not enough cameras for four distinct neighbours")
    centres = np.stack([np.asarray(c.camtoworld)[:3, 3] for c in cameras])
    distances = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1)
    result: list[list[int]] = []
    for index in range(len(cameras)):
        order = np.lexsort((np.arange(len(cameras)), distances[index]))
        result.append([int(i) for i in order if i != index][:count])
    return result


def triangulate_track(
    cameras: Sequence[CameraRecord],
    patch_xy: np.ndarray,
    *,
    grid_size: int = 36,
    near: float = 0.01,
    far: float = 1.0e10,
    tolerance_patches: float = 1.0,
) -> TriangulationResult:
    """DLT triangulation with rank, depth, range and reprojection gates."""

    if len(cameras) < 3 or len({c.view_id for c in cameras}) < 3:
        raise ValueError("track requires at least three distinct cameras")
    xy = np.asarray(patch_xy, dtype=np.float64)
    if xy.shape != (len(cameras), 2) or not np.isfinite(xy).all():
        raise ValueError("track patch coordinates are invalid")
    rows = []
    projections = []
    for camera, (x, y) in zip(cameras, xy):
        P = projection_matrix(camera, grid_size)
        projections.append(P)
        rows.extend((x * P[2] - P[0], y * P[2] - P[1]))
    design = np.stack(rows)
    rank = int(np.linalg.matrix_rank(design))
    if rank < 3:
        raise ValueError("triangulation design matrix rank is below three")
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) <= np.finfo(np.float64).eps:
        raise ValueError("triangulation produced a point at infinity")
    world = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(world).all():
        raise ValueError("triangulation produced non-finite coordinates")
    world_h = np.append(world, 1.0)
    errors = []
    for camera, observed, P in zip(cameras, xy, projections):
        camera_xyz = world_to_camera(camera) @ world_h
        depth = float(camera_xyz[2])
        if not near <= depth <= far:
            raise ValueError("triangulated point violates positive near/far depth")
        projected_h = P @ world_h
        projected = projected_h[:2] / projected_h[2]
        errors.append(float(np.linalg.norm(projected - observed)))
    error_array = np.asarray(errors, dtype=np.float64)
    if not np.isfinite(error_array).all() or np.any(error_array > tolerance_patches):
        raise ValueError("track reprojection exceeds one fine patch")
    return TriangulationResult(world.astype(np.float32), error_array.astype(np.float32), rank)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, item: tuple[int, int]) -> tuple[int, int]:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def components(self) -> list[list[tuple[int, int]]]:
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for item in sorted(self.parent):
            groups.setdefault(self.find(item), []).append(item)
        return [groups[key] for key in sorted(groups)]


def build_track_components(
    pair_matches: Mapping[tuple[int, int], Tensor],
    *,
    minimum_cameras: int = 3,
) -> list[list[tuple[int, int]]]:
    """Merge symmetric pair matches and reject ambiguous per-camera components."""

    uf = _UnionFind()
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for (a, b), pairs in sorted(pair_matches.items()):
        if a >= b:
            raise ValueError("pair-match keys must use ascending distinct view ids")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pair matches must be [N,2]")
        for pa, pb in pairs.tolist():
            left, right = (int(a), int(pa)), (int(b), int(pb))
            uf.union(left, right)
            edges.add((left, right))
    tracks = []
    for component in uf.components():
        views = [node[0] for node in component]
        nodes = set(component)
        degree = {node: 0 for node in component}
        for left, right in edges:
            if left in nodes and right in nodes:
                degree[left] += 1
                degree[right] += 1
        cycle_supported = all(value >= 2 for value in degree.values())
        if len(set(views)) >= minimum_cameras and len(set(views)) == len(views) and cycle_supported:
            tracks.append(component)
    return tracks


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash metadata and tensor bytes deterministically, excluding its own digest."""

    digest = hashlib.sha256()
    for key in sorted(k for k in payload if k != "payload_sha256"):
        digest.update(key.encode("utf-8") + b"\0")
        value = payload[key]
        if isinstance(value, Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            )
        digest.update(b"\n")
    return digest.hexdigest()


def validate_static_track_payload(payload: Mapping[str, Any]) -> None:
    forbidden_exact = {
        "test_basenames",
        "test_rgb",
        "test_dino",
        "b1_depth",
        "ru_render",
        "checkpoint",
    }
    forbidden = sorted(key for key in payload if str(key).lower() in forbidden_exact)
    if forbidden:
        raise ValueError(f"static-track cache contains forbidden test/runtime fields: {forbidden}")
    required = {
        "schema_version", "dataset_root_canonical", "dataset_split_sha256",
        "camera_manifest_sha256", "dino_feature_manifest_sha256", "train_basenames",
        "test_basenames_hash_only", "neighbor_view_ids", "track_ids", "track_view_ids",
        "track_patch_xy", "track_world_xyz", "track_evidence_binary",
        "track_match_cosine_for_audit_only", "track_reprojection_error_patch_units",
        "track_rgb_median", "build_config", "build_git_commit", "payload_sha256",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"static-track cache is missing fields: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("static-track cache schema mismatch")
    if not payload["train_basenames"] or not isinstance(payload["test_basenames_hash_only"], str):
        raise ValueError("static-track split metadata is invalid")
    track_ids = payload["track_ids"]
    xyz = payload["track_world_xyz"]
    evidence = payload["track_evidence_binary"]
    if not isinstance(track_ids, Tensor) or track_ids.ndim != 1 or len(track_ids) == 0:
        raise ValueError("static-track cache contains no valid tracks")
    if not isinstance(xyz, Tensor) or xyz.shape != (len(track_ids), 3) or not torch.isfinite(xyz).all():
        raise ValueError("static-track world coordinates are invalid")
    if not isinstance(evidence, Tensor) or evidence.ndim != 3 or evidence.shape[1:] != (36, 36):
        raise ValueError("track evidence must be [train_views,36,36]")
    unique = torch.unique(evidence)
    if not all(float(v) in (0.0, 1.0) for v in unique):
        raise ValueError("fine-grid track evidence must be binary")
    expected = canonical_payload_sha256(payload)
    if payload["payload_sha256"] != expected:
        raise ValueError("static-track cache payload SHA-256 mismatch")


def load_static_track_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("static-track cache must contain a mapping")
    validate_static_track_payload(payload)
    return payload


def evidence_upsample(evidence: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Reuse RU's bilinear, ``align_corners=False`` fine-grid mapping."""

    if evidence.ndim == 2:
        evidence = evidence[None, None]
    elif evidence.ndim == 3:
        evidence = evidence[:, None]
    if evidence.ndim != 4 or evidence.shape[-2:] != (36, 36):
        raise ValueError("static evidence must originate on the 36x36 fine grid")
    return F.interpolate(evidence.float(), size=image_size, mode="bilinear", align_corners=False)


def build_config_dict(config: TrackBuildConfig) -> dict[str, Any]:
    return asdict(config)
