#!/usr/bin/env python3
"""Build the Garden RU-PART static-track cache without opening test images/features."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.dino_features import FeatureCache, sha256_file
from puri_gs.static_tracks import (
    SCHEMA_VERSION,
    CameraRecord,
    TrackBuildConfig,
    build_config_dict,
    build_track_components,
    canonical_payload_sha256,
    fundamental_matrix,
    hash_basenames,
    mutual_epipolar_matches,
    nearest_pose_neighbors,
    patch_centers,
    triangulate_track,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--dino-weight-path", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=4, choices=(4,))
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1.0e10)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _git_commit(path: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _rel_files(path: Path) -> list[str]:
    return sorted(str(item.relative_to(path)).replace("\\", "/") for item in path.rglob("*") if item.is_file())


def load_camera_records(data_dir: Path, gsplat_dir: Path, factor: int) -> tuple[list[CameraRecord], list[Path]]:
    from pycolmap import SceneManager

    examples = gsplat_dir / "examples"
    sys.path.insert(0, str(examples))
    from datasets.normalize import align_principal_axes, similarity_from_cameras, transform_cameras, transform_points

    manager = SceneManager(str(data_dir / "sparse" / "0"))
    manager.load_cameras(); manager.load_images(); manager.load_points3D()
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
    rows = []
    for image_id, image in manager.images.items():
        camera = manager.cameras[image.camera_id]
        w2c = np.concatenate((np.concatenate((image.R(), image.tvec.reshape(3, 1)), axis=1), bottom), axis=0)
        K = np.array([[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        K[:2] /= factor
        rows.append((image.name, image.camera_id, np.linalg.inv(w2c), K, camera.width // factor, camera.height // factor))
    rows.sort(key=lambda item: item[0])
    camtoworlds = np.stack([item[2] for item in rows])
    points = manager.points3D.astype(np.float64)
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds); points = transform_points(T1, points)
    T2 = align_principal_axes(points)
    camtoworlds = transform_cameras(T2, camtoworlds); points = transform_points(T2, points)
    if np.median(points[:, 2]) > np.mean(points[:, 2]):
        T3 = np.diag([1.0, -1.0, -1.0, 1.0])
        camtoworlds = transform_cameras(T3, camtoworlds)
    originals = sorted(_rel_files(data_dir / "images"))
    downsized = sorted(_rel_files(data_dir / f"images_{factor}"))
    if len(originals) != len(downsized) or len(rows) != len(originals):
        raise ValueError("Garden COLMAP/original/downsampled image mapping is incomplete")
    mapped = dict(zip(originals, downsized))
    image_paths = [data_dir / f"images_{factor}" / mapped[item[0]] for item in rows]
    records = [CameraRecord(i, item[0], camtoworlds[i], item[3], int(item[4]), int(item[5])) for i, item in enumerate(rows)]
    return records, image_paths


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    started = perf_counter()
    config = TrackBuildConfig()
    all_cameras, all_paths = load_camera_records(args.data_dir, args.gsplat_dir, args.factor)
    test_global = list(range(0, len(all_cameras), 8))
    train_global = [i for i in range(len(all_cameras)) if i not in set(test_global)]
    if len(train_global) != 161 or len(test_global) != 24:
        raise ValueError(f"Garden split mismatch: train={len(train_global)} test={len(test_global)}")
    train_names = [all_cameras[i].basename for i in train_global]
    test_names = [all_cameras[i].basename for i in test_global]
    test_set = set(test_names)
    opened: set[str] = set()
    calibration_global = train_global[0]
    calibration_name = all_cameras[calibration_global].basename
    calibration_rgb = imageio.imread(all_paths[calibration_global])
    opened.add(calibration_name)
    actual_height, actual_width = calibration_rgb.shape[:2]
    nominal = all_cameras[calibration_global]
    width_scale, height_scale = actual_width / nominal.width, actual_height / nominal.height
    train_cameras = []
    for local, global_id in enumerate(train_global):
        source = all_cameras[global_id]
        K = source.K.copy(); K[0] *= width_scale; K[1] *= height_scale
        train_cameras.append(CameraRecord(local, source.basename, source.camtoworld, K, int(round(source.width * width_scale)), int(round(source.height * height_scale))))

    cache = FeatureCache(args.feature_cache_dir, expected_weight_sha256=sha256_file(args.dino_weight_path))
    features = []
    for name in train_names:
        if name in test_set:
            raise RuntimeError("test feature access rejected")
        opened.add(name)
        features.append(cache.load(name, 36))

    neighbors = nearest_pose_neighbors(train_cameras)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical static-track matching")
    match_device = torch.device(args.device)
    pair_matches: dict[tuple[int, int], torch.Tensor] = {}
    pair_cosines: dict[tuple[int, int], torch.Tensor] = {}
    for a, adjacent in enumerate(neighbors):
        for b in adjacent:
            key = tuple(sorted((a, b)))
            if key in pair_matches:
                continue
            F_ab = fundamental_matrix(train_cameras[key[0]], train_cameras[key[1]])
            matches, cosine = mutual_epipolar_matches(
                features[key[0]].to(match_device), features[key[1]].to(match_device), F_ab
            )
            pair_matches[key] = matches
            pair_cosines[key] = cosine
    components = build_track_components(pair_matches)

    rgb_grids = []
    for global_id in train_global:
        name = all_cameras[global_id].basename
        if name in test_set:
            raise RuntimeError("test RGB access rejected")
        opened.add(name)
        rgb = torch.from_numpy(np.asarray(imageio.imread(all_paths[global_id])[..., :3]).copy()).float() / 255.0
        rgb = F.interpolate(rgb.permute(2, 0, 1)[None], size=(36, 36), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        rgb_grids.append(rgb)

    centres = patch_centers()
    track_views, track_xy, track_xyz, track_reproj, track_rgb, track_cos = [], [], [], [], [], []
    failures = defaultdict(int)
    for component in components:
        cameras = [train_cameras[view] for view, _ in component]
        xy = np.stack([centres[patch] for _, patch in component])
        try:
            result = triangulate_track(cameras, xy, near=args.near, far=args.far)
        except ValueError as error:
            failures[str(error)] += 1
            continue
        views = [view for view, _ in component]
        patches = [patch for _, patch in component]
        colors = [rgb_grids[v][p // 36, p % 36] for v, p in zip(views, patches)]
        node_cos = []
        for node_index, (view, patch) in enumerate(component):
            incident = []
            for other_view, other_patch in component:
                key = tuple(sorted((view, other_view)))
                if view == other_view or key not in pair_matches:
                    continue
                pairs = pair_matches[key]
                oriented = (patch, other_patch) if view < other_view else (other_patch, patch)
                hit = torch.where((pairs[:, 0] == oriented[0]) & (pairs[:, 1] == oriented[1]))[0]
                if len(hit): incident.append(float(pair_cosines[key][int(hit[0])]))
            node_cos.append(float(np.mean(incident)) if incident else 0.0)
        track_views.append(views); track_xy.append(xy); track_xyz.append(result.world_xyz)
        track_reproj.append(result.reprojection_error_patch_units); track_rgb.append(torch.stack(colors).median(0).values)
        track_cos.append(node_cos)
    if not track_xyz:
        raise RuntimeError("no valid >=3-camera static tracks; stop without training")

    max_nodes = max(len(value) for value in track_views)
    count = len(track_xyz)
    view_tensor = torch.full((count, max_nodes), -1, dtype=torch.int64)
    xy_tensor = torch.full((count, max_nodes, 2), float("nan"), dtype=torch.float32)
    reproj_tensor = torch.full((count, max_nodes), float("nan"), dtype=torch.float32)
    cosine_tensor = torch.full((count, max_nodes), float("nan"), dtype=torch.float32)
    evidence = torch.zeros((len(train_names), 36, 36), dtype=torch.uint8)
    for track_id, (views, xy, reproj, cosine) in enumerate(zip(track_views, track_xy, track_reproj, track_cos)):
        length = len(views)
        view_tensor[track_id, :length] = torch.tensor(views)
        xy_tensor[track_id, :length] = torch.from_numpy(xy)
        reproj_tensor[track_id, :length] = torch.from_numpy(reproj)
        cosine_tensor[track_id, :length] = torch.tensor(cosine)
        for view, point in zip(views, xy):
            x, y = np.floor(point).astype(int)
            evidence[view, y, x] = 1

    camera_manifest = [{"view_id": c.view_id, "basename": c.basename, "camtoworld": np.asarray(c.camtoworld).round(12).tolist(), "K": np.asarray(c.K).round(12).tolist(), "width": c.width, "height": c.height} for c in train_cameras]
    split_hash = hash_basenames(["train:" + value for value in train_names] + ["test:" + value for value in test_names])
    feature_manifest_path = args.feature_cache_dir / "manifest.json"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root_canonical": str(args.data_dir.resolve()),
        "dataset_split_sha256": split_hash,
        "camera_manifest_sha256": hash_basenames([json.dumps(item, sort_keys=True, separators=(",", ":")) for item in camera_manifest]),
        "dino_feature_manifest_sha256": sha256_file(feature_manifest_path),
        "train_basenames": train_names,
        "test_basenames_hash_only": hash_basenames(test_names),
        "neighbor_view_ids": torch.tensor(neighbors, dtype=torch.int64),
        "track_ids": torch.arange(count, dtype=torch.int64),
        "track_view_ids": view_tensor,
        "track_patch_xy": xy_tensor,
        "track_world_xyz": torch.from_numpy(np.stack(track_xyz)).float(),
        "track_evidence_binary": evidence,
        "track_match_cosine_for_audit_only": cosine_tensor,
        "track_reprojection_error_patch_units": reproj_tensor,
        "track_rgb_median": torch.stack(track_rgb).float(),
        "build_config": {**build_config_dict(config), "near": args.near, "far": args.far, "factor": args.factor},
        "build_git_commit": _git_commit(Path(__file__).resolve().parents[2]),
    }
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    if canonical_payload_sha256(payload) != payload["payload_sha256"]:
        raise RuntimeError("deterministic payload hash repeat failed")
    test_opened = len(opened.intersection(test_set))
    if test_opened:
        raise RuntimeError("test basename was opened during static-track build")
    finite_reproj = reproj_tensor[torch.isfinite(reproj_tensor)]
    audit = {
        "train_count": len(train_names), "test_count": len(test_names), "opened_train_basenames": len(opened),
        "test_file_opened_count": test_opened, "pose_neighbors_per_view": 4,
        "neighbor_view_ids": neighbors,
        "mutual_matches": sum(len(value) for value in pair_matches.values()),
        "three_camera_components": len(components), "valid_tracks": count,
        "triangulation_failures": dict(sorted(failures.items())),
        "positive_depth_pass": count,
        "reprojection_error_patch_units": {"p50": float(torch.quantile(finite_reproj, 0.5)), "p95": float(torch.quantile(finite_reproj, 0.95)), "max": float(finite_reproj.max())},
        "track_coverage_per_view": evidence.float().mean((1, 2)).tolist(),
        "world_xyz_finite": bool(torch.isfinite(payload["track_world_xyz"]).all()),
        "world_xyz_min": payload["track_world_xyz"].amin(0).tolist(), "world_xyz_max": payload["track_world_xyz"].amax(0).tolist(),
        "payload_sha256": payload["payload_sha256"], "deterministic_hash_repeat_match": True,
        "build_time_seconds": perf_counter() - started,
    }
    return payload, audit


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir.resolve(); args.feature_cache_dir = args.feature_cache_dir.resolve()
    args.dino_weight_path = args.dino_weight_path.resolve(); args.gsplat_dir = args.gsplat_dir.resolve(); args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite static-track output: {args.output_dir}")
    wall_started = perf_counter()
    payload, audit = build_payload(args)
    repeated_payload, _ = build_payload(args)
    if repeated_payload["payload_sha256"] != payload["payload_sha256"]:
        raise RuntimeError("same-input repeated static-track build produced a different hash")
    audit["deterministic_hash_repeat_match"] = True
    audit["build_time_seconds"] = perf_counter() - wall_started
    args.output_dir.mkdir(parents=True)
    cache_path = args.output_dir / "garden_factor4_static_tracks.pt"
    torch.save(payload, cache_path)
    audit["cache_bytes"] = cache_path.stat().st_size
    audit["cache_file_sha256"] = sha256_file(cache_path)
    (args.output_dir / "static_track_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = ["# RU-PART static-track audit", "", *(f"- {key}: `{value}`" for key, value in audit.items() if key != "track_coverage_per_view")]
    (args.output_dir / "static_track_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("train_count", "test_count", "test_file_opened_count", "valid_tracks", "payload_sha256", "cache_file_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
