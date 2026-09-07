import numpy as np
import pytest
import torch

from puri_gs.static_tracks import (
    SCHEMA_VERSION,
    CameraRecord,
    build_track_components,
    fundamental_matrix,
    mutual_epipolar_matches,
    nearest_pose_neighbors,
    patch_centers,
    triangulate_track,
    canonical_payload_sha256,
    load_static_track_cache,
)


def camera(view_id, center):
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = center
    K = np.array([[30.0, 0.0, 18.0], [0.0, 30.0, 18.0], [0.0, 0.0, 1.0]])
    return CameraRecord(view_id, f"train_{view_id}.png", c2w, K, 36, 36)


def project(cam, point):
    local = np.linalg.inv(cam.camtoworld) @ np.append(point, 1.0)
    pixel = cam.K @ local[:3]
    return pixel[:2] / pixel[2]


def test_three_camera_triangulation_recovers_known_point():
    cameras = [camera(0, (-1, 0, 0)), camera(1, (1, 0, 0)), camera(2, (0, -1, 0))]
    expected = np.array([0.2, 0.1, 5.0])
    xy = np.stack([project(value, expected) for value in cameras])
    result = triangulate_track(cameras, xy)
    assert result.rank >= 3
    assert np.allclose(result.world_xyz, expected, atol=1e-5)
    assert result.reprojection_error_patch_units.max() < 1e-5


def test_tracks_with_fewer_than_three_cameras_are_rejected():
    cameras = [camera(0, (-1, 0, 0)), camera(1, (1, 0, 0))]
    with pytest.raises(ValueError, match="three distinct"):
        triangulate_track(cameras, np.array([[18.0, 18.0], [18.0, 18.0]]))


def test_negative_depth_rank_deficiency_and_bad_reprojection_are_rejected():
    cameras = [camera(0, (-1, 0, 0)), camera(1, (1, 0, 0)), camera(2, (0, -1, 0))]
    behind = np.array([0.0, 0.0, -5.0])
    with pytest.raises(ValueError, match="depth"):
        triangulate_track(cameras, np.stack([project(value, behind) for value in cameras]))
    identical = [camera(i, (0, 0, 0)) for i in range(3)]
    with pytest.raises(ValueError, match="rank"):
        triangulate_track(identical, np.array([[18.0, 18.0]] * 3))
    point = np.array([0.0, 0.0, 5.0])
    observations = np.stack([project(value, point) for value in cameras])
    observations[2] += np.array([8.0, 8.0])
    with pytest.raises(ValueError, match="reprojection"):
        triangulate_track(cameras, observations)


def test_mutual_epipolar_matching_preserves_row_zero_and_xy_order():
    a, b = camera(0, (-1, 0, 0)), camera(1, (1, 0, 0))
    features = torch.eye(36 * 36).reshape(36 * 36, 36, 36)
    pairs, cosine = mutual_epipolar_matches(features, features, fundamental_matrix(a, b))
    assert len(pairs) == 36 * 36
    assert torch.equal(pairs[:, 0], pairs[:, 1])
    assert torch.allclose(cosine, torch.ones_like(cosine))
    centres = patch_centers()
    assert np.array_equal(centres[0], [0.5, 0.5])
    assert np.array_equal(centres[36], [0.5, 1.5])


def test_track_components_require_unique_three_camera_cycle_support():
    tracks = build_track_components({
        (0, 1): torch.tensor([[4, 5]]),
        (1, 2): torch.tensor([[5, 6]]),
        (0, 2): torch.tensor([[4, 6]]),
    })
    assert tracks == [[(0, 4), (1, 5), (2, 6)]]
    assert build_track_components({(0, 1): torch.tensor([[4, 5]])}) == []


def test_pose_neighbor_list_is_four_distinct_deterministic_views():
    cameras = [camera(i, (float(i), 0.0, 0.0)) for i in range(7)]
    first = nearest_pose_neighbors(cameras)
    second = nearest_pose_neighbors(cameras)
    assert first == second
    assert all(len(row) == 4 and len(set(row)) == 4 for row in first)
    assert all(index not in row for index, row in enumerate(first))


def test_cache_schema_is_binary_deterministic_and_contains_test_hash_only(tmp_path):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root_canonical": "/data/garden",
        "dataset_split_sha256": "a", "camera_manifest_sha256": "b",
        "dino_feature_manifest_sha256": "c", "train_basenames": ["train.jpg"],
        "test_basenames_hash_only": "d", "neighbor_view_ids": torch.tensor([[1, 2, 3, 4]]),
        "track_ids": torch.tensor([0]), "track_view_ids": torch.tensor([[0, 1, 2]]),
        "track_patch_xy": torch.zeros(1, 3, 2), "track_world_xyz": torch.ones(1, 3),
        "track_evidence_binary": torch.zeros(1, 36, 36, dtype=torch.uint8),
        "track_match_cosine_for_audit_only": torch.ones(1, 3),
        "track_reprojection_error_patch_units": torch.zeros(1, 3),
        "track_rgb_median": torch.full((1, 3), 0.5), "build_config": {},
        "build_git_commit": "commit",
    }
    payload["track_evidence_binary"][0, 4, 5] = 1
    digest = canonical_payload_sha256(payload)
    assert digest == canonical_payload_sha256(payload)
    payload["payload_sha256"] = digest
    path = tmp_path / "tracks.pt"; torch.save(payload, path)
    loaded = load_static_track_cache(path)
    assert loaded["payload_sha256"] == digest
    assert "test.jpg" not in repr(loaded)


@pytest.mark.parametrize(
    "forbidden_key",
    ["test_basenames", "test_rgb", "test_dino", "b1_depth", "ru_render", "checkpoint"],
)
def test_cache_schema_rejects_forbidden_test_or_runtime_content(tmp_path, forbidden_key):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root_canonical": "/data/garden",
        "dataset_split_sha256": "a", "camera_manifest_sha256": "b",
        "dino_feature_manifest_sha256": "c", "train_basenames": ["train.jpg"],
        "test_basenames_hash_only": "d", "neighbor_view_ids": torch.tensor([[1, 2, 3, 4]]),
        "track_ids": torch.tensor([0]), "track_view_ids": torch.tensor([[0, 1, 2]]),
        "track_patch_xy": torch.zeros(1, 3, 2), "track_world_xyz": torch.ones(1, 3),
        "track_evidence_binary": torch.zeros(1, 36, 36, dtype=torch.uint8),
        "track_match_cosine_for_audit_only": torch.ones(1, 3),
        "track_reprojection_error_patch_units": torch.zeros(1, 3),
        "track_rgb_median": torch.full((1, 3), 0.5), "build_config": {},
        "build_git_commit": "commit", forbidden_key: ["forbidden"],
    }
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    path = tmp_path / "tracks.pt"; torch.save(payload, path)
    with pytest.raises(ValueError, match="forbidden test/runtime fields"):
        load_static_track_cache(path)
