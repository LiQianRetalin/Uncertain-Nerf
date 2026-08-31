import numpy as np
from PIL import Image

from tools.analyze_clean_responsibility_postmortem import (
    HISTOGRAM_EDGES,
    compare_scenes,
    draw_distribution_plot,
    effective_sample_size,
    nearest_training_views,
    pose_neighbor_scores,
    rank_worst_test_views,
    responsibility_image_statistics,
    summarize_scene,
)


def test_effective_sample_size_and_requested_q_statistics():
    q = np.array([[1.0, 0.8], [0.5, 0.2]], dtype=np.float32)
    gradient = np.array([[0.0, 0.3], [0.7, 1.0]], dtype=np.float32)
    residual = 1.0 - q
    valid = np.ones_like(q, dtype=bool)

    neff, ratio = effective_sample_size(q, valid)
    expected = float(q.sum() ** 2 / (np.square(q).sum() + 1e-6))
    assert np.isclose(neff, expected)
    assert np.isclose(ratio, expected / q.size)

    stats = responsibility_image_statistics(q, gradient, residual, valid)
    assert np.isclose(stats["q_mean"], q.mean())
    assert np.isclose(stats["q_lt_0_8_ratio"], 0.5)
    assert np.isclose(stats["q_lt_0_5_ratio"], 0.25)
    assert np.isclose(stats["q_eq_0_2_ratio"], 0.25)
    assert 0.0 <= stats["edge_overlap_q_0_8"] <= 1.0
    assert 0.0 <= stats["edge_overlap_q_0_5"] <= 1.0
    assert 0.0 < stats["neff_ratio"] <= 1.0


def test_worst_test_ranking_requires_paired_b1_and_a1_metrics():
    rows = []
    for name, b1, a1 in (("a.jpg", 30.0, 29.5), ("b.jpg", 30.0, 27.0)):
        rows.extend(
            [
                {
                    "scene": "room",
                    "profile": "b1",
                    "image_name": name,
                    "psnr": b1,
                    "ssim": 0.9,
                    "lpips": 0.1,
                },
                {
                    "scene": "room",
                    "profile": "a1",
                    "image_name": name,
                    "psnr": a1,
                    "ssim": 0.89,
                    "lpips": 0.11,
                },
            ]
        )
    ranked = rank_worst_test_views(rows, count=2)
    assert [row["image_name"] for row in ranked] == ["b.jpg", "a.jpg"]
    assert [row["worst_rank"] for row in ranked] == [1, 2]
    assert np.isclose(ranked[0]["a1_minus_b1_psnr"], -3.0)


def test_pose_neighbors_are_deterministic_and_do_not_mutate_poses():
    test_pose = np.eye(4, dtype=np.float64)
    train_poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    train_poses[0, 0, 3] = 0.1
    train_poses[1, 0, 3] = 0.4
    train_poses[2, 0, 3] = 0.2
    train_poses[2, :3, 2] = np.array([0.0, 0.6, 0.8])
    original = train_poses.copy()

    score, position, angle = pose_neighbor_scores(
        test_pose, train_poses, scene_scale=1.0, angular_weight=0.25
    )
    assert score.shape == position.shape == angle.shape == (3,)
    np.testing.assert_array_equal(train_poses, original)

    neighbors = nearest_training_views(
        test_name="test.jpg",
        test_pose=test_pose,
        train_names=["near.jpg", "far.jpg", "angled.jpg"],
        train_poses=train_poses,
        scene_scale=1.0,
        count=2,
    )
    assert [row["train_image_name"] for row in neighbors] == [
        "near.jpg",
        "angled.jpg",
    ]
    assert [row["neighbor_rank"] for row in neighbors] == [1, 2]


def _synthetic_row(scene: str, index: int, offset: float) -> dict:
    return {
        "scene": scene,
        "train_index": index,
        "parser_index": index,
        "image_name": f"{scene}_{index}.jpg",
        "valid_pixel_count": 16,
        "q_mean": 0.9 + offset,
        "q_median": 0.9 + offset,
        "q_p10": 0.7 + offset,
        "q_p90": 1.0,
        "q_lt_0_8_ratio": 0.2 - offset,
        "q_lt_0_5_ratio": 0.1 - offset,
        "q_eq_0_2_ratio": 0.02,
        "edge_overlap_q_0_8": 0.25,
        "edge_overlap_q_0_5": 0.2,
        "spearman_degradation_vs_gradient": 0.3,
        "spearman_degradation_vs_residual": 0.8,
        "neff_pixels": 14.0 + offset,
        "neff_ratio": 0.875 + offset,
        "residual_mean": 0.04 - offset,
    }


def test_scene_summary_comparison_and_distribution_png(tmp_path):
    scene_rows = {
        "garden": [_synthetic_row("garden", 0, 0.0), _synthetic_row("garden", 1, 0.01)],
        "room": [_synthetic_row("room", 0, -0.02), _synthetic_row("room", 1, -0.01)],
    }
    scene_summary = {}
    for scene, rows in scene_rows.items():
        counts = np.arange(1, len(HISTOGRAM_EDGES), dtype=np.int64)
        scene_summary[scene] = summarize_scene(rows, counts)

    comparison = compare_scenes(scene_summary)
    assert comparison["q_mean"]["room_minus_garden"] < 0
    assert comparison["q_lt_0_8_ratio"]["room_minus_garden"] > 0

    output = tmp_path / "distribution.png"
    draw_distribution_plot(output, scene_rows, scene_summary)
    assert output.is_file()
    assert Image.open(output).size == (1500, 900)
