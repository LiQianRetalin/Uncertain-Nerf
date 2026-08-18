import argparse
import json

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("scipy")

from v6.aggregate import summarize
from v6.evaluation import evaluate


def test_complete_v6_evaluation_and_seed_aggregation(tmp_path):
    render = tmp_path / "render"
    gt_depth = tmp_path / "gt-depth"
    render.mkdir()
    gt_depth.mkdir()
    target = np.full((16, 16, 3), 0.5, dtype=np.float32)
    prediction = target.copy()
    prediction[4:8, 4:8] += 0.1
    imageio.imwrite(render / "rgb_000.png", (prediction * 255).astype(np.uint8))
    imageio.imwrite(render / "gt_rgb_000.png", (target * 255).astype(np.uint8))
    uncertainty = np.full((16, 16), 0.1, dtype=np.float32)
    uncertainty[4:8, 4:8] = 0.9
    opacity = np.zeros((16, 16), dtype=np.float32)
    opacity[4:12, 4:12] = 0.9
    np.save(render / "uncertainty_000.npy", uncertainty)
    np.save(render / "acc_000.npy", opacity)
    background = np.ones((16, 16), dtype=np.uint8) * 255
    background[4:12, 4:12] = 0
    imageio.imwrite(render / "background_mask_000.png", background)
    depth = np.ones((16, 16), dtype=np.float32)
    np.save(render / "depth_000.npy", depth * 1.01)
    np.save(gt_depth / "depth_000.npy", depth)
    predicted_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    target_points = predicted_points + 0.001
    np.save(tmp_path / "pred-points.npy", predicted_points)
    np.save(tmp_path / "gt-points.npy", target_points)
    with open(render / "render_efficiency.json", "w", encoding="utf-8") as handle:
        json.dump({"fps": 10.0}, handle)
    training_summary = tmp_path / "training_summary.json"
    with open(training_summary, "w", encoding="utf-8") as handle:
        json.dump({"total_training_seconds": 12.0}, handle)
    args = argparse.Namespace(
        pred_dir=str(render),
        method="uncertain-nerf-v6",
        seed=0,
        compute_lpips=False,
        device="cpu",
        gt_depth_dir=str(gt_depth),
        pred_pointcloud=str(tmp_path / "pred-points.npy"),
        gt_pointcloud=str(tmp_path / "gt-points.npy"),
        training_summary=str(training_summary),
        variance_min=1.0e-4,
        variance_max=0.05,
        bad_error_quantile=0.8,
        floater_threshold=0.1,
        fscore_threshold=0.01,
        output=str(render / "metrics.json"),
    )
    metrics = evaluate(args)
    for key in (
        "PSNR", "SSIM", "AbsRel", "depth_RMSE", "Chamfer_L1", "F_score",
        "background_false_opacity", "floater_pixel_rate", "NLL", "AUSE",
        "AURG", "AURC", "bad_pixel_AUROC", "UCE", "fps",
        "total_training_seconds",
    ):
        assert metrics[key] is not None
    metric_paths = []
    for seed in range(3):
        path = tmp_path / f"metrics-seed{seed}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({**metrics, "seed": seed}, handle)
        metric_paths.append(str(path))
    summary = summarize(metric_paths)
    assert summary["seeds"] == [0, 1, 2]
    assert any(row["metric"] == "PSNR" for row in summary["metrics"])
