import copy

import pytest

from v8_robot.gate import evaluate_candidate
from v8_robot.pose_contract import parse_pose_packet


LIMITS = {
    "protocol": "puri-v8-robot-screen-1",
    "clean_psnr_db_min": 27.2,
    "robust_static_psnr_db_min": 20.0,
    "robust_floater_pixel_rate_max": 0.1,
    "runtime_width": 640,
    "runtime_height": 480,
    "render_fps_min": 30.0,
    "render_p95_ms_max": 33.4,
    "map_update_p95_ms_max": 200.0,
    "peak_vram_mb_max": 4096.0,
    "model_size_mb_max": 512.0,
    "sh_degree_max": 3,
}


def passing_metrics():
    return {
        "protocol": "puri-v8-robot-screen-1",
        "candidate": "gsplat-test",
        "input_mode": "rgb_only",
        "integration": {
            "external_pose_api": True,
            "single_raster_pass": True,
            "rendering_mlp": False,
            "sh_degree": 3,
        },
        "clean": {"psnr_db": 27.2, "ssim": 0.9, "lpips": 0.1},
        "robust": {"static_psnr_db": 20.0, "floater_pixel_rate": 0.1},
        "runtime": {
            "width": 640,
            "height": 480,
            "render_fps": 30.0,
            "render_p95_ms": 33.4,
            "map_update_p95_ms": 200.0,
            "peak_vram_mb": 4096.0,
            "model_size_mb": 512.0,
        },
    }


def test_candidate_at_limits_passes():
    result = evaluate_candidate(passing_metrics(), LIMITS)
    assert result["decision"] == "PASS"
    assert result["failed_checks"] == []


def test_candidate_fails_all_material_regressions():
    metrics = passing_metrics()
    metrics["clean"]["psnr_db"] = 27.19
    metrics["runtime"]["render_fps"] = 29.9
    metrics["integration"]["rendering_mlp"] = True
    result = evaluate_candidate(metrics, LIMITS)
    assert result["decision"] == "FAIL"
    assert set(result["failed_checks"]) >= {
        "clean.psnr_db",
        "runtime.render_fps",
        "rendering_mlp",
    }


def test_missing_metric_is_not_silently_accepted():
    metrics = passing_metrics()
    del metrics["robust"]["static_psnr_db"]
    with pytest.raises(ValueError, match="robust.static_psnr_db"):
        evaluate_candidate(metrics, LIMITS)


def test_non_boolean_contract_and_fractional_resolution_are_rejected():
    metrics = passing_metrics()
    metrics["integration"]["external_pose_api"] = 1
    with pytest.raises(ValueError, match="must be a boolean"):
        evaluate_candidate(metrics, LIMITS)
    metrics = passing_metrics()
    metrics["runtime"]["width"] = 640.5
    with pytest.raises(ValueError, match="must be an integer"):
        evaluate_candidate(metrics, LIMITS)


def test_pose_packet_accepts_both_rgb_and_fast_livo_sources():
    packet = {
        "timestamp_ns": 123,
        "frame_id": "camera",
        "pose_source": "rgb_sfm",
        "world_from_camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "intrinsics": [500, 0, 320, 0, 500, 240, 0, 0, 1],
        "image_width": 640,
        "image_height": 480,
    }
    assert parse_pose_packet(packet).pose_source == "rgb_sfm"
    live = copy.deepcopy(packet)
    live["pose_source"] = "fast_livo2"
    live["pose_covariance"] = [0.0] * 36
    assert parse_pose_packet(live).pose_covariance == (0.0,) * 36


def test_pose_packet_rejects_unplanned_sensor_source():
    with pytest.raises(ValueError, match="pose_source"):
        parse_pose_packet(
            {
                "timestamp_ns": 1,
                "frame_id": "camera",
                "pose_source": "ad_hoc_lidar_depth",
                "world_from_camera": [0.0] * 16,
                "intrinsics": [0.0] * 9,
                "image_width": 640,
                "image_height": 480,
            }
        )
