import copy
import json
from pathlib import Path

import pytest

from v8_robot.reproduction_gate import evaluate_reproduction


ROOT = Path(__file__).resolve().parents[1]
LIMITS = {
    "protocol": "puri-gsplat153-garden-reproduction-1",
    "candidate": "gsplat-1.5.3-default",
    "dataset": "mipnerf360-garden-factor4-test-every8",
    "seed": 42,
    "steps": 30000,
    "reference": {"psnr": 27.32, "ssim": 0.865, "lpips": 0.075, "num_GS": 5840000},
    "acceptance": {
        "psnr_min": 26.82,
        "ssim_min": 0.85,
        "lpips_max": 0.1,
        "num_GS_min": 4500000,
        "num_GS_max": 7000000,
    },
}


def passing_stats():
    return {
        "psnr": 27.32,
        "ssim": 0.865,
        "lpips": 0.075,
        "ellipse_time": 0.01,
        "num_GS": 5840000,
    }


def test_reference_result_passes():
    result = evaluate_reproduction(passing_stats(), LIMITS)
    assert result["decision"] == "PASS"
    assert result["failed_checks"] == []
    assert result["seed"] == 42


def test_repository_limits_match_the_pinned_reference():
    path = ROOT / "configs" / "gsplat153_garden_reproduction_limits.json"
    repository_limits = json.loads(path.read_text(encoding="utf-8"))
    assert repository_limits == LIMITS


def test_material_metric_and_density_regressions_fail():
    stats = copy.deepcopy(passing_stats())
    stats.update(psnr=26.0, lpips=0.2, num_GS=1000000)
    result = evaluate_reproduction(stats, LIMITS)
    assert result["decision"] == "FAIL"
    assert set(result["failed_checks"]) == {"psnr", "lpips", "num_GS_min"}


def test_missing_or_non_finite_metric_is_rejected():
    stats = passing_stats()
    del stats["ssim"]
    with pytest.raises(ValueError, match="ssim must be a number"):
        evaluate_reproduction(stats, LIMITS)
    stats = passing_stats()
    stats["psnr"] = float("nan")
    with pytest.raises(ValueError, match="psnr must be finite"):
        evaluate_reproduction(stats, LIMITS)


def test_fractional_gaussian_count_is_rejected():
    stats = passing_stats()
    stats["num_GS"] = 5840000.5
    with pytest.raises(ValueError, match="num_GS must be an integer"):
        evaluate_reproduction(stats, LIMITS)
