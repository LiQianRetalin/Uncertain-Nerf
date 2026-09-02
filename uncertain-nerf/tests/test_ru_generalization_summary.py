import pytest

from tools.summarize_puri_gs_ru_backbone import decide as decide_backbone
from tools.summarize_puri_gs_ru_generalization import decide


def _run(profile, psnr, ssim, lpips, count=24, gaussian=100, vram=2.0):
    return {
        "path": profile,
        "profile": profile,
        "git_commit": "a" * 40,
        "evaluation_git_commit": "a" * 40,
        "config": {"seed": 42, "total_steps": 30000},
        "split": {"protocol": "every-nth-test", "train": ["train"], "test": [f"{i}.png" for i in range(count)]},
        "environment": {"gpu": "NVIDIA L20"},
        "evaluation_environment": {"gpu": "NVIDIA L20"},
        "metrics": {"psnr": psnr, "ssim": ssim, "lpips": lpips},
        "gaussian_count": gaussian,
        "inference_vram_gib": vram,
        "training_time_seconds": 100.0,
        "checkpoint": {"path": "checkpoint.pt", "bytes": 1, "sha256": "a" * 64},
        "per_image": [
            {"image_name": f"{i}.png", "psnr": psnr, "ssim": ssim, "lpips": lpips}
            for i in range(count)
        ],
    }


def test_garden_pass_requires_quality_compactness_and_efficiency():
    b1 = _run("b1", 30.0, 0.90, 0.10)
    ru = _run("ru", 29.9, 0.896, 0.109, gaussian=105, vram=2.1)
    report, rows = decide(
        "garden", b1, ru, {"decision": "EFFICIENCY_AUDIT_PASS", "fps_ratio": 0.96, "pass": True}
    )
    assert report["decision"] == "GARDEN_CLEAN_PASS"
    assert report["gate"]["pass"] is True
    assert len(rows) == 24


def test_ontogo_accepts_lpips_route_and_view_gate():
    b1 = _run("b1", 20.0, 0.70, 0.20, count=45)
    ru = _run("ru", 20.1, 0.71, 0.17, count=45)
    report, _ = decide(
        "ontogo", b1, ru, {"decision": "EFFICIENCY_AUDIT_PASS", "fps_ratio": 1.0, "pass": True}
    )
    assert report["decision"] == "ONTOGO_DYNAMIC_PASS"


def test_quality_pass_waits_for_required_efficiency_audit():
    b1 = _run("b1", 30.0, 0.90, 0.10)
    ru = _run("ru", 30.0, 0.90, 0.10)
    with pytest.raises(RuntimeError, match=r"3\+3 efficiency audit"):
        decide("garden", b1, ru, None)


def test_backbone_four_outcomes():
    base = {
        "provenance": "RU_PROVENANCE_COMPLETE",
        "android_pairing": "ANDROID_PAIRING_VALID",
        "android_efficiency": "ANDROID_EFFICIENCY_PASS",
        "phase_r": "RU_RECONSTRUCTION_PASS",
        "room_efficiency": "EFFICIENCY_AUDIT_PASS",
    }
    expected = {
        (True, True): "RU_BACKBONE_ACCEPTED",
        (True, False): "RU_CLEAN_COMPACT_ONLY",
        (False, True): "RU_DYNAMIC_WITH_CLEAN_TRADEOFF",
        (False, False): "RU_SCENE_SPECIFIC",
    }
    for (garden, ontogo), decision in expected.items():
        inputs = {
            **base,
            "garden": "GARDEN_CLEAN_PASS" if garden else "GARDEN_CLEAN_FAIL",
            "ontogo": "ONTOGO_DYNAMIC_PASS" if ontogo else "ONTOGO_DYNAMIC_FAIL",
        }
        assert decide_backbone(inputs) == decision
