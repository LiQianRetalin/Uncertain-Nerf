from tools.summarize_clean_screen import (
    choose_clean_baseline,
    decide_clean_numeric,
    decide_phase2,
)


def _profile(psnr, ssim, lpips, gs, train, fps):
    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "num_GS": gs,
        "train_seconds": train,
        "render_fps": fps,
    }


def test_clean_baseline_selection_and_noninferior_gate():
    scenes = {}
    for scene in ("garden", "room"):
        scenes[scene] = {
            "b0": _profile(25.0, 0.80, 0.20, 1_000_000, 200, 40),
            "b1": _profile(24.95, 0.80, 0.202, 700_000, 150, 42),
            "a1": _profile(24.92, 0.798, 0.205, 680_000, 155, 40),
        }
    baseline = choose_clean_baseline(scenes)
    assert baseline["primary_baseline"] == "b1"
    decision = decide_clean_numeric(scenes, baseline)
    assert decision["numeric_decision"] == "CLEAN_NONINFERIOR"


def test_phase2_decision_matrix():
    assert (
        decide_phase2(
            pairwise_decision="PAIRWISE_STABLE",
            clean_decision="CLEAN_NONINFERIOR",
            edge_classification="LOW_EDGE_SENSITIVITY",
            metrics_stably_improved=2,
            efficiency_pass=True,
            gaussian_or_vram_ok=True,
        )
        == "A1_CONFIRMED_CANDIDATE"
    )
    assert (
        decide_phase2(
            pairwise_decision="PAIRWISE_BORDERLINE",
            clean_decision="CLEAN_NONINFERIOR",
            edge_classification="MODERATE_EDGE_SENSITIVITY",
            metrics_stably_improved=1,
            efficiency_pass=True,
            gaussian_or_vram_ok=True,
        )
        == "A1_BORDERLINE_RETAIN"
    )
    assert (
        decide_phase2(
            pairwise_decision="PAIRWISE_BORDERLINE",
            clean_decision="CLEAN_NONINFERIOR",
            edge_classification="LOW_EDGE_SENSITIVITY",
            metrics_stably_improved=1,
            efficiency_pass=True,
            gaussian_or_vram_ok=True,
        )
        == "A1_BORDERLINE_RETAIN"
    )
    assert (
        decide_phase2(
            pairwise_decision="PAIRWISE_UNSTABLE",
            clean_decision="CLEAN_NONINFERIOR",
            edge_classification="LOW_EDGE_SENSITIVITY",
            metrics_stably_improved=3,
            efficiency_pass=True,
            gaussian_or_vram_ok=True,
        )
        == "A1_REJECT"
    )
