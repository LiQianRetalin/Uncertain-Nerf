import torch

from tools.compare_puri_gs_b1_checkpoints import compare


def _checkpoint(offset: float = 0.0):
    return {
        "step": 99,
        "splats": {
            "means": torch.zeros(4, 3) + offset,
            "opacities": torch.zeros(4) + offset,
        },
    }


def _metrics(psnr=25.0, ssim=0.9, lpips=0.1):
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}


def test_b1_checkpoint_comparison_uses_metrics_not_tensor_equality():
    result = compare(
        _checkpoint(),
        _checkpoint(1e-3),
        reference_metrics=_metrics(),
        candidate_metrics=_metrics(psnr=25.04, ssim=0.8991, lpips=0.1019),
    )
    assert result["status"] == "B1_REGRESSION_PASS"
    assert result["reference_gaussian_count"] == 4
    assert result["tensor_equality_required"] is False
    assert result["tensor_differences_diagnostic_only"]


def test_b1_checkpoint_comparison_fails_outside_metric_tolerance():
    result = compare(
        _checkpoint(),
        _checkpoint(1e-3),
        reference_metrics=_metrics(),
        candidate_metrics=_metrics(psnr=25.06),
    )
    assert result["status"] == "B1_REGRESSION_FAIL"
