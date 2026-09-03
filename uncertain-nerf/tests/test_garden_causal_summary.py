import csv
import json
from pathlib import Path

import pytest
import torch

from puri_gs.delayed_absgrad import topology_event_summary
from tools.summarize_puri_gs_garden_causal_2x2 import (
    PROTOCOL,
    _load_run,
    attribution_label,
    build_summary,
    ensure_output_targets_absent,
    paired_bootstrap_ci95,
    render_markdown,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_run(
    root: Path,
    *,
    config_name: str,
    quadrant: str,
    metrics: dict[str, float],
) -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / config_name).read_text(encoding="utf-8")
    )
    names = [f"garden_{index:03d}.JPG" for index in range(24)]
    split = {
        "protocol": "every-nth-test",
        "train": [f"garden_train_{index:03d}.JPG" for index in range(161)],
        "test": names,
    }
    _write_json(root / "config.yaml", config)
    _write_json(root / "dataset_split.json", split)
    _write_json(root / "independent_eval" / "dataset_split.json", split)
    (root / "git_commit.txt").write_text("a" * 40 + "\n", encoding="utf-8")
    (root / "independent_eval" / "git_commit.txt").write_text(
        "a" * 40 + "\n", encoding="utf-8"
    )
    (root / "run_command.txt").write_text(
        "python simple_trainer.py --max_steps 30000\n", encoding="utf-8"
    )
    environment = {
        "repository_commit": "a" * 40,
        "torch": {"gpu": "NVIDIA L20", "version": "2.4.0+cu121", "cuda_runtime": "12.1"},
        "gsplat": {"version": "1.5.3", "commit": "b" * 40},
        "packages": {"torchmetrics": "1.4.0"},
    }
    _write_json(root / "environment.json", environment)
    _write_json(
        root / "train_metrics.json",
        {
            "step": 29_999,
            "training_time_seconds": 1000.0,
            "peak_vram_gib": 6.0,
            "gaussian_count": 1000,
        },
    )
    _write_json(root / "independent_eval" / "test_metrics.json", metrics)
    _write_json(
        root / "independent_eval" / "efficiency_metrics.json",
        {
            "render_fps": 100.0,
            "latency_p50_ms": 10.0,
            "latency_p95_ms": 11.0,
            "gaussian_count": 1000,
            "inference_vram_gib": 2.0,
        },
    )
    _write_json(
        root / "independent_eval" / "ru_validation.json",
        {
            "standard_checkpoint_load_pass": True,
            "evaluation_imported_dino": False,
            "evaluation_loaded_mask_head": False,
            "evaluation_rasterization_count_ratio": 1.0,
        },
    )
    metrics_path = root / "independent_eval" / "per_image_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["image_name", "psnr", "ssim", "lpips"]
        )
        writer.writeheader()
        for name in names:
            writer.writerow({"image_name": name, **metrics})
    checkpoint_path = root / "ckpts" / "ckpt_29999_rank0.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 29_999, "splats": {}}, checkpoint_path)

    M, T = {
        "Y00": (0, 0),
        "Y01": (0, 1),
        "Y10": (1, 0),
        "Y11": (1, 1),
    }[quadrant]
    topology = topology_event_summary(
        delayed_topology=bool(T),
        total_steps=30_000,
        mask_pause_after_reset=300 if M else 0,
    )
    if quadrant in {"Y01", "Y10"}:
        _write_json(
            root / "causal_contract.json",
            {
                "protocol": PROTOCOL,
                "semantic_mask_enabled": bool(M),
                "delayed_topology_enabled": bool(T),
                "M": M,
                "T": T,
                "topology_events": topology,
                "expected_mask_update_count": (
                    30_000 - topology["mask_pause_step_count"] if M else 0
                ),
            },
        )
    if M:
        _write_json(
            root / "DINO_time.json",
            {
                "render_feature_seconds": 123.0,
                "mask_update_count": 30_000 - topology["mask_pause_step_count"],
                "mask_pause_count": topology["mask_pause_step_count"],
            },
        )
        _write_json(
            root / "aux" / "dino_environment.json",
            {
                "trainable_parameter_count": 0,
                "weight_sha256": "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",
                "repository_commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
                "mask_head_parameter_count": 1234,
            },
        )
        _write_json(
            root / "aux" / "training_schedule.json",
            {
                "semantic_mask_enabled": True,
                "delayed_topology_enabled": bool(T),
                "topology_events": topology,
            },
        )


def _gate(passed: bool, psnr: float, ssim: float, lpips: float) -> dict:
    return {
        "pass": passed,
        "delta_psnr": psnr,
        "delta_ssim": ssim,
        "delta_lpips": lpips,
    }


def test_attribution_rule_is_deterministic():
    clean = _gate(True, 0.0, 0.0, 0.0)
    fail = _gate(False, -1.0, -0.01, 0.02)
    assert attribution_label(clean, fail, fail) == "GARDEN_MASK_DOMINANT"
    assert attribution_label(fail, clean, fail) == "GARDEN_TOPOLOGY_DOMINANT"
    assert attribution_label(clean, clean, fail) == "GARDEN_NEGATIVE_INTERACTION"
    assert attribution_label(fail, fail, fail) == "GARDEN_BOTH_CONTRIBUTE"
    conflict = _gate(False, 0.1, -0.01, 0.02)
    assert attribution_label(conflict, conflict, fail) == "GARDEN_CAUSAL_INCONCLUSIVE"


def test_bootstrap_is_seeded_and_output_overwrite_is_rejected(tmp_path: Path):
    assert paired_bootstrap_ci95([1.0, 2.0, 3.0]) == paired_bootstrap_ci95(
        [1.0, 2.0, 3.0]
    )
    markdown, report_json = ensure_output_targets_absent(tmp_path)
    markdown.touch()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        ensure_output_targets_absent(tmp_path)
    markdown.unlink()
    report_json.touch()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        ensure_output_targets_absent(tmp_path)


def test_full_synthetic_four_quadrant_summary(tmp_path: Path):
    specifications = {
        "Y00": ("puri_gs_b1_full30k.yaml", {"psnr": 30.0, "ssim": 0.90, "lpips": 0.10}),
        "Y01": (
            "puri_gs_garden_dg_only_full30k.yaml",
            {"psnr": 30.0, "ssim": 0.90, "lpips": 0.10},
        ),
        "Y10": (
            "puri_gs_garden_mask_only_full30k.yaml",
            {"psnr": 29.0, "ssim": 0.88, "lpips": 0.12},
        ),
        "Y11": ("puri_gs_ru_full30k.yaml", {"psnr": 28.5, "ssim": 0.85, "lpips": 0.15}),
    }
    runs = {}
    for quadrant, (config, metrics) in specifications.items():
        root = tmp_path / quadrant
        _make_run(root, config_name=config, quadrant=quadrant, metrics=metrics)
        if quadrant != "Y00":
            # Historical Garden artifacts predate this provenance-only field,
            # whereas current runs record it. Pairing is still defined by the
            # protocol and the exact ordered train/test filename lists.
            for split_path in (
                root / "dataset_split.json",
                root / "independent_eval" / "dataset_split.json",
            ):
                split = json.loads(split_path.read_text(encoding="utf-8"))
                split["dataset_format"] = "colmap"
                _write_json(split_path, split)
        runs[quadrant] = _load_run(root, quadrant)

    summary = build_summary(runs)
    assert summary["pairing_audit"]["pass"] is True
    assert summary["runs"]["Y10"]["mask"]["mask_update_count"] == 30_000
    assert summary["runs"]["Y10"]["mask"]["mask_pause_count"] == 0
    assert summary["runs"]["Y11"]["mask"]["mask_update_count"] == 29_400
    assert summary["runs"]["Y11"]["mask"]["mask_pause_count"] == 600
    assert summary["attribution"]["label"] == "GARDEN_MASK_DOMINANT"
    assert len(summary["representative_images"]["images"]) == 6
    markdown = render_markdown(summary)
    assert "阶段 U 未实现、未启动" in markdown
    assert "没有执行 3×3" in markdown


def test_checkpoint_with_extra_training_state_is_rejected(tmp_path: Path):
    root = tmp_path / "Y00"
    _make_run(
        root,
        config_name="puri_gs_b1_full30k.yaml",
        quadrant="Y00",
        metrics={"psnr": 30.0, "ssim": 0.9, "lpips": 0.1},
    )
    torch.save(
        {"step": 29_999, "splats": {}, "optimizer": {}},
        root / "ckpts" / "ckpt_29999_rank0.pt",
    )
    with pytest.raises(ValueError, match="only standard step/splats"):
        _load_run(root, "Y00")
