import argparse
import csv
import json
from pathlib import Path

import pytest

import run_puri_gs
from puri_gs.config import load_experiment_config
from run_puri_gs import PROJECT_ROOT, _build_command, _verify_gsplat
from tools.summarize_puri_gs_ru_efficiency_audit import (
    _load_run,
    decide,
    render_markdown,
)


def _run(method: str, run_index: int, fps: float) -> dict:
    return {
        "method": method,
        "run_index": run_index,
        "path": f"/{method}{run_index}",
        "test_image_count": 39,
        "warmup_render_count": 10,
        "gaussian_count": 1000 if method == "b1" else 600,
        "inference_vram_gib": 1.0 if method == "b1" else 0.6,
        "psnr": 30.0 if method == "b1" else 31.0,
        "ssim": 0.9 if method == "b1" else 0.92,
        "lpips": 0.1 if method == "b1" else 0.08,
        "render_fps": fps,
        "latency_mean_ms": 1000.0 / fps,
        "latency_p50_ms": 4.0 if method == "b1" else 3.0,
        "latency_p95_ms": 5.0 if method == "b1" else 4.0,
        "image_names": [f"image_{index:03d}.png" for index in range(39)],
    }


def test_audit_uses_median_of_three_paired_fps_ratios():
    b1 = [_run("b1", index, fps) for index, fps in enumerate((100, 102, 98), 1)]
    ru = [_run("ru", index, fps) for index, fps in enumerate((96, 100, 80), 1)]
    report = decide(b1, ru)
    assert report["original_phase_r_decision_preserved"] is True
    assert report["summary"]["paired_fps_ratios"] == pytest.approx(
        [0.96, 100 / 102, 80 / 98]
    )
    assert report["summary"]["median_paired_fps_ratio"] == pytest.approx(0.96)
    assert report["summary"]["throughput_gate_pass"] is True
    assert report["decision"] == "EFFICIENCY_AUDIT_PASS"
    assert "不覆盖或改写原始 Phase R" in render_markdown(report)


def test_audit_fails_when_median_paired_ratio_is_below_fixed_gate():
    b1 = [_run("b1", index, 100.0) for index in range(1, 4)]
    ru = [_run("ru", index, fps) for index, fps in enumerate((90, 94, 96), 1)]
    report = decide(b1, ru)
    assert report["summary"]["median_paired_fps_ratio"] == pytest.approx(0.94)
    assert report["summary"]["throughput_gate_pass"] is False
    assert report["decision"] == "EFFICIENCY_AUDIT_FAIL"


def test_android_audit_requires_both_fps_and_p95_gates():
    b1 = [_run("b1", index, 100.0) for index in range(1, 4)]
    ru = [_run("ru", index, 100.0) for index in range(1, 4)]
    for run in ru:
        run["latency_p95_ms"] = 5.3
    report = decide(b1, ru, scene="android", p95_ratio_gate=1.05)
    assert report["summary"]["throughput_gate_pass"] is True
    assert report["summary"]["p95_ratio_gate_pass"] is False
    assert report["decision"] == "ANDROID_EFFICIENCY_FAIL"


def test_android_uses_ratio_of_method_medians_for_fps_gate():
    b1 = [_run("b1", 1, 1000.0), _run("b1", 2, 100.0), _run("b1", 3, 1.0)]
    ru = [_run("ru", 1, 96.0), _run("ru", 2, 1.0), _run("ru", 3, 100.0)]
    report = decide(b1, ru, scene="android", p95_ratio_gate=1.05)
    assert report["summary"]["median_paired_fps_ratio"] < 0.95
    assert report["summary"]["median_fps_ratio"] == pytest.approx(0.96)
    assert report["decision"] == "ANDROID_EFFICIENCY_PASS"


def test_generic_efficiency_script_is_counterbalanced_and_serial():
    script = (
        PROJECT_ROOT / "scripts" / "run_puri_gs_efficiency_audit.sh"
    ).read_text(encoding="utf-8")
    order = [
        "run_evaluation b1 1",
        "run_evaluation ru 1",
        "run_evaluation ru 2",
        "run_evaluation b1 2",
        "run_evaluation b1 3",
        "run_evaluation ru 3",
    ]
    assert [script.index(item) for item in order] == sorted(
        script.index(item) for item in order
    )
    assert "--eval-warmup-renders 10" in script
    assert "--eval-disable-image-save" in script
    assert "--p95-ratio-gate 1.05" in script
    assert "checkpoint_sha256.txt" in script
    assert "ANDROID-PAIRING-PRECONDITION-PASS" in script
    assert "AUDIT-OUTPUT-ALREADY-EXISTS" in script
    assert "&" not in "\n".join(
        line for line in script.splitlines() if line.strip().startswith("run_evaluation")
    )
    prepare = (
        PROJECT_ROOT / "scripts" / "prepare_puri_gs_efficiency_audit.sh"
    ).read_text(encoding="utf-8")
    assert "EFFICIENCY-AUDIT-PATCH-READY" in prepare
    assert "EFFICIENCY-PREPARE-UNRECOGNIZED-PATCH-STATE" in prepare


def test_checkpoint_audit_command_enables_only_warmup_and_standard_eval(tmp_path: Path):
    config = load_experiment_config(PROJECT_ROOT / "configs" / "puri_gs_ru_full30k.yaml")
    checkpoint = tmp_path / "ckpt_29999_rank0.pt"
    checkpoint.touch()
    args = argparse.Namespace(
        data_factor=None,
        train_keyword=None,
        test_keyword=None,
        checkpoint=checkpoint,
        resume_checkpoint=None,
        cvtr_mask_dir=None,
        dino_repo_dir=None,
        dino_weight_path=None,
        feature_cache_dir=None,
        max_steps=None,
        eval_warmup_renders=10,
        eval_disable_image_save=True,
    )
    command = _build_command(
        args,
        config,
        tmp_path / "gsplat",
        tmp_path / "data",
        tmp_path / "eval",
    )
    joined = " ".join(map(str, command))
    assert "--eval_warmup_renders 10" in joined
    assert "--eval_disable_image_save" in joined
    assert "--ckpt" in command
    assert "--puri_gs_ru_enabled" not in command
    assert "--dino_repo_dir" not in command


def test_audit_patch_records_warmup_and_raw_per_image_latency():
    patch = (
        PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_efficiency_audit.patch"
    ).read_text(encoding="utf-8")
    assert "eval_warmup_renders" in patch
    assert "per_image_latency.csv" in patch
    assert 'fieldnames=["image_index", "image_name", "latency_ms"]' in patch
    assert "raw_latency_sample_count" in patch
    assert "eval_disable_image_save" in patch


def test_audit_verification_handles_patch_stacked_on_ru_superset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trainer = tmp_path / "examples" / "simple_trainer.py"
    dataset = tmp_path / "examples" / "datasets" / "colmap.py"
    trainer.parent.mkdir(parents=True)
    dataset.parent.mkdir(parents=True)
    trainer.write_text(
        "\n".join(
            (
                "puri_gs_ru_enabled",
                "train_keyword",
                "eval_warmup_renders",
                "eval_disable_image_save",
                "per_image_latency.csv",
            )
        ),
        encoding="utf-8",
    )
    dataset.write_text("_is_png_file\n", encoding="utf-8")

    verified_patches = []
    monkeypatch.setattr(
        run_puri_gs,
        "_git",
        lambda *args, cwd: run_puri_gs.EXPECTED_GSPLAT_COMMIT,
    )
    monkeypatch.setattr(
        run_puri_gs,
        "_verify_applied_patch",
        lambda gsplat_dir, patch_path: verified_patches.append(patch_path.name),
    )

    _verify_gsplat(
        tmp_path,
        require_ru=True,
        require_efficiency_audit=True,
    )

    assert verified_patches == [
        "gsplat_v1.5.3_puri_gs_efficiency_audit.patch"
    ]


def test_training_verification_accepts_efficiency_superset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trainer = tmp_path / "examples" / "simple_trainer.py"
    dataset = tmp_path / "examples" / "datasets" / "colmap.py"
    trainer.parent.mkdir(parents=True)
    dataset.parent.mkdir(parents=True)
    trainer.write_text(
        "puri_gs_ru_enabled\ntrain_keyword\neval_warmup_renders\n"
        "eval_disable_image_save\nper_image_latency.csv\n",
        encoding="utf-8",
    )
    dataset.write_text("_is_png_file\n", encoding="utf-8")
    monkeypatch.setattr(
        run_puri_gs,
        "_git",
        lambda *args, cwd: run_puri_gs.EXPECTED_GSPLAT_COMMIT,
    )
    monkeypatch.setattr(run_puri_gs, "_verify_applied_patch", lambda *args: None)
    _verify_gsplat(tmp_path, require_ru=True)


def test_audit_loader_recomputes_metrics_from_raw_latency_csv(tmp_path: Path):
    names = [f"image_{index:03d}.png" for index in range(39)]
    (tmp_path / "config.yaml").write_text(
        json.dumps({"profile": "b1"}), encoding="utf-8"
    )
    (tmp_path / "dataset_split.json").write_text(
        json.dumps(
            {
                "protocol": "every-nth-test",
                "train": [f"train_{index:03d}.png" for index in range(272)],
                "test": names,
            }
        ),
        encoding="utf-8",
    )
    with (tmp_path / "per_image_latency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["image_index", "image_name", "latency_ms"]
        )
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow(
                {"image_index": index, "image_name": name, "latency_ms": 10.0}
            )
    (tmp_path / "efficiency_metrics.json").write_text(
        json.dumps(
            {
                "render_fps": 100.0,
                "latency_p50_ms": 10.0,
                "latency_p95_ms": 10.0,
                "gaussian_count": 1000,
                "inference_vram_gib": 1.0,
                "warmup_render_count": 10,
                "raw_latency_sample_count": 39,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ru_validation.json").write_text(
        json.dumps(
            {
                "standard_checkpoint_load_pass": True,
                "evaluation_imported_dino": False,
                "evaluation_loaded_mask_head": False,
                "evaluation_rasterization_count_ratio": 1.0,
                "evaluation_warmup_render_count": 10,
                "raw_latency_sample_count": 39,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "test_metrics.json").write_text(
        json.dumps({"psnr": 30.0, "ssim": 0.9, "lpips": 0.1}),
        encoding="utf-8",
    )
    (tmp_path / "run_command.txt").write_text(
        "simple_trainer.py --ckpt /checkpoint.pt --eval_warmup_renders 10\n",
        encoding="utf-8",
    )

    run = _load_run(tmp_path, "b1", 1)
    assert run["render_fps"] == pytest.approx(100.0)
    assert run["latency_mean_ms"] == pytest.approx(10.0)
    assert run["latency_p50_ms"] == pytest.approx(10.0)
    assert run["latency_p95_ms"] == pytest.approx(10.0)
