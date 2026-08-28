from pathlib import Path

from puri_gs.config import load_experiment_config, trainer_method_args
from run_puri_gs import _apply_runtime_overrides
from scripts.verify_gsplat_robot_install import _has_compatible_architecture


ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return load_experiment_config(ROOT / "configs" / name)


def test_b0_b1_a1_profiles_have_only_intended_method_differences():
    b0 = _load("puri_gs_b0_default.yaml")
    b1 = _load("puri_gs_b1_absgrad.yaml")
    a1 = _load("puri_gs_a1_responsibility.yaml")
    assert b0["training"] == b1["training"] == a1["training"]
    assert b0["strategy"] == {
        "type": "default",
        "absgrad": False,
        "grow_grad2d": 0.0002,
    }
    assert b1["strategy"] == a1["strategy"] == {
        "type": "default",
        "absgrad": True,
        "grow_grad2d": 0.0006,
    }
    assert not b0["responsibility"]["enabled"]
    assert not b1["responsibility"]["enabled"]
    assert a1["responsibility"] == {
        "enabled": True,
        "start_step": 3000,
        "threshold": 1.5,
        "min_weight": 0.2,
        "pool_size": 3,
        "epsilon": 1e-6,
    }


def test_cli_translation_enables_only_the_selected_features():
    b0_args = trainer_method_args(_load("puri_gs_b0_default.yaml"))
    b1_args = trainer_method_args(_load("puri_gs_b1_absgrad.yaml"))
    a1_args = trainer_method_args(_load("puri_gs_a1_responsibility.yaml"))
    assert "--strategy.absgrad" not in b0_args
    assert "--responsibility_enabled" not in b0_args
    assert "--strategy.absgrad" in b1_args
    assert "--responsibility_enabled" not in b1_args
    assert "--strategy.absgrad" in a1_args
    assert "--responsibility_enabled" in a1_args


def test_patch_keeps_a1_in_training_loss_and_writes_explicit_split():
    complete_patch = (
        ROOT / "patches" / "gsplat_v1.5.3_robot_screen.patch"
    ).read_text(encoding="utf-8")
    a1_patch = (ROOT / "patches" / "gsplat_v1.5.3_puri_gs_a1.patch").read_text(
        encoding="utf-8"
    )
    assert "loss.backward()" in complete_patch
    for patch in (complete_patch, a1_patch):
        assert "responsibility_weighted_l1" in patch
        assert "dataset_split.json" in patch
        assert "train_keyword" in patch
        assert "test_keyword" in patch
        assert "_is_png_file" in patch
        assert patch.count("rasterization(") == 0
    assert "l1loss, responsibility = responsibility_weighted_l1" in a1_patch
    assert "responsibility_step{step:04d}.png" in a1_patch


def test_smoke_override_changes_only_effective_a1_start_step():
    checked_in = _load("puri_gs_a1_responsibility.yaml")
    effective = _apply_runtime_overrides(checked_in, 3)
    assert checked_in["responsibility"]["start_step"] == 3000
    assert effective["responsibility"]["start_step"] == 3
    assert effective["strategy"] == checked_in["strategy"]


def test_smoke_override_rejects_baseline_profiles():
    import pytest

    with pytest.raises(ValueError, match="only override the A1"):
        _apply_runtime_overrides(_load("puri_gs_b0_default.yaml"), 3)


def test_cuda_architecture_gate_accepts_l20_but_rejects_blackwell_major_12():
    compiled = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    assert _has_compatible_architecture((8, 9), compiled)
    assert _has_compatible_architecture((9, 0), compiled)
    assert not _has_compatible_architecture((12, 0), compiled)
