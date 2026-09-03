import argparse
import copy
from pathlib import Path

import pytest

from puri_gs.config import (
    causal_factors,
    load_experiment_config,
    trainer_method_args,
    validate_experiment_config,
)
from puri_gs.delayed_absgrad import (
    DelayedAbsGradSchedule,
    mask_update_paused_after_resets,
    topology_event_summary,
)
from run_puri_gs import PROJECT_ROOT, _build_command, _causal_contract


def _load(name: str):
    return load_experiment_config(PROJECT_ROOT / "configs" / name)


def _args(tmp_path: Path, *, checkpoint=None):
    return argparse.Namespace(
        data_factor=None,
        dataset_format="colmap",
        train_keyword=None,
        test_keyword=None,
        checkpoint=checkpoint,
        resume_checkpoint=None,
        cvtr_mask_dir=None,
        dino_repo_dir=tmp_path / "dinov2",
        dino_weight_path=tmp_path / "dino.pth",
        feature_cache_dir=tmp_path / "features",
        max_steps=None,
    )


def test_four_quadrants_resolve_to_exact_orthogonal_factors():
    profiles = {
        "Y00": _load("puri_gs_b1_full30k.yaml"),
        "Y01": _load("puri_gs_garden_dg_only_full30k.yaml"),
        "Y10": _load("puri_gs_garden_mask_only_full30k.yaml"),
        "Y11": _load("puri_gs_ru_full30k.yaml"),
    }
    assert {name: causal_factors(cfg) for name, cfg in profiles.items()} == {
        "Y00": (False, False),
        "Y01": (False, True),
        "Y10": (True, False),
        "Y11": (True, True),
    }
    for config in profiles.values():
        training = config["training"]
        assert config["total_steps"] == 30_000
        assert config["seed"] == 42
        assert config.get("sh_degree", training.get("sh_degree")) == 3
        assert training["data_factor"] == 4
        assert training["test_every"] == 8
        assert config.get("ssim_lambda", training.get("ssim_lambda")) == 0.2
        if config["profile"] == "b1":
            assert config["strategy"]["absgrad"] is True
            assert config["strategy"]["grow_grad2d"] == 0.0006
        else:
            assert config["absgrad"] is True
            assert config["grow_grad2d"] == 0.0006


def test_causal_configs_contain_no_inactive_subsystem_fields():
    dg = _load("puri_gs_garden_dg_only_full30k.yaml")
    mask = _load("puri_gs_garden_mask_only_full30k.yaml")
    assert "dino_model" not in dg
    assert "mask_begin_step" not in dg
    assert "mask_pause_after_reset" not in dg
    assert "densify_start_step" not in mask
    assert "opacity_reset_start_step" not in mask


def test_illegal_or_mislabeled_causal_mixes_are_rejected():
    dg = _load("puri_gs_garden_dg_only_full30k.yaml")
    duplicate_b1 = copy.deepcopy(dg)
    duplicate_b1["delayed_densification"] = False
    with pytest.raises(ValueError, match="missing .* quadrants"):
        validate_experiment_config(duplicate_b1)

    mislabeled = copy.deepcopy(dg)
    mislabeled["causal_variant"] = "mask_only"
    with pytest.raises(ValueError, match="causal_variant"):
        validate_experiment_config(mislabeled)

    inactive_leak = copy.deepcopy(dg)
    inactive_leak["mask_threshold"] = 0.25
    with pytest.raises(ValueError, match="inactive causal subsystem"):
        validate_experiment_config(inactive_leak)


def test_cli_translation_loads_only_the_selected_subsystem(tmp_path):
    dg = _load("puri_gs_garden_dg_only_full30k.yaml")
    mask = _load("puri_gs_garden_mask_only_full30k.yaml")
    ru = _load("puri_gs_ru_full30k.yaml")

    dg_args = trainer_method_args(dg)
    assert "--puri_gs_delayed_topology_enabled" in dg_args
    assert "--puri_gs_mask_enabled" not in dg_args
    assert not any("dino" in item.casefold() for item in dg_args)
    assert not any("mask_" in item for item in dg_args)

    mask_args = trainer_method_args(mask)
    assert "--puri_gs_mask_enabled" in mask_args
    assert "--puri_gs_delayed_topology_enabled" not in mask_args
    assert "--dino_model" in mask_args
    assert not any("densify" in item for item in mask_args)
    assert not any("opacity_reset" in item for item in mask_args)

    ru_args = trainer_method_args(ru)
    assert ru_args[0] == "--puri_gs_ru_enabled"
    assert "--puri_gs_mask_enabled" not in ru_args
    assert "--puri_gs_delayed_topology_enabled" not in ru_args

    dg_command = _build_command(
        _args(tmp_path), dg, tmp_path / "gsplat", tmp_path / "data", tmp_path / "dg"
    )
    dg_joined = " ".join(map(str, dg_command)).casefold()
    assert "dino" not in dg_joined
    assert "feature_cache" not in dg_joined

    mask_command = _build_command(
        _args(tmp_path),
        mask,
        tmp_path / "gsplat",
        tmp_path / "data",
        tmp_path / "mask",
    )
    mask_joined = " ".join(map(str, mask_command))
    assert "--dino_repo_dir" in mask_joined
    assert "--dino_weight_path" in mask_joined
    assert "--feature_cache_dir" in mask_joined


def test_checkpoint_evaluation_for_both_new_quadrants_has_no_training_assets(tmp_path):
    for name in (
        "puri_gs_garden_dg_only_full30k.yaml",
        "puri_gs_garden_mask_only_full30k.yaml",
    ):
        config = _load(name)
        args = _args(tmp_path, checkpoint=tmp_path / "ckpt_29999_rank0.pt")
        args.dino_repo_dir = None
        args.dino_weight_path = None
        args.feature_cache_dir = None
        command = _build_command(
            args,
            config,
            tmp_path / "gsplat",
            tmp_path / "data",
            tmp_path / "eval",
        )
        joined = " ".join(map(str, command)).casefold()
        assert "--ckpt" in command
        assert "dino" not in joined
        assert "feature_cache" not in joined
        assert "puri_gs_mask_enabled" not in joined
        assert "puri_gs_delayed_topology_enabled" not in joined


def test_dg_topology_matches_ru_and_mask_topology_matches_actual_b1():
    b1 = topology_event_summary(delayed_topology=False)
    ru = topology_event_summary(delayed_topology=True)
    dg = _causal_contract(_load("puri_gs_garden_dg_only_full30k.yaml"))
    mask = _causal_contract(_load("puri_gs_garden_mask_only_full30k.yaml"))
    assert dg is not None and mask is not None
    assert dg["topology_events"] == ru
    assert mask["topology_events"]["refine_steps"] == b1["refine_steps"]
    assert mask["topology_events"]["reset_steps"] == b1["reset_steps"] == []
    assert b1["refine_steps"][0] == 600
    assert b1["refine_steps"][-1] == 14_900
    assert b1["refine_event_count"] == 144
    assert ru["refine_steps"][0] == 10_000
    assert ru["refine_steps"][-1] == 19_900
    assert ru["refine_event_count"] == 100
    assert ru["reset_steps"] == [15_000, 18_000]


def test_mask_pause_tracks_only_actual_reset_events():
    schedule = DelayedAbsGradSchedule()
    resets = schedule.reset_steps(30_000)
    paused = [
        step
        for step in range(30_000)
        if mask_update_paused_after_resets(step, resets, 300)
    ]
    assert paused[:2] == [15_001, 15_002]
    assert paused[299] == 15_300
    assert paused[300] == 18_001
    assert paused[-1] == 18_300
    assert len(paused) == 600
    assert not any(
        mask_update_paused_after_resets(step, (), 300) for step in range(30_000)
    )

    mask = _causal_contract(_load("puri_gs_garden_mask_only_full30k.yaml"))
    ru = _causal_contract(_load("puri_gs_ru_full30k.yaml"))
    assert mask is not None and ru is not None
    assert mask["topology_events"]["mask_pause_segments"] == []
    assert mask["expected_mask_update_count"] == 30_000
    assert ru["topology_events"]["mask_pause_step_count"] == 600
    assert ru["expected_mask_update_count"] == 29_400


def test_additive_trainer_patch_preserves_legacy_ru_and_has_no_scene_branch():
    patch = (
        PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_garden_causal.patch"
    ).read_text(encoding="utf-8")
    assert "cfg.puri_gs_ru_enabled or cfg.puri_gs_mask_enabled" in patch
    assert (
        "cfg.puri_gs_ru_enabled or cfg.puri_gs_delayed_topology_enabled" in patch
    )
    assert "if scene" not in patch
    assert "PURI-GS-FACTORS" in patch
    assert "delayed_topology_enabled=self.puri_gs_delayed_topology_enabled" in patch


def test_pinned_b1_reset_expression_is_audited_not_silently_fixed():
    source = (
        PROJECT_ROOT
        / "external"
        / "gsplat-v1.5.3-fresh"
        / "gsplat"
        / "strategy"
        / "default.py"
    ).read_text(encoding="utf-8")
    assert "if step % self.reset_every == 0 & step > 0:" in source
    summary = topology_event_summary(delayed_topology=False)
    assert summary["reset_event_count"] == 0
    assert summary["reset_behavior"].startswith("pinned_gsplat_1.5.3")
