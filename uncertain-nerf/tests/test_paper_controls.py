import argparse
import copy
import csv
import json
from types import SimpleNamespace

import pytest
import torch

from puri_gs.config import load_experiment_config, parse_refine_windows, validate_experiment_config
from puri_gs.delayed_absgrad import DelayedAbsGradSchedule, delayed_strategy_from_default
from puri_gs.paper_controls import (
    MASK_SAMPLE_STEPS, PaperControlRecorder, check_control_diffs, resolved_schedule,
    schedule_from_config, validate_event_files,
)
from puri_gs.ru_training import PURIGSRUTraining
from puri_gs.semantic_mask import hard_static_mask
from run_puri_gs import PROJECT_ROOT, _build_command


def config(name):
    return load_experiment_config(PROJECT_ROOT / "configs" / f"puri_gs_{name}_full30k.yaml")


@pytest.mark.parametrize("name,count,switch,stop", [
    ("ru", 100, 20000, 20000), ("ru_align", 100, 10000, 20000), ("ru_tar", 120, 10000, 24000),
])
def test_real_scale_and_schedule_boundaries(name, count, switch, stop):
    cfg = config(name)
    schedule = schedule_from_config(cfg)
    owner = object.__new__(PURIGSRUTraining)
    owner.cfg = SimpleNamespace(**cfg)
    expected = list(range(10000, 20000, 100))
    if name == "ru_tar":
        expected += list(range(20000, 24000, 200))
    assert list(schedule.refine_steps(30000)) == expected
    assert len(expected) == count == len(set(expected))
    assert schedule.reset_steps(30000) == (15000, 18000)
    assert schedule.statistics_stop_step == stop
    for step in (9999, 10000, 19900, 19999, 20000, 20200, 23800, 23900, 24000, 29999):
        assert schedule.should_refine(step) == (step in expected)
        assert owner.grid_for_step(step) == (16 if step < switch else 36)
    assert not any(schedule.should_reset(s) for s in range(20000, 30000))
    for step, paused in ((15000, False), (15001, True), (15300, True), (15301, False),
                         (18000, False), (18001, True), (18300, True), (18301, False)):
        assert schedule.mask_update_paused(step) is paused


@pytest.mark.parametrize("windows", [
    [], [{"start": 20, "stop": 10, "every": 1}],
    [{"start": 0, "stop": 30001, "every": 1}],
    [{"start": 0, "stop": 20, "every": 0}],
    [{"start": True, "stop": 20, "every": 1}],
    [{"start": 0, "stop": 20, "every": 1}, {"start": 19, "stop": 30, "every": 1}],
])
def test_invalid_windows_are_rejected(windows):
    with pytest.raises(ValueError):
        parse_refine_windows(windows)


def test_exact_single_factor_diffs_and_no_hidden_config_fields():
    ru, align, tar = [config(name) for name in ("ru", "ru_align", "ru_tar")]
    diff = check_control_diffs(ru, align, tar)
    assert diff["RU_Align_minus_RU"]["algorithm_changes"] == {
        "bootstrap_switch_step": {"before": 20000, "after": 10000}}
    assert set(diff["RU_TAR_minus_RU_Align"]["algorithm_changes"]) == {"refine_windows"}
    for name, value in (("mask_threshold", 0.2), ("grow_grad2d", 0.0005),
                        ("densify_stop_step", 24000), ("total_steps", 30001), ("seed", 43),
                        ("new_score", 1), ("sh_degree", 2)):
        changed = dict(tar, **{name: value})
        with pytest.raises(ValueError):
            validate_experiment_config(changed)
    with pytest.raises(ValueError):
        validate_experiment_config(dict(ru, refine_windows=tar["refine_windows"]))
    with pytest.raises(ValueError):
        validate_experiment_config(dict(align, training={"data_factor": 2, "test_every": 8}))


def test_checkpoint_commands_do_not_pass_training_assets(tmp_path):
    args = argparse.Namespace(data_factor=None, dataset_format="colmap", train_keyword=None,
        test_keyword=None, checkpoint=tmp_path / "ckpt.pt", resume_checkpoint=None,
        cvtr_mask_dir=None, max_steps=None, dino_repo_dir=None, dino_weight_path=None,
        feature_cache_dir=None)
    for name in ("ru_align", "ru_tar"):
        command = _build_command(args, config(name), tmp_path, tmp_path, tmp_path)
        assert "--ckpt" in command
        assert not any(token in " ".join(command) for token in ("dino", "feature_cache", "puri_gs_paper_control", "refine_windows"))


def test_real_strategy_split_clone_prune_and_count_identity(monkeypatch):
    from gsplat.strategy import DefaultStrategy
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    strategy = delayed_strategy_from_default(DefaultStrategy(grow_grad2d=0.0006), DelayedAbsGradSchedule())
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.zeros(4, 3)),
        "scales": torch.nn.Parameter(torch.tensor([0.001, 0.03, 0.2, 0.001]).log()[:, None].expand(4, 3).clone()),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]]).expand(4, 4).clone()),
        "opacities": torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.001]).logit()),
    })
    optimizers = {key: torch.optim.Adam([value], lr=0.001) for key, value in params.items()}
    events = []
    strategy.event_recorder = lambda **event: events.append(event)
    state = strategy.initialize_state(scene_scale=1)
    means2d = torch.zeros(1, 4, 2, requires_grad=True)
    means2d.absgrad = torch.tensor([[[0.01, 0.01], [0.01, 0.01], [0., 0.], [0., 0.]]])
    info = dict(width=100, height=100, n_cameras=1, means2d=means2d,
                radii=torch.ones(1, 4, 2), gaussian_ids=torch.arange(4))
    strategy.step_post_backward(params, optimizers, state, 10000, info)
    event = events[0]
    assert event == dict(step=10000, gaussian_count_before=4, clone_count=1,
                         split_count=1, prune_count=2, gaussian_count_after=4, reset_event=0)
    assert state["grad2d"].count_nonzero() == 0
    assert state["count"].count_nonzero() == 0
    assert all(len(value) == 4 for value in params.values())
    assert all(opt.param_groups[0]["params"][0] is params[key] for key, opt in optimizers.items())


@pytest.mark.parametrize("name", ["ru_align", "ru_tar"])
def test_all_iterations_actual_strategy_event_log_and_raster_contract(tmp_path, monkeypatch, name):
    from gsplat.strategy import DefaultStrategy
    import puri_gs.delayed_absgrad as delayed
    cfg = config(name)
    strategy = delayed_strategy_from_default(DefaultStrategy(), schedule_from_config(cfg))
    recorder = PaperControlRecorder(tmp_path, cfg, 30000)
    strategy.event_recorder = recorder.record_topology
    visited, resets = [], []
    current = [0]
    def update(*args, **kwargs):
        visited.append(current[0])
    monkeypatch.setattr(strategy, "_update_state", update)
    monkeypatch.setattr(strategy, "_grow_gs", lambda *args: (0, 0))
    monkeypatch.setattr(strategy, "_prune_gs", lambda *args: 0)
    monkeypatch.setattr(delayed, "reset_opa", lambda **kwargs: resets.append(current[0]))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    probability = torch.linspace(0, 1, 81).reshape(1, 1, 9, 9)
    safe = hard_static_mask(probability)
    for step in range(30000):
        current[0] = step
        recorder.begin_iteration(step, 900, 700)
        recorder.note_rasterization(900, 700)
        if step < 10000:
            recorder.note_rasterization(224, 224)
        strategy.step_post_backward({"means": torch.zeros(1, 3)}, {},
                                    {"grad2d": torch.zeros(1), "count": torch.zeros(1)}, step, {})
        state = SimpleNamespace(probability=probability, safe_mask=safe,
            residual=torch.ones(1, 1, 9, 9) * 0.1, paused=strategy.schedule.mask_update_paused(step))
        recorder.record_iteration(step, state)
    recorder.finish()
    result = validate_event_files(tmp_path, cfg)
    assert visited == list(range(strategy.schedule.statistics_stop_step))
    assert resets == [15000, 18000]
    assert result["event_count"] == (120 if name == "ru_tar" else 100)
    with (tmp_path / "mask_aggregates.csv").open() as stream:
        masks = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in masks] == list(MASK_SAMPLE_STEPS)
    assert float(masks[0]["eroded_static_retention"]) == pytest.approx(safe.mean().item())
    with pytest.raises(FileExistsError):
        PaperControlRecorder(tmp_path, cfg, 30000)
    path = tmp_path / "topology_events.csv"
    text = path.read_text()
    path.write_text(text.replace("not_available", "nan", 1))
    with pytest.raises(ValueError, match="non-finite"):
        validate_event_files(tmp_path, cfg)


def test_recorder_rejects_duplicate_event_bad_counts_and_extra_render(tmp_path):
    recorder = PaperControlRecorder(tmp_path, config("ru_tar"), 30000)
    event = dict(step=10000, gaussian_count_before=10, clone_count=1,
                 split_count=2, prune_count=1, gaussian_count_after=12, reset_event=0)
    with pytest.raises(ValueError, match="identity"):
        recorder.record_topology(**dict(event, gaussian_count_after=13))
    recorder.record_topology(**event)
    with pytest.raises(ValueError, match="duplicate"):
        recorder.record_topology(**event)
    recorder.begin_iteration(0, 900, 700)
    recorder.note_rasterization(900, 700)
    recorder.note_rasterization(900, 700)
    with pytest.raises(ValueError, match="rasterization"):
        recorder.record_iteration(0, None)
