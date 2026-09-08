from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from puri_gs.config import load_experiment_config, trainer_method_args
from puri_gs.delayed_absgrad import DelayedAbsGradSchedule
from puri_gs.prospective_topology import SparseFootprint
from puri_gs.ru_part import RUPARTController


ROOT = Path(__file__).resolve().parents[1]


def test_ru_part_config_is_single_fixed_profile():
    config = load_experiment_config(ROOT / "configs" / "puri_gs_ru_part_garden30k.yaml")
    assert config["profile"] == "ru_part"
    assert config["rho0"] == 0.5 and config["gaussian_hard_cap"] == 2_312_002
    assert config["bootstrap_switch_step"] == 20_000
    assert config["intervention_mode"] == "current"
    args = trainer_method_args(config)
    assert "--puri_gs_ru_part_enabled" in args
    assert args[args.index("--ru_part_mode") + 1] == "current"
    assert "--puri_gs_paper_control" not in args and "--refine_windows" not in args


def test_ru_part_control_configs_differ_only_by_intervention_mode():
    paths = {
        mode: ROOT / "configs" / name
        for mode, name in {
            "parent": "puri_gs_ru_part_parent_garden30k.yaml",
            "noop": "puri_gs_ru_part_noop_garden30k.yaml",
            "current": "puri_gs_ru_part_garden30k.yaml",
        }.items()
    }
    configs = {mode: load_experiment_config(path) for mode, path in paths.items()}
    common = {
        mode: {key: value for key, value in config.items() if key != "intervention_mode"}
        for mode, config in configs.items()
    }
    assert common["parent"] == common["noop"] == common["current"]
    for mode, config in configs.items():
        args = trainer_method_args(config)
        assert args[args.index("--ru_part_mode") + 1] == mode


def test_ru_part_rejects_unknown_intervention_mode(tmp_path):
    source = (ROOT / "configs" / "puri_gs_ru_part_garden30k.yaml").read_text()
    changed = tmp_path / "changed.json"
    changed.write_text(source.replace('"intervention_mode": "current"', '"intervention_mode": "invalid"'))
    with pytest.raises(ValueError, match="intervention_mode"):
        load_experiment_config(changed)


def test_ru_part_schedule_has_exactly_100_events_and_no_tar_window():
    schedule = DelayedAbsGradSchedule()
    assert schedule.refine_steps(30_000) == tuple(range(10_000, 20_000, 100))
    assert len(schedule.refine_steps(30_000)) == 100
    assert not schedule.should_refine(20_000) and not schedule.should_refine(23_800)


def test_trainer_patch_is_training_only_and_preserves_standard_checkpoint():
    patch = (ROOT / "patches" / "gsplat_v1.5.3_puri_gs_ru_part.patch").read_text()
    assert "puri_gs_ru_part_enabled" in patch
    assert "opacity_override=0.5" in patch
    assert "ru_part_replay_ckpt" in patch and "replay_ckpts" in patch
    assert "cuda/" not in patch and "csrc/" not in patch
    assert "evaluation_loaded_track_cache" in patch
    assert "calibration_file = image_files[calibration_index]" in patch
    assert 'prepared_dir = image_dir + "_png"' in patch
    assert "set(prepared_files) != set(" in patch
    assert "requires a complete pre-generated PNG image directory" in patch
    assert 'cfg.ru_part_mode == "current"' in patch
    assert "self.cfg.strategy.part_controller = self.ru_training.ru_part" in patch
    assert "from dataclasses import asdict, dataclass, field" in patch
    assert "+                yaml.dump(asdict(cfg), f)" in patch
    assert "-                yaml.dump(vars(cfg), f)" in patch
    assert "fixed_camera_sequence(len(self.trainset), max_steps, 42)" in patch
    assert '"seed": 42' in patch
    assert "cfg.seed" not in patch
    assert (
        'if cfg.puri_gs_ru_part_enabled and cfg.ru_part_mode != "parent" '
        "and self.cfg.strategy.schedule.should_refine(step):"
    ) in patch


def test_no_forbidden_dense_or_second_backward_patterns():
    source = (ROOT / "puri_gs" / "ru_part.py").read_text()
    topology = (ROOT / "puri_gs" / "prospective_topology.py").read_text()
    assert ".cdist(" not in source + topology
    assert "N×H×W" not in source + topology
    assert ".backward(" not in source + topology
    assert "for candidate" not in source


def test_fixed_profile_rejects_parameter_drift(tmp_path):
    source = (ROOT / "configs" / "puri_gs_ru_part_garden30k.yaml").read_text()
    changed = tmp_path / "changed.json"
    changed.write_text(source.replace('"rho0": 0.5', '"rho0": 0.4'))
    with pytest.raises(ValueError, match="fixed fields"):
        load_experiment_config(changed)


def test_four_fixed_representative_steps_write_the_registered_panels(tmp_path):
    controller = RUPARTController.__new__(RUPARTController)
    controller.aux_dir = tmp_path
    controller.parser = SimpleNamespace(image_names=["train.jpg"])
    controller._representatives = []
    event = {
        "step": 10_000, "global_image_id": 0,
        "support": torch.zeros(36, 36), "evidence": torch.ones(36, 36),
        "alpha": torch.zeros(36, 36), "standard_rgb": torch.zeros(36, 36, 3),
        "target_rgb": torch.ones(36, 36, 3),
    }
    footprint = SparseFootprint(7, 4, 6, 5, 8, torch.ones(2, 3))
    controller.save_representative(event, [footprint])
    assert (tmp_path / "representative_step10000.png").is_file()
    assert controller._representatives[0]["accepted_track_ids"] == [7]

    event["step"] = 10_100
    controller.save_representative(event, [footprint])
    assert len(controller._representatives) == 1


def test_rescue_gradient_is_local_and_detached_from_alpha_and_mask():
    controller = RUPARTController.__new__(RUPARTController)
    controller.device = torch.device("cpu")
    controller.global_to_train = {0: 0}
    controller.global_to_train_lookup = torch.tensor([0])
    controller.evidence = torch.ones(1, 36, 36)
    render = torch.zeros(1, 4, 4, 3, requires_grad=True)
    target = torch.ones_like(render)
    alpha = torch.zeros(1, 4, 4, 1, requires_grad=True)
    mask = torch.zeros(1, 1, 4, 4, requires_grad=True)
    loss = controller.rescue_loss(render, target, alpha, mask, 0)
    loss.backward()
    assert render.grad is not None and torch.count_nonzero(render.grad) == render.numel()
    assert alpha.grad is None and mask.grad is None


def test_rescue_is_exactly_zero_without_track_evidence():
    controller = RUPARTController.__new__(RUPARTController)
    controller.device = torch.device("cpu")
    controller.global_to_train = {0: 0}
    controller.global_to_train_lookup = torch.tensor([0])
    controller.evidence = torch.zeros(1, 36, 36)
    render = torch.rand(1, 4, 4, 3, requires_grad=True)
    loss = controller.rescue_loss(
        render, torch.zeros_like(render), torch.zeros(1, 4, 4, 1),
        torch.zeros(1, 1, 4, 4), 0,
    )
    assert loss == 0


def test_noop_rescue_is_connected_exact_zero():
    controller = RUPARTController.__new__(RUPARTController)
    controller.intervention_mode = "noop"
    render = torch.rand(1, 4, 4, 3, requires_grad=True)
    loss = controller.rescue_loss(
        render,
        torch.zeros_like(render),
        torch.zeros(1, 4, 4, 1),
        torch.zeros(1, 1, 4, 4),
        0,
    )
    loss.backward()
    assert loss.item() == 0.0
    assert render.grad is not None and torch.count_nonzero(render.grad) == 0


def test_all_garden_training_views_use_matching_evidence_and_event():
    controller = RUPARTController.__new__(RUPARTController)
    indices = [i for i in range(185) if i % 8 != 0]
    controller.trainset = SimpleNamespace(indices=indices)
    controller.device = torch.device("cpu")
    controller.global_to_train = {v: i for i, v in enumerate(indices)}
    controller.evidence = torch.arange(161).view(161, 1, 1).expand(161, 36, 36).float()
    controller.fixed_support_probe_count = 0
    controller._camera = lambda item: SimpleNamespace(K=torch.eye(3).numpy(), width=36, height=36)
    rgb = torch.zeros(1, 36, 36, 3)
    alpha = torch.zeros(1, 36, 36, 1)
    for item, expected_global_id in enumerate(indices):
        global_id = controller.global_image_id_from_train(torch.tensor([item]))
        assert global_id == expected_global_id
        assert torch.all(controller.static_evidence(torch.tensor(global_id)) == item)
        controller.prepare_event(
            step=10200, global_image_id=global_id,
            standard_alpha=alpha, fixed_support=alpha,
            standard_rgb=rgb, target_rgb=rgb,
            camtoworld=torch.eye(4)[None], K=torch.eye(3)[None],
            fixed_support_probe_ms=0.0,
        )
        event = controller.take_event(10200)
        assert event["train_view_id"] == item
        assert event["global_image_id"] == expected_global_id
        assert torch.all(event["evidence"] == item)
    assert controller.global_image_id_from_train(120) == 138
    for bad in [-1, 0, 120, 184, 185]:
        for value in (bad, torch.tensor([bad])):
            with pytest.raises(ValueError, match="non-training"):
                controller.static_evidence(value)
    for bad in [-1, 161]:
        with pytest.raises(ValueError, match="out of range"):
            controller.global_image_id_from_train(bad)
    patch = (ROOT / "patches" / "gsplat_v1.5.3_puri_gs_ru_part.patch").read_text()
    assert patch.count("global_image_id=self.ru_training.ru_part.global_image_id_from_train(image_ids)") == 2
