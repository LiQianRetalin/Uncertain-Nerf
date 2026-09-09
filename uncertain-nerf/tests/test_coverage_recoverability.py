"""CPU contract tests; these do not certify the server CUDA precheck."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puri_gs.coverage_recoverability import (
    aggregate_coverage, coverage_row, diagnostic_decision, diagnostic_loss,
    polygon_mask, rank_check_views, region_metrics, temporal_summary, validate_config,
)
from puri_gs.recoverability_runtime import (
    FinalStateRuntime, read_runtime_yaml, source_functions, terminal_learning_rates, validate_splats,
)
from puri_gs.ru_part_v3 import static_rescue_l1
from puri_gs.semantic_mask import masked_photo_loss
from tools import ru_part_v3_recoverability as cli

ROOT = Path(__file__).resolve().parents[1]


def differentiable_ssim(x, y, **kwargs):
    return 1 - (x - y).square().mean()


def test_loss_oracle_adds_only_missing_support_and_global_rgb_mean():
    rgb = torch.tensor([.7, .6, .5, .4] * 3).reshape(1, 2, 2, 3).requires_grad_()
    target = torch.zeros_like(rgb)
    m = torch.tensor([1., 0., 0., 0.]).reshape(1, 1, 2, 2).requires_grad_()
    c = torch.tensor([.2, 1., .4, 0.]).reshape_as(m).requires_grad_()
    s = torch.tensor([1., 1., 1., 0.]).reshape_as(m).requires_grad_()
    va = diagnostic_loss(rgb, target, m, c, s, group="Va", fused_ssim_fn=differentiable_ssim)[0]
    vb = diagnostic_loss(rgb, target, m, c, s, group="Vb", fused_ssim_fn=differentiable_ssim)[0]
    oracle = diagnostic_loss(rgb, target, m, c, s, group="O", fused_ssim_fn=differentiable_ssim)[0]
    base = masked_photo_loss(rgb, target, m.detach(), fused_ssim_fn=differentiable_ssim)[0]
    assert torch.equal(va, vb)
    torch.testing.assert_close(va, base + .8 * static_rescue_l1(rgb, target, m, c, step=500))
    delta = .8 * (((1-m.detach()) * (1-c.detach()) * s.detach()).permute(0, 2, 3, 1) * rgb.abs()).mean()
    torch.testing.assert_close(oracle-va, delta)
    grad = torch.autograd.grad(oracle-va, rgb, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(delta, rgb)[0])
    oracle.backward()
    assert m.grad is c.grad is s.grad is None


def test_missing_masks_and_zero_denominators_stay_missing():
    c = torch.tensor([[0., .5], [1., 0.]])
    missing = coverage_row("missing", c)
    accepted = coverage_row("accepted", c, torch.ones_like(c))
    rejected = coverage_row("rejected", c, torch.zeros_like(c))
    assert missing["mean_Q"] is None
    assert accepted["rejected_mean_support"] is None
    assert accepted["mean_Q_when_positive"] is None
    assert rejected["rejected_mean_support"] == .375
    assert rejected["mean_Q_when_positive"] == .75
    summary = aggregate_coverage([missing, accepted, rejected])
    assert summary["views_with_M"] == 2
    assert summary["equal_view_mean"]["mean_Q"] == .1875


def test_equal_view_and_pixel_weighted_coverage_differ():
    a = coverage_row("a", torch.ones(1, 1))
    b = coverage_row("b", torch.zeros(3, 3))
    result = aggregate_coverage([a, b])
    assert result["equal_view_mean"]["mean_C"] == .5
    assert result["pixel_weighted_mean"]["mean_C"] == .1


def test_sparse_temporal_boundaries_are_exact_but_means_remain_sampled():
    rows = [dict(step=499, active_samples=0, mean_Q=0),
            dict(step=500, active_samples=1, mean_Q=.1, added_l1=.2),
            dict(step=10000, active_samples=100, mean_Q=.1, added_l1=.3),
            dict(step=20000, active_samples=200, mean_Q=0, added_l1=0),
            dict(step=29999, active_samples=300, mean_Q=.1, added_l1=.1)]
    windows = temporal_summary(rows)["windows"]
    assert [row["active_updates"] for row in windows] == [99, 101, 100]
    assert windows[0]["metrics"]["added_l1"]["logged_sample_mean"] == .2
    assert all(row["metrics"]["added_l1"]["whole_window_sum"] is None for row in windows)
    assert temporal_summary([])["windows"][0]["active_updates"] is None


@pytest.mark.parametrize("rows", [
    [dict(step=500, active_samples=2), dict(step=501, active_samples=1)],
    [dict(step=500, active_samples=1), dict(step=500, active_samples=1)],
    [dict(step=500, mean_Q=float("nan"))],
])
def test_invalid_progress_rejected(rows):
    with pytest.raises(ValueError):
        temporal_summary(rows)


def test_region_metrics_are_unclamped_and_protection_has_explicit_fallback():
    roi = torch.tensor([[True, False], [False, False]])
    rgb = torch.full((2, 2, 3), 2.)
    result = region_metrics(rgb, torch.zeros_like(rgb), torch.zeros(2, 2), roi, torch.zeros(2, 2))
    assert result["roi_mae"] == 2
    assert result["protection_pixels"] == 3
    assert result["protection_fallback_to_outside_S"]
    assert result["roi_low_alpha_fraction"] == 1


def metrics_fixture():
    row = dict(roi_mae=.1, protection_mae=.01, full_mae=.05, roi_alpha_mean=.5,
               roi_low_alpha_fraction=.1, roi_pixels=4, protection_pixels=8,
               protection_fallback_to_outside_S=False)
    return {g: {str(u): {name: dict(row) for name in ("a", "b", "c", "d")}
                for u in (0, 100, 200, 400)} for g in ("Va", "Vb", "O")}


@pytest.mark.parametrize("local,checks,damage,status", [
    (.08, .08, False, "PROMISING_LOCAL_RECOVERABILITY"),
    (.08, .1, False, "LOCAL_ONLY_RESPONSE"),
    (.1, .1, False, "NO_CLEAR_RECOVERABILITY_SIGNAL"),
    (.08, .08, True, "COLLATERAL_ERROR_DETECTED"),
])
def test_mechanical_classification_and_protection_precedence(local, checks, damage, status):
    metrics = metrics_fixture()
    for name in ("a", "b"):
        metrics["O"]["400"][name]["roi_mae"] = local
    for name in ("c", "d"):
        metrics["O"]["400"][name]["roi_mae"] = checks
    if damage:
        metrics["O"]["400"]["d"]["protection_mae"] = .02
    result = diagnostic_decision(metrics, ["a", "b"], ["c", "d"])
    assert result["status"] == status
    assert len(result["protected_regions"]) == 6
    assert result["local"]["threshold"] == pytest.approx(.005)


def test_initial_ranges_and_repeat_controls_both_contribute_to_noise():
    m = metrics_fixture()
    for name in ("a", "b"):
        m["O"]["0"][name]["roi_mae"] = .11
        m["Va"]["400"][name]["roi_mae"] = .08
        m["Vb"]["400"][name]["roi_mae"] = .085
        m["O"]["400"][name]["roi_mae"] = .06
    result = diagnostic_decision(m, ["a", "b"], ["c", "d"])
    assert result["local"]["E0"] == pytest.approx(.31 / 3)
    assert result["local"]["n"] == pytest.approx(.01)
    assert result["local"]["threshold"] == pytest.approx(.03)
    assert not result["LOCAL_GAIN"]  # improves initial, but not better control by .03


@pytest.mark.parametrize("change", ["missing_100", "nan", "region_changed", "alpha_only", "empty"])
def test_invalid_or_alpha_only_results_cannot_pass(change):
    m = metrics_fixture()
    if change == "missing_100":
        del m["Vb"]["100"]
    elif change == "nan":
        m["O"]["400"]["a"]["roi_mae"] = float("nan")
    elif change == "region_changed":
        m["Vb"]["400"]["a"]["roi_pixels"] = 5
    elif change == "empty":
        m["O"]["400"]["a"]["protection_pixels"] = 0
    else:
        m["O"]["400"]["a"]["roi_alpha_mean"] = 1
    result = diagnostic_decision(m, ["a", "b"], ["c", "d"])
    assert result["status"] == ("NO_CLEAR_RECOVERABILITY_SIGNAL" if change == "alpha_only" else "DIAGNOSTIC_INVALID")


@pytest.mark.parametrize("polygon", [[], [[[0, 0], [9, 0], [9, 9], [0, 9]]], [[[0, 0], [10, 0], [0, 1]]]])
def test_empty_full_or_outside_roi_rejected(polygon):
    with pytest.raises(ValueError):
        polygon_mask(10, 10, polygon)


def test_polygon_and_geometric_view_order_use_training_coordinates():
    assert polygon_mask(10, 10, [[[1, 1], [3, 1], [3, 3], [1, 3]]]).sum() == 9
    cameras = []
    for i, (name, x) in enumerate([("a", 0), ("b", 5), ("c", 1), ("d", 4), ("e", 10)]):
        pose = np.eye(4)
        pose[0, 3] = x
        cameras.append(dict(basename=name, camtoworld=pose.tolist(), view_id=i))
    assert [r["image_name"] for r in rank_check_views(cameras, ["a", "b"])] == ["c", "d", "e"]


def test_original_update_29999_lr_matches_real_exponential_scheduler():
    cfg = dict(max_steps=30000, batch_size=1, steps_scaler=1, sparse_grad=False, visible_adam=False,
               means_lr=.00016, scales_lr=.005, quats_lr=.001, opacities_lr=.05, sh0_lr=.0025, shN_lr=.000125)
    param = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.Adam([param], lr=cfg["means_lr"] * 1.234)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=.01 ** (1/30000))
    for _ in range(29999):
        opt.step()
        scheduler.step()
    rates = terminal_learning_rates(cfg, 1.234)
    assert rates["means"] == opt.param_groups[0]["lr"]
    scheduler.step()
    assert rates["means"] != opt.param_groups[0]["lr"]
    assert rates["opacities"] == .05


def test_saved_yaml_is_read_without_constructing_python_objects(tmp_path):
    path = tmp_path / "cfg.yml"
    path.write_text("strategy: !!python/object:puri_gs.delayed_absgrad.DelayedAbsGradStrategy\n  absgrad: true\nvalue: null\n")
    value = read_runtime_yaml(path)
    assert value["strategy"]["absgrad"] is True and value["value"] is None
    assert value["strategy"]["__yaml_type__"].endswith("DelayedAbsGradStrategy")


def small_splats(n=2):
    return {key: torch.zeros(shape) for key, shape in dict(means=(n, 3), scales=(n, 3), quats=(n, 4),
             opacities=(n,), sh0=(n, 1, 3), shN=(n, 15, 3)).items()}


def test_actual_source_renderer_and_optimizer_blocks(tmp_path):
    candidates = [ROOT / f"external/{name}/examples/simple_trainer.py" for name in
                  ("gsplat-v1.5.3-ru-part-v3", "gsplat-v1.5.3-v3-verify")]
    trainer = next((p for p in candidates if p.is_file()), None)
    if trainer is None:
        pytest.skip("prepared gsplat source is required for this integration check")
    calls = []
    def rasterization(**kwargs):
        calls.append(kwargs)
        return torch.zeros(1, 4, 4, 3), torch.zeros(1, 4, 4, 1), {}
    class DefaultStrategy:
        absgrad = True
    factory, renderer, digest = source_functions(trainer, rasterization=rasterization, default_strategy=DefaultStrategy)
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in small_splats().items()})
    opt = factory(splats, [(k, None, .001) for k in splats], 1, 1, False, False)
    engine = FinalStateRuntime.__new__(FinalStateRuntime)
    engine.optimizers = opt
    description = engine.optimizer_description()
    assert description == json.loads(json.dumps(description))
    assert all(row["initial_state_entries"] == 0 and row["settings"]["eps"] == 1e-15 for row in description.values())
    runner = SimpleNamespace(splats=splats, world_size=1, cfg=SimpleNamespace(app_opt=False, antialiased=False,
        camera_model="pinhole", packed=True, sparse_grad=False, with_ut=False, with_eval3d=False, strategy=DefaultStrategy()))
    renderer(runner, torch.eye(4)[None], torch.eye(3)[None], 4, 4, sh_degree=3, near_plane=.01, far_plane=1e10)
    assert len(calls) == 1 and calls[0]["absgrad"] and calls[0]["colors"].shape == (2, 16, 3)
    assert calls[0]["near_plane"] == .01 and len(digest) == 64
    sum(p.sum() for p in splats.values()).backward()
    for optimizer in opt.values():
        optimizer.step()
    fresh = factory(splats, [(k, None, .001) for k in splats], 1, 1, False, False)
    assert all(not optimizer.state for optimizer in fresh.values())


@pytest.mark.parametrize("bad", ["step", "shape", "nan", "keys"])
def test_nonstandard_checkpoint_rejected(bad):
    value = {"step": 29999, "splats": small_splats()}
    if bad == "step":
        value["step"] = 30000
    elif bad == "shape":
        value["splats"]["shN"] = torch.zeros(2, 3, 3)
    elif bad == "nan":
        value["splats"]["means"][0, 0] = float("nan")
    else:
        value["splats"]["unknown"] = torch.ones(2)
    with pytest.raises(ValueError):
        validate_splats(value, count=2, step=29999)


def test_training_data_guard_rejects_test_name_before_dataset_access():
    engine = FinalStateRuntime.__new__(FinalStateRuntime)
    engine.local_ids = {"DSC07987.JPG": 0}
    engine.identity = {"test_basenames": ["DSC07988.JPG"]}
    with pytest.raises(ValueError, match="TEST_IMAGE_ACCESS_REJECTED"):
        engine.data("DSC07988.JPG")


def test_protocol_and_gpu_zero_lock(monkeypatch):
    import puri_gs.v3_gpu as gpu
    cfg = cli.protocol()
    assert cfg["preferred_gpu"] == 0
    changed = copy.deepcopy(cfg)
    changed["updates_per_group"] = 800
    with pytest.raises(ValueError):
        validate_config(changed)
    rows = [dict(index=i, uuid=f"GPU-{i}", memory_used_mib=0, utilization_percent=0, compute_process_present=False) for i in (0, 6)]
    monkeypatch.setattr(gpu, "gpu_inventory", lambda: rows)
    assert cli.choose_gpu("auto")["index"] == 0
    rows[0]["compute_process_present"] = True
    with pytest.raises(ValueError):
        cli.choose_gpu("auto", locked=rows[0])
    assert cli.final_flags({})["ACTUAL_UPDATES_VA"] == 0
    assert cli.final_flags({})["FRESH_DIAGNOSTIC_OPTIMIZERS"] != True


@pytest.fixture
def diagnostic_run(tmp_path, monkeypatch):
    """Real orchestration/loss/Adam with a tiny CPU renderer, no GPU claims."""
    import sys
    import puri_gs.recoverability_runtime as runtime_module
    torch.set_num_threads(2)
    names = ["DSC07987.JPG", "DSC07989.JPG", "check1.JPG", "check2.JPG"]
    checkpoint = tmp_path / "source.pt"
    torch.save({"step": 29999, "splats": small_splats()}, checkpoint)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    roi = {"optimization_views": names[:2], "check_views": names[2:], "views": {}}
    for name in names:
        s = np.array([[1, 0], [0, 0]], dtype=np.bool_)
        path = tmp_path / f"{name}_S.npy"
        np.save(path, s)
        roi["views"][name] = {"path": str(path)}
        if name in names[:2]:
            for key, value in (("M", np.zeros((2, 2), dtype=np.bool_)), ("C", np.full((2, 2), .2, dtype=np.float32))):
                np.save(prepared / f"{Path(name).stem}_{key}.npy", value)
    cli.new_json(tmp_path / "roi_manifest.json", roi)

    class CPUEngine:
        loads = []
        def __init__(self, paths, protocol):
            self.paths, self.protocol = paths, protocol
            self.device = torch.device("cpu")
            self.counts = dict.fromkeys(("training_rasterization", "evaluation_rasterization", "precheck_rasterization",
                "preparation_rasterization", "gaussian_backward", "optimizer_updates", "precheck_updates", "head_updates", "topology_events"), 0)
            self.opened_training_images = set()
            self.splats = None
        def load_parameters(self):
            assert self.splats is None
            payload = torch.load(self.paths["checkpoint"], weights_only=True)
            self.splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in payload["splats"].items()})
            self.optimizers = {k: torch.optim.Adam([{"params": p, "name": k}], lr=1e-4, eps=1e-15) for k, p in self.splats.items()}
            digest = cli.tensor_digest(self.splats)
            self.loads.append(digest)
            return digest
        optimizer_description = FinalStateRuntime.optimizer_description
        def render(self, name, *, kind):
            assert name in names and (kind != "training" or name in names[:2])
            self.opened_training_images.add(name)
            self.counts[kind + "_rasterization"] += 1
            scalar = .5 + sum(p.mean() for p in self.splats.values())
            rgb = scalar.expand(1, 2, 2, 3)
            return rgb, torch.zeros_like(rgb), torch.ones(1, 2, 2, 1) * .5
        def discard(self):
            self.splats, self.optimizers = None, {}

    runtime = dict(paths={"checkpoint": str(checkpoint)}, config={"gaussian_count": 2})
    seed_engine = CPUEngine(runtime["paths"], runtime["config"])
    seed_engine.load_parameters()
    source_digest = cli.tensor_digest(seed_engine.splats)
    settings = seed_engine.optimizer_description()
    seed_engine.discard()
    CPUEngine.loads = []
    prereg = dict(code_sha256={}, source_sha256={}, frozen_sha256={}, source_checkpoint=str(checkpoint),
        source_parameter_digest=source_digest,
        source_checkpoint_sha256=cli.sha(checkpoint), runtime_environment={}, optimizer=settings,
        camera_sequence=names[:2] * 200, intervention_qualification={names[0]: {"delta_Q_weighted_residual_sum": 1}})
    cli.new_json(tmp_path / "diagnostic_preregistration.json", prereg)
    (tmp_path / "diagnostic_preregistration.sha256").write_text(cli.sha(tmp_path / "diagnostic_preregistration.json"))
    monkeypatch.setattr(runtime_module, "FinalStateRuntime", CPUEngine)
    monkeypatch.setattr(runtime_module, "runtime_environment", lambda device: {})
    monkeypatch.setitem(sys.modules, "fused_ssim", SimpleNamespace(fused_ssim=differentiable_ssim))
    return tmp_path, runtime, CPUEngine


def test_serial_three_group_workflow_counts_outputs_and_independent_starts(diagnostic_run):
    root, runtime, engine = diagnostic_run
    initial_sha = cli.sha(runtime["paths"]["checkpoint"])
    assert cli.run_diagnostic(root, runtime, {"status": "RUNNING"}) == "DIAGNOSTIC_COMPLETE"
    result = cli.read(root / "diagnostic_result.json")
    assert len(engine.loads) == 4 and len(set(engine.loads)) == 1  # precheck, Va, Vb, O
    assert cli.sha(runtime["paths"]["checkpoint"]) == initial_sha
    assert result["counts"]["training_rasterization"] == result["counts"]["gaussian_backward"] == 1200
    assert result["counts"]["evaluation_rasterization"] == 48
    assert result["counts"]["precheck_updates"] == result["counts"]["precheck_rasterization"] == 1
    for group in ("Va", "Vb", "O"):
        state = result["groups"][group]
        assert state["GROUP_STATUS"] == "COMPLETE" and state["updates_completed"] == 400
        artifact = torch.load(root / group / "diagnostic_splats.pt", weights_only=True)
        assert artifact["step"] == 399 and artifact["source_step"] == 29999 and artifact["diagnostic_only"]
        assert len(list((root / group).glob("metrics_*.json"))) == 4
    assert result["groups"]["O"]["actual_intervention_updates"] == 400


def test_failed_update_retains_exact_count_and_stops_before_vb(diagnostic_run, monkeypatch):
    root, runtime, _ = diagnostic_run
    original = cli.perform_update
    def failing(engine, inputs, name, group, *, precheck=False):
        result = original(engine, inputs, name, group, precheck=precheck)
        if not precheck and engine.counts["optimizer_updates"] == 7:
            raise RuntimeError("injected post-update failure")
        return result
    monkeypatch.setattr(cli, "perform_update", failing)
    with pytest.raises(RuntimeError, match="injected"):
        cli.run_diagnostic(root, runtime, {"status": "RUNNING"})
    assert cli.read(root / "Va/status.json")["updates_completed"] == 7
    assert not (root / "Vb").exists() and not (root / "O").exists()
    assert not (root / "diagnostic_result.json").exists()


def test_preregistration_tampering_fails_before_precheck(diagnostic_run):
    root, runtime, engine = diagnostic_run
    (root / "diagnostic_preregistration.json").write_text("{}")
    with pytest.raises(ValueError, match="preregistration changed"):
        cli.run_diagnostic(root, runtime, {})
    assert not engine.loads
