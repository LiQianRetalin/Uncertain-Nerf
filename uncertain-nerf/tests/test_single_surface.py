"""Fixed-role mathematics and serial orchestration; CPU, not a CUDA certificate."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puri_gs.coverage_recoverability import diagnostic_loss, canonical_sha
from puri_gs.single_surface import (A, B, LRS, PROTOCOL, candidate_views, validate_config,
    validate_proposal, validate_frozen, intervention_qualification, view_metrics, decide)
from puri_gs.recoverability_runtime import FinalStateRuntime
from tools import ru_part_v3_single_surface as single
from tools import ru_part_v3_recoverability as shared
from test_coverage_recoverability import small_splats, differentiable_ssim


@pytest.mark.parametrize("m,c,s,coefficient", [(0., 0., 1., .8), (1., 0., 1., 0.), (0., 1., 1., 0.), (0., 0., 0., 0.)])
def test_exact_missing_support_gradient_and_unchanged_dssim(m, c, s, coefficient):
    rgb = torch.ones(1, 2, 2, 3, requires_grad=True)
    target = torch.zeros_like(rgb)
    mask, support, region = [torch.full((1, 1, 2, 2), x, requires_grad=True) for x in (m, c, s)]
    seen = []
    def ssim(x, y, **kwargs):
        seen.append((x.detach().clone(), y.detach().clone()))
        return differentiable_ssim(x, y)
    va = diagnostic_loss(rgb, target, mask, support, region, group="Va", fused_ssim_fn=ssim)[0]
    oracle = diagnostic_loss(rgb, target, mask, support, region, group="O", fused_ssim_fn=ssim)[0]
    for x, y in zip(seen[0], seen[1]):
        assert torch.equal(x, y)
    gradient = torch.autograd.grad(oracle-va, rgb, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.full_like(rgb, coefficient/rgb.numel()))
    oracle.backward()
    assert mask.grad is support.grad is region.grad is None


def test_context_keeps_original_v3_loss_and_target_roi_is_null():
    rgb = torch.full((1, 2, 2, 3), .4, requires_grad=True)
    target = torch.zeros_like(rgb)
    m = torch.tensor([0., 1., 0., 1.]).reshape(1, 1, 2, 2)
    c = torch.ones_like(m) * .25
    s = torch.zeros_like(m)
    losses = [diagnostic_loss(rgb, target, m, c, s, group=g, fused_ssim_fn=differentiable_ssim) for g in ("Va", "Vb", "O")]
    assert all(torch.equal(losses[0][0], loss[0]) for loss in losses)
    assert losses[0][2] > 0  # B keeps its pre-existing recovery loss
    row = view_metrics(rgb[0], target[0], torch.ones(2, 2), role="B", roi=s[0, 0].bool(), mask=m[0, 0])
    assert row["roi_mae"] is row["roi_alpha_mean"] is row["roi_low_alpha_fraction"] is None
    assert row["accepted_pixels"] == 2 and row["full_pixels"] == 4
    assert row["roi_pixels"] == 0


def test_numeric_qualification_reports_S_and_full_image_denominators():
    m, c = torch.zeros(2, 2), torch.full((2, 2), .25)
    s = torch.tensor([[True, False], [False, False]])
    q = intervention_qualification(m, c, s, torch.ones(2, 2, 3), torch.zeros(2, 2, 3))
    assert q["qualified"] and q["proposal_status"] == "UNCONFIRMED_PROPOSAL"
    assert q["delta_Q_nonzero_pixels"] == 1 and q["delta_Q_fraction_in_S"] == 1
    assert q["delta_Q_full_image_mean"] == .75/4 and q["delta_Q_mean_in_S"] == .75
    assert q["delta_Q_weighted_residual_sum"] == 2.25
    assert not intervention_qualification(torch.ones_like(m), c, s, torch.ones(2, 2, 3), torch.zeros(2, 2, 3))["qualified"]
    assert not validate_frozen(m, c, torch.zeros_like(s), context=True).any()


def test_acceptance_domain_includes_accepted_pixels_inside_A_roi():
    roi = torch.tensor([[True, True], [False, False]])
    mask = torch.tensor([[0., 1.], [0., 1.]])
    rgb = torch.tensor([[[2., 2., 2.], [4., 4., 4.]], [[6., 6., 6.], [8., 8., 8.]]])
    row = view_metrics(rgb, torch.zeros_like(rgb), torch.ones(2, 2), role="A", roi=roi, mask=mask, support=torch.zeros(2, 2))
    assert row["roi_mae"] == 3 and row["outside_mae"] == 7 and row["accepted_mae"] == 6
    assert row["delta_Q_mae"] == 2  # float RGB is deliberately not clipped


def metric_fixture():
    row = dict(full_mae=.1, full_pixels=20, roi_mae=.1, roi_pixels=4, outside_mae=.1, outside_pixels=16,
        accepted_mae=.1, accepted_pixels=10, roi_alpha_mean=.5, roi_low_alpha_fraction=.2, delta_Q_mae=.1, delta_Q_pixels=2)
    h = "DSC07986.JPG"
    a = dict(row, role="A")
    b = dict(row, role="B", roi_mae=None, roi_pixels=0, outside_mae=None, outside_pixels=0,
             roi_alpha_mean=None, roi_low_alpha_fraction=None, delta_Q_mae=None, delta_Q_pixels=0)
    hr = dict(row, role="H", accepted_mae=None, accepted_pixels=0, delta_Q_mae=None, delta_Q_pixels=0)
    return {g: {str(u): copy.deepcopy({A: a, B: b, h: hr}) for u in (0, 400)} for g in ("Va", "Vb", "O")}, h


@pytest.mark.parametrize("local,check_gain,status", [
    (True, True, "LOCAL_RESPONSE_WITH_CHECK_VIEW_GAIN"),
    (True, False, "LOCAL_RESPONSE_WITH_PROTECTION"),
    (False, False, "NO_CLEAR_LOCAL_RESPONSE"),
])
def test_H_gain_is_extra_evidence_not_a_requirement_for_local_response(local, check_gain, status):
    values, h = metric_fixture()
    values["O"]["400"][A]["roi_mae"] = .08 if local else .1
    values["O"]["400"][h]["roi_mae"] = .08 if check_gain else .1
    result = decide(values, h)
    assert result["status"] == status
    assert result["LOCAL_GAIN"] == local and result["CHECK_VIEW_GAIN"] == check_gain
    assert len(result["protected_regions"]) == 6


def test_initial_protection_budget_cannot_expand_with_control_noise():
    values, h = metric_fixture()
    values["Va"]["400"][B]["full_mae"] = .08
    values["Vb"]["400"][B]["full_mae"] = .12
    values["O"]["400"][B]["full_mae"] = .102
    result = decide(values, h)
    row = result["protected_regions"][B+":full_mae"]
    assert row["d_initial"] == pytest.approx(.001)
    assert row["b_control"] == pytest.approx(.12)
    assert not row["initial_budget_pass"] and row["control_budget_pass"]
    assert result["status"] == "PROTECTION_NOT_ESTABLISHED"


def test_no_B_averaging_and_all_three_initials_contribute_to_noise():
    values, h = metric_fixture()
    values["O"]["0"][A]["roi_mae"] = .13
    values["O"]["400"][A]["roi_mae"] = .04
    result = decide(values, h)
    assert result["local"]["E0"] == pytest.approx(.11)
    assert result["local"]["n"] == pytest.approx(.03)
    assert result["local"]["threshold"] == pytest.approx(.09)
    assert not result["LOCAL_GAIN"]


@pytest.mark.parametrize("bad", ["B_roi_zero_metric", "H_missing", "accepted_empty", "nan", "changed_area"])
def test_invalid_domains_cannot_give_positive_results(bad):
    values, h = metric_fixture()
    if bad == "B_roi_zero_metric":
        values["O"]["400"][B]["roi_mae"] = 0.
    elif bad == "H_missing":
        del values["Vb"]["0"][h]
    elif bad == "accepted_empty":
        values["Va"]["0"][A]["accepted_pixels"] = 0
    elif bad == "nan":
        values["O"]["400"][h]["roi_mae"] = float("nan")
    else:
        values["O"]["400"][A]["roi_pixels"] += 1
    assert decide(values, h)["status"] == "DIAGNOSTIC_INVALID"


def test_candidate_ranking_is_distance_to_A_not_minimum_to_A_or_B():
    names = [A, B] + [f"train_{i}.JPG" for i in range(159)]
    cameras = []
    for i, name in enumerate(names):
        pose = np.eye(4)
        pose[0, 3] = 1000 if name == B else i
        cameras.append(dict(basename=name, view_id=i, camtoworld=pose.tolist()))
    identity = dict(train_basenames=names, test_basenames=[f"test_{i}.JPG" for i in range(24)], cameras=cameras)
    candidates = candidate_views(identity)
    assert [c["image_name"] for c in candidates] == names[2:10]
    proposal = dict(protocol=PROTOCOL, candidate_sha256=canonical_sha(candidates), check_view=names[2],
        candidates_reviewed=1, polygons={A: [[[1, 1], [4, 1], [3, 4]]], names[2]: [[[1, 1], [4, 1], [3, 4]]]},
        surface_description="house", correspondence_basis="gable brick pattern")
    assert validate_proposal(proposal, candidates) == names[2]
    proposal["check_view"] = names[3]
    with pytest.raises(ValueError):
        validate_proposal(proposal, candidates)


def test_config_preserves_old_protocol_and_fixed_new_settings():
    new = single.config()
    assert new["evaluation_updates"] == [0, 400]
    assert shared.protocol()["evaluation_updates"] == [0, 100, 200, 400]
    assert new["protocol"] != shared.protocol()["protocol"]
    changed = dict(new, updates_per_group=800)
    with pytest.raises(ValueError):
        validate_config(changed)


@pytest.fixture
def single_run(tmp_path, monkeypatch):
    import sys
    import puri_gs.recoverability_runtime as runtime_module
    torch.set_num_threads(2)
    h = "DSC07986.JPG"
    checkpoint = tmp_path / "source.pt"
    torch.save({"step": 29999, "splats": small_splats()}, checkpoint)
    (tmp_path / "prepared").mkdir()
    views = {}
    for name, role in ((A, "A"), (B, "B"), (h, "H")):
        region = np.array([[1, 0], [0, 0]], dtype=np.bool_) if role != "B" else np.zeros((2, 2), dtype=np.bool_)
        path = tmp_path / f"{role}_region.npy"
        np.save(path, region)
        views[name] = dict(path=str(path), sha256=shared.sha(path), role=role)
        if role != "H":
            np.save(tmp_path / "prepared" / f"{Path(name).stem}_M.npy", np.array([[0, 1], [0, 1]], dtype=np.bool_))
            np.save(tmp_path / "prepared" / f"{Path(name).stem}_C.npy", np.full((2, 2), .2, dtype=np.float32))
    class Engine:
        loads, training_views = [], []
        def __init__(self, paths, protocol, *, frozen_views=None):
            self.paths, self.protocol, self.device = paths, protocol, torch.device("cpu")
            self.learning_rates = LRS
            self.splats = None
            self.opened_training_images = set()
            self.counts = dict.fromkeys(("training_rasterization", "evaluation_rasterization", "precheck_rasterization",
                "preparation_rasterization", "gaussian_backward", "optimizer_updates", "precheck_updates", "head_updates", "topology_events"), 0)
        def load_parameters(self):
            assert self.splats is None
            values = torch.load(self.paths["checkpoint"], weights_only=True)["splats"]
            self.splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in values.items()})
            self.optimizers = {k: torch.optim.Adam([{"params": p, "name": k}], lr=LRS[k], eps=1e-15) for k, p in self.splats.items()}
            self.loads.append(shared.tensor_digest(self.splats))
            return self.loads[-1]
        optimizer_description = FinalStateRuntime.optimizer_description
        def render(self, name, *, kind):
            assert name in (A, B, h)
            if kind in ("training", "precheck"):
                assert name in (A, B)
                if kind == "training":
                    self.training_views.append(name)
            self.opened_training_images.add(name)
            self.counts[kind+"_rasterization"] += 1
            rgb = (.5+sum(p.mean() for p in self.splats.values())).expand(1, 2, 2, 3)
            return rgb, torch.zeros_like(rgb), torch.ones(1, 2, 2, 1)*.5
        def discard(self):
            self.splats, self.optimizers = None, {}
    runtime = dict(paths={"checkpoint": str(checkpoint)}, config=dict(gaussian_count=2,
        source_checkpoint_sha256=shared.sha(checkpoint), protocol=PROTOCOL), commit="test-only", code_sha256={}, gpu={"index": 0, "uuid": "CPU-TEST"})
    e = Engine(runtime["paths"], runtime["config"])
    digest = e.load_parameters()
    optimizer = e.optimizer_description()
    e.discard()
    Engine.loads = []
    shared.new_json(tmp_path / "candidate_views.json", {"candidates": [{"image_name": h}]})
    runtime["candidate_sha256"] = shared.sha(tmp_path / "candidate_views.json")
    shared.new_json(tmp_path / "runtime.json", runtime)
    panel = tmp_path / "panel.png"
    panel.write_bytes(b"test-only-panel-identity")
    roi = dict(views=views, check_view=h, review_panel_path=str(panel), review_panel_sha256=shared.sha(panel),
        candidate_views_sha256=runtime["candidate_sha256"], candidates_reviewed=1, intervention_qualification={"qualified": True, "context_S_zero": True})
    shared.new_json(tmp_path / "roi_manifest.json", roi)
    shared.new_json(tmp_path / "roi_confirmation.json", dict(confirmed=True, roi_manifest_sha256=shared.sha(tmp_path / "roi_manifest.json"), panel_sha256=shared.sha(panel)))
    shared.new_json(tmp_path / "preparation_reuse.json", dict(source_sha256={}, prepared_numeric_sha256={}, review_image_sha256={},
        learning_rates=LRS, runtime_environment={}, source_parameter_digest=digest, optimizer=optimizer))
    monkeypatch.setattr(runtime_module, "FinalStateRuntime", Engine)
    monkeypatch.setattr(runtime_module, "runtime_environment", lambda *a, **k: {})
    monkeypatch.setitem(sys.modules, "fused_ssim", SimpleNamespace(fused_ssim=differentiable_ssim))
    return tmp_path, runtime, Engine


def test_actual_serial_loop_2_prechecks_1200_training_and_18_evaluations(single_run):
    root, runtime, engine = single_run
    assert single.run_diagnostic(root, runtime, {"status": "RUNNING"}) == "DIAGNOSTIC_COMPLETE"
    result = shared.read(root / "diagnostic_result.json")
    assert len(engine.loads) == 5 and len(set(engine.loads)) == 1
    assert engine.training_views == [A, B]*600
    assert result["counts"]["training_rasterization"] == result["counts"]["gaussian_backward"] == 1200
    assert result["counts"]["evaluation_rasterization"] == 18
    assert result["counts"]["precheck_updates"] == 2
    assert 0 < result["groups"]["O"]["actual_intervention_updates"] <= 200
    for group in ("Va", "Vb", "O"):
        assert sorted(p.name for p in (root / group).glob("metrics_*.json")) == ["metrics_0.json", "metrics_400.json"]
        assert result["metrics"][group]["400"][B]["roi_mae"] is None
    assert result["PROTECTION_PASS"] is not None
    prereg = shared.read(root / single.PREREG)
    assert prereg["protocol"] == PROTOCOL and prereg["evaluation_updates"] == [0, 400]
    assert prereg["roi"]["review_panel_path"] in prereg["frozen_sha256"]
    assert shared.sha(runtime["paths"]["checkpoint"]) == runtime["config"]["source_checkpoint_sha256"]


def test_hard_failure_stops_following_group_and_keeps_exact_updates(single_run, monkeypatch):
    root, runtime, _ = single_run
    original = shared.perform_update
    def fail(engine, inputs, name, group, **kwargs):
        result = original(engine, inputs, name, group, **kwargs)
        if not kwargs.get("precheck") and engine.counts["optimizer_updates"] == 9:
            raise RuntimeError("injected update failure")
        return result
    monkeypatch.setattr(shared, "perform_update", fail)
    monkeypatch.setattr(single, "output_root", lambda args: root)
    monkeypatch.setattr(single, "verify_runtime", lambda runtime: None)
    monkeypatch.setattr(shared, "choose_gpu", lambda *a, **k: runtime["gpu"])
    shared.new_json(root / "run.status.json", {"status": "STARTING"})
    assert single.worker(SimpleNamespace(phase="run")) == 1
    assert shared.read(root / "Va/status.json")["updates_completed"] == 9
    assert not (root / "Vb").exists() and not (root / "O").exists()
    state = shared.read(root / "status.json")
    assert state["status"] == "DIAGNOSTIC_INVALID" and state["counts"]["optimizer_updates"] == 9
    assert state["current_group"] == "Va" and state["updates_completed"] == 9
    assert state["counts"]["precheck_backward"] == state["counts"]["precheck_updates"] == 2
    import zipfile
    with zipfile.ZipFile(root / "result_bundle.zip") as z:
        assert json.loads(z.read("status.json")) == state


def test_H_is_absent_from_loss_input_dictionary(single_run):
    root, runtime, engine_class = single_run
    training, regions = single.frozen_inputs(root, shared.read(root / "roi_manifest.json"), "cpu")
    assert set(training) == {A, B} and len(regions) == 3
    engine = engine_class(runtime["paths"], runtime["config"])
    with pytest.raises(KeyError):
        shared.perform_update(engine, training, "DSC07986.JPG", "O")
    assert engine.counts["training_rasterization"] == 0


@pytest.fixture
def proposed_regions(tmp_path, monkeypatch):
    from PIL import Image
    h = "DSC07986.JPG"
    candidates = [{"image_name": h, "train_view_id": 26, "distance_to_A": .1}]
    proposal = dict(protocol=PROTOCOL, candidate_sha256=canonical_sha(candidates), check_view=h,
        candidates_reviewed=1, earlier_candidate_exclusions={}, surface_description="same upper brick gable",
        correspondence_basis="same apex, roof slopes and brick pattern",
        polygons={A: [[[2, 2], [8, 2], [5, 6]]], h: [[[3, 3], [9, 3], [6, 7]]]})
    for folder in ("prepared", "review", "old"):
        (tmp_path / folder).mkdir()
    identity = dict(cameras=[dict(basename=n, width=12, height=10) for n in (A, B, h)],
                    train_image_sha256={n: "fixture-original" for n in (A, B, h)})
    for n in (A, B, h):
        Image.new("RGB", (12, 10), (64, 64, 64)).save(tmp_path / "review" / f"{Path(n).stem}_original.png")
        if n != h:
            np.save(tmp_path / "prepared" / f"{Path(n).stem}_M.npy", np.zeros((10, 12), dtype=np.bool_))
            np.save(tmp_path / "prepared" / f"{Path(n).stem}_C.npy", np.full((10, 12), .2, dtype=np.float32))
    np.save(tmp_path / "prepared/DSC07987_source_rgb.npy", np.full((10, 12, 3), .4, dtype=np.float32))
    shared.new_json(tmp_path / "candidate_views.json", {"candidates": candidates})
    shared.new_json(tmp_path / "runtime.json", {"previous_prepare": str(tmp_path / "old")})
    shared.new_json(tmp_path / "old/preparation.json", {"identity": identity})
    shared.new_json(tmp_path / "preparation_reuse.json", {"prepared_numeric_sha256": {}, "review_image_sha256": {}, "counts": {"optimizer_updates": 0}})
    monkeypatch.setattr(single, "output_root", lambda args: tmp_path)
    monkeypatch.setattr(single, "verify_runtime", lambda runtime: None)
    return tmp_path, proposal, {"identity": identity}


def test_real_polygon_preparation_confirmation_and_final_status_bundle(proposed_regions):
    root, proposal, prepared = proposed_regions
    roi = single.build_regions(root, {}, prepared, proposal)
    assert roi["views"][B]["pixels"] == 0 and roi["views"][B]["kind"] == "train_intervention_roi"
    assert roi["views"][proposal["check_view"]]["kind"] == "eval_roi"
    q = roi["intervention_qualification"]
    assert q["qualified"] and q["delta_Q_nonzero_pixels"] == q["roi_pixels"]
    assert q["delta_Q_weight_sum"] == pytest.approx(.8*q["roi_pixels"])
    shared.new_json(root / "prepare.status.json", {"status": "ROI_REVIEW_REQUIRED", "exit_code": 0})
    shared.new_json(root / "status.json", {"status": "ROI_REVIEW_REQUIRED", "exit_code": 0})
    args = SimpleNamespace(panel_sha256="wrong", statement="These A/H polygons mark the same static brick surface.")
    with pytest.raises(ValueError, match="panel identity"):
        single.confirm_roi(args)
    assert not (root / "roi_confirmation.json").exists()
    args.panel_sha256 = roi["review_panel_sha256"]
    assert single.confirm_roi(args) == 0
    locked = shared.sha(root / "roi_manifest.json")
    assert shared.read(root / "roi_manifest.json")["confirmation"]["confirmed"]
    assert shared.read(root / "roi_confirmation.json")["roi_manifest_sha256"] == locked
    with pytest.raises(ValueError, match="already confirmed"):
        single.confirm_roi(args)
    with pytest.raises(ValueError, match="already confirmed"):
        single.build_regions(root, {}, prepared, proposal)
    assert shared.sha(root / "roi_manifest.json") == locked
    single.bundle(root, "prepare")
    import zipfile
    with zipfile.ZipFile(root / "review_bundle.zip") as z:
        assert json.loads(z.read("status.json"))["status"] == "ROI_REVIEW_REQUIRED"
        assert "preparation_reuse.json" in z.namelist()


def test_same_surface_revision_keeps_prior_bundle_and_updates_qualification(proposed_regions):
    import base64
    root, proposal, prepared = proposed_regions
    first = single.build_regions(root, {}, prepared, proposal)
    shared.new_json(root / "prepare.status.json", {"status": "ROI_REVIEW_REQUIRED", "exit_code": 0})
    shared.new_json(root / "status.json", {"status": "ROI_REVIEW_REQUIRED", "exit_code": 0})
    single.bundle(root, "prepare")
    original = shared.sha(root / "review_bundle.zip")
    revised = copy.deepcopy(proposal)
    revised["polygons"][A] = [[[3, 3], [7, 3], [5, 5]]]
    args = SimpleNamespace(proposal_base64=base64.b64encode(json.dumps(revised).encode()).decode())
    assert single.revise_roi(args) == 0
    current = shared.read(root / "roi_manifest.json")
    assert current["revision"] == 2 and current["views"][A]["pixels"] < first["views"][A]["pixels"]
    assert shared.sha(root / "review_bundle.zip") == original
    assert (root / "review_bundle_proposal2.zip").exists()
    assert shared.read(root / "prepare.status.json")["roi_revision"] == 2


def test_malformed_single_polygon_payload_rejected_before_preparation(proposed_regions):
    root, proposal, _ = proposed_regions
    proposal["polygons"][A] = proposal["polygons"][A][0]
    with pytest.raises(ValueError, match="list of polygons"):
        validate_proposal(proposal, shared.read(root / "candidate_views.json")["candidates"])


def test_prepare_reuses_arrays_with_no_mask_feature_or_gaussian_pass(proposed_regions, monkeypatch):
    import shutil
    import puri_gs.recoverability_runtime as runtime_module
    root, proposal, prepared = proposed_regions
    previous = root / "old"
    (previous / "prepared").mkdir()
    (previous / "review").mkdir()
    # Separate the existing fixture into immutable previous inputs and a fresh output.
    output = root / "bounded"
    output.mkdir()
    for folder in ("prepared", "review"):
        for p in (root / folder).iterdir():
            shutil.copyfile(p, previous / folder / p.name)
    source = root / "source"
    source.mkdir()
    for name in ("cfg.yml", "config.yaml", "v3_input_manifest.json", "v3_run_manifest.json"):
        (source / name).write_text("test-only immutable source identity")
    checkpoint = source / "source.pt"
    checkpoint.write_bytes(b"test-only checkpoint")
    counts = dict.fromkeys(("training_rasterization", "evaluation_rasterization", "precheck_rasterization",
        "preparation_rasterization", "gaussian_backward", "optimizer_updates", "precheck_updates", "head_updates", "topology_events"), 0)
    environment = dict(torch="test", cuda_runtime="test", packages={})
    prepared.update(problems=[], learning_rates=LRS, extracted_source_sha256="fixed-renderer-and-adam",
        runtime_environment=environment, optimizer={"not_loaded_in_prepare": True}, source_parameter_digest="fixed",
        prepared_numeric_sha256={str(p): shared.sha(p) for p in (previous / "prepared").iterdir()},
        review_image_sha256={str(p): shared.sha(p) for p in (previous / "review").iterdir()},
        source_sha256={str(p): shared.sha(p) for p in source.iterdir()})
    shared.state_write(previous / "preparation.json", prepared)
    shared.new_json(output / "candidate_views.json", shared.read(root / "candidate_views.json"))
    class PrepareOnlyEngine:
        learning_rates, extracted_source_sha, device = LRS, "fixed-renderer-and-adam", "cpu"
        source_hashes, test_image_read_attempts, opened_training_images = {}, 0, set()
        def __init__(self, paths, cfg, *, frozen_views):
            assert frozen_views == [A, B, proposal["check_view"]]
            self.counts = dict(counts)
        # Any accidental load_parameters, mask, feature access or render raises AttributeError.
    monkeypatch.setattr(runtime_module, "FinalStateRuntime", PrepareOnlyEngine)
    monkeypatch.setattr(runtime_module, "runtime_environment", lambda *a, **k: environment)
    runtime = dict(proposal=proposal, previous_prepare=str(previous), commit="test", paths=dict(checkpoint=str(checkpoint), source_run=str(source)),
        config=dict(protocol=PROTOCOL, previous_preparation_sha256=shared.sha(previous / "preparation.json"),
                    source_checkpoint_sha256=shared.sha(checkpoint)))
    before = {str(p): shared.sha(p) for p in previous.rglob("*") if p.is_file()}
    assert single.prepare(output, runtime, {}) == "ROI_REVIEW_REQUIRED"
    assert shared.read(output / "preparation_reuse.json")["counts"] == counts
    assert shared.read(output / "diagnostic_result.json")["flags"]["ROI_CONFIRMED"] is False
    assert shared.read(output / "roi_manifest.json")["intervention_qualification"]["qualified"]
    shared.verify_hashes(before)
