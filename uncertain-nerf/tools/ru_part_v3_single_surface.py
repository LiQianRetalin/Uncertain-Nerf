#!/usr/bin/env python3
"""Single house surface: A intervention, B context, one evaluation-only H."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ru_part_v3_recoverability as shared

CONFIG = ROOT / "configs/ru_part_v3_single_surface.json"
DEFAULT_RUN = "house_upper_v1"
PREREG = "single_surface_preregistration.json"
read, new_json, write, sha, check = shared.read, shared.new_json, shared.state_write, shared.sha, shared.check


def config():
    from puri_gs.single_surface import validate_config
    return validate_config(read(CONFIG))


def output_root(args):
    check(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id), "invalid run id")
    return ROOT / "logs-puri/ru_part_v3_single_surface_diag" / args.run_id


def decode_proposal(encoded):
    check(bool(encoded), "prepare requires the Codex-generated proposal payload")
    return json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))


def code_hashes(paths):
    result = shared.code_hashes(paths)
    for path in (CONFIG, Path(__file__), ROOT / "puri_gs/single_surface.py"):
        result[str(path)] = sha(path)
    return result


def verify_runtime(runtime):
    check(shared.current_commit() == runtime["commit"], "code commit changed after bounded preparation")
    shared.verify_hashes(runtime["code_sha256"])


def launch(args):
    from puri_gs.single_surface import candidate_views, validate_proposal, PROTOCOL
    check(os.name == "posix", "run background jobs in the existing Linux environment")
    root, cfg = output_root(args), config()
    commit = shared.current_commit()
    if args.phase == "prepare":
        check(not root.exists(), "output exists; preserve it, no overwrite or automatic retry")
        previous = ROOT / cfg["previous_prepare"]
        check(sha(previous / "preparation.json") == cfg["previous_preparation_sha256"], "previous preparation identity differs")
        prepared, old_runtime = read(previous / "preparation.json"), read(previous / "runtime.json")
        check(not prepared["problems"], "previous preparation has unresolved input problems")
        candidates = candidate_views(prepared["identity"])
        proposal = decode_proposal(args.proposal_base64)
        validate_proposal(proposal, candidates)
        gpu = shared.choose_gpu(args.gpu)
        paths = old_runtime["paths"]
        check(Path(paths["source_run"]).resolve() == (ROOT / cfg["source_run"]).resolve(), "source run path differs")
        check(sha(Path(paths["gsplat"], "examples/simple_trainer.py")) == paths["trainer_sha256"], "source trainer differs from V3")
        root.mkdir(parents=True)
        # Lock the metadata ranking before this worker reads/exports candidate images.
        new_json(root / "candidate_views.json", {"protocol": PROTOCOL, "candidates": candidates,
            "rule": "distance to A, then train view ID, first 8 excluding A/B", "locked_before_image_export": True,
            "selected_proposal": {"check_view": proposal["check_view"], "candidates_reviewed": proposal["candidates_reviewed"],
                "reason": proposal["correspondence_basis"], "earlier_exclusions": proposal.get("earlier_candidate_exclusions", {}),
                "label_status": "UNCONFIRMED_PROPOSAL"}})
        new_json(root / "runtime.json", {"config": cfg, "paths": paths, "previous_prepare": str(previous),
            "gpu": gpu, "commit": commit, "code_sha256": code_hashes(paths), "proposal": proposal,
            "python": sys.executable, "candidate_sha256": sha(root / "candidate_views.json")})
    else:
        runtime = read(root / "runtime.json")
        verify_runtime(runtime)
        check(read(root / "prepare.status.json")["status"] == "ROI_REVIEW_REQUIRED", "bounded preparation is incomplete/blocked")
        check(read(root / "roi_confirmation.json")["confirmed"] is True, "ROI_NOT_READY: explicit label confirmation missing")
        check(not (root / "run.status.json").exists() and not (root / PREREG).exists(), "run already started; no resume/retry")
        gpu = shared.choose_gpu(args.gpu, locked=runtime["gpu"])
    command = [sys.executable, str(Path(__file__).resolve()), "--run-id", args.run_id, "worker", args.phase]
    return shared.start_worker(root, args.phase, gpu, command)


def copy_verified(source, destination, expected):
    check(Path(source).is_file(), f"REQUIRED_INPUT_MISSING: {source}")
    check(sha(source) == expected, f"input identity differs: {source}")
    check(not Path(destination).exists(), f"output already exists: {destination}")
    shutil.copyfile(source, destination)


def build_regions(root, runtime, prepared, proposal):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from puri_gs.coverage_recoverability import polygon_mask
    from puri_gs.single_surface import A, B, PROTOCOL, validate_proposal, intervention_qualification, validate_frozen
    candidates = read(root / "candidate_views.json")["candidates"]
    h = validate_proposal(proposal, candidates)
    check(not (root / "roi_confirmation.json").exists() and not (root / PREREG).exists(), "ROI already confirmed/locked")
    revision = len(list((root / "roi").glob("proposal*"))) + 1
    directory = root / "roi" / f"proposal{revision}"
    directory.mkdir(parents=True)
    cameras = {row["basename"]: row for row in prepared["identity"]["cameras"]}
    views, panels = {}, []
    for name, role in ((A, "A"), (B, "B"), (h, "H")):
        camera = cameras[name]
        polygons = proposal["polygons"].get(name, [])
        region = (np.zeros((camera["height"], camera["width"]), dtype=np.bool_) if role == "B"
                  else polygon_mask(camera["width"], camera["height"], polygons))
        kind = "eval_roi" if role == "H" else "train_intervention_roi"
        path = directory / f"{Path(name).stem}_{kind}.npy"
        np.save(path, region)
        with Image.open(root / "review" / f"{Path(name).stem}_original.png") as image:
            check(image.size == (camera["width"], camera["height"]), "ROI / actual RGB dimensions differ")
            overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for polygon in polygons:
            points = [tuple(p) for p in polygon]
            draw.line(points + points[:1], fill=(255, 40, 30), width=2)
        boundary = directory / f"{Path(name).stem}_boundary.png"
        overlay.save(boundary)
        panels.append((f"{role}: {name} - " + ("context only, S=0" if role == "B" else kind), boundary))
        if polygons:
            points = [p for polygon in polygons for p in polygon]
            box = (max(0, min(p[0] for p in points)-35), max(0, min(p[1] for p in points)-25),
                   min(camera["width"], max(p[0] for p in points)+35), min(camera["height"], max(p[1] for p in points)+25))
            crop = overlay.crop(box)
            crop = crop.resize((crop.width*3, crop.height*3))
            crop_path = directory / f"{Path(name).stem}_boundary_detail.png"
            crop.save(crop_path)
            panels.append((f"{role} detail; coordinates remain in original factor4 pixels", crop_path))
        views[name] = {"role": role, "kind": kind, "path": str(path), "sha256": sha(path),
            "polygons": polygons, "width": camera["width"], "height": camera["height"], "pixels": int(region.sum()),
            "image_sha256": prepared["identity"]["train_image_sha256"][name]}
    panel = directory / "single_surface_roi_review.png"
    shared.contact_sheet(panels, panel, columns=2)
    arrays = {}
    for name in (A, B):
        arrays[name] = {k: torch.from_numpy(np.load(root / "prepared" / f"{Path(name).stem}_{k}.npy", allow_pickle=False)) for k in ("M", "C")}
        arrays[name]["S"] = torch.from_numpy(np.load(views[name]["path"], allow_pickle=False))
    rgb = torch.from_numpy(np.load(root / "prepared" / "DSC07987_source_rgb.npy", allow_pickle=False))
    with Image.open(root / "review/DSC07987_original.png") as image:
        target = torch.from_numpy(np.asarray(image).copy()).float() / 255.
    q = intervention_qualification(arrays[A]["M"], arrays[A]["C"], arrays[A]["S"], rgb, target)
    dq_b = validate_frozen(arrays[B]["M"], arrays[B]["C"], arrays[B]["S"], context=True)
    check(not bool(dq_b.any()), "context DeltaQ must be zero")
    q.update(context_S_zero=True, context_delta_Q_zero=True, check_ROI_used_for_training=False)
    manifest = {"protocol": PROTOCOL, "proposal_status": "UNCONFIRMED_PROPOSAL", "revision": revision,
        "target_view": A, "context_view": B, "check_view": h, "views": views,
        "surface_description": proposal["surface_description"], "correspondence_basis": proposal["correspondence_basis"],
        "candidates_reviewed": proposal["candidates_reviewed"], "earlier_candidate_exclusions": proposal.get("earlier_candidate_exclusions", {}),
        "candidate_views_sha256": sha(root / "candidate_views.json"), "review_panel_path": str(panel), "review_panel_sha256": sha(panel),
        "coordinate_space": "original factor4 training pixels", "intervention_qualification": q}
    new_json(directory / "roi_manifest.json", manifest)
    write(root / "roi_manifest.json", manifest)
    write(root / "intervention_qualification.json", q)
    print(json.dumps(q, ensure_ascii=False, indent=2), flush=True)
    print("ROI_PANEL", panel, "SHA256", sha(panel), flush=True)
    return manifest


def prepare(root, runtime, state):
    from puri_gs.recoverability_runtime import FinalStateRuntime, runtime_environment
    from puri_gs.single_surface import A, B, LRS
    cfg = runtime["config"]
    previous = Path(runtime["previous_prepare"])
    prepared = read(previous / "preparation.json")
    check(sha(previous / "preparation.json") == cfg["previous_preparation_sha256"], "previous preparation changed")
    check(sha(runtime["paths"]["checkpoint"]) == cfg["source_checkpoint_sha256"], "source checkpoint changed")
    h = runtime["proposal"]["check_view"]
    # No StaticSupport, head, feature-cache load, Gaussian load or rendering here.
    engine = FinalStateRuntime(runtime["paths"], cfg, frozen_views=[A, B, h])
    check(engine.learning_rates == prepared["learning_rates"] == LRS, f"terminal LR mismatch: {engine.learning_rates}")
    check(engine.extracted_source_sha == prepared["extracted_source_sha256"], "original renderer/Adam implementation differs")
    environment = runtime_environment(engine.device, package_hashes=False)
    for key in ("torch", "cuda_runtime", "packages"):
        check(environment[key] == prepared["runtime_environment"][key], f"source dependency environment differs: {key}")
    sources = dict(engine.source_hashes)
    source = Path(runtime["paths"]["source_run"])
    for path in [source / name for name in ("cfg.yml", "config.yaml", "v3_input_manifest.json", "v3_run_manifest.json")]:
        check(sha(path) == prepared["source_sha256"][str(path)], f"source identity changed: {path}")
        sources[str(path)] = sha(path)
    sources[str(previous / "preparation.json")] = cfg["previous_preparation_sha256"]
    (root / "prepared").mkdir()
    (root / "review").mkdir()
    for name in (A, B):
        for kind in ("M", "C"):
            old = previous / "prepared" / f"{Path(name).stem}_{kind}.npy"
            copy_verified(old, root / "prepared" / old.name, prepared["prepared_numeric_sha256"][str(old)])
            sources[str(old)] = prepared["prepared_numeric_sha256"][str(old)]
    rgb = previous / "prepared/DSC07987_source_rgb.npy"
    copy_verified(rgb, root / "prepared" / rgb.name, prepared["prepared_numeric_sha256"][str(rgb)])
    sources[str(rgb)] = prepared["prepared_numeric_sha256"][str(rgb)]
    for name in (A, B, h):
        filename = f"{Path(name).stem}_original.png"
        old = previous / "review" / filename
        if old.is_file() and str(old) in prepared["review_image_sha256"]:
            copy_verified(old, root / "review" / filename, prepared["review_image_sha256"][str(old)])
            sources[str(old)] = prepared["review_image_sha256"][str(old)]
        else:
            # Only the selected member of the already locked candidate list.
            shared.save_rgb(root / "review" / filename, engine.data(name)["image"] / 255.)
    prep = {"protocol": cfg["protocol"], "identity": prepared["identity"], "source_sha256": sources,
        "optimizer": prepared["optimizer"], "learning_rates": engine.learning_rates,
        "source_parameter_digest": prepared["source_parameter_digest"],
        "runtime_environment": environment,
        "prepared_numeric_sha256": {str(p): sha(p) for p in (root / "prepared").glob("*.npy")},
        "review_image_sha256": {str(p): sha(p) for p in (root / "review").glob("*.png")},
        "counts": engine.counts, "test_image_read_attempts": engine.test_image_read_attempts,
        "opened_training_images": sorted(engine.opened_training_images), "full_prepare_rerun": False}
    new_json(root / "preparation_reuse.json", prep)
    roi = build_regions(root, runtime, prepared, runtime["proposal"])
    result = result_record(root, "ROI_REVIEW_REQUIRED" if roi["intervention_qualification"]["qualified"] else "BLOCKED",
                           "ROI_CONFIRMATION_PENDING" if roi["intervention_qualification"]["qualified"] else "NO_ORACLE_INTERVENTION")
    result["counts"] = engine.counts
    write(root / "diagnostic_result.json", result)
    write_report(root, result)
    (root / "implementation_audit.md").write_text(
        "# 单表面准备审计\n\n复用原M/C与A的float source RGB。候选按到A距离预先锁定；没有重做161图prepare。\n\n"
        "原renderer、损失、fresh Adam及400步循环复用已有实现。B训练S=0，H区域仅供评价。\n\n"
        "本阶段没有CUDA更新预检或正式更新；本地自检见仓库SINGLE_SURFACE_RUNBOOK_CN.md，服务器未由此worker重跑单元测试。\n\n"
        + json.dumps({"commit": runtime["commit"], "learning_rates": engine.learning_rates, "counts": engine.counts,
                      "test_image_read_attempts": engine.test_image_read_attempts}, indent=2), encoding="utf-8")
    return result["PROCESS_STATUS"]


def confirm_roi(args):
    root = output_root(args)
    runtime, roi = read(root / "runtime.json"), read(root / "roi_manifest.json")
    verify_runtime(runtime)
    check(not (root / "roi_confirmation.json").exists() and not (root / PREREG).exists(), "ROI already confirmed/locked")
    check(read(root / "prepare.status.json")["status"] == "ROI_REVIEW_REQUIRED", "preparation not ready for confirmation")
    check(roi["intervention_qualification"]["qualified"], "NO_ORACLE_INTERVENTION")
    check(sha(roi["review_panel_path"]) == args.panel_sha256 == roi["review_panel_sha256"], "reviewed panel identity differs")
    check(len(args.statement.strip()) >= 8, "explicit static-surface and A/H correspondence confirmation required")
    shared.verify_hashes({r["path"]: r["sha256"] for r in roi["views"].values()})
    roi.update(proposal_status="CONFIRMED", confirmation={"confirmed": True, "statement": args.statement,
        "panel_sha256": args.panel_sha256, "time": time.time()})
    write(root / "roi_manifest.json", roi)
    new_json(root / "roi_confirmation.json", {**roi["confirmation"],
        "panel_sha256": args.panel_sha256, "roi_manifest_sha256": sha(root / "roi_manifest.json"),
        "source": "explicit user confirmation of the displayed A/H polygons and static correspondence"})
    print("ROI_CONFIRMED", root / "roi_confirmation.json")
    return 0


def revise_roi(args):
    root = output_root(args)
    runtime, roi = read(root / "runtime.json"), read(root / "roi_manifest.json")
    verify_runtime(runtime)
    proposal = decode_proposal(args.proposal_base64)
    check(proposal["check_view"] == roi["check_view"] and proposal["surface_description"] == roi["surface_description"],
          "boundary revision must keep the same proposed surface and H")
    prepared = read(Path(runtime["previous_prepare"]) / "preparation.json")
    shared.verify_hashes(read(root / "preparation_reuse.json")["prepared_numeric_sha256"])
    shared.verify_hashes(read(root / "preparation_reuse.json")["review_image_sha256"])
    updated = build_regions(root, runtime, prepared, proposal)
    ready = updated["intervention_qualification"]["qualified"]
    result = result_record(root, "ROI_REVIEW_REQUIRED" if ready else "BLOCKED",
                           "ROI_CONFIRMATION_PENDING" if ready else "NO_ORACLE_INTERVENTION")
    result["counts"] = read(root / "preparation_reuse.json")["counts"]
    write_report(root, result)
    write(root / "diagnostic_result.json", result)
    state = read(root / "prepare.status.json")
    state.update(status=result["PROCESS_STATUS"], exit_code=0 if ready else 2, roi_revision=updated["revision"],
                 revision_time=time.time())
    write(root / "prepare.status.json", state)
    write(root / "status.json", state)
    bundle(root, "prepare", filename=f"review_bundle_proposal{updated['revision']}.zip")
    return 0


def frozen_inputs(root, roi, device):
    import numpy as np
    import torch
    from puri_gs.single_surface import A, B, validate_frozen
    training, regions = {}, {}
    for name, record in roi["views"].items():
        region = torch.from_numpy(np.load(record["path"], allow_pickle=False)).to(device)
        check(region.dtype == torch.bool, "region must be a bool array")
        regions[name] = region
        if name in (A, B):
            values = {k: torch.from_numpy(np.load(root / "prepared" / f"{Path(name).stem}_{k}.npy", allow_pickle=False)).to(device).float() for k in ("M", "C")}
            validate_frozen(values["M"], values["C"], region, context=name == B)
            training[name] = {"M": values["M"][None, None], "C": values["C"][None, None], "S": region}
    check(set(training) == {A, B}, "H entered the training data flow")
    return training, regions


def evaluate(engine, training, regions, directory, *, update, h):
    import torch
    from puri_gs.single_surface import A, B, view_metrics
    rows, images = {}, []
    check(update in (0, 400), "single-surface has no intermediate evaluations")
    with torch.no_grad():
        for name, role in ((A, "A"), (B, "B"), (h, "H")):
            rgb, target, alpha = engine.render(name, kind="evaluation")
            values = training.get(name, {})
            rows[name] = view_metrics(rgb[0], target[0], alpha[0, ..., 0], role=role, roi=regions[name],
                mask=values["M"][0, 0] if values else None, support=values["C"][0, 0] if values else None)
            overlay = rgb[0].detach().clone()
            if role != "B":
                edge = regions[name] & (torch.nn.functional.max_pool2d((~regions[name])[None, None].float(), 3, 1, 1)[0, 0] > 0)
                overlay[edge] = overlay.new_tensor([1., .15, .1])
            path = directory / f"{Path(name).stem}_{update}_roi.png"
            shared.save_rgb(path, overlay)
            images.append((f"{role} {name} after {update} updates", path))
            if role != "B":
                alpha_path = directory / f"{Path(name).stem}_{update}_alpha.png"
                shared.save_rgb(alpha_path, alpha[0, ..., 0])
    shared.contact_sheet(images, directory / f"evaluation_{update}.png", columns=3, tile_width=430)
    new_json(directory / f"metrics_{update}.json", rows)
    return rows


def run_diagnostic(root, runtime, state):
    import torch
    from puri_gs.single_surface import A, B, LRS, decide, HISTORY
    from puri_gs.recoverability_runtime import FinalStateRuntime, runtime_environment
    roi, prep, confirmation = [read(root / p) for p in ("roi_manifest.json", "preparation_reuse.json", "roi_confirmation.json")]
    check(confirmation["confirmed"] and confirmation["roi_manifest_sha256"] == sha(root / "roi_manifest.json"), "ROI confirmation differs")
    check(roi["intervention_qualification"]["qualified"], "NO_ORACLE_INTERVENTION")
    check(sha(roi["review_panel_path"]) == confirmation["panel_sha256"] == roi["review_panel_sha256"], "ROI panel changed")
    check(sha(root / "candidate_views.json") == runtime["candidate_sha256"] == roi["candidate_views_sha256"], "candidate order changed")
    shared.verify_hashes(prep["source_sha256"])
    shared.verify_hashes(prep["prepared_numeric_sha256"])
    shared.verify_hashes(prep["review_image_sha256"])
    shared.verify_hashes({row["path"]: row["sha256"] for row in roi["views"].values()})
    check(sha(runtime["paths"]["checkpoint"]) == runtime["config"]["source_checkpoint_sha256"], "source checkpoint changed")
    h = roi["check_view"]
    engine = FinalStateRuntime(runtime["paths"], runtime["config"], frozen_views=[A, B, h])
    state["counts"] = engine.counts
    check(engine.learning_rates == prep["learning_rates"] == LRS, "terminal LR differs")
    environment = runtime_environment(engine.device, package_hashes=False)
    check(environment == prep["runtime_environment"], "environment differs from bounded preparation")
    training, regions = frozen_inputs(root, roi, engine.device)
    image_checks, prechecks = {}, {}
    state.update(current_stage="cuda_precheck", current_group=None, updates_completed=0)
    write(root / "run.status.json", state)
    for group in ("Va", "O"):
        shared.reset_rng()
        before = engine.load_parameters()
        check(before == prep["source_parameter_digest"] and engine.optimizer_description() == prep["optimizer"], "precheck source/optimizer mismatch")
        try:
            prechecks[group], image_checks[group] = shared.perform_update(engine, training, A, group, precheck=True, retain_image_gradient=True)
        finally:
            write(root / "run.status.json", state)
        check(prechecks[group]["some_parameter_gradient_nonzero"] and shared.tensor_digest(engine.splats) != before, "CUDA precheck did not change finite parameters")
        engine.discard()
    # Reuse retained RGB gradients from the two actual backwards; no VJP/probe pass.
    dq = ((1-training[A]["M"]) * (1-training[A]["C"]) * training[A]["S"][None, None]).cpu().permute(0, 2, 3, 1)
    expected = .8 * dq * (image_checks["O"]["rgb"]-image_checks["O"]["target"]).sign() / image_checks["O"]["rgb"].numel()
    actual = image_checks["O"]["gradient"] - image_checks["Va"]["gradient"]
    error = float((actual-expected).abs().max())
    check(torch.allclose(image_checks["Va"]["rgb"], image_checks["O"]["rgb"], atol=1e-6, rtol=1e-6), "precheck initial render mismatch")
    check(bool(expected.abs().sum() > 0) and torch.allclose(actual, expected, atol=1e-8, rtol=1e-4), "recovery image-gradient difference check failed")
    check(sha(runtime["paths"]["checkpoint"]) == runtime["config"]["source_checkpoint_sha256"], "precheck modified source")
    check(engine.counts["precheck_updates"] == engine.counts["precheck_rasterization"] == engine.counts["precheck_backward"] == 2,
          "precheck call budget violated")
    new_json(root / "cuda_precheck.json", {"status": "PASS", "updates": engine.counts["precheck_updates"],
        "rasterizations": engine.counts["precheck_rasterization"], "gaussian_backwards": engine.counts["precheck_backward"],
        "temporary_copies_discarded": engine.splats is None, "source_unchanged": True, "image_gradient_max_abs_error": error,
        "image_gradient_atol": 1e-8, "image_gradient_rtol": 1e-4, "checks": prechecks, "head_updates": 0, "topology_events": 0})
    del image_checks, actual, expected, dq
    frozen = {**prep["prepared_numeric_sha256"], **prep["review_image_sha256"], **{row["path"]: row["sha256"] for row in roi["views"].values()}}
    frozen[roi["review_panel_path"]] = roi["review_panel_sha256"]
    for name in ("runtime.json", "candidate_views.json", "roi_manifest.json", "roi_confirmation.json", "preparation_reuse.json", "cuda_precheck.json"):
        frozen[str(root / name)] = sha(root / name)
    prereg = {"protocol": runtime["config"]["protocol"], "config": runtime["config"], **HISTORY,
        "code_commit": runtime["commit"], "code_sha256": runtime["code_sha256"],
        "source_checkpoint": runtime["paths"]["checkpoint"], "source_checkpoint_sha256": runtime["config"]["source_checkpoint_sha256"],
        "source_parameter_digest": prep["source_parameter_digest"], "source_sha256": prep["source_sha256"], "frozen_sha256": frozen,
        "candidate_order": read(root / "candidate_views.json"), "roi": roi, "confirmation": confirmation,
        "camera_sequence": [A, B] * 200, "optimizer": prep["optimizer"], "learning_rates": LRS,
        "optimizer_initialization": "fresh zero-state per group; no source Adam state", "rng": {"seed": 42, "fully_deterministic_claim": False},
        "gpu": runtime["gpu"], "runtime_environment": environment, "evaluation_updates": [0, 400],
        "metric_rules": {"target": "A only; B target ROI null", "check": "H evaluation ROI only",
            "initial": "mean of Va0/Vb0/O0", "n": "max(abs(Va400-Vb400), max(initials)-min(initials))",
            "gain": "both E0-O and min(Va,Vb)-O >= max(fraction*E0,3n,1e-4)",
            "protection_initial": "O-E0 <= d=max(0.01*E0,1e-4)", "protection_control": "O-min(Va,Vb) <= max(d,3n)",
            "domains": ["A outside S", "A accepted M", "B full", "B accepted M", "H inside R", "H outside R"],
            "rgb": "unclamped float render; original target/255; ROI-area mean only for evaluation"},
        "output_root": str(root)}
    new_json(root / PREREG, prereg)
    (root / "single_surface_preregistration.sha256").write_text(sha(root / PREREG) + "\n")
    print("PREREGISTRATION_LOCKED", sha(root / PREREG), flush=True)
    all_metrics, groups = {}, {}
    for group in ("Va", "Vb", "O"):
        all_metrics[group], groups[group] = shared.run_group(root, engine, group, prereg, training, state,
            evaluate_group=lambda engine, directory, update: evaluate(engine, training, regions, directory, update=update, h=h),
            evaluation_updates=(0, 400), prereg_path=root / PREREG)
    decision = decide(all_metrics, h)
    check(decision["status"] != "DIAGNOSTIC_INVALID", f"invalid evaluation: {decision}")
    check(0 < groups["O"]["actual_intervention_updates"] <= 200, "invalid number of A-only O intervention updates")
    check(engine.counts["training_rasterization"] == engine.counts["gaussian_backward"] == 1200 and
          engine.counts["evaluation_rasterization"] == 18 and engine.counts["precheck_updates"] == 2, "single-surface call budget violated")
    shared.verify_hashes(frozen)
    shared.verify_hashes(prep["source_sha256"])
    shared.verify_hashes(runtime["code_sha256"])
    check(sha(root / PREREG) == (root / "single_surface_preregistration.sha256").read_text().strip(), "preregistration changed")
    check(sha(runtime["paths"]["checkpoint"]) == runtime["config"]["source_checkpoint_sha256"], "source changed during diagnostic")
    for group, values in groups.items():
        values.update(GROUP_STATUS="COMPLETE", process_exit_code=0)
        write(root / group / "status.json", values)
    result = result_record(root, "DIAGNOSTIC_COMPLETE", decision["status"], groups=groups, complete=True)
    result.update(decision=decision, metrics=all_metrics, counts=engine.counts, source_checkpoint_unchanged=True,
                  preregistration_sha256=sha(root / PREREG), opened_training_images=sorted(engine.opened_training_images))
    write_report(root, result)
    write(root / "diagnostic_result.json", result)
    return "DIAGNOSTIC_COMPLETE"


def result_record(root, process, diagnostic, *, groups=None, complete=False):
    from puri_gs.single_surface import A, B, PROTOCOL, HISTORY
    groups = groups or {}
    roi = read(root / "roi_manifest.json") if (root / "roi_manifest.json").exists() else {}
    confirmation = read(root / "roi_confirmation.json") if (root / "roi_confirmation.json").exists() else {}
    flags = {"BRANCH": "ru-part", "PROTOCOL": PROTOCOL, **HISTORY, "TARGET_VIEW": A, "CONTEXT_VIEW": B,
        "CONTEXT_INTERVENTION_ROI_ZERO": roi.get("intervention_qualification", {}).get("context_S_zero"),
        "CHECK_VIEW": roi.get("check_view"), "CANDIDATE_VIEWS_REVIEWED": roi.get("candidates_reviewed", 0),
        "ROI_CONFIRMED": confirmation.get("confirmed", False), "FRESH_DIAGNOSTIC_OPTIMIZERS": True if complete else None,
        "MASK_FROZEN": True if complete else None, "HEAD_UPDATES": 0, "TOPOLOGY_EVENTS": 0,
        **{f"ACTUAL_UPDATES_{g.upper()}": groups.get(g, {}).get("updates_completed", 0) for g in ("Va", "Vb", "O")},
        "CHECK_VIEW_USED_IN_SHORT_OPTIMIZATION": False, "TEST_IMAGES_USED": False, "FULL_PREPARE_RERUN": False,
        "NEW_10K_PREFIX_RUN": False, "FULL_30K_RUN": False, "MULTI_SEED_RUN": False, "CROSS_SCENE_RUN": False, "NEXT_ALGORITHM_STARTED": False}
    return {"PROCESS_STATUS": process, "DIAGNOSTIC_STATUS": diagnostic, "flags": flags, "groups": groups,
            "LOCAL_GAIN": None, "PROTECTION_PASS": None, "CHECK_VIEW_GAIN": None,
            "intervention_qualification": roi.get("intervention_qualification")}


def write_report(root, result):
    from puri_gs.single_surface import A, B
    lines = ["# V3单表面诊断", "", f"执行状态：{result['PROCESS_STATUS']}；诊断结论：{result['DIAGNOSTIC_STATUS']}。", "",
        "A=DSC07987.JPG的房屋砖面；B=DSC07989.JPG仅上下文、S=0但每组仍优化200次；H只评价、在本轮更新0次，参与过原V3训练。", "",
        f"H：{result['flags']['CHECK_VIEW']}；已核对候选：{result['flags']['CANDIDATE_VIEWS_REVIEWED']}；静态标签确认：{result['flags']['ROI_CONFIRMED']}。", "",
        "| 组 | 实际更新 | 组状态 | 退出码 |", "|---|---:|---|---|"]
    for g in ("Va", "Vb", "O"):
        row = result["groups"].get(g, {})
        lines.append(f"| {g} | {row.get('updates_completed',0)} | {row.get('GROUP_STATUS','NOT_RUN')} | {row.get('process_exit_code')} |")
    if result.get("reason"):
        lines += ["", "停止原因：" + result["reason"]]
    if "decision" in result:
        decision = result["decision"]
        for field in ("LOCAL_GAIN", "PROTECTION_PASS", "CHECK_VIEW_GAIN"):
            result[field] = decision[field]
            result["flags"][field] = decision[field]
        lines += ["", "| 目标 | E0 | Va400 | Vb400 | O400 | n | 阈值 | 增益 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for name in ("local", "check"):
            row = decision[name]
            lines.append("| " + name + " | " + " | ".join(f"{row[k]:.8f}" for k in ("E0", "Va", "Vb", "O", "n", "threshold")) + f" | {row['gain']} |")
        lines += ["", "保护逐域判定；d只用于初态，不随n变大：", "", "```json", json.dumps(decision["protected_regions"], indent=2), "```"]
        lines += ["", "各组0/400完整浮点评价、初态三次误差和alpha辅助量见diagnostic_result.json及组内metrics。B空ROI记null，不进入任何目标平均。"]
    lines += ["", "新增监督资格：", "", "```json", json.dumps(result.get("intervention_qualification"), indent=2), "```",
        "", "5%/2%/1%、1e-4和3n为预先固定的工程规则；3n不是三个标准差。初态均值及极差仅来自三组0次评价。",
        "保护未建立须查看初态预算与控制比较分别是否失败；不一概宣称损害已被可靠检出。",
        "局部响应不等于几何真值或自动C能达到同样效果。无响应只约束本终态、新Adam、固定拓扑及400步，不证明Jacobian为零或必须birth。",
        "历史V3仍为NO_GO、四图诊断仍为ROI_NOT_READY、旧重放仍为REPLAY_NOT_EQUIVALENT；完成后停止，不追加实验。", "",
        "```json", json.dumps(result["flags"], indent=2, ensure_ascii=False), "```", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def bundle(root, phase, *, filename=None):
    destination = root / (filename or ("review_bundle.zip" if phase == "prepare" else "result_bundle.zip"))
    paths = [root / name for name in ("status.json", f"{phase}.status.json", "diagnostic_result.json", "report.md", "candidate_views.json",
             "roi_manifest.json", "intervention_qualification.json", "roi_confirmation.json", "cuda_precheck.json", PREREG,
             "single_surface_preregistration.sha256", "implementation_audit.md", "preparation_reuse.json") if (root / name).exists()]
    paths += list((root / "roi").rglob("*.png")) + list((root / "roi").rglob("*.npy"))
    if phase == "prepare":
        paths += list((root / "review").glob("*.png"))
    else:
        for group in ("Va", "Vb", "O"):
            paths += list((root / group).glob("*.json")) + list((root / group).glob("progress.jsonl"))
            paths += list((root / group).glob("evaluation_*.png"))
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root))
    print("READBACK_BUNDLE", destination, flush=True)


def worker(args):
    root = output_root(args)
    runtime = read(root / "runtime.json")
    path = root / f"{args.phase}.status.json"
    state = read(path)
    check(state["status"] == "STARTING", "phase cannot resume/retry")
    state.update(status="RUNNING", pid=os.getpid())
    write(path, state)
    write(root / "status.json", state)
    try:
        verify_runtime(runtime)
        shared.choose_gpu("auto", locked=runtime["gpu"])
        outcome = prepare(root, runtime, state) if args.phase == "prepare" else run_diagnostic(root, runtime, state)
        state.update(status=outcome, exit_code=2 if outcome == "BLOCKED" else 0)
    except Exception as error:
        groups = {}
        for g in ("Va", "Vb", "O"):
            gp = root / g / "status.json"
            if gp.is_file():
                groups[g] = read(gp)
                groups[g].update(GROUP_STATUS="INCOMPLETE", process_exit_code=1, end_time=time.time())
                write(gp, groups[g])
        state.update(status="BLOCKED" if args.phase == "prepare" else "DIAGNOSTIC_INVALID", exit_code=1,
                     error=f"{type(error).__name__}: {error}")
        if groups:
            active = state.get("current_group") if state.get("current_group") in groups else next(reversed(groups))
            state.update(current_group=active, updates_completed=groups[active]["updates_completed"])
        result = result_record(root, state["status"], state["error"] if args.phase == "prepare" else "DIAGNOSTIC_INVALID", groups=groups)
        result["reason"] = state["error"]
        result["counts"] = state.get("counts", {})
        if "TEST_IMAGE_ACCESS" in state["error"]:
            result["flags"]["TEST_IMAGES_USED"] = "ACCESS_ATTEMPT_REJECTED"
        write_report(root, result)
        write(root / "diagnostic_result.json", result)
        print(state["error"], flush=True)
    state["end_time"] = time.time()
    write(path, state)
    write(root / "status.json", state)
    try:
        bundle(root, args.phase)
    except Exception as error:
        state.update(status="READBACK_BUNDLE_FAILED", exit_code=1, diagnostic_outcome_before_export=state["status"],
                     error=f"{type(error).__name__}: {error}")
        write(path, state)
        write(root / "status.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return state["exit_code"]


def status(args):
    root = output_root(args)
    for phase in ("prepare", "run"):
        path = root / f"{phase}.status.json"
        if path.exists():
            value = read(path)
            if value.get("pid") and value["status"] == "RUNNING":
                try:
                    os.kill(value["pid"], 0)
                except ProcessLookupError:
                    value["status"] = "INCOMPLETE_PROCESS_GONE"
            print(json.dumps(value, ensure_ascii=False, indent=2))
    for group in ("Va", "Vb", "O"):
        path = root / group / "status.json"
        if path.is_file():
            print(group, json.dumps(read(path), ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("check-config")
    actions.add_parser("status")
    for action in ("launch", "worker"):
        command = actions.add_parser(action)
        command.add_argument("phase", choices=("prepare", "run"))
        if action == "launch":
            command.add_argument("--gpu", default="0")
            command.add_argument("--proposal-base64")
    command = actions.add_parser("confirm-roi")
    command.add_argument("--panel-sha256", required=True)
    command.add_argument("--statement", required=True)
    command = actions.add_parser("revise-roi")
    command.add_argument("--proposal-base64", required=True)
    args = parser.parse_args(argv)
    if args.action == "check-config":
        print(json.dumps(config(), ensure_ascii=False, indent=2))
        return 0
    return globals()[args.action.replace("-", "_")](args)


if __name__ == "__main__":
    raise SystemExit(main())
