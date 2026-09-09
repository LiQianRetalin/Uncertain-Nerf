#!/usr/bin/env python3
"""Bounded V3 coverage / Va,Vb,O diagnostic; existing final state only."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs/ru_part_v3_coverage_recoverability.json"
DEFAULT_RUN = "garden_v3_final_diag_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def output_root(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id):
        raise ValueError("invalid run id")
    return ROOT / "logs-puri/ru_part_v3_coverage_recoverability_diag" / args.run_id


def current_commit():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != "ru-part":
        raise RuntimeError(f"current branch is {branch}; use the Git GUI to locate ru-part")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def protocol():
    from puri_gs.coverage_recoverability import validate_config
    return validate_config(read(CONFIG))


def new_json(path, value):
    from puri_gs.coverage_recoverability import write_new_json
    write_new_json(path, value)


def state_write(path, value):
    from puri_gs.ru_part_v3 import write_json
    write_json(path, value)


def sha(path):
    from puri_gs.static_tracks import sha256_file
    return sha256_file(path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def source_paths(args):
    env = read(ROOT / "logs-puri/ru_part_v3_screening/environment_resolved.json")
    source = ROOT / protocol()["source_run"]
    result = {key: env[key] for key in ("gsplat", "data", "feature_cache", "track_cache", "trainer_sha256")}
    result.update(source_run=str(source), checkpoint=str(source / "ckpts/ckpt_29999_rank0.pt"))
    result["mask_manifest"] = str(args.mask_manifest.resolve()) if getattr(args, "mask_manifest", None) else None
    return result


def code_hashes(paths):
    files = [CONFIG, Path(__file__), ROOT / "puri_gs/coverage_recoverability.py",
             ROOT / "puri_gs/recoverability_runtime.py", ROOT / "puri_gs/ru_part_v3.py",
             ROOT / "puri_gs/semantic_mask.py", ROOT / "puri_gs/dino_features.py",
             ROOT / "puri_gs/static_tracks.py", ROOT / "puri_gs/v3_gpu.py",
             ROOT / "tools/build_puri_gs_static_tracks.py", Path(paths["gsplat"], "examples/simple_trainer.py"),
             Path(paths["gsplat"], "examples/datasets/colmap.py"), Path(paths["gsplat"], "examples/datasets/normalize.py")]
    return {str(path): sha(path) for path in files}


def verify_hashes(mapping):
    for path, expected in mapping.items():
        check(sha(path) == expected, f"locked input changed: {path}")


def choose_gpu(requested, *, locked=None):
    from puri_gs.v3_gpu import gpu_inventory, gpu_idle
    inventory = gpu_inventory()
    if locked is not None:
        candidates = [row for row in inventory if row["index"] == locked["index"] and row["uuid"] == locked["uuid"]]
    else:
        order = (0, 6, 7, 1, 2, 3, 4, 5) if requested == "auto" else (int(requested),)
        candidates = [row for index in order for row in inventory if row["index"] == index]
    selected = next((row for row in candidates if gpu_idle(row)), None)
    check(selected is not None, "chosen GPU is busy/unavailable; no job started")
    return selected


def launch(args):
    check(os.name == "posix", "background GPU diagnostic runs in the existing Linux environment")
    root = output_root(args)
    commit = current_commit()
    cfg = protocol()
    if args.phase == "prepare":
        check(not root.exists(), "output directory exists; no overwrite or automatic rerun")
        paths = source_paths(args)
        check(sha(Path(paths["gsplat"], "examples/simple_trainer.py")) == paths["trainer_sha256"], "source trainer differs from V3 preflight")
        gpu = choose_gpu(args.gpu)
        root.mkdir(parents=True)
        new_json(root / "runtime.json", {"paths": paths, "gpu": gpu, "commit": commit, "code_sha256": code_hashes(paths),
                                       "config": cfg, "python": sys.executable})
    else:
        runtime = read(root / "runtime.json")
        check(read(root / "prepare.status.json")["status"] == "PREPARATION_COMPLETE", "preparation is incomplete")
        check(runtime["commit"] == commit, "code commit changed after preparation")
        verify_hashes(runtime["code_sha256"])
        prereg = read(root / "diagnostic_preregistration.json")
        check(sha(root / "diagnostic_preregistration.json") == (root / "diagnostic_preregistration.sha256").read_text().strip(), "preregistration changed")
        check(prereg["roi_confirmation"]["confirmed"] is True, "ROI_NOT_READY: static labels not confirmed")
        gpu = choose_gpu(args.gpu, locked=runtime["gpu"])
        check(not (root / "run.status.json").exists(), "diagnostic was already launched; no retry/resume")
    status_path = root / f"{args.phase}.status.json"
    check(not status_path.exists(), "phase already exists")
    new_json(status_path, {"phase": args.phase, "status": "STARTING", "pid": None, "start_time": time.time(),
                           "exit_code": None, "gpu": gpu["index"], "gpu_uuid": gpu["uuid"], "current_group": None,
                           "updates_completed": 0})
    command = [sys.executable, str(Path(__file__).resolve()), "--run-id", args.run_id, "worker", args.phase]
    with (root / f"{args.phase}.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu["index"])},
                                 stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"已启动 {args.phase}，PID={child.pid}，GPU={gpu['index']}，UUID={gpu['uuid']}")
    print(f"最近日志：tail -n 80 '{root / (args.phase + '.log')}'")
    print(f"持续查看：tail -n 40 -f '{root / (args.phase + '.log')}'")
    print(f"状态：{sys.executable} {Path(__file__).resolve()} --run-id {args.run_id} status")
    return 0


def save_rgb(path, tensor):
    from PIL import Image
    image = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
    import numpy as np
    image = np.clip(image, 0, 1)
    Image.fromarray((image * 255).round().astype("uint8")).save(path)


def contact_sheet(items, destination, *, columns=2, tile_width=600):
    from PIL import Image, ImageDraw
    tiles = []
    for label, path in items:
        with Image.open(path) as original:
            image = original.convert("RGB")
        image.thumbnail((tile_width, 430))
        tile = Image.new("RGB", (tile_width, 466), "white")
        tile.paste(image, (0, 32))
        ImageDraw.Draw(tile).text((6, 8), label, fill="black")
        tiles.append(tile)
    panel = Image.new("RGB", (columns * tile_width, ((len(tiles) + columns - 1) // columns) * 466), "white")
    for i, tile in enumerate(tiles):
        panel.paste(tile, ((i % columns) * tile_width, (i // columns) * 466))
    panel.save(destination)


def mask_fallback(paths, config, name, identity):
    import numpy as np
    import torch
    if not paths.get("mask_manifest"):
        return None, None
    manifest = read(paths["mask_manifest"])
    check(manifest["source_checkpoint_sha256"] == config["source_checkpoint_sha256"] and manifest["source_step"] == 29999,
          "MASK_STATE_UNAVAILABLE: exported masks refer to another state")
    record = manifest.get("views", {}).get(name)
    if not record:
        return None, None
    check(record["coordinate_space"] == "factor4_training_pixels" and
          record["image_sha256"] == identity["train_image_sha256"][name], "exported Mask provenance mismatch")
    path = Path(paths["mask_manifest"]).parent / record["path"]
    check(sha(path) == record["sha256"], "exported Mask changed")
    value = np.load(path, allow_pickle=False)
    camera = next(row for row in identity["cameras"] if row["basename"] == name)
    check(value.shape == (camera["height"], camera["width"]) and np.isin(value, [0, 1]).all(), "invalid exported numerical M")
    return torch.from_numpy(value.copy()).float(), {str(path): record["sha256"], str(Path(paths["mask_manifest"])): sha(paths["mask_manifest"])}


def prepare(root, runtime, state):
    import numpy as np
    import torch
    from puri_gs.coverage_recoverability import coverage_row, aggregate_coverage, temporal_summary, rank_check_views
    from puri_gs.recoverability_runtime import FinalStateRuntime, runtime_environment
    cfg, paths = runtime["config"], runtime["paths"]
    source = Path(paths["source_run"])
    problems = []
    (root / "prepared").mkdir()
    review = root / "review"
    review.mkdir()
    checkpoint_hash = sha(paths["checkpoint"]) if Path(paths["checkpoint"]).is_file() else None
    if checkpoint_hash != cfg["source_checkpoint_sha256"]:
        problems.append("SOURCE_CHECKPOINT_IDENTITY_MISMATCH")
    engine = FinalStateRuntime(paths, cfg)
    if engine.optimizer_problem:
        problems.append(engine.optimizer_problem)
    current_environment = runtime_environment(engine.device)
    sources = dict(engine.source_hashes)
    for path in (source / "cfg.yml", source / "config.yaml", source / "v3_input_manifest.json", source / "v3_run_manifest.json",
                 source / "v3_progress.jsonl", source / "v3_training_checks.json", source / "aux/dino_environment.json", source / "environment.json",
                 Path(paths["feature_cache"], "manifest.json")):
        if path.is_file():
            sources[str(path)] = sha(path)
    rows, masks = [], {}
    for index, name in enumerate(engine.names):
        camera = engine.identity["cameras"][index]
        c = engine.support.current(index, (camera["height"], camera["width"]))[0, 0]
        reason = engine.mask_problem
        mask_source = "same_run_final_head_and_fine_cache"
        try:
            mask = engine.mask(name)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            mask, reason = None, str(error)
        if mask is None:
            try:
                mask, provenance = mask_fallback(paths, cfg, name, engine.identity)
                if provenance:
                    sources.update(provenance)
                    mask = mask.to(engine.device)
                    mask_source = "validated_full_numerical_export"
            except (OSError, ValueError, KeyError) as error:
                reason = str(error)
        rows.append(coverage_row(name, c, mask, mask_reason=reason if mask is None else None))
        rows[-1]["mask_source"] = mask_source if mask is not None else "UNAVAILABLE"
        if name in cfg["optimization_views"]:
            stem = Path(name).stem
            np.save(root / "prepared" / f"{stem}_C.npy", c.cpu().numpy())
            if mask is not None:
                np.save(root / "prepared" / f"{stem}_M.npy", mask.cpu().numpy().astype(np.bool_))
                masks[name] = mask
        if (index + 1) % 20 == 0 or index + 1 == len(engine.names):
            state.update(current_stage="coverage", views_completed=index+1)
            state_write(root / "prepare.status.json", state)
            print(f"COVERAGE {index+1}/{len(engine.names)}", flush=True)
    with (root / "coverage_by_view.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records = [json.loads(line) for line in (source / "v3_progress.jsonl").read_text().splitlines() if line.strip()] if (source / "v3_progress.jsonl").is_file() else []
    summary = aggregate_coverage(rows)
    checks_path = source / "v3_training_checks.json"
    summary["timing_windows"] = temporal_summary(records, read(checks_path) if checks_path.is_file() else {})
    new_json(root / "coverage_summary.json", summary)
    if set(masks) != set(cfg["optimization_views"]):
        problems.append("MASK_STATE_UNAVAILABLE")
    rank = rank_check_views(engine.identity["cameras"])
    new_json(root / "check_view_candidates.json", {"ranking_rule": "minimum Euclidean camera-center distance to either optimization view; ties use train view ID", "candidates": rank})
    candidates = [row["image_name"] for row in rank[:cfg["neighbor_preview_count"]]]
    for name in cfg["optimization_views"] + candidates:
        target = engine.data(name)["image"] / 255.
        save_rgb(review / f"{Path(name).stem}_original.png", target)
    contact_sheet([(f"rank {i+1}: {name}", review / f"{Path(name).stem}_original.png") for i, name in enumerate(candidates)], review / "neighbor_candidates.png", columns=4, tile_width=400)
    if checkpoint_hash == cfg["source_checkpoint_sha256"]:
        initial_digest = engine.load_parameters(optimizers=engine.learning_rates is not None)
        optimizer_description = engine.optimizer_description() if engine.learning_rates is not None else None
        for name in cfg["optimization_views"]:
            stem = Path(name).stem
            with torch.no_grad():
                rgb, target, alpha = engine.render(name, kind="preparation")
            np.save(root / "prepared" / f"{stem}_source_rgb.npy", rgb[0].cpu().numpy())
            save_rgb(review / f"{stem}_render.png", rgb[0])
            save_rgb(review / f"{stem}_residual.png", (rgb[0] - target[0]).abs().mean(-1))
            c = np.load(root / "prepared" / f"{stem}_C.npy")
            save_rgb(review / f"{stem}_C.png", c)
            tiles = [(f"{name} source {target.shape[2]}x{target.shape[1]}", review / f"{stem}_original.png"),
                     ("source render", review / f"{stem}_render.png"),
                     ("RGB MAE, display clipped only", review / f"{stem}_residual.png"),
                     ("C (no threshold)", review / f"{stem}_C.png")]
            if name in masks:
                m = masks[name].cpu().numpy()
                q = (1 - m) * c
                save_rgb(review / f"{stem}_M.png", m)
                save_rgb(review / f"{stem}_Q.png", q)
                overlay = target[0].cpu().numpy().copy()
                overlay = overlay * (1 - .65 * q[..., None])
                overlay[..., 0] += .65 * q
                save_rgb(review / f"{stem}_Q_overlay.png", overlay)
                tiles += [("final M", review / f"{stem}_M.png"), ("Q", review / f"{stem}_Q.png"),
                          ("Q overlay (not a static label)", review / f"{stem}_Q_overlay.png")]
            contact_sheet(tiles, review / f"{stem}_alignment_panel.png", columns=3, tile_width=430)
        engine.discard()
    else:
        optimizer_description = None
        initial_digest = None
    sources.update(engine.source_hashes)
    new_json(root / "preparation.json", {"problems": problems, "checkpoint_sha256": checkpoint_hash,
        "runtime_environment": current_environment,
        "source_parameter_digest": initial_digest,
        "prepared_numeric_sha256": {str(path): sha(path) for path in sorted((root / "prepared").glob("*.npy"))},
        "review_image_sha256": {str(path): sha(path) for path in sorted(review.glob("*.png"))},
        "checkpoint_stat": {"size": Path(paths["checkpoint"]).stat().st_size, "mtime_ns": Path(paths["checkpoint"]).stat().st_mtime_ns} if checkpoint_hash else None,
        "source_sha256": sources, "optimization_views": cfg["optimization_views"], "preview_candidates": candidates,
        "identity": engine.identity, "optimizer": optimizer_description, "learning_rate_basis": "original step29999 before its optimizer update; 29999 ExponentialLR decays, not post-update 30000",
        "learning_rates": engine.learning_rates, "scene_scale": engine.scene_scale, "extracted_source_sha256": engine.extracted_source_sha,
        "mask_source": "same-run aux/mask_head_step29999.pt plus cached fine features; histogram not used in Mask inference" if engine.head is not None else "validated exported numerical arrays, where available",
        "opened_training_images": sorted(engine.opened_training_images), "test_image_read_attempts": engine.test_image_read_attempts,
        "render_counts": engine.counts})
    new_json(root / "preregistration_draft.json", {"status": "ROI_NOT_READY", "config": cfg, "source_checkpoint_sha256": checkpoint_hash,
             "optimizer": optimizer_description, "gpu": runtime["gpu"], "camera_sequence": cfg["optimization_views"] * 200,
             "static_labels_confirmed": False, "check_view_selection": "pending original-image same-surface review"})
    with zipfile.ZipFile(root / "review_bundle.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(review.glob("*.png")):
            archive.write(path, path.relative_to(root))
        for name in ("coverage_by_view.csv", "coverage_summary.json", "check_view_candidates.json", "preparation.json"):
            archive.write(root / name, name)
    (root / "implementation_audit.md").write_text(
        "# V3覆盖与局部恢复诊断：准备审计\n\n" +
        "源step29999；真实head仅用于冻结M，fine缓存不加载DINO模型。原renderer与optimizer构造尾部直接复用。\n\n" +
        "累计Q更新计数与稀疏日志均值分开记录；静态ROI尚未确认，Va/Vb/O均未运行。\n\n" +
        json.dumps({"commit": runtime["commit"], "problems": problems, "mask_scope_views": summary["views_with_M"],
                    "learning_rates": engine.learning_rates, "render_counts": engine.counts}, ensure_ascii=False, indent=2), encoding="utf-8")
    return "BLOCKED" if problems else "PREPARATION_COMPLETE"


def propose_roi(args):
    import numpy as np
    from PIL import Image, ImageDraw
    from puri_gs.coverage_recoverability import polygon_mask
    root = output_root(args)
    runtime, prepared = read(root / "runtime.json"), read(root / "preparation.json")
    check(not prepared["problems"], "preparation has unresolved problems")
    verify_hashes(prepared["review_image_sha256"])
    check(not (root / "roi_confirmation.json").exists() and not (root / "diagnostic_preregistration.json").exists(),
          "ROI was already confirmed/locked; it cannot be changed")
    check(re.fullmatch(r"[A-Za-z0-9_-]{1,60}", args.proposal_id), "invalid proposal id")
    # This small data payload is generated by Codex after image review. The user
    # never has to compose polygon JSON or source a static label from residuals.
    proposal = json.loads(base64.b64decode(args.proposal_base64, validate=True).decode("utf-8"))
    optimize = runtime["config"]["optimization_views"]
    checks = proposal["check_views"]
    check(len(checks) == 2 and len(set(optimize + checks)) == 4, "ROI_NOT_READY: four distinct training views required")
    check(all(name in prepared["preview_candidates"] for name in checks), "check views must come from the geometry-ranked preview")
    ranked = prepared["preview_candidates"]
    check([name for name in ranked if name in checks] == checks, "check views must preserve geometry order")
    excluded = proposal.get("earlier_candidate_exclusions", {})
    for name in ranked[:ranked.index(checks[-1])]:
        if name not in checks:
            check(bool(excluded.get(name)), "missing original-image explanation for skipping an earlier geometric candidate")
    check(set(proposal["polygons"]) == set(optimize + checks), "ROI polygon view set differs")
    check(bool(proposal.get("surface_description")), "static surface description is missing")
    roi_dir = root / "roi" / args.proposal_id
    roi_dir.mkdir(parents=True)
    records, panels = {}, []
    cameras = {row["basename"]: row for row in prepared["identity"]["cameras"]}
    for name in optimize + checks:
        camera = cameras[name]
        mask = polygon_mask(camera["width"], camera["height"], proposal["polygons"][name])
        stem = Path(name).stem
        path = roi_dir / f"{stem}_S.npy"
        np.save(path, mask)
        source_image = root / "review" / f"{stem}_original.png"
        with Image.open(source_image) as original:
            overlay = original.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for polygon in proposal["polygons"][name]:
            points = [tuple(point) for point in polygon]
            draw.line(points + points[:1], fill=(255, 50, 30), width=4)
        overlay_path = roi_dir / f"{stem}_boundary.png"
        overlay.save(overlay_path)
        panels.append((name + (" [optimize]" if name in optimize else " [check only]"), overlay_path))
        records[name] = {"path": str(path), "sha256": sha(path), "polygons": proposal["polygons"][name],
                         "width": camera["width"], "height": camera["height"], "roi_pixels": int(mask.sum()),
                         "image_sha256": prepared["identity"]["train_image_sha256"][name]}
    contact_sheet(panels, roi_dir / "four_view_static_roi_review.png")
    panel_hash = sha(roi_dir / "four_view_static_roi_review.png")
    manifest = {"source_checkpoint_sha256": prepared["checkpoint_sha256"], "views": records,
        "optimization_views": optimize, "check_views": checks, "surface_description": proposal["surface_description"],
        "earlier_candidate_exclusions": excluded, "rasterize_rule": "Pillow ImageDraw.polygon on integer original factor4 pixel coordinates; filled boundary included",
        "static_confirmation": "PENDING", "review_panel_sha256": panel_hash,
        "proposal_id": args.proposal_id, "review_panel_path": str(roi_dir / "four_view_static_roi_review.png")}
    new_json(roi_dir / "manifest.json", manifest)
    state_write(root / "roi_manifest.json", manifest)  # only an unconfirmed active proposal pointer
    print("ROI_REVIEW_REQUIRED", roi_dir / "four_view_static_roi_review.png")
    print("PANEL_SHA256", panel_hash)
    return 0


def confirm_roi(args):
    root = output_root(args)
    roi = read(root / "roi_manifest.json")
    check(args.panel_sha256 == roi["review_panel_sha256"] == sha(roi["review_panel_path"]), "ROI panel differs from the reviewed panel")
    check(len(args.statement.strip()) >= 8, "record the user's explicit static-surface/boundary confirmation")
    verify_hashes({row["path"]: row["sha256"] for row in roi["views"].values()})
    new_json(root / "roi_confirmation.json", {"confirmed": True, "statement": args.statement,
        "review_panel_sha256": args.panel_sha256, "roi_manifest_sha256": sha(root / "roi_manifest.json"),
        "confirmed_at": time.time(), "source": "explicit user confirmation after four-view original-image boundary review"})
    print("ROI_CONFIRMED")
    return 0


def lock(args):
    import numpy as np
    from puri_gs.coverage_recoverability import canonical_sha
    root = output_root(args)
    runtime, prepared, roi, confirmation = [read(root / name) for name in
        ("runtime.json", "preparation.json", "roi_manifest.json", "roi_confirmation.json")]
    check(not prepared["problems"], "preparation has unresolved problems")
    check(read(root / "prepare.status.json")["status"] == "PREPARATION_COMPLETE", "preparation is incomplete")
    check(current_commit() == runtime["commit"], "code changed after preparation")
    verify_hashes(runtime["code_sha256"])
    verify_hashes(prepared["source_sha256"])
    verify_hashes(prepared["prepared_numeric_sha256"])
    verify_hashes(prepared["review_image_sha256"])
    check(confirmation["confirmed"] is True and confirmation["roi_manifest_sha256"] == sha(root / "roi_manifest.json"), "ROI_NOT_READY")
    check(confirmation["review_panel_sha256"] == sha(roi["review_panel_path"]), "reviewed ROI panel changed")
    check(sha(runtime["paths"]["checkpoint"]) == prepared["checkpoint_sha256"], "source checkpoint changed")
    from puri_gs.v3_gpu import gpu_inventory
    check(any(row["uuid"] == runtime["gpu"]["uuid"] and row["index"] == runtime["gpu"]["index"] for row in gpu_inventory()), "GPU mapping changed")
    qualified, frozen = {}, {}
    for name in roi["optimization_views"]:
        stem = Path(name).stem
        m_path, c_path = (root / "prepared" / f"{stem}_{kind}.npy" for kind in ("M", "C"))
        m, c = np.load(m_path), np.load(c_path)
        s = np.load(roi["views"][name]["path"])
        rgb_path = root / "prepared" / f"{stem}_source_rgb.npy"
        rgb = np.load(rgb_path)
        from PIL import Image
        with Image.open(root / "review" / f"{stem}_original.png") as image:
            target = np.asarray(image).astype(np.float32) / 255.
        dq = (1 - m.astype(np.float32)) * (1 - c) * s
        weighted_residual = float((dq[..., None] * np.abs(rgb - target)).sum(dtype=np.float64))
        qualified[name] = {"delta_Q_nonzero_pixels": int((dq > 0).sum()), "delta_Q_weight_sum": float(dq.sum(dtype=np.float64)),
                           "delta_Q_nonzero_fraction": float((dq > 0).mean()), "delta_Q_weighted_residual_sum": weighted_residual}
        frozen.update({str(path): sha(path) for path in (m_path, c_path, rgb_path)})
    if not (sum(row["delta_Q_weight_sum"] for row in qualified.values()) > 0 and
            sum(row["delta_Q_weighted_residual_sum"] for row in qualified.values()) > 0):
        result = {"PROCESS_STATUS": "BLOCKED", "DIAGNOSTIC_STATUS": "NO_ORACLE_INTERVENTION",
                  "intervention_qualification": qualified, "flags": final_flags({})}
        write_report(root, result)
        new_json(root / "diagnostic_result.json", result)
        print("NO_ORACLE_INTERVENTION: no optimization launched")
        return 2
    frozen.update({row["path"]: row["sha256"] for row in roi["views"].values()})
    frozen[str(root / "roi_manifest.json")] = sha(root / "roi_manifest.json")
    frozen[str(root / "roi_confirmation.json")] = sha(root / "roi_confirmation.json")
    frozen[str(root / "preparation.json")] = sha(root / "preparation.json")
    frozen[str(root / "runtime.json")] = sha(root / "runtime.json")
    frozen[roi["review_panel_path"]] = sha(roi["review_panel_path"])
    verify_hashes(frozen)
    check(prepared["optimizer"] is not None, "OPTIMIZER_CONFIG_UNRESOLVED")
    environment = read(Path(runtime["paths"]["source_run"]) / "environment.json")
    prereg = {"protocol": runtime["config"], "source_checkpoint": runtime["paths"]["checkpoint"],
        "source_checkpoint_sha256": prepared["checkpoint_sha256"], "source_sha256": prepared["source_sha256"],
        "source_parameter_digest": prepared["source_parameter_digest"],
        "frozen_sha256": frozen, "code_sha256": runtime["code_sha256"], "code_commit": runtime["commit"],
        "source_environment": environment, "runtime_environment": prepared["runtime_environment"],
        "gpu": runtime["gpu"], "python": sys.executable,
        "optimization_views": roi["optimization_views"], "check_views": roi["check_views"],
        "camera_sequence": roi["optimization_views"] * 200, "optimizer": prepared["optimizer"],
        "optimizer_state_initialization": "fresh zero-moment/zero-step Adam per group",
        "learning_rate_basis": prepared["learning_rate_basis"], "criteria": runtime["config"]["criteria"],
        "aggregation": "equal mean of two per-view ROI MAEs; E0 mean and range over Va/Vb/O starts; controls Va/Vb final",
        "protection": "each view separately; optimization accepted pixels outside S, fallback outside S; check views outside S and each check S",
        "render_metric_scale": "unclamped render RGB and original factor4 target /255", "roi_confirmation": confirmation,
        "roi_manifest_sha256": sha(root / "roi_manifest.json"), "intervention_qualification": qualified,
        "test_images_forbidden": True, "head_frozen": True, "topology_frozen": True,
        "planned_updates_per_group": 400, "output_root": str(root),
        "historical_V3_status": "QUALITY_RECOVERY_FAIL / NO_GO", "historical_replay_status": "REPLAY_NOT_EQUIVALENT"}
    new_json(root / "diagnostic_preregistration.json", prereg)
    (root / "diagnostic_preregistration.sha256").write_text(sha(root / "diagnostic_preregistration.json") + "\n")
    print("PREREGISTRATION_LOCKED", root / "diagnostic_preregistration.json")
    print(json.dumps({"intervention": qualified, "gpu": runtime["gpu"], "optimizer": prepared["optimizer"]}, ensure_ascii=False))
    return 0


def frozen_inputs(root, roi, device):
    import numpy as np
    import torch
    result = {}
    for name, record in roi["views"].items():
        s = torch.from_numpy(np.load(record["path"], allow_pickle=False)).to(device).bool()
        row = {"S": s}
        if name in roi["optimization_views"]:
            for key in ("M", "C"):
                value = np.load(root / "prepared" / f"{Path(name).stem}_{key}.npy", allow_pickle=False)
                row[key] = torch.from_numpy(value).to(device).float()[None, None].detach()
        result[name] = row
    return result


def evaluate(engine, inputs, directory, *, update):
    import torch
    from puri_gs.coverage_recoverability import region_metrics
    metrics = {}
    with torch.no_grad():
        for name, frozen in inputs.items():
            rgb, target, alpha = engine.render(name, kind="evaluation")
            metrics[name] = region_metrics(rgb[0], target[0], alpha[0, ..., 0], frozen["S"],
                frozen["M"][0, 0] if "M" in frozen else None)
            if update in (0, 400):
                save_rgb(directory / f"{Path(name).stem}_{update}_rgb.png", rgb[0])
                save_rgb(directory / f"{Path(name).stem}_{update}_alpha.png", alpha[0, ..., 0])
                edge = frozen["S"] & ~torch.nn.functional.max_pool2d(
                    (~frozen["S"])[None, None].float(), 3, 1, 1)[0, 0].eq(0)
                overlay = rgb[0].detach().clone()
                overlay[edge] = overlay.new_tensor([1., .2, .1])
                save_rgb(directory / f"{Path(name).stem}_{update}_roi_boundary.png", overlay)
    new_json(directory / f"metrics_{update}.json", metrics)
    return metrics


def tensor_digest(parameters):
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(parameters):
        digest.update(key.encode())
        digest.update(parameters[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def perform_update(engine, inputs, name, group, *, precheck=False):
    import torch
    from fused_ssim import fused_ssim
    from puri_gs.coverage_recoverability import diagnostic_loss
    frozen = inputs[name]
    rgb, target, _ = engine.render(name, kind="precheck" if precheck else "training")
    loss, base, extra = diagnostic_loss(rgb, target, frozen["M"], frozen["C"], frozen["S"][None, None].float(),
                                      group=group, fused_ssim_fn=fused_ssim)
    check(bool(torch.isfinite(loss)), "non-finite diagnostic loss")
    loss.backward()
    if not precheck:
        engine.counts["gaussian_backward"] += 1
    gradients = [p.grad for p in engine.splats.values() if p.grad is not None]
    check(gradients and bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()), "non-finite Gaussian gradients")
    nonzero = bool(torch.stack([(g != 0).any() for g in gradients]).any()) if precheck else None
    delta_q = (1 - frozen["M"]) * (1 - frozen["C"]) * frozen["S"][None, None]
    intervened = group == "O" and bool((delta_q.permute(0, 2, 3, 1) * (rgb - target).abs()).detach().sum() > 0)
    for optimizer in engine.optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    engine.counts["precheck_updates" if precheck else "optimizer_updates"] += 1
    check(bool(torch.stack([torch.isfinite(p).all() for p in engine.splats.values()]).all()), "non-finite updated Gaussian parameters")
    check(len(engine.splats["means"]) == engine.protocol["gaussian_count"], "frozen Gaussian count changed")
    return {"base_loss": float(base.detach()), "recovery_loss": float(extra.detach()),
            "actual_intervention": intervened, "some_parameter_gradient_nonzero": nonzero}


def run_diagnostic(root, runtime, state):
    import random
    import numpy as np
    import torch
    from puri_gs.coverage_recoverability import diagnostic_decision
    from puri_gs.recoverability_runtime import FinalStateRuntime, validate_splats, runtime_environment
    check(sha(root / "diagnostic_preregistration.json") == (root / "diagnostic_preregistration.sha256").read_text().strip(), "preregistration changed")
    prereg = read(root / "diagnostic_preregistration.json")
    verify_hashes(prereg["code_sha256"])
    verify_hashes(prereg["source_sha256"])
    verify_hashes(prereg["frozen_sha256"])
    check(sha(prereg["source_checkpoint"]) == prereg["source_checkpoint_sha256"], "source checkpoint changed")
    engine = FinalStateRuntime(runtime["paths"], runtime["config"])
    check(runtime_environment(engine.device) == prereg["runtime_environment"], "diagnostic environment differs from preregistration")
    roi = read(root / "roi_manifest.json")
    inputs = frozen_inputs(root, roi, engine.device)
    engine.head = engine.feature_cache = None  # all diagnostic M/C/S come from frozen numerical arrays
    def reset_seed():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    reset_seed()
    before = engine.load_parameters()
    check(before == prereg["source_parameter_digest"], "source parameters differ from preparation")
    check(engine.optimizer_description() == prereg["optimizer"], "OPTIMIZER_CONFIG_UNRESOLVED: runtime optimizer differs")
    name = next(name for name, row in prereg["intervention_qualification"].items() if row["delta_Q_weighted_residual_sum"] > 0)
    precheck = perform_update(engine, inputs, name, "O", precheck=True)
    after = tensor_digest(engine.splats)
    check(precheck["some_parameter_gradient_nonzero"] and before != after, "CUDA precheck did not update parameters")
    check(sha(prereg["source_checkpoint"]) == prereg["source_checkpoint_sha256"], "precheck modified source checkpoint")
    engine.discard()
    new_json(root / "cuda_precheck.json", {"status": "PASS", "temporary_copy_discarded": engine.splats is None,
        "training_rasterization": 0, "precheck_rasterization": 1, "precheck_backward": 1,
        "parameters_changed": before != after, "gaussian_count_unchanged": True, "source_file_unchanged": True, **precheck})
    all_metrics, group_states = {}, {}
    initial_digest, optimizer_settings = None, None
    for group in ("Va", "Vb", "O"):
        directory = root / group
        directory.mkdir()
        reset_seed()
        digest = engine.load_parameters()
        settings = engine.optimizer_description()
        if initial_digest is None:
            initial_digest, optimizer_settings = digest, settings
        check(digest == initial_digest == before and settings == optimizer_settings == prereg["optimizer"], "group initial state/settings differ")
        start_counts = dict(engine.counts)
        group_state = {"GROUP_STATUS": "RUNNING", "process_exit_code": None, "group": group,
                       "updates_completed": 0, "last_diagnostic_step": -1, "source_step": 29999,
                       "diagnostic_only": True, "gaussian_count_initial": len(engine.splats["means"]),
                       "initial_parameter_digest": digest, "fresh_optimizer": True, "actual_intervention_updates": 0}
        state_write(directory / "status.json", group_state)
        metrics = {"0": evaluate(engine, inputs, directory, update=0)}
        state.update(current_stage="optimization", current_group=group, updates_completed=0)
        state_write(root / "run.status.json", state)
        started = time.perf_counter()
        with (directory / "progress.jsonl").open("x") as log:
            for step, name in enumerate(prereg["camera_sequence"]):
                try:
                    values = perform_update(engine, inputs, name, group)
                finally:
                    # Preserve the actual number of fully applied optimizer updates,
                    # including an update that subsequently fails the finite check.
                    completed = engine.counts["optimizer_updates"] - start_counts["optimizer_updates"]
                    group_state.update(updates_completed=completed, last_diagnostic_step=completed-1)
                    state_write(directory / "status.json", group_state)
                group_state["actual_intervention_updates"] += int(values["actual_intervention"])
                updates = step + 1
                group_state.update(updates_completed=updates, last_diagnostic_step=step)
                if updates % 50 == 0:
                    row = {"group": group, "updates_completed": updates, "image": name,
                           "gaussian_count": len(engine.splats["means"]), "elapsed_seconds": time.perf_counter()-started, **values}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                    state.update(current_group=group, updates_completed=updates)
                    state_write(root / "run.status.json", state)
                    state_write(directory / "status.json", group_state)
                if updates in (100, 200, 400):
                    metrics[str(updates)] = evaluate(engine, inputs, directory, update=updates)
        artifact = directory / "diagnostic_splats.pt"
        torch.save({"step": 399, "splats": {key: p.detach().cpu() for key, p in engine.splats.items()},
                    "diagnostic_only": True, "source_step": 29999, "diagnostic_updates": 400}, artifact)
        saved = torch.load(artifact, map_location="cpu", weights_only=True)
        validate_splats(saved, count=runtime["config"]["gaussian_count"], step=399)
        del saved
        counts = {key: engine.counts[key] - start_counts[key] for key in engine.counts}
        check(counts["training_rasterization"] == counts["gaussian_backward"] == counts["optimizer_updates"] == 400 and
              counts["evaluation_rasterization"] == 16 and counts["head_updates"] == counts["topology_events"] == 0, "diagnostic call budget violated")
        group_state.update(GROUP_STATUS="UPDATES_COMPLETE", gaussian_count_final=len(engine.splats["means"]),
                           final_parameter_artifact_exists=True, final_parameter_artifact_loadable=True,
                           final_metrics_finite=True, head_updates=0, topology_events=0, counts=counts,
                           artifact_sha256=sha(artifact))
        new_json(directory / "manifest.json", {"source_step": 29999, "diagnostic_updates": 400, "diagnostic_only": True,
            "preregistration_sha256": sha(root / "diagnostic_preregistration.json"), "optimizer": settings,
            "M_C_S_sha256": prereg["frozen_sha256"], "initial_parameter_digest": digest})
        state_write(directory / "status.json", group_state)
        all_metrics[group], group_states[group] = metrics, group_state
        engine.discard()
    decision = diagnostic_decision(all_metrics, roi["optimization_views"], roi["check_views"])
    check(decision["status"] != "DIAGNOSTIC_INVALID", f"invalid final evaluation: {decision}")
    check(group_states["O"]["actual_intervention_updates"] > 0, "NO_ORACLE_INTERVENTION during actual O updates")
    verify_hashes(prereg["frozen_sha256"])
    check(sha(prereg["source_checkpoint"]) == prereg["source_checkpoint_sha256"], "source checkpoint changed during diagnostic")
    for group, value in group_states.items():
        value.update(GROUP_STATUS="COMPLETE", process_exit_code=0)
        state_write(root / group / "status.json", value)
    flags = final_flags(group_states, complete=True)
    result = {"PROCESS_STATUS": "DIAGNOSTIC_COMPLETE", "DIAGNOSTIC_STATUS": decision["status"],
              "decision": decision, "metrics": all_metrics, "groups": group_states, "flags": flags,
              "preregistration_sha256": sha(root / "diagnostic_preregistration.json"),
              "intervention_qualification": prereg["intervention_qualification"], "counts": engine.counts,
              "source_checkpoint_unchanged": True, "opened_training_images": sorted(engine.opened_training_images)}
    write_report(root, result)
    new_json(root / "diagnostic_result.json", result)
    return "DIAGNOSTIC_COMPLETE"


def final_flags(groups, *, complete=False):
    return {"BRANCH": "ru-part", "SOURCE_CHECKPOINT_STEP": 29999, "DIAGNOSTIC_ONLY": True,
            "ORIGINAL_V3_STATUS": "QUALITY_RECOVERY_FAIL / NO_GO", "HISTORICAL_REPLAY_STATUS": "REPLAY_NOT_EQUIVALENT",
            "HISTORICAL_REPLAY_RECLASSIFIED": False, "EXACT_REPLAY_REQUIRED": False,
            "FRESH_DIAGNOSTIC_OPTIMIZERS": True if complete else "NOT_RUN_OR_INCOMPLETE",
            "MASK_FROZEN": True if complete else "NOT_RUN_OR_INCOMPLETE",
            "HEAD_UPDATES": 0 if complete else "NOT_RUN_OR_INCOMPLETE",
            "TOPOLOGY_EVENTS": 0 if complete else "NOT_RUN_OR_INCOMPLETE",
            "DEDICATED_BIRTH_ENABLED": False, "TEST_IMAGES_USED_FOR_DIAGNOSTIC": False,
            "TRAIN_CHECK_VIEWS_USED_IN_SHORT_OPTIMIZATION": False, "PLANNED_UPDATES_PER_GROUP": 400,
            **{f"ACTUAL_UPDATES_{g.upper()}": groups.get(g, {}).get("updates_completed", 0) for g in ("Va", "Vb", "O")},
            "NEW_10K_PREFIX_RUN": False, "FULL_30K_RUN": False, "MULTI_SEED_RUN": False,
            "CROSS_SCENE_RUN": False, "NEXT_ALGORITHM_STARTED": False}


def write_report(root, result):
    lines = ["# V3覆盖与局部可恢复性诊断", "", f"执行状态：{result['PROCESS_STATUS']}",
             f"诊断状态：{result['DIAGNOSTIC_STATUS']}", "",
             "这是固定V3终态、新Adam状态、固定终态学习率、固定Mask与拓扑的400步局部诊断。检查图参与过原训练，仅未参与本轮短优化。",
             "人工静态标签只用于O，不属于可部署自动方法。历史V3仍为NO_GO，旧重放仍为REPLAY_NOT_EQUIVALENT。", ""]
    if result.get("reason"):
        lines += [f"停止原因：{result['reason']}", ""]
    coverage_path = root / "coverage_summary.json"
    if coverage_path.is_file():
        coverage = read(coverage_path)
        lines += [f"C统计覆盖{coverage['views_with_C']}张训练图；终态M可用{coverage['views_with_M']}张。缺失M不当作Q=0。", "",
                  "| 统计口径 | C非零比例 | mean C | M拒绝比例 | Q非零比例 | mean Q |",
                  "|---|---:|---:|---:|---:|---:|"]
        for field, label in (("equal_view_mean", "逐图等权"), ("pixel_weighted_mean", "按图像像素数加权")):
            lines.append("| " + label + " | " + " | ".join(str(coverage[field][key]) for key in
                ("C_nonzero_fraction", "mean_C", "mask_rejected_fraction", "Q_nonzero_fraction", "mean_Q")) + " |")
        lines += ["", "| 原训练步窗口 | 可核实激活步数 | 日志采样点数 | 采样 mean Q | 采样新增L1 | 全窗口新增L1总和 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in coverage["timing_windows"]["windows"]:
            q, loss = row["metrics"]["mean_Q"], row["metrics"]["added_l1"]
            lines.append(f"| {row['start']}–{row['end']} | {row['active_updates']} | {row['logged_rows']} | "
                         f"{q['logged_sample_mean']} | {loss['logged_sample_mean']} | {loss['whole_window_sum']} |")
        lines += ["", "None表示历史文件无法给出该数值；不按稀疏点插值或补跑训练。窗口计数的边界推导记录在coverage_summary.json。", ""]
    prepared_path = root / "preparation.json"
    if prepared_path.is_file():
        prepared = read(prepared_path)
        lines += [f"Mask来源：{prepared['mask_source']}。", f"优化器学习率口径：{prepared['learning_rate_basis']}。", "",
                  "```json", json.dumps(prepared["learning_rates"], ensure_ascii=False, indent=2), "```", ""]
    decision = result.get("decision", {})
    if "local" in decision:
        lines += ["| 集合ROI MAE | 初态均值 | Va400 | Vb400 | O400 | 波动n | 实际阈值 | 优于初态 | 优于更好控制 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for label in ("local", "check"):
            row = decision[label]
            lines.append("| " + label + " | " + " | ".join(f"{row[key]:.8f}" for key in ("E0", "Va", "Vb", "O", "n", "threshold", "improvement_vs_initial", "improvement_vs_better_control")) + " |")
        lines += ["", "各图保护区域的判定值：", "", "```json", json.dumps(decision["protected_regions"], ensure_ascii=False, indent=2), "```"]
        lines += ["", "| 图像 | O初态ROI MAE | Va400 | Vb400 | O400 | O初态ROI alpha | O400 ROI alpha | O400低alpha比例 | O400保护区MAE |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, start in result["metrics"]["O"]["0"].items():
            finish = result["metrics"]["O"]["400"][name]
            values = (start["roi_mae"], result["metrics"]["Va"]["400"][name]["roi_mae"],
                      result["metrics"]["Vb"]["400"][name]["roi_mae"], finish["roi_mae"],
                      start["roi_alpha_mean"], finish["roi_alpha_mean"], finish["roi_low_alpha_fraction"], finish["protection_mae"])
            lines.append("| " + name + " | " + " | ".join(f"{v:.8f}" for v in values) + " |")
        lines += ["", f"实际O干预步数：{result['groups']['O']['actual_intervention_updates']}。预先核验的新增Q及残差加权量：", "",
                  "```json", json.dumps(result["intervention_qualification"], ensure_ascii=False, indent=2), "```",
                  "", "训练/评估/临时CUDA预检分别计数：", "", "```json", json.dumps(result["counts"], indent=2), "```"]
    lines += ["", "覆盖及时序统计见coverage_summary.json和coverage_by_view.csv。稀疏日志均值不当作全窗口均值；原Mask来源与ROI确认见preparation.json、roi_manifest.json和roi_confirmation.json。",
              "", "三组初态误差取算术均值及max−min；3n是工程波动余量，不是三个标准差。负结果只约束本状态、终态小学习率、零动量和400步预算，不能证明Jacobian为零或必须birth。",
              "", "完整数值、每图ROI/保护区MAE及alpha辅助指标见diagnostic_result.json。只提高alpha不判恢复。",
              "", "完成后停止，不修改C、不追加第四组、不改步长或门槛、不做24张test评测、FPS基准或新的30k。", "",
              "```json", json.dumps(result.get("flags", {}), indent=2, ensure_ascii=False), "```", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def worker(args):
    root = output_root(args)
    state_path = root / f"{args.phase}.status.json"
    state = read(state_path)
    check(state["status"] == "STARTING", "phase cannot be resumed or rerun")
    runtime = read(root / "runtime.json")
    state.update(status="RUNNING", pid=os.getpid())
    state_write(state_path, state)
    try:
        choose_gpu("auto", locked=runtime["gpu"])
        check(current_commit() == runtime["commit"], "code commit changed")
        verify_hashes(runtime["code_sha256"])
        result = prepare(root, runtime, state) if args.phase == "prepare" else run_diagnostic(root, runtime, state)
        state.update(status=result, exit_code=2 if result == "BLOCKED" else 0)
        if result == "BLOCKED":
            minimal = {"PROCESS_STATUS": "BLOCKED", "DIAGNOSTIC_STATUS": ";".join(read(root / "preparation.json")["problems"]),
                       "flags": final_flags({})}
            new_json(root / "diagnostic_result.json", minimal)
            write_report(root, minimal)
    except Exception as error:
        state.update(status="DIAGNOSTIC_INVALID" if args.phase == "run" else "BLOCKED", exit_code=1,
                     error=f"{type(error).__name__}: {error}")
        groups = {}
        for group in ("Va", "Vb", "O"):
            path = root / group / "status.json"
            if path.exists():
                groups[group] = read(path)
                groups[group].update(GROUP_STATUS="INCOMPLETE", process_exit_code=1)
                state_write(path, groups[group])
        result = {"PROCESS_STATUS": state["status"], "DIAGNOSTIC_STATUS": "DIAGNOSTIC_INVALID" if args.phase == "run" else state["error"],
                  "reason": state["error"], "flags": final_flags(groups), "groups": groups}
        if not (root / "diagnostic_result.json").exists():
            new_json(root / "diagnostic_result.json", result)
        write_report(root, result)
        print(state["error"], flush=True)
    state["end_time"] = time.time()
    state_write(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)
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
        if path.exists():
            print(group, json.dumps(read(path), ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("check-config")
    commands.add_parser("status")
    for action in ("launch", "worker"):
        p = commands.add_parser(action)
        p.add_argument("phase", choices=("prepare", "run"))
        if action == "launch":
            p.add_argument("--gpu", default="0")
            p.add_argument("--mask-manifest", type=Path)
    p = commands.add_parser("propose-roi")
    p.add_argument("--proposal-base64", required=True)
    p.add_argument("--proposal-id", default="proposal1")
    p = commands.add_parser("confirm-roi")
    p.add_argument("--panel-sha256", required=True)
    p.add_argument("--statement", required=True)
    commands.add_parser("lock")
    args = parser.parse_args(argv)
    if args.action == "check-config":
        print(json.dumps(protocol(), indent=2, ensure_ascii=False))
        return 0
    return globals()[args.action.replace("-", "_")](args)


if __name__ == "__main__":
    raise SystemExit(main())
