#!/usr/bin/env python3
"""Small Garden-only V3 preflight, background wrapper, and screening report."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from puri_gs.ru_part_v3 import PROTOCOL, DISABLED_COUNTS, write_json
from puri_gs.static_tracks import sha256_file
from puri_gs.v3_gpu import GPU_PRIORITY, gpu_inventory, select_gpu

STAGES = ("cache", "smoke-parent", "smoke-v3", "parent", "v3", "eval-parent", "eval-v3")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def output_root():
    return ROOT / "logs-puri/ru_part_v3_screening"


def environment():
    return read(output_root() / "environment_resolved.json")


def preflight(args):
    import torch
    out = output_root()
    out.mkdir(parents=True, exist_ok=True)
    pinned = None
    for stage in STAGES:
        state_path = out / f"{stage}.status.json"
        if state_path.is_file():
            state = read(state_path)
            if state.get("status") in ("STARTING", "RUNNING"):
                raise RuntimeError(f"{stage} is active or interrupted; inspect its status before re-running preflight")
            if stage != "cache":
                pinned = environment()["gpu"]
    inventory = gpu_inventory()
    selected = select_gpu(inventory, args.gpu, pinned=pinned)
    print(f"GPU_SELECTED={selected['index']} ({selected['name']}); priority={list(GPU_PRIORITY)}", flush=True)
    paths = {
        "root": str(ROOT), "python": str(Path(sys.executable).resolve()),
        "gsplat": str(ROOT / "external/gsplat-v1.5.3-ru-part-v3"),
        "data": str(ROOT / "data/mipnerf360/360_v2/garden"),
        "dino_repo": str(ROOT / "external/dinov2"),
        "dino_weight": str(ROOT / "data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth"),
        "feature_cache": str(ROOT / "data/PURI-GS-derived/semantic_features/garden"),
        "gpu": selected["index"], "gpu_uuid": selected["uuid"],
        "gpu_priority": list(GPU_PRIORITY), "gpu_inventory_at_preflight": inventory,
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    problems = []
    if branch != "ru-part":
        problems.append(f"branch is {branch}, expected ru-part")
    if os.name != "posix":
        problems.append("training wrapper requires the existing Linux environment")
    for key, suffix in (("data", "sparse/0/points3D.bin"), ("data", "images_4_png"),
                        ("dino_repo", "hubconf.py"), ("dino_weight", ""),
                        ("feature_cache", "manifest.json")):
        path = Path(paths[key]) / suffix
        if not path.exists():
            problems.append(f"missing {path}")
    if not torch.cuda.is_available():
        problems.append("CUDA unavailable in the existing Python environment")
    try:
        if importlib.metadata.version("gsplat").split("+")[0] != "1.5.3":
            problems.append("installed gsplat must be 1.5.3")
    except importlib.metadata.PackageNotFoundError:
        problems.append("gsplat not installed")
    candidates = sorted((ROOT / "data/PURI-GS-derived").rglob("garden_factor4_static_tracks.pt"))
    if args.track_cache:
        candidates = [args.track_cache.resolve()] if args.track_cache.is_file() else []
    if len(candidates) == 1:
        paths["track_cache"] = str(candidates[0].resolve())
    elif not candidates:
        paths["track_cache"] = str(ROOT / "data/PURI-GS-derived/static_tracks/garden_v3_original_builder/garden_factor4_static_tracks.pt")
        paths["cache_build_required"] = True
    else:
        problems.append("multiple static caches; select an accepted cache with --track-cache")
    paths.update({"commit": commit, "branch": branch, "torch": torch.__version__})
    cache_audit = None
    if paths.get("track_cache") and Path(paths["track_cache"]).is_file():
        from puri_gs.static_tracks import load_static_track_cache
        try:
            payload = load_static_track_cache(paths["track_cache"])
            cache_audit = {"file_sha256": sha256_file(paths["track_cache"]),
                           "payload_sha256": payload["payload_sha256"],
                           "build_config": payload["build_config"], "build_git_commit": payload["build_git_commit"],
                           "train_count": len(payload["train_basenames"]),
                           "grid_nonzero_fraction": float(payload["track_evidence_binary"].float().mean()),
                           "feature_manifest_sha256": payload["dino_feature_manifest_sha256"]}
        except (ValueError, OSError) as error:
            problems.append(f"static cache validation failed: {error}")
    old_parents = []
    for config in (ROOT / "logs-puri").glob("**/config.yaml"):
        if out in config.parents:
            continue
        try:
            c = read(config)
        except (ValueError, OSError):
            continue
        command_path = config.parent / "run_command.txt"
        command_text = command_path.read_text() if command_path.is_file() else ""
        if "garden" not in str(config.parent).casefold() and "garden" not in command_text.casefold():
            continue
        if ((c.get("profile") == "ru" and not c.get("paper_control"))
                or (c.get("profile") == "ru_part" and c.get("intervention_mode") == "parent")):
            old_parents.append({"run": str(config.parent), "checkpoint_exists": (config.parent / "ckpts/ckpt_29999_rank0.pt").is_file(),
                                "data_identity_exists": (config.parent / "v3_input_manifest.json").is_file(),
                                "status": "REQUIRES_HISTORICAL_COMPARABILITY_AUDIT"})
    audit = {"paths": paths, "problems": problems, "cache": cache_audit, "existing_standard_ru_candidates": old_parents,
             "old_part_parent": "lineage wrapper, no cap in current source; historical data/algorithm/cost still require evidence",
             "parent_plan": "No historical Parent is automatically accepted. If identity/algorithm cannot be established, one fresh standard Parent is necessary.",
             "status": "PREFLIGHT_BLOCKED" if problems else "PATHS_READY"}
    write_json(out / "preflight.json", audit)
    if problems:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 2
    if not Path(paths["gsplat"]).exists():
        sources = [ROOT / "external/gsplat-v1.5.3-ru-part-controls", ROOT / "external/gsplat-v1.5.3"]
        source = next((p for p in sources if (p / ".git").exists()), None)
        if source is None:
            raise FileNotFoundError("No existing pinned gsplat source; no dependency downloads attempted")
        env = {**os.environ, "PURI_GSPLAT_REPOSITORY": str(source)}
    else:
        env = os.environ.copy()
    subprocess.run(["bash", str(ROOT / "scripts/prepare_puri_gs_ru_part_v3.sh"), paths["gsplat"]], env=env, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_ru_part_v3.py", "tests/test_v3_gpu.py",
                    "tests/test_v3_parent_reference.py", "-q"], cwd=ROOT, check=True)
    # Trainer help is checked in the exact Python environment before any launch.
    subprocess.run([sys.executable, "simple_trainer.py", "default", "--help"],
                   cwd=Path(paths["gsplat"]) / "examples", env={**env, "PYTHONPATH": str(ROOT)},
                   stdout=(out / "trainer_help.txt").open("w"), stderr=subprocess.STDOUT, check=True)
    paths["gpu_name"] = selected["name"]
    paths["trainer_sha256"] = sha256_file(Path(paths["gsplat"], "examples/simple_trainer.py"))
    write_json(out / "environment_resolved.json", paths)
    audit["status"] = "PREFLIGHT_CACHE_REQUIRED" if paths.get("cache_build_required") else "PREFLIGHT_READY"
    write_json(out / "preflight.json", audit)
    print(audit["status"], out / "preflight.json")
    if old_parents:
        print("发现历史标准 RU 候选，先核对 preflight.json 中的对照记录；未核对前不应直接增加 Parent 训练。")
    else:
        print("未找到可核对的标准 RU Parent。将需要增加一次标准 Parent 从头训练，与一次 V3 比较。")
    return 0


def stage_paths(stage):
    base = output_root()
    return base / stage, base / (stage + ".log"), base / (stage + ".status.json")


def command_for(stage, env):
    if stage == "cache":
        return [env["python"], str(ROOT / "tools/build_puri_gs_static_tracks.py"),
                "--data-dir", env["data"], "--feature-cache-dir", env["feature_cache"],
                "--dino-weight-path", env["dino_weight"], "--gsplat-dir", env["gsplat"],
                "--output-dir", str(Path(env["track_cache"]).parent), "--factor", "4", "--device", "cuda:0"]
    mode = "parent" if "parent" in stage else "v3"
    run, _, _ = stage_paths(stage)
    cmd = [env["python"], str(ROOT / "run_puri_gs.py"), "--config",
           str(ROOT / f"configs/puri_gs_ru_part_v3_{mode}_garden30k.yaml"),
           "--gsplat-dir", env["gsplat"], "--data-dir", env["data"],
           "--result-dir", str(run), "--gpu", str(env["gpu"])]
    if stage.startswith("eval-"):
        checkpoint = stage_paths(mode)[0] / "ckpts/ckpt_29999_rank0.pt"
        cmd += ["--checkpoint", str(checkpoint), "--eval-warmup-renders", "10"]
    else:
        cmd += ["--dino-repo-dir", env["dino_repo"], "--dino-weight-path", env["dino_weight"],
                "--feature-cache-dir", env["feature_cache"], "--v3-stop-step",
                "599" if stage.startswith("smoke-") else "29999"]
        if mode == "v3":
            cmd += ["--track-cache", env["track_cache"]]
    return cmd


def validate_checkpoint(path, expected_step):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("step") != expected_step:
        raise ValueError("checkpoint step differs from terminal step")
    splats = checkpoint["splats"]
    count = len(splats["means"])
    shapes = {"means": (count, 3), "scales": (count, 3), "quats": (count, 4),
              "opacities": (count,), "sh0": (count, 1, 3), "shN": (count, 15, 3)}
    if count <= 0 or set(splats) != set(shapes):
        raise ValueError("invalid standard Gaussian checkpoint keys/count")
    for name, shape in shapes.items():
        if tuple(splats[name].shape) != shape or not torch.isfinite(splats[name]).all():
            raise ValueError(f"invalid standard Gaussian tensor {name}")
    return count


def require_complete(stage, expected):
    state = read(stage_paths(stage)[2])
    if state["status"] != expected or state["exit_code"] != 0:
        raise ValueError(f"{stage} has not completed: {state}")


def require_parent_ready():
    from puri_gs.v3_parent_reference import load_reference
    reference = load_reference(output_root(), verify=True)
    if reference is None:
        require_complete("parent", "TRAIN_COMPLETE")
    return reference


def register_parent(args):
    from puri_gs.v3_parent_reference import register_reference
    env = environment()
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current_commit != env["commit"]:
        raise ValueError("code commit changed; rerun preflight before registration")
    if sha256_file(Path(env["gsplat"], "examples/simple_trainer.py")) != env["trainer_sha256"]:
        raise ValueError("trainer source changed since preflight")
    for stage in STAGES:
        path = stage_paths(stage)[2]
        if path.is_file() and read(path).get("status") in ("STARTING", "RUNNING"):
            raise RuntimeError(f"{stage} is active; finish it before registering a Parent")
    reference = register_reference(output_root(), ROOT, args.run, args.eval_dir, env, validate_checkpoint)
    print("PARENT_REFERENCE_READY", output_root() / "parent_reference.json")
    print(json.dumps({key: reference[key] for key in (
        "decision", "source_run", "evaluation_dir", "gaussian_count", "psnr_reproduction_abs_difference",
        "quality_comparability", "historical_training_gpu", "reevaluation_gpu", "training_time_comparability",
        "historical_input_hashes", "historical_camera_sequence", "limitations")}, indent=2, ensure_ascii=False))
    return 0


def launch(args):
    env = environment()
    run, log, status_path = stage_paths(args.stage)
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current_commit != env["commit"]:
        raise ValueError("code commit changed; rerun preflight before launch")
    if sha256_file(Path(env["gsplat"], "examples/simple_trainer.py")) != env["trainer_sha256"]:
        raise ValueError("trainer source changed since preflight")
    for other in STAGES:
        other_path = stage_paths(other)[2]
        if other_path.is_file() and read(other_path).get("status") in ("STARTING", "RUNNING"):
            raise RuntimeError(f"{other} is active or interrupted; inspect its status before launching another job")
    if run.exists() or log.exists() or status_path.exists():
        raise FileExistsError("stage output exists; refusing automatic overwrite/retry/resume")
    if args.stage == "cache":
        if Path(env["track_cache"]).parent.exists():
            raise FileExistsError("cache build destination exists; do not overwrite/rebuild accepted evidence")
    elif not Path(env["track_cache"]).is_file():
        raise FileNotFoundError("complete the original cache build before smoke")
    if args.stage == "smoke-v3":
        require_complete("smoke-parent", "SMOKE_COMPLETE")
    elif args.stage in ("parent", "v3"):
        require_complete("smoke-parent", "SMOKE_COMPLETE")
        require_complete("smoke-v3", "SMOKE_COMPLETE")
        if args.stage == "parent":
            if (output_root() / "parent_reference.json").exists():
                raise RuntimeError("Historical Parent already registered; no additional Parent training is needed")
            audit = read(output_root() / "preflight.json")
            if audit["existing_standard_ru_candidates"] and not (output_root() / "parent_comparability_audit.json").is_file():
                raise RuntimeError("Historical Parent candidates exist. Complete their comparability audit before adding a full Parent run; send preflight.json for review.")
            decision_path = output_root() / "parent_comparability_audit.json"
            if decision_path.is_file() and read(decision_path).get("decision") != "NO_COMPARABLE_EXISTING_PARENT":
                raise RuntimeError("Existing Parent audit does not authorize a replacement Parent run")
        p = read(stage_paths("smoke-parent")[0] / "v3_training_checks.json")
        v = read(stage_paths("smoke-v3")[0] / "v3_training_checks.json")
        ratio = v["profile_median_ms"] / p["profile_median_ms"]
        if ratio > 1.08:
            raise RuntimeError(f"smoke step time ratio {ratio:.4f} > 1.08; inspect implementation overhead first")
        if args.stage == "v3":
            require_parent_ready()
            evidence = read(stage_paths("smoke-v3")[0] / "evidence_check/evidence_check.json")
            if evidence["status"] == "EVIDENCE_INACTIVE":
                raise RuntimeError("EVIDENCE_INACTIVE")
    elif args.stage.startswith("eval-"):
        if args.stage == "eval-parent" and (output_root() / "parent_reference.json").exists():
            raise RuntimeError("Registered Parent re-evaluation is already complete; use launch eval-v3 after V3 training")
        require_complete(args.stage[5:], "TRAIN_COMPLETE")
        source_mode = args.stage[5:]
        if sha256_file(stage_paths(source_mode)[0] / "ckpts/ckpt_29999_rank0.pt") != read(stage_paths(source_mode)[2])["checkpoint_sha256"]:
            raise ValueError("checkpoint changed since TRAIN_COMPLETE")
    selected = select_gpu(gpu_inventory(), pinned=env["gpu"])
    if env.get("gpu_uuid") and selected["uuid"] != env["gpu_uuid"]:
        raise RuntimeError("GPU UUID changed since preflight; stop and inspect device mapping")
    write_json(status_path, {"stage": args.stage, "status": "STARTING", "start_time": time.time(),
                             "pid": None, "exit_code": None, "last_step": -1,
                             "gpu": env["gpu"], "gpu_uuid": selected["uuid"]})
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([env["python"], str(Path(__file__).resolve()), "worker", args.stage],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, stdin=subprocess.DEVNULL)
    print(f"已启动 {args.stage}，包装进程 PID={process.pid}")
    print(f"最近日志：tail -n 80 '{log}'")
    print(f"持续查看：tail -n 40 -f '{log}'")
    print(f"状态：{env['python']} {Path(__file__).resolve()} status {args.stage}")
    return 0


def worker(args):
    env = environment()
    run, _, status_path = stage_paths(args.stage)
    state = read(status_path)
    state.update(status="RUNNING", pid=os.getpid())
    write_json(status_path, state)
    exit_code = -1
    try:
        # Recheck at worker start; selection does not reserve a shared server GPU.
        select_gpu(gpu_inventory(), pinned=env["gpu"])
        cmd = command_for(args.stage, env)
        state["command"] = cmd
        write_json(status_path, state)
        print(json.dumps({"command": cmd}), flush=True)
        started = time.perf_counter()
        child = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(env["gpu"])})
        state["child_pid"] = child.pid
        write_json(status_path, state)
        exit_code = child.wait()
        state["process_wall_seconds"] = time.perf_counter() - started
        state["exit_code"] = exit_code
        if exit_code != 0:
            raise RuntimeError(f"child process exited {exit_code}")
        if args.stage == "cache":
            from puri_gs.static_tracks import load_static_track_cache
            cache = load_static_track_cache(env["track_cache"])
            state.update(status="CACHE_COMPLETE", cache_sha256=sha256_file(env["track_cache"]),
                         payload_sha256=cache["payload_sha256"])
        elif args.stage.startswith("eval-"):
            rows = list(csv.DictReader((run / "per_image_metrics.csv").open()))
            mode = args.stage[5:]
            identity = read(stage_paths(mode)[0] / "v3_input_manifest.json")
            if len(rows) != 24 or sorted(r["image_name"] for r in rows) != sorted(identity["test_basenames"]):
                raise ValueError("evaluation must cover exactly the registered 24 test views")
            metrics = read(run / "test_metrics.json")
            import math
            for key in ("psnr", "ssim", "lpips"):
                average = sum(float(row[key]) for row in rows) / 24
                if not math.isfinite(average) or abs(average - metrics[key]) > 1e-5:
                    raise ValueError("evaluation summary differs from per-view metrics")
            validation = read(run / "ru_validation.json")
            if (not validation["standard_checkpoint_load_pass"] or validation["evaluation_loaded_mask_head"]
                    or validation["evaluation_imported_dino"] or validation["evaluation_rasterization_count_ratio"] != 1):
                raise ValueError("invalid independent evaluator path")
            from puri_gs.v3_parent_reference import evaluation_fingerprint
            state["evaluation_fingerprint"] = evaluation_fingerprint(run, env)
            state["status"] = "EVAL_COMPLETE"
        else:
            final_step = 599 if args.stage.startswith("smoke-") else 29999
            checks = read(run / "v3_training_checks.json")
            state["last_step"] = checks["last_step"]
            if state["last_step"] != final_step or checks["disabled_call_counts"] != DISABLED_COUNTS:
                raise ValueError("terminal step or disabled path checks failed")
            checkpoint = run / f"ckpts/ckpt_{final_step}_rank0.pt"
            count = validate_checkpoint(checkpoint, final_step)
            state.update(standard_checkpoint_exists=True, standard_checkpoint_loadable=True,
                         checkpoint_sha256=sha256_file(checkpoint), gaussian_count=count)
            if args.stage.endswith("v3"):
                if not (run / "evidence_check/evidence_check.json").is_file():
                    raise ValueError("evidence check output missing")
            state["status"] = "SMOKE_COMPLETE" if final_step == 599 else "TRAIN_COMPLETE"
    except Exception as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        if state.get("exit_code") is None:
            state["exit_code"] = exit_code
        print(state["error"], flush=True)
    state["end_time"] = time.time()
    write_json(status_path, state)
    print(json.dumps(state, indent=2), flush=True)
    return 0 if state["status"].endswith("COMPLETE") else 1


def status(args):
    run, log, path = stage_paths(args.stage)
    value = read(path)
    progress = run / "v3_progress.json"
    if progress.is_file():
        value["last_step"] = read(progress)["step"]
    pid = value.get("pid")
    if pid and value["status"] == "RUNNING":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            value["status"] = "INCOMPLETE_PROCESS_GONE"
    print(json.dumps(value, indent=2, ensure_ascii=False))
    print(f"失败时发回以上状态和：tail -n 80 '{log}'")
    return 0


def report(args):
    from puri_gs.v3_report import build_report
    build_report(output_root(), roi_path=args.roi)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--gpu", default="auto", help="auto selects an idle GPU in order 6,7,0,1,2,3,4,5; an index pins a device")
    p.add_argument("--track-cache", type=Path)
    p = sub.add_parser("register-parent", help="validate and reference the audited existing standard RU Parent")
    p.add_argument("--run", type=Path, default=ROOT / "logs-puri/ru-generalization-rerun-9e292309/garden_ru_30k")
    p.add_argument("--eval-dir", type=Path, default=output_root() / "eval-existing-parent")
    for action in ("launch", "worker", "status"):
        p = sub.add_parser(action)
        p.add_argument("stage", choices=STAGES)
    p = sub.add_parser("report")
    p.add_argument("--roi", type=Path)
    args = parser.parse_args()
    return globals()[args.action.replace("-", "_")](args)


if __name__ == "__main__":
    raise SystemExit(main())
