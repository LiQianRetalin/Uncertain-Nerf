"""Prepare and launch the sole P04-T Android RU evidence identity on server."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf")
WORK = Path("/home/chenglong/P04T-work")
RUN_ID = "P04T-android-ru-seed42-evidence-20k"
OUTPUT = WORK / "outputs" / RUN_ID
ORIGINAL = ROOT / "logs-puri/ru-generalization-rerun-9e292309/android_ru_30k"
SOURCE = ROOT / "external/gsplat-v1.5.3-ru"
FROZEN_SOURCE_SHA = "dea00d8a11071789d4afb01c01d5969683b53fc96db20d5582a6c5278bec328c"
FROZEN_PATCHED_SHA = "921c89f59fadc398ce3ccb8baa8cd2c85edfdadd52aa0a443edb7413750f0531"
INPUT_SHA = {
    "data/nerf_robustnerf/robustnerf/android/sparse/0/cameras.bin": "c422e39f54e9791ce15eb795d239cb8a0c8e6584d009b192c59ac3937d11147d",
    "data/nerf_robustnerf/robustnerf/android/sparse/0/images.bin": "fc6d910e9b6f87a59cd420ef2b658f48cf64d8153284bd075775eb3ebc5c7973",
    "data/nerf_robustnerf/robustnerf/android/sparse/0/points3D.bin": "69e3066cfc11635c9fe4d32822d716bcc3cad39036332c1a1ecb0cd42834e3c5",
    "data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth": "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",
    "data/PURI-GS-derived/semantic_features/android/manifest.json": "459cf0694d82056f4b4b7064a9f125222a4b13f53244189cdc582b12a22dc782",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def original_dry_command():
    cmd = [sys.executable, str(ROOT / "run_puri_gs.py"),
           "--config", str(ROOT / "configs/puri_gs_ru_full30k.yaml"),
           "--gsplat-dir", str(SOURCE),
           "--data-dir", str(ROOT / "data/nerf_robustnerf/robustnerf/android"),
           "--result-dir", str(OUTPUT), "--gpu", "7",
           "--train-keyword", "clutter", "--test-keyword", "extra",
           "--dino-repo-dir", str(ROOT / "external/dinov2"),
           "--dino-weight-path", str(ROOT / "data/PURI-GS-assets/dinov2/dinov2_vits14_reg4_pretrain.pth"),
           "--feature-cache-dir", str(ROOT / "data/PURI-GS-derived/semantic_features/android"),
           "--dry-run"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ensure(result.returncode == 0, "official RU dry-run failed: " + result.stderr[-1000:])
    lines = result.stdout.strip().splitlines()
    ensure(lines and lines[-1].startswith("CUDA_VISIBLE_DEVICES=7 "), "official RU dry command not found")
    return shlex.split(lines[-1])


def prepare():
    ensure(not OUTPUT.exists(), "P04T output identity already exists")
    ensure(not (WORK / "run_plan.csv").exists(), "P04T run plan already frozen")
    ensure(json.loads((WORK / "preflight.json").read_text())["status"] == "P04T_PREFLIGHT_PASS",
           "independent attribution preflight did not pass")
    runtime = json.loads((WORK / "runtime_preflight.json").read_text())
    ensure(runtime["status"] == "P04T_REAL_STATE_D_ISOLATION_PASS"
           and runtime["charged_extra_calls_total"] == 17,
           "real-state isolation preflight did not pass")
    with (WORK / "call_budget_ledger.csv").open(newline="", encoding="utf-8") as stream:
        charged = sum(int(row["charged"]) for row in csv.DictReader(stream))
    ensure(charged == 17, "P04T preflight call ledger changed")
    ensure(sha(SOURCE / "examples/simple_trainer.py") == FROZEN_SOURCE_SHA,
           "original RU trainer changed")
    ensure(sha(WORK / "examples/p04t_simple_trainer.py") == FROZEN_PATCHED_SHA,
           "P04T isolated trainer changed")
    for relative, expected in INPUT_SHA.items():
        ensure(sha(ROOT / relative) == expected, "frozen input changed: " + relative)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    ensure(branch == "ru-part", "P04T requires ru-part; no branch change allowed")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits", "--id=7"], text=True).strip()
    used, utilization = [int(item.strip()) for item in gpu.split(",")]
    ensure(used < 1024 and utilization < 10, "GPU 7 is occupied")
    old = shlex.split((ORIGINAL / "run_command.txt").read_text().strip())
    new = original_dry_command()
    ensure(old[0].startswith("CUDA_VISIBLE_DEVICES=") and new[0] == "CUDA_VISIBLE_DEVICES=7",
           "RU device command changed")
    ensure(old[1:3] == new[1:3] and old[3:] != new[3:], "RU command structure changed")
    old_compare, new_compare = old.copy(), new.copy()
    old_compare[0] = new_compare[0]
    old_compare[old_compare.index("--result_dir") + 1] = str(OUTPUT)
    ensure(old_compare == new_compare, "P04T command differs beyond GPU and output identity")
    cfg_old = json.loads((ORIGINAL / "config.yaml").read_text())
    ensure(cfg_old["total_steps"] == 30000 and cfg_old["seed"] == 42
           and cfg_old["training"]["data_factor"] == 4,
           "historical Android RU identity differs")
    shutil.copytree(SOURCE / "examples", WORK / "examples", dirs_exist_ok=True)
    ensure(sha(WORK / "examples/p04t_simple_trainer.py") == FROZEN_PATCHED_SHA,
           "isolated trainer changed during examples copy")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "config.yaml").write_text(json.dumps(cfg_old, indent=2) + "\n")
    (OUTPUT / "run_command.txt").write_text(shlex.join(new) + "\n")
    (OUTPUT / "git_commit.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True))
    new[2] = "p04t_simple_trainer.py"
    plan = {"run_id": RUN_ID, "method": "original GS-RU", "scene": "android", "seed": 42,
            "planned_schedule_steps": 30000, "controlled_stop_completed_step": 19999,
            "normal_updates": 20000, "gpu": 7, "extra_call_limit": 300,
            "all_attempt_update_limit": 40000, "gpu_seconds_limit": 7200,
            "window_1": "10901..11000", "window_2": "18901..19000",
            "pollution_offsets": "0;33;66;99", "probe_count": 512,
            "ratio_epsilon": 1e-12, "b_support_threshold": 1e-6,
            "kappa": 4, "same_condition_cap": 2,
            "min_distinct_cameras": 3, "u_support_threshold": .05,
            "native_grad_threshold": .0006,
            "attribution_tolerance": "absolute<=2e-4+relative*2e-4",
            "gradient_alignment_tolerance": "abs<=2e-5 OR rel<=2e-4",
            "isolated_trainer_sha256": FROZEN_PATCHED_SHA,
            "original_trainer_sha256": FROZEN_SOURCE_SHA,
            "original_config_sha256": sha(ORIGINAL / "config.yaml"),
            "original_split_sha256": sha(ORIGINAL / "dataset_split.json"),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "gpu_prelaunch_used_mib": used, "gpu_prelaunch_utilization_percent": utilization}
    with (WORK / "run_plan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=plan.keys()); writer.writeheader(); writer.writerow(plan)
    (WORK / "frozen_command.json").write_text(json.dumps({"cwd": str(WORK / "examples"),
        "argv": new, "environment": {"CUDA_VISIBLE_DEVICES": "7",
        "PYTHONPATH": str(WORK / "code") + ":" + str(ROOT),
        "P04T_RUN_ID": RUN_ID, "P04T_WORK": str(WORK), "P04T_ATTEMPT": "1",
        "P04T_PRIOR_GPU_SECONDS": "90"}}, indent=2) + "\n")
    with (WORK / "run_ledger.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(("attempt", "phase", "status", "updates", "extra_calls", "reason", "seconds"))
        csv.writer(stream).writerow((0, "preflight", "PASS_WITH_TWO_FIXTURE_FAILURES", 0, 17,
            "synthetic reference PASS; real-state D isolation PASS; fixture missing _region/camera_model corrected without changing tolerances", ""))
    print("P04T_PREPARED_NO_TRAINING", flush=True)


def run():
    frozen = json.loads((WORK / "frozen_command.json").read_text())
    ensure(not (WORK / "training_updates_ledger.csv").exists(), "P04T training already attempted")
    ensure(sha(WORK / "examples/p04t_simple_trainer.py") == FROZEN_PATCHED_SHA,
           "isolated trainer SHA changed after freeze")
    env = os.environ.copy()
    env.update(frozen["environment"])
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "DIFFUSERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    started = time.monotonic()
    result = subprocess.run(frozen["argv"][1:], cwd=frozen["cwd"], env=env)
    seconds = time.monotonic() - started
    status = json.loads((WORK / "status.json").read_text()) if (WORK / "status.json").exists() else {}
    complete = (result.returncode == 0 and status.get("stage") == "COMPLETE"
                and status.get("completed_step") == 19999
                and status.get("cumulative_updates") == 20000)
    with (WORK / "run_ledger.csv").open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow((1, "formal_parent_training", "COMPLETE" if complete else "FAILED_PARTIAL",
            status.get("cumulative_updates", ""), status.get("charged_extra_calls", ""),
            "normal stop at step19999" if complete else f"exit={result.returncode}; status={status.get('stage')}",
            round(seconds, 3)))
    print("P04T_TRAINING_COMPLETE_STOP" if complete else "P04T_TRAINING_FAILED_PARTIAL", flush=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
