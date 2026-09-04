#!/usr/bin/env python3
"""Read-only config/run validation for the two fixed Garden paper controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puri_gs.config import load_experiment_config, trainer_method_args
from puri_gs.paper_controls import (
    check_control_diffs, resolved_schedule, schedule_from_config, validate_event_files, write_json_new,
)
from puri_gs.delayed_absgrad import topology_event_summary
from tools.summarize_puri_gs_garden_causal_2x2 import (
    _checkpoint_inventory, _read_per_image, _split_identity,
)

TEST_NAMES = [f"DSC{number:05d}.JPG" for number in range(7956, 8141, 8)]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def config_audit() -> dict:
    configs = {name: load_experiment_config(ROOT / "configs" / f"puri_gs_{name}_full30k.yaml")
               for name in ("ru", "ru_align", "ru_tar")}
    diff = check_control_diffs(configs["ru"], configs["ru_align"], configs["ru_tar"])
    return {"pass": True, "resolved_config_diff": diff,
            "schedules": {name: resolved_schedule(cfg) for name, cfg in configs.items()},
            "trainer_method_args": {name: trainer_method_args(cfg) for name, cfg in configs.items()},
            "test_names": TEST_NAMES,
            "test_names_sha256": hashlib.sha256("\n".join(TEST_NAMES).encode()).hexdigest()}


def _finite_tree(value, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite {label}")


def audit_run(directory: Path, *, evaluation: bool = False) -> dict:
    config = load_experiment_config(directory / "config.yaml")
    if not config.get("paper_control"):
        raise ValueError("expected a paper-control run")
    event_audit = validate_event_files(directory, config)
    if read(directory / "resolved_config_diff.json") != config_audit()["resolved_config_diff"]:
        raise ValueError("resolved config diff differs")
    validation = read(directory / "paper_control_validation.json")
    expected = {"training_steps": 30000, "full_rasterization_count": 30000,
                "full_rasterizations_per_step": 1.0, "coarse_rasterization_count": 10000,
                "refine_events_observed": event_audit["event_count"], "mask_samples_observed": 11,
                "statistics_affect_training_decisions": False}
    if any(validation.get(key) != value for key, value in expected.items()):
        raise ValueError("training render/event contract differs")
    split = read(directory / "dataset_split.json")
    if len(split.get("train", [])) != 161 or split.get("test") != TEST_NAMES:
        raise ValueError("Garden split differs")
    train = read(directory / "train_metrics.json")
    if train["step"] != 29999 or train["gaussian_count"] != event_audit["final_topology_count"]:
        raise ValueError("final step/count differs from topology evidence")
    for field in ("training_time_seconds", "peak_vram_gib"):
        if not math.isfinite(float(train[field])) or train[field] <= 0:
            raise ValueError(f"invalid training {field}")
    aux = read(directory / "aux/ru_training_validation.json")
    timing = read(directory / "DINO_time.json")
    dino = read(directory / "aux/dino_environment.json")
    saved_schedule = read(directory / "aux/training_schedule.json")
    expected_events = topology_event_summary(delayed_topology=True, total_steps=30000,
        mask_pause_after_reset=300, delayed_schedule=schedule_from_config(config))
    if saved_schedule.get("topology_events") != expected_events:
        raise ValueError("aux topology schedule differs")
    for key in ("bootstrap_switch_step", "mask_begin_step", "mask_threshold", "mask_erode_kernel",
                "mask_pause_after_reset", "densify_start_step", "densify_stop_step", "densify_every",
                "opacity_reset_start_step", "opacity_reset_every"):
        if saved_schedule.get(key) != config[key]:
            raise ValueError(f"aux schedule differs: {key}")
    if aux.get("gradient_isolation_pass") is not True or aux.get("dino_trainable_parameter_count") != 0:
        raise ValueError("DINO/gradient isolation failed")
    if timing.get("mask_update_count") != 29400 or timing.get("mask_pause_count") != 600:
        raise ValueError("actual head update/pause counts differ")
    if timing.get("render_feature_calls") != 30000 or dino.get("trainable_parameter_count") != 0:
        raise ValueError("DINO call/freeze contract differs")
    if dino.get("weight_sha256") != "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb":
        raise ValueError("DINO weight differs")
    if dino.get("repository_commit") != "7764ea0f912e53c92e82eb78a2a1631e92725fc8":
        raise ValueError("DINO source differs")
    checkpoint = _checkpoint_inventory(directory / "ckpts/ckpt_29999_rank0.pt")
    import torch
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True, mmap=True)
    splats = payload["splats"]
    shapes = {"means": (3,), "scales": (3,), "quats": (4,), "opacities": (),
              "sh0": (1, 3), "shN": (15, 3)}
    if set(splats) != set(shapes):
        raise ValueError("checkpoint does not contain exactly standard Gaussian fields")
    for name, shape in shapes.items():
        tensor = splats[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (train["gaussian_count"], *shape) or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid checkpoint Gaussian tensor: {name}")
    del payload, splats
    commit = (directory / "git_commit.txt").read_text().strip()
    environment = read(directory / "environment.json")
    if environment.get("repository_commit") != commit:
        raise ValueError("training environment/commit differs")
    result = {"pass": True, "path": str(directory), "config": config,
              "git_commit": commit, "environment": environment, "split": split,
              "topology": event_audit, "schedule": resolved_schedule(config),
              "training": train, "checkpoint": checkpoint, "dino": dino,
              "rasterization_validation": validation}
    if evaluation:
        target = directory / "independent_eval"
        if load_experiment_config(target / "config.yaml") != config:
            raise ValueError("evaluation config differs")
        if (target / "git_commit.txt").read_text().strip() != commit:
            raise ValueError("training/evaluation commits differ")
        if _split_identity(read(target / "dataset_split.json")) != _split_identity(split):
            raise ValueError("training/evaluation split differs")
        validation = read(target / "ru_validation.json")
        required = {"standard_checkpoint_load_pass": True, "evaluation_imported_dino": False,
                    "evaluation_loaded_mask_head": False, "evaluation_rasterization_count_ratio": 1.0}
        if any(validation.get(key) != value for key, value in required.items()):
            raise ValueError("independent inference isolation failed")
        command = (target / "run_command.txt").read_text()
        if any(flag in command for flag in ("--dino_", "--feature_cache", "--puri_gs_ru_enabled", "--puri_gs_paper_control")):
            raise ValueError("training assets leaked into evaluation")
        per_image = _read_per_image(target / "per_image_metrics.csv")
        if [row["image_name"] for row in per_image] != TEST_NAMES:
            raise ValueError("evaluation image order differs")
        metrics, efficiency = read(target / "test_metrics.json"), read(target / "efficiency_metrics.json")
        if efficiency["gaussian_count"] != train["gaussian_count"]:
            raise ValueError("training/evaluation Gaussian counts differ")
        for field in ("render_fps", "latency_p50_ms", "latency_p95_ms", "inference_vram_gib"):
            if not math.isfinite(float(efficiency[field])) or efficiency[field] <= 0:
                raise ValueError(f"invalid evaluation {field}")
        for name in ("psnr", "ssim", "lpips"):
            value = float(metrics[name])
            mean = sum(row[name] for row in per_image) / 24
            if not math.isfinite(value) or abs(value - mean) > 1e-4:
                raise ValueError("mean metrics differ from paired images")
        result.update(metrics=metrics, efficiency=efficiency, per_image=per_image,
                      evaluation_git_commit=commit, evaluation_validation=validation)
    _finite_tree(result, "run")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.evaluation and args.run is None:
        parser.error("--evaluation requires --run")
    result = audit_run(args.run.resolve(), evaluation=args.evaluation) if args.run else config_audit()
    if args.output:
        write_json_new(args.output, result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print("PAPER-CONTROL-AUDIT-PASS")


if __name__ == "__main__":
    main()
