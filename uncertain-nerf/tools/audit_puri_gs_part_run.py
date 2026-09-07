#!/usr/bin/env python3
"""Audit RU-PART run integrity and emit the preregistered STOP report label."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--quality-metrics", type=Path)
    parser.add_argument("--coverage-diagnostic", type=Path)
    parser.add_argument("--b1-efficiency", type=Path)
    parser.add_argument("--ru-train-metrics", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); run = args.run_dir.resolve()
    if (run / "NON_SCIENTIFIC_SMOKE.json").exists():
        raise ValueError("NON_SCIENTIFIC_SMOKE output is not accepted by formal evaluation")
    events_path = run / "ru_part_topology_events.csv"
    checkpoint = run / "ckpts" / "ckpt_29999_rank0.pt"
    if not events_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("formal RU-PART event log or step-29999 checkpoint is missing")
    with events_path.open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    accepted_path = run / "ru_part_accepted_candidates.csv"
    if not accepted_path.is_file():
        raise FileNotFoundError("accepted-candidate audit log is missing")
    with accepted_path.open(newline="", encoding="utf-8") as stream:
        accepted = list(csv.DictReader(stream))
    expected_steps = list(range(10000, 20000, 100))
    actual_steps = [int(row["step"]) for row in events]
    event_integrity = actual_steps == expected_steps
    clone_slot_match = all(int(row["coverage_birth_accepted"]) == int(row["regular_clone_replaced"]) for row in events)
    budget_ok = all(int(row["gaussians_after"]) <= 2_312_002 for row in events)
    candidate_integrity = (
        len(accepted) == sum(int(row["coverage_birth_accepted"]) for row in events)
        and len({int(row["track_id"]) for row in accepted}) == len(accepted)
        and all(
            float(row["B_birth"]) > 0
            and float(row["B_birth"]) > float(row["H0_birth"])
            and float(row["B_birth"]) > float(row["B_clone"])
            and float(row["H0_birth"]) <= float(row["H0_clone"])
            and float(row["prospective_alpha_mass"]) > 0
            for row in accepted
        )
        and all(
            int(row["birth_pareto_dominates_clone"]) >= int(row["coverage_birth_accepted"])
            and int(row["accepted_track_id_unique"]) == 1
            for row in events
        )
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_schema = isinstance(payload, dict) and set(payload) == {"step", "splats"} and payload.get("step") == 29999
    final_gaussians = len(payload["splats"]["means"]) if checkpoint_schema else None
    summary = json.loads((run / "ru_part_global_summary.json").read_text())
    no_leak = summary.get("test_images_opened_during_training") == 0
    integrity_gate = all((event_integrity, clone_slot_match, budget_ok, candidate_integrity, checkpoint_schema, no_leak))

    quality_gate = coverage_gate = efficiency_gate = False
    if args.quality_metrics:
        quality = json.loads(args.quality_metrics.read_text())
        quality_gate = quality["psnr"] >= 27.566728 and quality["ssim"] >= 0.868957 and quality["lpips"] <= 0.073151 and quality["DSC07988_psnr"] >= 19.491301
    if args.coverage_diagnostic:
        coverage = json.loads(args.coverage_diagnostic.read_text())
        coverage_gate = coverage["all24_S_cov"] <= 0.20 and coverage["DSC07988_S_cov"] <= 0.25 and coverage["control3_S_cov"] <= 0.01 and sum(int(row["coverage_birth_accepted"]) for row in events) > 0 and clone_slot_match and candidate_integrity
    if bool(args.b1_efficiency) != bool(args.ru_train_metrics):
        raise ValueError("efficiency audit requires both B1 inference and RU training baselines")
    if args.b1_efficiency and args.ru_train_metrics:
        b1_inference = json.loads(args.b1_efficiency.read_text())
        ru_training = json.loads(args.ru_train_metrics.read_text())
        training = json.loads((run / "train_metrics.json").read_text())
        evaluation = json.loads((run / "independent_eval" / "efficiency_metrics.json").read_text())
        validation = json.loads((run / "independent_eval" / "ru_validation.json").read_text())
        efficiency_gate = (
            final_gaussians <= 2_312_002
            and training["training_time_seconds"] <= 1.10 * ru_training["training_time_seconds"]
            and training["peak_vram_gib"] <= 1.10 * ru_training["peak_vram_gib"]
            and evaluation["render_fps"] >= b1_inference["render_fps"]
            and validation.get("evaluation_rasterization_count_ratio") == 1.0
            and validation.get("evaluation_imported_dino") is False
            and validation.get("evaluation_loaded_track_cache") is False
            and validation.get("evaluation_loaded_mask_head") is False
        )
    result = {
        "METHOD": "RU-PART", "SEED": 42,
        "GARDEN_QUALITY_GATE": quality_gate,
        "COVERAGE_ASSOCIATION_GATE": coverage_gate,
        "BUDGET_EFFICIENCY_GATE": efficiency_gate,
        "INTEGRITY_GATE": integrity_gate,
        "OVERALL_GATE": quality_gate and coverage_gate and efficiency_gate and integrity_gate,
        "NEXT_ACTION": "STOP_AND_RETURN_REPORT",
    }
    audit = {
        "branch_commit": (run / "git_commit.txt").read_text().strip(),
        "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_schema_unchanged": checkpoint_schema,
        "topology_event_count": len(events), "event_steps_exact": event_integrity,
        "clone_slot_match": clone_slot_match, "candidate_integrity": candidate_integrity,
        "all_event_counts_under_cap": budget_ok, "final_gaussians": final_gaussians,
        "test_images_opened_during_training": summary.get("test_images_opened_during_training"),
        "decision": result,
    }
    output = run / "ru_part_formal_audit.json"
    if output.exists(): raise FileExistsError(f"refusing to overwrite audit: {output}")
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if integrity_gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
