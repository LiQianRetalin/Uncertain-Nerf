"""RU-PART controlled mechanism-diagnostic entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from puri_gs.mechanism_diagnostic import (
    build_final_report, build_roi_evidence, compare_replay_runs,
    confirm_static_rois, record_vjp_decision, run_restore_smoke,
    sha256_file, verify_current_sources, verify_reproduction,
    write_new_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_training_paths(command):
    command.add_argument("--gsplat-dir", type=Path, required=True)
    command.add_argument("--data-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--gpu", type=int, required=True)
    command.add_argument("--dino-repo-dir", type=Path, required=True)
    command.add_argument("--dino-weight-path", type=Path, required=True)
    command.add_argument("--feature-cache-dir", type=Path, required=True)


def launch_training(args, stage, replay=None, roi_dir=None, confirmed=False):
    if args.output_dir.exists():
        raise FileExistsError(f"output directory exists: {args.output_dir}")
    command = [
        sys.executable, str(PROJECT_ROOT / "run_puri_gs.py"),
        "--config", str(PROJECT_ROOT / "configs/puri_gs_ru_part_parent_garden30k.yaml"),
        "--gsplat-dir", str(args.gsplat_dir), "--data-dir", str(args.data_dir),
        "--result-dir", str(args.output_dir), "--gpu", str(args.gpu),
        "--dino-repo-dir", str(args.dino_repo_dir),
        "--dino-weight-path", str(args.dino_weight_path),
        "--feature-cache-dir", str(args.feature_cache_dir),
        "--diagnostic-stage", stage,
    ]
    if stage in {"U", "R1", "R2"}:
        command.extend(["--diagnostic-stop-after-step", "10199"])
    elif stage == "SMOKE":
        command.extend(["--diagnostic-stop-after-step", "99"])
    elif stage in {"P1", "P2", "O"}:
        command.extend(["--diagnostic-stop-after-step", "10399"])
    if replay is not None:
        command.extend(["--replay-checkpoint", str(replay)])
    if roi_dir is not None:
        command.extend(["--diagnostic-roi-dir", str(roi_dir)])
    if confirmed:
        command.append("--diagnostic-roi-confirmed")
    replay_hash_before = sha256_file(Path(replay)) if replay is not None else None
    returncode = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    if replay is not None:
        replay_hash_after = sha256_file(Path(replay))
        if replay_hash_before != replay_hash_after:
            raise RuntimeError("source replay checkpoint changed during diagnostic")
        if args.output_dir.is_dir():
            write_new_json(args.output_dir / "source_checkpoint_immutability.json", {
                "path": str(Path(replay).resolve()), "sha256_before": replay_hash_before,
                "sha256_after": replay_hash_after, "unchanged": True,
            })
    return returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="verify existing records; no RGB or GPU work")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--per-image", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--verify-current-sources", action="store_true",
                       help="also hash source checkpoint/config on their owning server")
    smoke = commands.add_parser("smoke", help="tiny full-state restore smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    trainer_smoke = commands.add_parser(
        "trainer-smoke", help="run 100 real trainer updates with max_steps retained at 30000"
    )
    add_training_paths(trainer_smoke)
    prefix = commands.add_parser(
        "prefix", help="run the one permitted uninterrupted U through step10199"
    )
    add_training_paths(prefix)
    replay = commands.add_parser("replay", help="restore step9999 for R1 or R2")
    add_training_paths(replay)
    replay.add_argument("--stage", choices=("R1", "R2"), required=True)
    replay.add_argument("--checkpoint", type=Path, required=True)
    compare = commands.add_parser("compare", help="compare U/R1/R2 terminal states")
    compare.add_argument("--u", type=Path, required=True)
    compare.add_argument("--r1", type=Path, required=True)
    compare.add_argument("--r2", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    probe = commands.add_parser("probe", help="derive fixed train-only ROI/evidence")
    probe.add_argument("--u-dir", type=Path, required=True)
    probe.add_argument("--track-cache", type=Path, required=True)
    probe.add_argument("--data-dir", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    vjp = commands.add_parser("vjp", help="run finite VJPs from the step9999 copy")
    add_training_paths(vjp)
    vjp.add_argument("--checkpoint", type=Path, required=True)
    vjp.add_argument("--roi-dir", type=Path, required=True)
    vjp.add_argument("--confirmed", action="store_true")
    confirm = commands.add_parser(
        "confirm", help="freeze the whole candidate ROI after explicit user confirmation"
    )
    confirm.add_argument("--roi-dir", type=Path, required=True)
    confirm.add_argument("--note", required=True)
    decide = commands.add_parser("decide-vjp", help="record the reviewed formal VJP gate")
    decide.add_argument("--roi-dir", type=Path, required=True)
    decide.add_argument("--vjp-report", type=Path, required=True)
    decide.add_argument("--status", choices=(
        "LOCAL_CONTROLLABILITY_PRESENT", "CONTROLLABILITY_INCONCLUSIVE",
        "LOCAL_CONTROLLABILITY_NOT_DETECTED",
    ), required=True)
    oracle = commands.add_parser("oracle", help="run fixed 400-step P1/P2/O branch")
    add_training_paths(oracle)
    oracle.add_argument("--checkpoint", type=Path, required=True)
    oracle.add_argument("--roi-dir", type=Path, required=True)
    oracle.add_argument("--stage", choices=("P1", "P2", "O"), required=True)
    report = commands.add_parser("report", help="assemble report and hard-stop")
    report.add_argument("--audit", type=Path, required=True)
    report.add_argument("--replay", type=Path, required=True)
    report.add_argument("--roi", type=Path, required=True)
    report.add_argument("--vjp-decision", type=Path)
    report.add_argument("--p1-dir", type=Path)
    report.add_argument("--p2-dir", type=Path)
    report.add_argument("--o-dir", type=Path)
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        result = run_restore_smoke(args.output_dir, device=args.device)
        print(result["status"])
        print(f"RESULT={args.output_dir / 'smoke_result.json'}")
        return 0 if result["status"] == "SMOKE_ACCEPTED" else 2
    if args.command == "prefix":
        return launch_training(args, "U")
    if args.command == "trainer-smoke":
        return launch_training(args, "SMOKE")
    if args.command == "replay":
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
        return launch_training(args, args.stage, args.checkpoint)
    if args.command == "compare":
        if args.output.exists():
            parser.error("comparison output exists; refusing overwrite")
        result = compare_replay_runs({"U": args.u, "R1": args.r1, "R2": args.r2})
        write_new_json(args.output, result)
        print(result["status"])
        return 0 if result["status"] == "REPLAY_ACCEPTED" else 2
    if args.command == "probe":
        result = build_roi_evidence(
            args.u_dir, args.track_cache, args.data_dir, args.output_dir
        )
        print(result["status"])
        print("MONITORS=" + ",".join(result["monitor_names"]))
        print(f"RESULT={args.output_dir / 'roi_evidence.json'}")
        return 0 if result["status"] == "ROI_ESTABLISHED" else 2
    if args.command == "vjp":
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
        return launch_training(
            args, "VJP", args.checkpoint, args.roi_dir, args.confirmed
        )
    if args.command == "confirm":
        result = confirm_static_rois(args.roi_dir, note=args.note)
        print(result["status"])
        print(f"RESULT={args.roi_dir / 'static_confirmation.json'}")
        return 0
    if args.command == "decide-vjp":
        result = record_vjp_decision(
            args.roi_dir, args.vjp_report, status=args.status
        )
        print(result["status"])
        return 0
    if args.command == "oracle":
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
        return launch_training(
            args, args.stage, args.checkpoint, args.roi_dir, confirmed=True
        )
    if args.command == "report":
        supplied = (args.p1_dir, args.p2_dir, args.o_dir)
        if any(value is not None for value in supplied) and not all(
            value is not None for value in supplied
        ):
            parser.error("P1/P2/O directories must be supplied together")
        branches = None if not all(value is not None for value in supplied) else {
            "P1": args.p1_dir, "P2": args.p2_dir, "O": args.o_dir,
        }
        result = build_final_report(
            audit_path=args.audit, replay_path=args.replay, roi_path=args.roi,
            vjp_decision_path=args.vjp_decision, branch_dirs=branches,
            output_dir=args.output_dir,
        )
        label = result["branch_analysis"]
        if isinstance(label, dict):
            label = label["label"]
        print(f"REPORT_COMPLETE label={label}")
        print("HARD_STOP：不得自动启动后续实验")
        return 0
    if args.output_dir.exists():
        parser.error("output directory exists; refusing overwrite")
    try:
        result = verify_reproduction(args.manifest, args.per_image)
        if args.verify_current_sources:
            result["current_sources"] = verify_current_sources(args.manifest)
            result["checkpoint_current_hash_verified"] = True
        result["scope"] = "historical_protocol_evidence_only"
        result["replay_equivalence"] = "not_run"
        result["gpu_smoke"] = "not_run"
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_new_json(args.output_dir / "audit.json", result)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"AUDIT_FAIL: {error}\n")
    print(f"{result['status']} max_error={result['max_abs_delta_db']:.9g} dB")
    print(f"CURRENT_SOURCE_HASH_VERIFIED={result['checkpoint_current_hash_verified']}")
    print(f"AUDIT_FILE={args.output_dir / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
