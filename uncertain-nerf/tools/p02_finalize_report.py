#!/usr/bin/env python3
"""Assemble the bounded P02 server outputs into the frozen report tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


RUNS = (
    ("P02-android-robustsplat", "android", "robustsplat"),
    ("P02-android-sls-mlp", "android", "sls-mlp"),
    ("P02-patio_high-robustsplat", "patio_high", "robustsplat"),
    ("P02-patio_high-sls-mlp", "patio_high", "sls-mlp"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--status", choices=("COMPLETE", "PARTIAL"), required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    root = args.work_root.expanduser().resolve()
    report = root / "report"
    state = root / "state"
    outputs = root / "outputs"
    summaries: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []

    for run_id, scene, method in RUNS:
        run_report = report / "runs" / run_id
        summary_path = run_report / "summary.json"
        if summary_path.is_file():
            summaries.append(load_json(summary_path))
        rows = read_csv(run_report / "per_image_metrics.csv")
        per_image.extend(rows)
        if rows:
            ordered = sorted(rows, key=lambda row: float(row["psnr"]))
            picks = (
                ("lowest_psnr", ordered[0]),
                ("median_psnr", ordered[len(ordered) // 2]),
                ("highest_psnr", ordered[-1]),
            )
            for reason, row in picks:
                representatives.append(
                    {
                        "run_id": run_id,
                        "scene": scene,
                        "method": method,
                        "selection_reason": reason,
                        "image_name": row["image_name"],
                        "psnr": row["psnr"],
                        "ssim": row["ssim"],
                        "lpips": row["lpips"],
                    }
                )

        if method == "robustsplat":
            timing_path = outputs / run_id / "test" / "ours_30000" / "p02_timing.json"
        else:
            timing_path = outputs / run_id / "stats" / "p02_timing_step29999.json"
        if timing_path.is_file():
            native = load_json(timing_path)
            for record in native["records"]:
                timing.append(
                    {
                        "run_id": run_id,
                        "scene": scene,
                        "method": method,
                        "seed": 42,
                        "warmup": native["warmup"],
                        "repeat": record["repeat"],
                        "frames": record["frames"],
                        "seconds": record["seconds"],
                        "fps": record["fps"],
                        "boundary": native["boundary"],
                    }
                )

    summary_fields = [
        "run_id", "scene", "method", "seed", "test_image_count", "psnr", "ssim",
        "lpips", "checkpoint_sha256", "protocol_sha256", "evaluator_id", "source",
    ]
    write_csv(report / "summary.csv", summary_fields, summaries)
    per_fields = list(per_image[0]) if per_image else [
        "run_id", "scene", "method", "seed", "image_index", "image_name", "psnr",
        "ssim", "lpips", "prediction_sha256", "checkpoint_sha256", "evaluator_id", "source",
    ]
    write_csv(report / "per_image_metrics.csv", per_fields, per_image)
    timing_fields = [
        "run_id", "scene", "method", "seed", "warmup", "repeat", "frames", "seconds",
        "fps", "boundary",
    ]
    write_csv(report / "timing_raw.csv", timing_fields, timing)
    write_csv(
        report / "representative_selection.csv",
        ["run_id", "scene", "method", "selection_reason", "image_name", "psnr", "ssim", "lpips"],
        representatives,
    )

    stage_rows = read_csv(state / "stage_timing.csv")
    cost_rows = []
    for row in stage_rows:
        started = int(row["started_epoch"])
        ended = int(row["ended_epoch"])
        cost_rows.append(
            {
                **row,
                "elapsed_seconds": ended - started,
                "cost_scope": "wall_clock; one selected L20 at a time for GPU stages",
            }
        )
    write_csv(
        report / "cost_breakdown.csv",
        ["stage", "started_epoch", "ended_epoch", "exit_code", "elapsed_seconds", "cost_scope"],
        cost_rows,
    )

    smoke_rows = read_csv(state / "smoke_ledger.csv")
    formal_rows = read_csv(state / "formal_ledger.csv")
    write_csv(
        report / "run_ledger.csv",
        ["run_id", "status", "seed", "updates", "gpu", "started_epoch", "ended_epoch"],
        formal_rows,
    )
    budget = {
        "schema": "puri-gs-p02-budget-v1",
        "smoke": {
            "per_attempt_update_limit": 100,
            "global_update_limit": 800,
            "attempted_updates": sum(int(row["updates"]) for row in smoke_rows),
            "attempt_count": len(smoke_rows),
        },
        "formal": {
            "official_updates_per_run": 30000,
            "max_attempts_per_identity": 2,
            "automatic_retry": False,
            "ledger_rows": formal_rows,
        },
    }
    (report / "budget.json").write_text(
        json.dumps(budget, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    timing_by_run: dict[str, list[float]] = {}
    for row in timing:
        timing_by_run.setdefault(str(row["run_id"]), []).append(float(row["fps"]))
    status = {
        "schema": "puri-gs-p02-status-v1",
        "status": args.status,
        "run_count_expected": 4,
        "run_count_evaluated": len(summaries),
        "timing_repeat_count": len(timing),
        "median_fps": {
            run_id: statistics.median(values) for run_id, values in timing_by_run.items()
        },
        "next_work_package_started": False,
    }
    if args.status == "COMPLETE":
        expected_ids = {run_id for run_id, _, _ in RUNS}
        summary_ids = {str(row["run_id"]) for row in summaries}
        timing_counts = {
            run_id: sum(row["run_id"] == run_id for row in timing)
            for run_id in expected_ids
        }
        last_formal_status = {
            row["run_id"]: row["status"] for row in formal_rows
        }
        expected_test_counts = {
            "P02-android-robustsplat": 19,
            "P02-android-sls-mlp": 19,
            "P02-patio_high-robustsplat": 45,
            "P02-patio_high-sls-mlp": 45,
        }
        invalid_counts = [
            row["run_id"]
            for row in summaries
            if int(row["test_image_count"]) != expected_test_counts[row["run_id"]]
        ]
        if (
            summary_ids != expected_ids
            or len(per_image) != 128
            or any(count != 3 for count in timing_counts.values())
            or any(last_formal_status.get(run_id) != "COMPLETE" for run_id in expected_ids)
            or invalid_counts
        ):
            raise RuntimeError(
                "refusing COMPLETE report: required four-run evidence is incomplete"
            )
    (report / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    decision = (
        "P02_COMPLETE_STOP_AND_RETURN_TO_PLAN_ASSISTANT"
        if args.status == "COMPLETE"
        else "P02_PARTIAL_STOP_AND_RETURN_FAILURE_LEDGER"
    )
    (report / "NEXT_DECISION.md").write_text(
        f"# NEXT_DECISION\n\n`{decision}`\n\n"
        "不自动启动 Corner、OAC、Room factor2、Garden 或 PART 修补。\n",
        encoding="utf-8",
    )
    lines = [
        "# REPORT_P02",
        "",
        f"状态：`{args.status}`",
        "",
        f"完成独立评测：{len(summaries)}/4；严格计时记录：{len(timing)}/12。",
        "",
        "所有 Reproduced 数字均来自共同输入、seed42、固定源代码与独立评测器；作者协议数字不混入本表。",
        "",
        "本工作包到此停止，后续决策交回方案助手。",
    ]
    (report / "REPORT_P02.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"P02_REPORT={args.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
