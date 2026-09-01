#!/usr/bin/env python3
"""Summarize the fixed three-repeat Room B1/RU latency audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


EXPECTED_TEST_IMAGES = 39
EXPECTED_WARMUP_RENDERS = 10
EXPECTED_REPEATS = 3
FPS_RATIO_GATE = 0.95


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required audit artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_latency_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"raw latency CSV is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != EXPECTED_TEST_IMAGES:
        raise ValueError(
            f"expected {EXPECTED_TEST_IMAGES} latency rows, got {len(source)}: {path}"
        )
    rows = [
        {
            "image_index": int(row["image_index"]),
            "image_name": row["image_name"],
            "latency_ms": float(row["latency_ms"]),
        }
        for row in source
    ]
    if [row["image_index"] for row in rows] != list(range(EXPECTED_TEST_IMAGES)):
        raise ValueError(f"latency image indices are not ordered 0..38: {path}")
    names = [row["image_name"] for row in rows]
    if len(set(names)) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"latency image names are not unique: {path}")
    if not all(
        math.isfinite(row["latency_ms"]) and row["latency_ms"] > 0
        for row in rows
    ):
        raise ValueError(f"latencies must be finite and positive: {path}")
    return rows


def _load_run(path: Path, method: str, run_index: int) -> dict[str, Any]:
    config = _read_json(path / "config.yaml")
    expected_profile = "b1" if method == "b1" else "ru"
    if config.get("profile") != expected_profile:
        raise ValueError(f"unexpected profile for {method} run {run_index}: {path}")

    split = _read_json(path / "dataset_split.json")
    if (
        split.get("protocol") != "every-nth-test"
        or len(split.get("train", [])) != 272
        or len(split.get("test", [])) != EXPECTED_TEST_IMAGES
    ):
        raise ValueError(f"unexpected Room split: {path}")

    rows = _read_latency_rows(path / "per_image_latency.csv")
    if {row["image_name"] for row in rows} != set(split["test"]):
        raise ValueError(f"raw latency names differ from the test split: {path}")

    efficiency = _read_json(path / "efficiency_metrics.json")
    validation = _read_json(path / "ru_validation.json")
    test_metrics = _read_json(path / "test_metrics.json")
    if efficiency.get("warmup_render_count") != EXPECTED_WARMUP_RENDERS:
        raise ValueError(f"warmup count must be 10: {path}")
    if efficiency.get("raw_latency_sample_count") != EXPECTED_TEST_IMAGES:
        raise ValueError(f"raw latency count must be 39: {path}")
    if validation.get("evaluation_warmup_render_count") != EXPECTED_WARMUP_RENDERS:
        raise ValueError(f"validation warmup count must be 10: {path}")
    if validation.get("raw_latency_sample_count") != EXPECTED_TEST_IMAGES:
        raise ValueError(f"validation latency count must be 39: {path}")
    if not (
        validation.get("standard_checkpoint_load_pass") is True
        and validation.get("evaluation_imported_dino") is False
        and validation.get("evaluation_loaded_mask_head") is False
        and validation.get("evaluation_rasterization_count_ratio") == 1.0
    ):
        raise ValueError(f"evaluation did not use the standard inference path: {path}")

    latencies = np.asarray([row["latency_ms"] for row in rows], dtype=np.float64)
    calculated = {
        "render_fps": float(1000.0 / latencies.mean()),
        "latency_mean_ms": float(latencies.mean()),
        "latency_p50_ms": float(np.quantile(latencies, 0.50)),
        "latency_p95_ms": float(np.quantile(latencies, 0.95)),
    }
    for key in ("render_fps", "latency_p50_ms", "latency_p95_ms"):
        if not math.isclose(
            calculated[key], float(efficiency[key]), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"raw latency does not reproduce {key}: {path}")

    command = (path / "run_command.txt").read_text(encoding="utf-8")
    if "--eval_warmup_renders 10" not in command:
        raise ValueError(f"run command did not request ten warmups: {path}")
    if "--puri_gs_ru_enabled" in command or "--dino_repo_dir" in command:
        raise ValueError(f"audit evaluation unexpectedly enabled training-only RU: {path}")

    for key in ("psnr", "ssim", "lpips"):
        if not math.isfinite(float(test_metrics[key])):
            raise ValueError(f"non-finite {key}: {path}")

    return {
        "method": method,
        "run_index": run_index,
        "path": str(path),
        "test_image_count": EXPECTED_TEST_IMAGES,
        "warmup_render_count": EXPECTED_WARMUP_RENDERS,
        "gaussian_count": int(efficiency["gaussian_count"]),
        "inference_vram_gib": float(efficiency["inference_vram_gib"]),
        "psnr": float(test_metrics["psnr"]),
        "ssim": float(test_metrics["ssim"]),
        "lpips": float(test_metrics["lpips"]),
        **calculated,
        "image_names": [row["image_name"] for row in rows],
    }


def decide(
    b1_runs: list[dict[str, Any]], ru_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(b1_runs) != EXPECTED_REPEATS or len(ru_runs) != EXPECTED_REPEATS:
        raise ValueError("the audit requires exactly three B1 and three RU runs")
    b1_runs = sorted(b1_runs, key=lambda item: item["run_index"])
    ru_runs = sorted(ru_runs, key=lambda item: item["run_index"])
    reference_names = b1_runs[0]["image_names"]
    if any(run["image_names"] != reference_names for run in b1_runs + ru_runs):
        raise ValueError("B1/RU audit runs use different test-image orders")
    if len({run["gaussian_count"] for run in b1_runs}) != 1:
        raise ValueError("B1 runs loaded different checkpoints")
    if len({run["gaussian_count"] for run in ru_runs}) != 1:
        raise ValueError("RU runs loaded different checkpoints")

    paired_fps_ratios = [
        ru["render_fps"] / b1["render_fps"]
        for b1, ru in zip(b1_runs, ru_runs, strict=True)
    ]
    median_paired_fps_ratio = float(median(paired_fps_ratios))
    throughput_pass = median_paired_fps_ratio >= FPS_RATIO_GATE

    def med(runs: list[dict[str, Any]], key: str) -> float:
        return float(median(run[key] for run in runs))

    summary = {
        "fps_ratio_gate": FPS_RATIO_GATE,
        "paired_fps_ratios": paired_fps_ratios,
        "median_paired_fps_ratio": median_paired_fps_ratio,
        "b1_median": {
            key: med(b1_runs, key)
            for key in (
                "render_fps",
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p95_ms",
            )
        },
        "ru_median": {
            key: med(ru_runs, key)
            for key in (
                "render_fps",
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p95_ms",
            )
        },
        "throughput_gate_pass": throughput_pass,
    }
    for key in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms"):
        summary[f"median_{key}_ratio"] = (
            summary["ru_median"][key] / summary["b1_median"][key]
        )
    return {
        "protocol": "puri-gs-ru-room-efficiency-audit-1",
        "original_phase_r_decision_preserved": True,
        "warmup_render_count_per_run": EXPECTED_WARMUP_RENDERS,
        "repeat_count_per_method": EXPECTED_REPEATS,
        "decision": (
            "EFFICIENCY_AUDIT_PASS" if throughput_pass else "EFFICIENCY_AUDIT_FAIL"
        ),
        "summary": summary,
        "runs": {
            "b1": [
                {key: value for key, value in run.items() if key != "image_names"}
                for run in b1_runs
            ],
            "ru": [
                {key: value for key, value in run.items() if key != "image_names"}
                for run in ru_runs
            ],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = []
    for method in ("b1", "ru"):
        for run in report["runs"][method]:
            rows.append(
                f"| {method.upper()} | {run['run_index']} | "
                f"{run['render_fps']:.4f} | {run['latency_mean_ms']:.4f} | "
                f"{run['latency_p50_ms']:.4f} | {run['latency_p95_ms']:.4f} |"
            )
    return "\n".join(
        [
            "# Phase R：PURI-GS-RU Room 推理效率复核",
            "",
            f"复核结论：`{report['decision']}`",
            "",
            "本复核不覆盖或改写原始 Phase R 结论；它只检查 Room 平均 FPS 门。",
            "",
            "| 方法 | 重复 | FPS | mean ms | p50 ms | p95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            f"三组配对 FPS 比例：{', '.join(f'{value:.4f}' for value in summary['paired_fps_ratios'])}",
            "",
            f"配对 FPS 比例中位数：{summary['median_paired_fps_ratio']:.4f}；固定门槛：{FPS_RATIO_GATE:.2f}。",
            "",
            "所有重复均使用 10 次不计时预热、39 张逐图原始延迟、标准 checkpoint 推理路径。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for method in ("b1", "ru"):
        for run_index in range(1, EXPECTED_REPEATS + 1):
            parser.add_argument(
                f"--{method}-run{run_index}", type=Path, required=True
            )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    json_path = output_dir / "phase_r_puri_gs_ru_efficiency_audit.json"
    md_path = output_dir / "PHASE_R_PURI_GS_RU_EFFICIENCY_AUDIT.md"
    for target in (json_path, md_path):
        if target.exists():
            raise RuntimeError(f"refusing to overwrite an existing audit report: {target}")

    runs = {
        method: [
            _load_run(
                getattr(args, f"{method}_run{run_index}").expanduser().resolve(),
                method,
                run_index,
            )
            for run_index in range(1, EXPECTED_REPEATS + 1)
        ]
        for method in ("b1", "ru")
    }
    report = decide(runs["b1"], runs["ru"])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(report["decision"])
    print(
        "MEDIAN_PAIRED_FPS_RATIO="
        f"{report['summary']['median_paired_fps_ratio']:.12f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
