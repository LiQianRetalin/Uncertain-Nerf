#!/usr/bin/env python3
"""Summarize a fixed three-repeat B1/RU latency audit."""

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


def _read_latency_rows(
    path: Path, expected_test_images: int = EXPECTED_TEST_IMAGES
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"raw latency CSV is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != expected_test_images:
        raise ValueError(
            f"expected {expected_test_images} latency rows, got {len(source)}: {path}"
        )
    rows = [
        {
            "image_index": int(row["image_index"]),
            "image_name": row["image_name"],
            "latency_ms": float(row["latency_ms"]),
        }
        for row in source
    ]
    if [row["image_index"] for row in rows] != list(range(expected_test_images)):
        raise ValueError(f"latency image indices are not contiguous and ordered: {path}")
    names = [row["image_name"] for row in rows]
    if len(set(names)) != expected_test_images:
        raise ValueError(f"latency image names are not unique: {path}")
    if not all(
        math.isfinite(row["latency_ms"]) and row["latency_ms"] > 0
        for row in rows
    ):
        raise ValueError(f"latencies must be finite and positive: {path}")
    return rows


def _load_run(
    path: Path,
    method: str,
    run_index: int,
    *,
    expected_test_images: int = EXPECTED_TEST_IMAGES,
    expected_train_images: int = 272,
    expected_split_protocol: str = "every-nth-test",
    require_checkpoint_sha: bool = False,
    require_dataset_protocol: bool = False,
) -> dict[str, Any]:
    config = _read_json(path / "config.yaml")
    expected_profile = "b1" if method == "b1" else "ru"
    if config.get("profile") != expected_profile:
        raise ValueError(f"unexpected profile for {method} run {run_index}: {path}")

    split = _read_json(path / "dataset_split.json")
    if (
        split.get("protocol") != expected_split_protocol
        or len(split.get("train", [])) != expected_train_images
        or len(split.get("test", [])) != expected_test_images
    ):
        raise ValueError(f"unexpected dataset split: {path}")
    if require_dataset_protocol and split.get("dataset_format") != "ontogo-patio-high":
        raise ValueError(f"Patio-High dataset format is not recorded: {path}")

    rows = _read_latency_rows(
        path / "per_image_latency.csv", expected_test_images=expected_test_images
    )
    if {row["image_name"] for row in rows} != set(split["test"]):
        raise ValueError(f"raw latency names differ from the test split: {path}")

    efficiency = _read_json(path / "efficiency_metrics.json")
    validation = _read_json(path / "ru_validation.json")
    test_metrics = _read_json(path / "test_metrics.json")
    if efficiency.get("warmup_render_count") != EXPECTED_WARMUP_RENDERS:
        raise ValueError(f"warmup count must be 10: {path}")
    if efficiency.get("raw_latency_sample_count") != expected_test_images:
        raise ValueError(f"raw latency count must be {expected_test_images}: {path}")
    if validation.get("evaluation_warmup_render_count") != EXPECTED_WARMUP_RENDERS:
        raise ValueError(f"validation warmup count must be 10: {path}")
    if validation.get("raw_latency_sample_count") != expected_test_images:
        raise ValueError(
            f"validation latency count must be {expected_test_images}: {path}"
        )
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
    if require_dataset_protocol and "--dataset_format ontogo-patio-high" not in command:
        raise ValueError(f"Patio-High audit command omitted its dataset format: {path}")

    for key in ("psnr", "ssim", "lpips"):
        if not math.isfinite(float(test_metrics[key])):
            raise ValueError(f"non-finite {key}: {path}")

    checkpoint_sha_path = path / "checkpoint_sha256.txt"
    checkpoint_sha = (
        checkpoint_sha_path.read_text(encoding="utf-8").strip()
        if checkpoint_sha_path.is_file()
        else None
    )
    if require_checkpoint_sha and (
        checkpoint_sha is None
        or len(checkpoint_sha) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha.lower())
    ):
        raise ValueError(f"valid checkpoint_sha256.txt is required: {path}")

    dataset_protocol = None
    if require_dataset_protocol:
        record = _read_json(path / "dataset_protocol.json")
        embedded = record.get("protocol", {})
        if not (
            embedded.get("schema") == "puri-gs-ontogo-patio-high-v1"
            and embedded.get("dataset_format") == "ontogo-patio-high"
            and embedded.get("train_count") == expected_train_images
            and embedded.get("test_count") == expected_test_images
            and embedded.get("unassigned_frame_indices") == [266]
        ):
            raise ValueError(f"invalid Patio-High dataset protocol: {path}")
        dataset_protocol = {
            "protocol_sha256": record.get("protocol_sha256"),
            "initial_point_count": embedded.get("initial_point_count"),
            "initial_points_sha256": embedded.get("initial_points_sha256"),
            "initial_colors_sha256": embedded.get("initial_colors_sha256"),
        }
        if any(value is None for value in dataset_protocol.values()):
            raise ValueError(f"incomplete Patio-High dataset protocol: {path}")

    return {
        "method": method,
        "run_index": run_index,
        "path": str(path),
        "test_image_count": expected_test_images,
        "warmup_render_count": EXPECTED_WARMUP_RENDERS,
        "gaussian_count": int(efficiency["gaussian_count"]),
        "inference_vram_gib": float(efficiency["inference_vram_gib"]),
        "psnr": float(test_metrics["psnr"]),
        "ssim": float(test_metrics["ssim"]),
        "lpips": float(test_metrics["lpips"]),
        "checkpoint_sha256": checkpoint_sha,
        "dataset_protocol": dataset_protocol,
        **calculated,
        "image_names": [row["image_name"] for row in rows],
        "latency_rows": rows,
    }


def decide(
    b1_runs: list[dict[str, Any]],
    ru_runs: list[dict[str, Any]],
    *,
    scene: str = "room",
    p95_ratio_gate: float | None = None,
) -> dict[str, Any]:
    if len(b1_runs) != EXPECTED_REPEATS or len(ru_runs) != EXPECTED_REPEATS:
        raise ValueError("the audit requires exactly three B1 and three RU runs")
    b1_runs = sorted(b1_runs, key=lambda item: item["run_index"])
    ru_runs = sorted(ru_runs, key=lambda item: item["run_index"])
    reference_names = b1_runs[0]["image_names"]
    if any(run["image_names"] != reference_names for run in b1_runs + ru_runs):
        raise ValueError("B1/RU audit runs use different test-image orders")
    if scene == "ontogo":
        protocols = [run.get("dataset_protocol") for run in b1_runs + ru_runs]
        if any(protocol is None for protocol in protocols) or any(
            protocol != protocols[0] for protocol in protocols[1:]
        ):
            raise ValueError("On-the-go audit runs use different dataset initializations")
    if len({run["gaussian_count"] for run in b1_runs}) != 1:
        raise ValueError("B1 runs loaded different checkpoints")
    if len({run["gaussian_count"] for run in ru_runs}) != 1:
        raise ValueError("RU runs loaded different checkpoints")
    for method, runs in (("B1", b1_runs), ("RU", ru_runs)):
        for key in ("psnr", "ssim", "lpips"):
            if len({run[key] for run in runs}) != 1:
                raise ValueError(f"{method} repeats produced different {key} values")
        hashes = {run.get("checkpoint_sha256") for run in runs}
        if hashes != {None} and len(hashes) != 1:
            raise ValueError(f"{method} repeats used different checkpoint SHA-256 values")

    paired_fps_ratios = [
        ru["render_fps"] / b1["render_fps"]
        for b1, ru in zip(b1_runs, ru_runs, strict=True)
    ]
    median_paired_fps_ratio = float(median(paired_fps_ratios))

    def med(runs: list[dict[str, Any]], key: str) -> float:
        return float(median(run[key] for run in runs))

    b1_p95 = med(b1_runs, "latency_p95_ms")
    ru_p95 = med(ru_runs, "latency_p95_ms")
    median_fps_ratio = med(ru_runs, "render_fps") / med(b1_runs, "render_fps")
    gate_fps_ratio = (
        median_paired_fps_ratio if scene == "room" else median_fps_ratio
    )
    throughput_pass = gate_fps_ratio >= FPS_RATIO_GATE
    p95_ratio = ru_p95 / b1_p95
    p95_pass = p95_ratio_gate is None or p95_ratio <= p95_ratio_gate
    summary = {
        "fps_ratio_gate": FPS_RATIO_GATE,
        "paired_fps_ratios": paired_fps_ratios,
        "median_paired_fps_ratio": median_paired_fps_ratio,
        "median_fps_ratio": median_fps_ratio,
        "gate_fps_ratio": gate_fps_ratio,
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
        "p95_ratio_gate": p95_ratio_gate,
        "p95_ratio_gate_pass": p95_pass,
    }
    for key in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms"):
        summary[f"median_{key}_ratio"] = (
            summary["ru_median"][key] / summary["b1_median"][key]
        )
    passed = throughput_pass and p95_pass
    decision = (
        "EFFICIENCY_AUDIT_PASS" if passed else "EFFICIENCY_AUDIT_FAIL"
    )
    if scene == "android":
        decision = "ANDROID_EFFICIENCY_PASS" if passed else "ANDROID_EFFICIENCY_FAIL"
    return {
        "protocol": f"puri-gs-ru-{scene}-efficiency-audit-1",
        "scene": scene,
        "original_phase_r_decision_preserved": True,
        "warmup_render_count_per_run": EXPECTED_WARMUP_RENDERS,
        "repeat_count_per_method": EXPECTED_REPEATS,
        "decision": decision,
        "summary": summary,
        "runs": {
            "b1": [
                {
                    key: value
                    for key, value in run.items()
                    if key not in {"image_names", "latency_rows"}
                }
                for run in b1_runs
            ],
            "ru": [
                {
                    key: value
                    for key, value in run.items()
                    if key not in {"image_names", "latency_rows"}
                }
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
    scene = report.get("scene", "room")
    title_scene = {
        "room": "Room",
        "android": "Android",
        "garden": "Garden",
        "ontogo": "On-the-go",
    }.get(scene, scene)
    scope_note = (
        "本复核不覆盖或改写原始 Phase R 结论；它只检查 Room 平均 FPS 门。"
        if scene == "room"
        else "本复核只读取固定 checkpoint，执行交错 3+3 标准推理，不修改训练结果。"
    )
    p95_note = ""
    if summary["p95_ratio_gate"] is not None:
        p95_note = (
            f"p95 latency 中位比：{summary['median_latency_p95_ms_ratio']:.4f}；"
            f"固定上限：{summary['p95_ratio_gate']:.2f}。"
        )
    return "\n".join(
        [
            f"# PURI-GS-RU {title_scene} 推理效率复核",
            "",
            f"复核结论：`{report['decision']}`",
            "",
            scope_note,
            "",
            "| 方法 | 重复 | FPS | mean ms | p50 ms | p95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            f"三组配对 FPS 比例：{', '.join(f'{value:.4f}' for value in summary['paired_fps_ratios'])}",
            "",
            f"门禁 FPS 比例：{summary['gate_fps_ratio']:.4f}；固定门槛：{FPS_RATIO_GATE:.2f}。",
            "",
            p95_note,
            "" if p95_note else "",
            "所有重复均使用 10 次不计时预热、固定测试集逐图原始延迟、标准 checkpoint 推理路径。",
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
    parser.add_argument(
        "--scene", choices=("room", "android", "garden", "ontogo"), default="room"
    )
    parser.add_argument("--expected-test-images", type=int, default=EXPECTED_TEST_IMAGES)
    parser.add_argument("--expected-train-images", type=int, default=272)
    parser.add_argument("--expected-split-protocol", default="every-nth-test")
    parser.add_argument("--p95-ratio-gate", type=float)
    parser.add_argument("--require-checkpoint-sha", action="store_true")
    parser.add_argument("--analysis-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    names = {
        "room": (
            "phase_r_puri_gs_ru_efficiency_audit.json",
            "PHASE_R_PURI_GS_RU_EFFICIENCY_AUDIT.md",
            "room_ru_latency_raw.csv",
        ),
        "android": (
            "android_ru_efficiency_audit.json",
            "ANDROID_RU_EFFICIENCY_AUDIT.md",
            "android_ru_latency_raw.csv",
        ),
        "garden": (
            "garden_ru_efficiency_audit.json",
            "GARDEN_RU_EFFICIENCY_AUDIT.md",
            "garden_ru_latency_raw.csv",
        ),
        "ontogo": (
            "ontogo_ru_efficiency_audit.json",
            "ONTOGO_RU_EFFICIENCY_AUDIT.md",
            "ontogo_ru_latency_raw.csv",
        ),
    }
    json_name, md_name, raw_name = names[args.scene]
    json_path = output_dir / json_name
    md_path = output_dir / md_name
    for target in (json_path, md_path):
        if target.exists():
            raise RuntimeError(f"refusing to overwrite an existing audit report: {target}")

    runs = {
        method: [
            _load_run(
                getattr(args, f"{method}_run{run_index}").expanduser().resolve(),
                method,
                run_index,
                expected_test_images=args.expected_test_images,
                expected_train_images=args.expected_train_images,
                expected_split_protocol=args.expected_split_protocol,
                require_checkpoint_sha=args.require_checkpoint_sha,
                require_dataset_protocol=args.scene == "ontogo",
            )
            for run_index in range(1, EXPECTED_REPEATS + 1)
        ]
        for method in ("b1", "ru")
    }
    report = decide(
        runs["b1"],
        runs["ru"],
        scene=args.scene,
        p95_ratio_gate=args.p95_ratio_gate,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    if args.analysis_dir is not None:
        analysis_dir = args.analysis_dir.expanduser().resolve()
        raw_path = analysis_dir / raw_name
        if raw_path.exists():
            raise RuntimeError(f"refusing to overwrite raw latency output: {raw_path}")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "scene",
                    "method",
                    "run_index",
                    "image_index",
                    "image_name",
                    "latency_ms",
                    "checkpoint_sha256",
                ),
            )
            writer.writeheader()
            for method in ("b1", "ru"):
                for run in runs[method]:
                    for row in run["latency_rows"]:
                        writer.writerow(
                            {
                                "scene": args.scene,
                                "method": method,
                                "run_index": run["run_index"],
                                **row,
                                "checkpoint_sha256": run["checkpoint_sha256"],
                            }
                        )
    print(report["decision"])
    print(
        "MEDIAN_PAIRED_FPS_RATIO="
        f"{report['summary']['median_paired_fps_ratio']:.12f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
