#!/usr/bin/env python3
"""Apply the single fixed Phase-R gate to four completed 30k experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


EXPECTED_COUNTS = {"android": 19, "room": 39}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_per_image(path: Path, expected_count: int) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"required per-image metrics are missing: {path}")
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["image_name"]
            if name in rows:
                raise ValueError(f"duplicate per-image metric: {name}")
            rows[name] = {
                metric: float(row[metric]) for metric in ("psnr", "ssim", "lpips")
            }
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} per-image rows, got {len(rows)}")
    return rows


def paired_bootstrap_ci95(values: list[float], samples: int = 10_000) -> list[float]:
    if not values:
        raise ValueError("paired bootstrap requires non-empty values")
    generator = random.Random(42)
    estimates = sorted(
        mean(generator.choice(values) for _ in values) for _ in range(samples)
    )

    def percentile(fraction: float) -> float:
        position = fraction * (len(estimates) - 1)
        lower = int(position)
        upper = min(lower + 1, len(estimates) - 1)
        weight = position - lower
        return estimates[lower] * (1 - weight) + estimates[upper] * weight

    return [percentile(0.025), percentile(0.975)]


def _load_run(path: Path, scene: str, method: str) -> dict[str, Any]:
    checkpoint = path / "ckpts" / "ckpt_29999_rank0.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"standard 30k checkpoint is missing: {checkpoint}")
    eval_dir = path / "independent_eval"
    validation = _read_json(eval_dir / "ru_validation.json")
    if method == "ru":
        validation.update(_read_json(path / "aux" / "ru_training_validation.json"))
    result = {
        "scene": scene,
        "method": method,
        "path": str(path),
        "test": _read_json(eval_dir / "test_metrics.json"),
        "train": _read_json(path / "train_metrics.json"),
        "efficiency": _read_json(eval_dir / "efficiency_metrics.json"),
        "validation": validation,
        "per_image": _read_per_image(
            eval_dir / "per_image_metrics.csv", EXPECTED_COUNTS[scene]
        ),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
    }
    if method == "ru":
        result["dino_time"] = _read_json(path / "DINO_time.json")
        result["dino_environment"] = _read_json(path / "aux" / "dino_environment.json")
    return result


def decide(runs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    android_b1 = runs["android"]["b1"]
    android_ru = runs["android"]["ru"]
    room_b1 = runs["room"]["b1"]
    room_ru = runs["room"]["ru"]

    android_names = sorted(android_b1["per_image"])
    room_names = sorted(room_b1["per_image"])
    if android_names != sorted(android_ru["per_image"]):
        raise ValueError("Android B1/RU per-image names differ")
    if room_names != sorted(room_ru["per_image"]):
        raise ValueError("Room B1/RU per-image names differ")

    android_psnr_delta = (
        android_ru["test"]["psnr"] - android_b1["test"]["psnr"]
    )
    android_ssim_delta = (
        android_ru["test"]["ssim"] - android_b1["test"]["ssim"]
    )
    android_lpips_relative = (
        android_b1["test"]["lpips"] - android_ru["test"]["lpips"]
    ) / android_b1["test"]["lpips"]
    android_per_image_delta = [
        android_ru["per_image"][name]["psnr"]
        - android_b1["per_image"][name]["psnr"]
        for name in android_names
    ]
    android_improved_ratio = sum(value > 0 for value in android_per_image_delta) / len(
        android_per_image_delta
    )
    android_ci95 = paired_bootstrap_ci95(android_per_image_delta)
    android_gate = {
        "delta_psnr": android_psnr_delta,
        "delta_ssim": android_ssim_delta,
        "relative_lpips_improvement": android_lpips_relative,
        "per_image_psnr_improved_ratio": android_improved_ratio,
        "paired_bootstrap_psnr_ci95": android_ci95,
    }
    android_gate["pass"] = bool(
        (android_psnr_delta >= 0.8 or android_lpips_relative >= 0.10)
        and android_ssim_delta > 0
        and android_improved_ratio >= 0.70
        and android_ci95[0] > 0
    )

    room_psnr_delta = room_ru["test"]["psnr"] - room_b1["test"]["psnr"]
    room_ssim_delta = room_ru["test"]["ssim"] - room_b1["test"]["ssim"]
    room_lpips_delta = room_ru["test"]["lpips"] - room_b1["test"]["lpips"]
    room_worst_psnr_delta = min(
        room_ru["per_image"][name]["psnr"] - room_b1["per_image"][name]["psnr"]
        for name in room_names
    )
    room_gate = {
        "delta_psnr": room_psnr_delta,
        "delta_ssim": room_ssim_delta,
        "delta_lpips": room_lpips_delta,
        "worst_per_image_psnr_delta": room_worst_psnr_delta,
    }
    room_gate["pass"] = bool(
        room_psnr_delta >= -0.15
        and room_lpips_delta <= 0.01
        and room_ssim_delta >= -0.005
        and room_worst_psnr_delta >= -1.0
    )

    efficiency_scenes: dict[str, Any] = {}
    for scene in ("android", "room"):
        baseline = runs[scene]["b1"]
        candidate = runs[scene]["ru"]
        fps_ratio = candidate["efficiency"]["render_fps"] / baseline["efficiency"][
            "render_fps"
        ]
        gaussian_ratio = candidate["efficiency"]["gaussian_count"] / baseline[
            "efficiency"
        ]["gaussian_count"]
        vram_ratio = candidate["efficiency"]["inference_vram_gib"] / baseline[
            "efficiency"
        ]["inference_vram_gib"]
        validation = candidate["validation"]
        standard_path_pass = bool(
            validation["standard_checkpoint_load_pass"]
            and not validation["evaluation_imported_dino"]
            and not validation["evaluation_loaded_mask_head"]
            and validation["evaluation_rasterization_count_ratio"] == 1.0
        )
        scene_gate = {
            "fps_ratio": fps_ratio,
            "gaussian_count_ratio": gaussian_ratio,
            "inference_vram_ratio": vram_ratio,
            "standard_inference_path_pass": standard_path_pass,
            "gradient_isolation_pass": bool(validation["gradient_isolation_pass"]),
        }
        scene_gate["pass"] = bool(
            fps_ratio >= 0.95
            and gaussian_ratio <= 1.20
            and vram_ratio <= 1.05
            and standard_path_pass
            and scene_gate["gradient_isolation_pass"]
        )
        efficiency_scenes[scene] = scene_gate
    efficiency_pass = all(item["pass"] for item in efficiency_scenes.values())
    training_observations: dict[str, Any] = {}
    for scene in ("android", "room"):
        baseline = runs[scene]["b1"]
        candidate = runs[scene]["ru"]
        training_ratio = (
            candidate["train"]["training_time_seconds"]
            / baseline["train"]["training_time_seconds"]
        )
        training_observations[scene] = {
            "training_time_ratio": training_ratio,
            "within_suggested_3x": training_ratio <= 3.0,
            "checkpoint_size_ratio": (
                candidate["checkpoint_size_bytes"] / baseline["checkpoint_size_bytes"]
            ),
            "dino_render_feature_seconds": candidate["dino_time"][
                "render_feature_seconds"
            ],
            "dino_render_feature_fraction_of_training": candidate["dino_time"][
                "render_feature_fraction_of_training"
            ],
            "mask_head_parameter_count": candidate["dino_environment"][
                "mask_head_parameter_count"
            ],
        }
    passed = android_gate["pass"] and room_gate["pass"] and efficiency_pass
    return {
        "decision": "RU_RECONSTRUCTION_PASS" if passed else "RU_RECONSTRUCTION_FAIL",
        "android_robustness_gate": android_gate,
        "room_clean_gate": room_gate,
        "inference_efficiency_gate": {
            "pass": efficiency_pass,
            "scenes": efficiency_scenes,
        },
        "training_observations_not_a_core_gate": training_observations,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    android = summary["android_robustness_gate"]
    room = summary["room_clean_gate"]
    efficiency = summary["inference_efficiency_gate"]
    training = summary["training_observations_not_a_core_gate"]
    return "\n".join(
        [
            "# Phase R：PURI-GS-RU 决策报告",
            "",
            f"最终结论：`{summary['decision']}`",
            "",
            "| 门禁 | 结果 | 关键事实 |",
            "| --- | --- | --- |",
            (
                f"| Android 鲁棒性 | {'PASS' if android['pass'] else 'FAIL'} | "
                f"ΔPSNR={android['delta_psnr']:.4f} dB，ΔSSIM={android['delta_ssim']:.6f}，"
                f"LPIPS 相对改善={android['relative_lpips_improvement']:.2%}，"
                f"逐图改善={android['per_image_psnr_improved_ratio']:.2%}，"
                f"bootstrap CI95=[{android['paired_bootstrap_psnr_ci95'][0]:.4f}, "
                f"{android['paired_bootstrap_psnr_ci95'][1]:.4f}] |"
            ),
            (
                f"| Room clean | {'PASS' if room['pass'] else 'FAIL'} | "
                f"ΔPSNR={room['delta_psnr']:.4f} dB，ΔSSIM={room['delta_ssim']:.6f}，"
                f"ΔLPIPS={room['delta_lpips']:.6f}，最差逐图 ΔPSNR="
                f"{room['worst_per_image_psnr_delta']:.4f} dB |"
            ),
            f"| 推理效率与独立路径 | {'PASS' if efficiency['pass'] else 'FAIL'} | 逐场景事实见 JSON |",
            (
                "| 训练时长建议（非核心门） | INFO | "
                f"Android={training['android']['training_time_ratio']:.3f}×，"
                f"Room={training['room']['training_time_ratio']:.3f}×；"
                "完整 DINO 用时、checkpoint 比例和 mask 参数量见 JSON |"
            ),
            "",
            "本报告只执行附件规定的唯一门禁；不会自动改变 B1、调整阈值或进入阶段 U。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android-b1", type=Path, required=True)
    parser.add_argument("--android-ru", type=Path, required=True)
    parser.add_argument("--room-b1", type=Path, required=True)
    parser.add_argument("--room-ru", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    for target in (
        output_dir / "PHASE_R_PURI_GS_RU.md",
        output_dir / "phase_r_puri_gs_ru.json",
    ):
        if target.exists():
            raise RuntimeError(f"refusing to overwrite an existing report: {target}")
    runs = {
        "android": {
            "b1": _load_run(args.android_b1.resolve(), "android", "b1"),
            "ru": _load_run(args.android_ru.resolve(), "android", "ru"),
        },
        "room": {
            "b1": _load_run(args.room_b1.resolve(), "room", "b1"),
            "ru": _load_run(args.room_ru.resolve(), "room", "ru"),
        },
    }
    summary = {"protocol": "puri-gs-ru-phase-r-1", **decide(runs)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_r_puri_gs_ru.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "PHASE_R_PURI_GS_RU.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(summary["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
