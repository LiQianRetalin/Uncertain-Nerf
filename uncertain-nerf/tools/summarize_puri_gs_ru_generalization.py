#!/usr/bin/env python3
"""Summarize the single fixed Garden or On-the-go B1/RU generalization pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_TEST_COUNTS = {"garden": 24, "ontogo": 45}
OUTPUT_NAMES = {
    "garden": (
        "GARDEN_RU_CLEAN_GENERALIZATION.md",
        "garden_ru_clean_generalization.json",
        "garden_ru_per_image_metrics.csv",
    ),
    "ontogo": (
        "ONTOGO_RU_DYNAMIC_GENERALIZATION.md",
        "ontogo_ru_dynamic_generalization.json",
        "ontogo_ru_per_image_metrics.csv",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required result is missing: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"required result is empty: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _per_image(path: Path, expected_count: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"per-image metrics are missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != expected_count:
        raise ValueError(
            f"expected {expected_count} per-image rows, got {len(source)}: {path}"
        )
    rows = []
    for row in source:
        values = {metric: float(row[metric]) for metric in ("psnr", "ssim", "lpips")}
        if not row["image_name"] or not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"invalid per-image metric row: {path}")
        rows.append({"image_name": row["image_name"], **values})
    if len({row["image_name"] for row in rows}) != expected_count:
        raise ValueError(f"per-image names are not unique: {path}")
    return rows


def _environment_fingerprint(environment: dict[str, Any]) -> dict[str, Any]:
    torch_info = environment.get("torch", {})
    gsplat = environment.get("gsplat", {})
    packages = environment.get("packages", {})
    return {
        "repository_commit": environment.get("repository_commit"),
        "gpu": torch_info.get("gpu"),
        "torch_version": torch_info.get("version"),
        "cuda_runtime": torch_info.get("cuda_runtime"),
        "gsplat_version": gsplat.get("version"),
        "gsplat_commit": gsplat.get("commit"),
        "torchmetrics_version": packages.get("torchmetrics"),
    }


def _shared_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    return {
        "seed": config.get("seed"),
        "total_steps": config.get("total_steps"),
        "data_factor": training.get("data_factor"),
        "test_every": training.get("test_every"),
        "sh_degree": config.get("sh_degree", training.get("sh_degree")),
        "ssim_lambda": config.get("ssim_lambda", training.get("ssim_lambda")),
        "gsplat_version": config.get("gsplat_version"),
    }


def _load_run(
    path: Path,
    profile: str,
    expected_count: int,
    *,
    require_dataset_protocol: bool = False,
) -> dict[str, Any]:
    config = _read_json(path / "config.yaml")
    if config.get("profile") != profile:
        raise ValueError(f"expected profile {profile}: {path}")
    checkpoint = path / "ckpts" / "ckpt_29999_rank0.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"30k checkpoint is missing: {checkpoint}")
    split = _read_json(path / "dataset_split.json")
    evaluation = path / "independent_eval"
    eval_split = _read_json(evaluation / "dataset_split.json")
    if split != eval_split:
        raise ValueError(f"training/evaluation split differs: {path}")
    test_names = split.get("test")
    if not isinstance(test_names, list) or len(test_names) != expected_count:
        raise ValueError(f"unexpected test split count: {path}")
    rows = _per_image(evaluation / "per_image_metrics.csv", expected_count)
    if [row["image_name"] for row in rows] != test_names:
        raise ValueError(f"per-image rows do not follow test split order: {path}")
    validation = _read_json(evaluation / "ru_validation.json")
    if not (
        validation.get("standard_checkpoint_load_pass") is True
        and validation.get("evaluation_imported_dino") is False
        and validation.get("evaluation_loaded_mask_head") is False
        and validation.get("evaluation_rasterization_count_ratio") == 1.0
    ):
        raise ValueError(f"evaluation is not the standard splats-only path: {path}")
    metrics = _read_json(evaluation / "test_metrics.json")
    efficiency = _read_json(evaluation / "efficiency_metrics.json")
    for key in ("psnr", "ssim", "lpips"):
        if not math.isfinite(float(metrics[key])):
            raise ValueError(f"non-finite {key}: {path}")
    dataset_protocol = None
    if require_dataset_protocol:
        training_protocol = _read_json(path / "dataset_protocol.json")
        evaluation_protocol = _read_json(evaluation / "dataset_protocol.json")
        if training_protocol != evaluation_protocol:
            raise ValueError(f"training/evaluation dataset protocols differ: {path}")
        embedded = training_protocol.get("protocol", {})
        if not (
            embedded.get("schema") == "puri-gs-ontogo-patio-high-v1"
            and embedded.get("dataset_format") == "ontogo-patio-high"
            and embedded.get("train_count") == 221
            and embedded.get("test_count") == 45
            and embedded.get("unassigned_frame_indices") == [266]
        ):
            raise ValueError(f"invalid Patio-High dataset protocol: {path}")
        dataset_protocol = {
            "protocol_sha256": training_protocol.get("protocol_sha256"),
            "transforms_sha256": embedded.get("transforms_sha256"),
            "split_sha256": embedded.get("split_sha256"),
            "initial_point_count": embedded.get("initial_point_count"),
            "initial_points_sha256": embedded.get("initial_points_sha256"),
            "initial_colors_sha256": embedded.get("initial_colors_sha256"),
        }
        if any(value is None for value in dataset_protocol.values()):
            raise ValueError(f"incomplete Patio-High dataset protocol: {path}")
    return {
        "path": str(path),
        "profile": profile,
        "git_commit": _read_text(path / "git_commit.txt"),
        "evaluation_git_commit": _read_text(evaluation / "git_commit.txt"),
        "config": _shared_config(config),
        "split": split,
        "environment": _environment_fingerprint(_read_json(path / "environment.json")),
        "evaluation_environment": _environment_fingerprint(
            _read_json(evaluation / "environment.json")
        ),
        "metrics": {key: float(metrics[key]) for key in ("psnr", "ssim", "lpips")},
        "gaussian_count": int(efficiency["gaussian_count"]),
        "inference_vram_gib": float(efficiency["inference_vram_gib"]),
        "training_time_seconds": float(
            _read_json(path / "train_metrics.json")["training_time_seconds"]
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
        },
        "per_image": rows,
        "render_dir": evaluation / "renders",
        "dataset_protocol": dataset_protocol,
    }


def _paired_rows(b1: dict[str, Any], ru: dict[str, Any]) -> list[dict[str, Any]]:
    if b1["split"] != ru["split"]:
        raise ValueError("B1/RU dataset splits differ")
    checks = {
        "training_commit": b1["git_commit"] == ru["git_commit"],
        "evaluation_commit": b1["evaluation_git_commit"] == ru["evaluation_git_commit"],
        "shared_config": b1["config"] == ru["config"],
        "training_environment": b1["environment"] == ru["environment"],
        "evaluation_environment": b1["evaluation_environment"] == ru["evaluation_environment"],
        "dataset_protocol": b1.get("dataset_protocol") == ru.get("dataset_protocol"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("B1/RU pairing failed: " + ", ".join(failed))
    result = []
    for b1_row, ru_row in zip(b1["per_image"], ru["per_image"], strict=True):
        if b1_row["image_name"] != ru_row["image_name"]:
            raise ValueError("B1/RU per-image names or order differ")
        delta_psnr = ru_row["psnr"] - b1_row["psnr"]
        delta_ssim = ru_row["ssim"] - b1_row["ssim"]
        delta_lpips = ru_row["lpips"] - b1_row["lpips"]
        result.append(
            {
                "image_name": b1_row["image_name"],
                "b1_psnr": b1_row["psnr"],
                "ru_psnr": ru_row["psnr"],
                "delta_psnr": delta_psnr,
                "b1_ssim": b1_row["ssim"],
                "ru_ssim": ru_row["ssim"],
                "delta_ssim": delta_ssim,
                "b1_lpips": b1_row["lpips"],
                "ru_lpips": ru_row["lpips"],
                "delta_lpips": delta_lpips,
                "psnr_or_lpips_improved": delta_psnr > 0 or delta_lpips < 0,
            }
        )
    return result


def _load_efficiency(path: Path | None, scene: str) -> dict[str, Any] | None:
    if path is None:
        return None
    report = _read_json(path)
    if report.get("scene") != scene:
        raise ValueError(f"efficiency report is for another scene: {path}")
    fps_ratio = report.get("summary", {}).get("gate_fps_ratio")
    if not isinstance(fps_ratio, (int, float)) or not math.isfinite(fps_ratio):
        raise ValueError(f"efficiency report lacks a finite gate FPS ratio: {path}")
    return {
        "path": str(path),
        "decision": report.get("decision"),
        "fps_ratio": float(fps_ratio),
        "pass": (
            report.get("decision") == "EFFICIENCY_AUDIT_PASS"
            and float(fps_ratio) >= 0.95
        ),
    }


def decide(
    scene: str,
    b1: dict[str, Any],
    ru: dict[str, Any],
    efficiency: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _paired_rows(b1, ru)
    delta_psnr = ru["metrics"]["psnr"] - b1["metrics"]["psnr"]
    delta_ssim = ru["metrics"]["ssim"] - b1["metrics"]["ssim"]
    delta_lpips = ru["metrics"]["lpips"] - b1["metrics"]["lpips"]
    gaussian_ratio = ru["gaussian_count"] / b1["gaussian_count"]
    vram_ratio = ru["inference_vram_gib"] / b1["inference_vram_gib"]
    common = {
        "delta_psnr": delta_psnr,
        "delta_ssim": delta_ssim,
        "delta_lpips": delta_lpips,
        "gaussian_count_ratio": gaussian_ratio,
        "inference_vram_ratio": vram_ratio,
    }
    if scene == "garden":
        quality_pass = bool(
            delta_psnr >= -0.15
            and delta_ssim >= -0.005
            and delta_lpips <= 0.01
            and gaussian_ratio <= 1.05
            and vram_ratio <= 1.05
        )
        decision_name = "GARDEN_CLEAN_PASS" if quality_pass else "GARDEN_CLEAN_FAIL"
        gate = {**common, "quality_and_compactness_pass": quality_pass}
    else:
        relative_lpips_improvement = (
            b1["metrics"]["lpips"] - ru["metrics"]["lpips"]
        ) / b1["metrics"]["lpips"]
        improved_ratio = sum(row["psnr_or_lpips_improved"] for row in rows) / len(rows)
        quality_pass = bool(
            (delta_psnr >= 0.50 or relative_lpips_improvement >= 0.10)
            and delta_ssim >= 0
            and improved_ratio >= 0.70
        )
        decision_name = "ONTOGO_DYNAMIC_PASS" if quality_pass else "ONTOGO_DYNAMIC_FAIL"
        gate = {
            **common,
            "relative_lpips_improvement": relative_lpips_improvement,
            "psnr_or_lpips_improved_view_ratio": improved_ratio,
            "dynamic_quality_pass": quality_pass,
        }
    if quality_pass and efficiency is None:
        raise RuntimeError(
            f"{scene} quality passed; complete the required 3+3 efficiency audit before finalizing"
        )
    efficiency_pass = efficiency is not None and efficiency["pass"]
    passed = quality_pass and efficiency_pass
    if quality_pass and not efficiency_pass:
        decision_name = decision_name.replace("PASS", "FAIL")
    gate["efficiency"] = efficiency
    gate["pass"] = passed
    return {
        "protocol": f"puri-gs-ru-{scene}-generalization-1",
        "scene": scene,
        "decision": decision_name,
        "gate": gate,
        "pairing": {
            "git_commit": b1["git_commit"],
            "evaluation_git_commit": b1["evaluation_git_commit"],
            "shared_config": b1["config"],
            "split_protocol": b1["split"].get("protocol"),
            "train_count": len(b1["split"].get("train", [])),
            "test_count": len(b1["split"].get("test", [])),
            "dataset_protocol": b1.get("dataset_protocol"),
        },
        "b1": {key: b1[key] for key in ("path", "metrics", "gaussian_count", "inference_vram_gib", "training_time_seconds", "checkpoint")},
        "ru": {key: ru[key] for key in ("path", "metrics", "gaussian_count", "inference_vram_gib", "training_time_seconds", "checkpoint")},
    }, rows


def _comparison_images(
    rows: list[dict[str, Any]],
    b1_render_dir: Path,
    ru_render_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    from PIL import Image, ImageChops

    if output_dir.exists():
        raise RuntimeError(f"comparison output already exists: {output_dir}")
    ranked = sorted(enumerate(rows), key=lambda item: item[1]["delta_psnr"])
    last = len(ranked) - 1
    positions = sorted(
        {
            0,
            round(last * 0.14),
            round(last * 0.29),
            round(last * 0.43),
            last // 2,
            round(last * 0.71),
            round(last * 0.86),
            last,
        }
    )
    if len(positions) != 8:
        raise RuntimeError("could not select eight distinct comparison views")
    output_dir.mkdir(parents=True)
    manifest = []
    for selection_index, position in enumerate(positions):
        original_index, row = ranked[position]
        b1_path = b1_render_dir / f"test_step29999_{original_index:04d}.png"
        ru_path = ru_render_dir / f"test_step29999_{original_index:04d}.png"
        with Image.open(b1_path) as b1_canvas, Image.open(ru_path) as ru_canvas:
            b1_image = b1_canvas.convert("RGB")
            ru_image = ru_canvas.convert("RGB")
            if b1_image.size != ru_image.size or b1_image.width % 2:
                raise ValueError(f"incompatible evaluation canvases for {row['image_name']}")
            width = b1_image.width // 2
            height = b1_image.height
            gt_b1 = b1_image.crop((0, 0, width, height))
            gt_ru = ru_image.crop((0, 0, width, height))
            if ImageChops.difference(gt_b1, gt_ru).getbbox() is not None:
                raise ValueError(f"B1/RU ground truth panels differ for {row['image_name']}")
            comparison = Image.new("RGB", (width * 3, height))
            comparison.paste(gt_b1, (0, 0))
            comparison.paste(b1_image.crop((width, 0, width * 2, height)), (width, 0))
            comparison.paste(ru_image.crop((width, 0, width * 2, height)), (width * 2, 0))
            output_path = output_dir / f"comparison_{selection_index + 1:02d}.png"
            comparison.save(output_path)
        role = (
            "minimum"
            if position == 0
            else "maximum"
            if position == last
            else "median"
            if position == last // 2
            else "quantile"
        )
        manifest.append(
            {
                "file": str(output_path),
                "panels": ["ground_truth", "b1", "ru"],
                "selection_role": role,
                "psnr_rank": position,
                "image_name": row["image_name"],
                "delta_psnr": row["delta_psnr"],
                "delta_lpips": row["delta_lpips"],
            }
        )
    return manifest


def render_markdown(report: dict[str, Any]) -> str:
    gate = report["gate"]
    dynamic = "" if report["scene"] == "garden" else (
        f"逐图 PSNR/LPIPS 至少一项改善={gate['psnr_or_lpips_improved_view_ratio']:.2%}，"
    )
    comparison = report.get("comparisons")
    comparison_note = ""
    if comparison is not None:
        comparison_note = f"对比图：{len(comparison)} 张，按逐图 ΔPSNR 从最差到最好固定分位选择；每张依次为 GT、B1、RU。"
    return "\n".join(
        [
            f"# PURI-GS-RU {report['scene']} 泛化判定",
            "",
            f"最终状态：`{report['decision']}`",
            "",
            f"ΔPSNR={gate['delta_psnr']:.6f} dB，ΔSSIM={gate['delta_ssim']:.6f}，ΔLPIPS={gate['delta_lpips']:.6f}。",
            "",
            f"Gaussian 比={gate['gaussian_count_ratio']:.6f}，VRAM 比={gate['inference_vram_ratio']:.6f}，{dynamic}严格 3+3 效率={'PASS' if gate['efficiency'] and gate['efficiency']['pass'] else 'FAIL/未执行'}。",
            "",
            comparison_note,
            "" if comparison_note else "",
            "本报告只应用冻结门槛，不修改 RU、不调参，也不自动选择其他场景。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(EXPECTED_TEST_COUNTS), required=True)
    parser.add_argument("--b1-run", type=Path, required=True)
    parser.add_argument("--ru-run", type=Path, required=True)
    parser.add_argument("--efficiency-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    md_name, json_name, csv_name = OUTPUT_NAMES[args.scene]
    targets = (output_dir / md_name, output_dir / json_name, analysis_dir / csv_name)
    if any(target.exists() for target in targets):
        raise RuntimeError("refusing to overwrite an existing generalization output")
    expected_count = EXPECTED_TEST_COUNTS[args.scene]
    require_dataset_protocol = args.scene == "ontogo"
    b1 = _load_run(
        args.b1_run.expanduser().resolve(),
        "b1",
        expected_count,
        require_dataset_protocol=require_dataset_protocol,
    )
    ru = _load_run(
        args.ru_run.expanduser().resolve(),
        "ru",
        expected_count,
        require_dataset_protocol=require_dataset_protocol,
    )
    efficiency = _load_efficiency(
        args.efficiency_audit.expanduser().resolve() if args.efficiency_audit else None,
        args.scene,
    )
    try:
        report, rows = decide(args.scene, b1, ru, efficiency)
    except RuntimeError as error:
        if "3+3 efficiency audit" not in str(error):
            raise
        print(f"{args.scene.upper()}_QUALITY_PASS_EFFICIENCY_REQUIRED")
        return 10
    if args.scene == "ontogo":
        if args.comparison_dir is None:
            raise RuntimeError("On-the-go summary requires --comparison-dir")
        report["comparisons"] = _comparison_images(
            rows, b1["render_dir"], ru["render_dir"], args.comparison_dir.resolve()
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with targets[2].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    targets[1].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    targets[0].write_text(render_markdown(report), encoding="utf-8")
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
