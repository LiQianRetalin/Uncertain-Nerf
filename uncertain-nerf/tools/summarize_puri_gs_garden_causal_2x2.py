#!/usr/bin/env python3
"""Validate and summarize the fixed Garden PURI-GS-RU 2x2 causal experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

# Keep the documented direct CLI invocation usable from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.config import causal_factors, validate_experiment_config
from puri_gs.delayed_absgrad import topology_event_summary


PROTOCOL = "puri-gs-ru-garden-causal-2x2-v1"
EXPECTED_IMAGE_COUNT = 24
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42
REPORT_NAMES = (
    "PHASE_R_GARDEN_CAUSAL_2X2.md",
    "phase_r_garden_causal_2x2.json",
)
QUADRANTS = {
    "Y00": {"method": "B1", "M": 0, "T": 0},
    "Y01": {"method": "DG-only", "M": 0, "T": 1},
    "Y10": {"method": "Mask-only", "M": 1, "T": 0},
    "Y11": {"method": "RU", "M": 1, "T": 1},
}
METRIC_DIRECTIONS = {
    "psnr": "higher_is_better",
    "ssim": "higher_is_better",
    "lpips": "lower_is_better",
    "render_fps": "higher_is_better",
    "gaussian_count": "raw_count_no_quality_direction",
    "latency_mean_ms": "lower_is_better",
    "latency_p50_ms": "lower_is_better",
    "latency_p95_ms": "lower_is_better",
    "inference_vram_gib": "lower_is_better",
    "training_time_seconds": "lower_is_better",
    "peak_vram_gib": "lower_is_better",
    "checkpoint_size_bytes": "lower_is_better",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _split_identity(split: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that define Garden train/test pairing.

    Current COLMAP runs record ``dataset_format=colmap`` while the pinned
    historical B1/RU artifacts predate that provenance-only field.
    """
    return {
        key: split.get(key)
        for key in ("protocol", "train_keyword", "test_keyword", "train", "test")
    }


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required text file is missing: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"required text file is empty: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_targets_absent(output_dir: Path) -> tuple[Path, Path]:
    targets = tuple(output_dir / name for name in REPORT_NAMES)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise RuntimeError(
            "refusing to overwrite existing Garden causal reports: " + ", ".join(existing)
        )
    return targets  # type: ignore[return-value]


def paired_bootstrap_ci95(
    values: list[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    if not values or samples <= 0:
        raise ValueError("paired bootstrap requires values and a positive sample count")
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(values) for _ in values) for _ in range(samples)
    )

    def percentile(fraction: float) -> float:
        position = fraction * (len(estimates) - 1)
        lower = int(position)
        upper = min(lower + 1, len(estimates) - 1)
        weight = position - lower
        return estimates[lower] * (1.0 - weight) + estimates[upper] * weight

    return [percentile(0.025), percentile(0.975)]


def _finite_number(value: Any, *, field: str, path: Path) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {path}")
    return result


def _read_per_image(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"per-image metrics are missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_IMAGE_COUNT} per-image rows, got {len(source)}: {path}"
        )
    rows: list[dict[str, Any]] = []
    for row in source:
        name = row.get("image_name", "")
        if not name:
            raise ValueError(f"empty image name: {path}")
        rows.append(
            {
                "image_name": name,
                **{
                    metric: _finite_number(row[metric], field=metric, path=path)
                    for metric in ("psnr", "ssim", "lpips")
                },
            }
        )
    if len({row["image_name"] for row in rows}) != EXPECTED_IMAGE_COUNT:
        raise ValueError(f"per-image names are not unique: {path}")
    return rows


def _checkpoint_inventory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"30k checkpoint is missing: {path}")
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    keys = sorted(checkpoint)
    if keys != ["splats", "step"] or checkpoint.get("step") != 29_999:
        raise ValueError(
            f"checkpoint must contain only standard step/splats at step 29999: {path}"
        )
    if not isinstance(checkpoint.get("splats"), dict):
        raise ValueError(f"checkpoint splats are not a mapping: {path}")
    del checkpoint
    return {
        "path": str(path),
        "step": 29_999,
        "keys": keys,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "standard_gaussian_only": True,
    }


def _effective_baseline(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    strategy = config.get("strategy", {})
    return {
        "total_steps": config.get("total_steps"),
        "seed": config.get("seed"),
        "sh_degree": config.get("sh_degree", training.get("sh_degree")),
        "ssim_lambda": config.get("ssim_lambda", training.get("ssim_lambda")),
        "data_factor": training.get("data_factor"),
        "test_every": training.get("test_every"),
        "absgrad": config.get("absgrad", strategy.get("absgrad")),
        "grow_grad2d": config.get("grow_grad2d", strategy.get("grow_grad2d")),
        "gsplat_version": config.get("gsplat_version"),
    }


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


def _expected_topology(M: int, T: int) -> dict[str, Any]:
    return topology_event_summary(
        delayed_topology=bool(T),
        total_steps=30_000,
        mask_pause_after_reset=300 if M else 0,
    )


def _validate_contract(
    root: Path, quadrant: str, config: dict[str, Any]
) -> dict[str, Any]:
    expected = QUADRANTS[quadrant]
    factors = causal_factors(config)
    if factors is None:
        raise ValueError(f"profile cannot be mapped to Garden M/T factors: {root}")
    M, T = (int(factors[0]), int(factors[1]))
    if (M, T) != (expected["M"], expected["T"]):
        raise ValueError(f"{quadrant} resolved to M={M}, T={T}: {root}")
    topology = _expected_topology(M, T)
    contract_path = root / "causal_contract.json"
    if quadrant in {"Y01", "Y10"}:
        contract = _read_json(contract_path)
        expected_contract = {
            "protocol": PROTOCOL,
            "semantic_mask_enabled": bool(M),
            "delayed_topology_enabled": bool(T),
            "M": M,
            "T": T,
            "topology_events": topology,
            "expected_mask_update_count": (
                30_000 - topology["mask_pause_step_count"] if M else 0
            ),
        }
        if contract != expected_contract:
            raise ValueError(f"causal contract differs from the resolved factors: {root}")
    elif contract_path.exists():
        raise ValueError(f"historical B1/RU unexpectedly contains a new causal contract: {root}")
    return {"M": M, "T": T, "topology_events": topology}


def _mask_inventory(root: Path, quadrant: str, contract: dict[str, Any]) -> dict[str, Any] | None:
    if contract["M"] == 0:
        forbidden = [root / "DINO_time.json", root / "aux" / "dino_environment.json"]
        present = [str(path) for path in forbidden if path.exists()]
        if present:
            raise ValueError(f"M=0 run contains semantic-mask artifacts: {present}")
        if quadrant == "Y01":
            command = _read_text(root / "run_command.txt")
            leaked = [
                token
                for token in ("--dino_repo_dir", "--dino_weight_path", "--feature_cache_dir")
                if token in command
            ]
            if leaked:
                raise ValueError(f"DG-only command touches DINO/cache: {leaked}")
        return None

    dino_time = _read_json(root / "DINO_time.json")
    dino_environment = _read_json(root / "aux" / "dino_environment.json")
    if dino_environment.get("trainable_parameter_count") != 0:
        raise ValueError(f"DINO must be frozen: {root}")
    if dino_environment.get("weight_sha256", "").casefold() != (
        "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"
    ):
        raise ValueError(f"unexpected DINO weight: {root}")
    if dino_environment.get("repository_commit") != (
        "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    ):
        raise ValueError(f"unexpected DINO repository commit: {root}")
    schedule = _read_json(root / "aux" / "training_schedule.json")
    expected_topology = contract["topology_events"]
    expected_updates = 30_000 - expected_topology["mask_pause_step_count"]
    observed_updates = dino_time.get("mask_update_count")
    observed_pauses = dino_time.get("mask_pause_count")
    if observed_updates != expected_updates:
        raise ValueError(
            f"mask update count is {observed_updates}, expected {expected_updates}: {root}"
        )
    if observed_pauses != expected_topology["mask_pause_step_count"]:
        raise ValueError(f"mask pause count does not match actual reset events: {root}")
    if quadrant == "Y10":
        checks = {
            "semantic_mask_enabled": schedule.get("semantic_mask_enabled") is True,
            "delayed_topology_enabled": (
                schedule.get("delayed_topology_enabled") is False
            ),
            "topology_events": schedule.get("topology_events") == expected_topology,
            "no_resets": schedule.get("topology_events", {}).get("reset_steps")
            == [],
            "no_pause_segments": expected_topology["mask_pause_segments"] == [],
            "all_updates": observed_updates == 30_000,
            "zero_pauses": observed_pauses == 0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError("Mask-only schedule validation failed: " + ", ".join(failed))
    return {
        "DINO_time": dino_time,
        "DINO_environment": dino_environment,
        "reset_events": expected_topology["reset_steps"],
        "mask_update_count": observed_updates,
        "mask_pause_count": observed_pauses,
        "mask_pause_segments": expected_topology["mask_pause_segments"],
        "training_schedule": schedule,
    }


def _load_run(root: Path, quadrant: str) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"run root is missing: {root}")
    config = _read_json(root / "config.yaml")
    validate_experiment_config(config)
    expected = QUADRANTS[quadrant]
    expected_profile = {
        "Y00": "b1",
        "Y01": "ru_causal",
        "Y10": "ru_causal",
        "Y11": "ru",
    }[quadrant]
    if config.get("profile") != expected_profile:
        raise ValueError(f"{quadrant} requires profile={expected_profile}: {root}")
    if quadrant in {"Y01", "Y10"}:
        expected_variant = "dg_only" if quadrant == "Y01" else "mask_only"
        if config.get("causal_variant") != expected_variant:
            raise ValueError(f"unexpected causal_variant: {root}")

    baseline = _effective_baseline(config)
    locked = {
        "total_steps": 30_000,
        "seed": 42,
        "sh_degree": 3,
        "ssim_lambda": 0.2,
        "data_factor": 4,
        "test_every": 8,
        "absgrad": True,
        "grow_grad2d": 0.0006,
        "gsplat_version": "1.5.3",
    }
    if baseline != locked:
        raise ValueError(f"fixed Garden baseline differs: {baseline}, expected {locked}: {root}")

    git_commit = _read_text(root / "git_commit.txt")
    evaluation = root / "independent_eval"
    evaluation_commit = _read_text(evaluation / "git_commit.txt")
    if evaluation_commit != git_commit:
        raise ValueError(f"training/evaluation commits differ: {root}")
    split = _read_json(root / "dataset_split.json")
    if split != _read_json(evaluation / "dataset_split.json"):
        raise ValueError(f"training/evaluation split differs: {root}")
    if split.get("dataset_format", "colmap") != "colmap":
        raise ValueError(f"Garden split must use the COLMAP dataset format: {root}")
    if len(split.get("train", [])) != 161 or len(split.get("test", [])) != 24:
        raise ValueError(f"Garden split must contain 161 train and 24 test images: {root}")
    per_image = _read_per_image(evaluation / "per_image_metrics.csv")
    if [row["image_name"] for row in per_image] != split["test"]:
        raise ValueError(f"per-image CSV order differs from the test split: {root}")

    validation = _read_json(evaluation / "ru_validation.json")
    validation_checks = {
        "standard_checkpoint_load_pass": validation.get("standard_checkpoint_load_pass")
        is True,
        "evaluation_imported_dino": validation.get("evaluation_imported_dino") is False,
        "evaluation_loaded_mask_head": validation.get("evaluation_loaded_mask_head") is False,
        "evaluation_rasterization_count_ratio": validation.get(
            "evaluation_rasterization_count_ratio"
        )
        == 1.0,
    }
    failed = [name for name, passed in validation_checks.items() if not passed]
    if failed:
        raise ValueError("non-standard independent evaluation: " + ", ".join(failed))

    test_metrics_source = _read_json(evaluation / "test_metrics.json")
    metrics = {
        metric: _finite_number(test_metrics_source[metric], field=metric, path=evaluation)
        for metric in ("psnr", "ssim", "lpips")
    }
    train_source = _read_json(root / "train_metrics.json")
    efficiency_source = _read_json(evaluation / "efficiency_metrics.json")
    render_fps = _finite_number(
        efficiency_source["render_fps"], field="render_fps", path=evaluation
    )
    efficiency = {
        "render_fps": render_fps,
        "latency_mean_ms": _finite_number(
            efficiency_source.get("latency_mean_ms", 1000.0 / render_fps),
            field="latency_mean_ms",
            path=evaluation,
        ),
        "latency_p50_ms": _finite_number(
            efficiency_source["latency_p50_ms"], field="latency_p50_ms", path=evaluation
        ),
        "latency_p95_ms": _finite_number(
            efficiency_source["latency_p95_ms"], field="latency_p95_ms", path=evaluation
        ),
        "gaussian_count": int(efficiency_source["gaussian_count"]),
        "inference_vram_gib": _finite_number(
            efficiency_source["inference_vram_gib"],
            field="inference_vram_gib",
            path=evaluation,
        ),
    }
    training = {
        "training_time_seconds": _finite_number(
            train_source["training_time_seconds"],
            field="training_time_seconds",
            path=root,
        ),
        "peak_vram_gib": _finite_number(
            train_source["peak_vram_gib"], field="peak_vram_gib", path=root
        ),
        "gaussian_count": int(train_source["gaussian_count"]),
    }
    if training["gaussian_count"] != efficiency["gaussian_count"]:
        raise ValueError(f"training/evaluation Gaussian counts differ: {root}")

    contract = _validate_contract(root, quadrant, config)
    mask = _mask_inventory(root, quadrant, contract)
    checkpoint = _checkpoint_inventory(root / "ckpts" / "ckpt_29999_rank0.pt")
    checkpoint["size_bytes"] = checkpoint.pop("bytes")
    return {
        "quadrant": quadrant,
        "method": expected["method"],
        "path": str(root),
        "git_commit": git_commit,
        "evaluation_git_commit": evaluation_commit,
        "config": config,
        "effective_fixed_baseline": baseline,
        "environment": _environment_fingerprint(_read_json(root / "environment.json")),
        "split": split,
        "contract": contract,
        "mask": mask,
        "metrics": metrics,
        "training": training,
        "efficiency": efficiency,
        "checkpoint": checkpoint,
        "evaluation_validation": validation,
        "per_image": per_image,
    }


def _pairing_audit(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = runs["Y00"]
    test_names = reference["split"]["test"]
    checks: dict[str, bool] = {}
    for quadrant, run in runs.items():
        checks[f"{quadrant}_same_split"] = _split_identity(
            run["split"]
        ) == _split_identity(reference["split"])
        checks[f"{quadrant}_same_test_names_and_order"] = (
            [row["image_name"] for row in run["per_image"]] == test_names
        )
        checks[f"{quadrant}_same_fixed_baseline"] = (
            run["effective_fixed_baseline"] == reference["effective_fixed_baseline"]
        )
    topology_keys = (
        "name",
        "delayed_topology",
        "refine_steps",
        "refine_event_count",
        "split_steps",
        "split_event_count",
        "clone_steps",
        "clone_event_count",
        "prune_steps",
        "prune_event_count",
        "statistics_window_inclusive",
        "reset_steps",
        "reset_event_count",
        "post_backward_order",
        "reset_behavior",
    )

    def topology_core(quadrant: str) -> dict[str, Any]:
        events = runs[quadrant]["contract"]["topology_events"]
        return {key: events[key] for key in topology_keys}

    checks["DG_topology_equals_RU"] = topology_core("Y01") == topology_core("Y11")
    checks["Mask_topology_equals_B1"] = topology_core("Y10") == topology_core("Y00")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("four-quadrant pairing audit failed: " + ", ".join(failed))
    return {
        "pass": True,
        "checks": checks,
        "train_image_count": len(reference["split"]["train"]),
        "test_image_count": len(test_names),
        "test_names": test_names,
    }


def _metric_effect(values: dict[str, float]) -> dict[str, float]:
    y00, y01, y10, y11 = (values[key] for key in ("Y00", "Y01", "Y10", "Y11"))
    return {
        "mask_at_T0_Y10_minus_Y00": y10 - y00,
        "mask_at_T1_Y11_minus_Y01": y11 - y01,
        "topology_at_M0_Y01_minus_Y00": y01 - y00,
        "topology_at_M1_Y11_minus_Y10": y11 - y10,
        "interaction_Y11_minus_Y10_minus_Y01_plus_Y00": y11 - y10 - y01 + y00,
    }


def _per_image_effects(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons = {
        "mask_at_T0_Y10_minus_Y00": (-1.0, 0.0, 1.0, 0.0),
        "mask_at_T1_Y11_minus_Y01": (0.0, -1.0, 0.0, 1.0),
        "topology_at_M0_Y01_minus_Y00": (-1.0, 1.0, 0.0, 0.0),
        "topology_at_M1_Y11_minus_Y10": (0.0, 0.0, -1.0, 1.0),
        "interaction_Y11_minus_Y10_minus_Y01_plus_Y00": (1.0, -1.0, -1.0, 1.0),
    }
    run_rows = {key: run["per_image"] for key, run in runs.items()}
    result: dict[str, Any] = {}
    for name, coefficients in comparisons.items():
        result[name] = {}
        for metric in ("psnr", "ssim", "lpips"):
            deltas = [
                sum(
                    coefficient * run_rows[quadrant][index][metric]
                    for coefficient, quadrant in zip(
                        coefficients, ("Y00", "Y01", "Y10", "Y11"), strict=True
                    )
                )
                for index in range(EXPECTED_IMAGE_COUNT)
            ]
            lower_is_better = metric == "lpips"
            improvement_count = sum(
                delta < 0 if lower_is_better else delta > 0 for delta in deltas
            )
            worst_decline = max(deltas) if lower_is_better else min(deltas)
            result[name][metric] = {
                "direction": METRIC_DIRECTIONS[metric],
                "mean_difference": mean(deltas),
                "median_difference": median(deltas),
                "improvement_count": improvement_count,
                "image_count": EXPECTED_IMAGE_COUNT,
                "paired_bootstrap_mean_difference_ci95": paired_bootstrap_ci95(deltas),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "worst_single_image_decline": worst_decline,
                "per_image_differences": deltas,
            }
    return result


def clean_tolerance(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_by_name = {row["image_name"]: row for row in baseline["per_image"]}
    candidate_by_name = {row["image_name"]: row for row in candidate["per_image"]}
    if list(baseline_by_name) != list(candidate_by_name):
        raise ValueError("clean-tolerance rows are not paired by image name and order")
    psnr_delta = candidate["metrics"]["psnr"] - baseline["metrics"]["psnr"]
    ssim_delta = candidate["metrics"]["ssim"] - baseline["metrics"]["ssim"]
    lpips_delta = candidate["metrics"]["lpips"] - baseline["metrics"]["lpips"]
    worst_psnr_delta = min(
        candidate_by_name[name]["psnr"] - baseline_by_name[name]["psnr"]
        for name in baseline_by_name
    )
    checks = {
        "psnr_delta_at_least_minus_0_15": psnr_delta >= -0.15,
        "ssim_delta_at_least_minus_0_005": ssim_delta >= -0.005,
        "lpips_delta_at_most_plus_0_01": lpips_delta <= 0.01,
        "worst_image_psnr_delta_at_least_minus_1_0": worst_psnr_delta >= -1.0,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "delta_psnr": psnr_delta,
        "delta_ssim": ssim_delta,
        "delta_lpips": lpips_delta,
        "worst_per_image_psnr_delta": worst_psnr_delta,
    }


def _directionally_fails(gate: dict[str, Any]) -> bool:
    return bool(
        gate["delta_psnr"] < 0
        and gate["delta_ssim"] < 0
        and gate["delta_lpips"] > 0
    )


def attribution_label(
    dg: dict[str, Any], mask: dict[str, Any], ru: dict[str, Any]
) -> str:
    if dg["pass"] and not mask["pass"]:
        return "GARDEN_MASK_DOMINANT"
    if mask["pass"] and not dg["pass"]:
        return "GARDEN_TOPOLOGY_DOMINANT"
    if dg["pass"] and mask["pass"] and not ru["pass"]:
        return "GARDEN_NEGATIVE_INTERACTION"
    if (
        not dg["pass"]
        and not mask["pass"]
        and _directionally_fails(dg)
        and _directionally_fails(mask)
    ):
        return "GARDEN_BOTH_CONTRIBUTE"
    return "GARDEN_CAUSAL_INCONCLUSIVE"


def _representative_images(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    b1 = runs["Y00"]["per_image"]
    ru = runs["Y11"]["per_image"]
    ranked = sorted(
        (
            {
                "image_name": before["image_name"],
                "test_index_zero_based": index,
                "RU_minus_B1_psnr": after["psnr"] - before["psnr"],
            }
            for index, (before, after) in enumerate(zip(b1, ru, strict=True))
        ),
        key=lambda row: (row["RU_minus_B1_psnr"], row["image_name"]),
    )
    positions = [round(index * (EXPECTED_IMAGE_COUNT - 1) / 5) for index in range(6)]
    selected = []
    for position in positions:
        row = dict(ranked[position])
        row["rank_zero_based_after_sort"] = position
        selected.append(row)
    return {
        "selection_rule": (
            "Sort all 24 images by RU-minus-B1 PSNR ascending, then select the "
            "six evenly spaced ranks round(i*23/5), i=0..5."
        ),
        "selected_rank_positions_zero_based": positions,
        "images": selected,
    }


def _scalar_effects(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "psnr": ("metrics", "psnr"),
        "ssim": ("metrics", "ssim"),
        "lpips": ("metrics", "lpips"),
        "render_fps": ("efficiency", "render_fps"),
        "gaussian_count": ("efficiency", "gaussian_count"),
        "latency_mean_ms": ("efficiency", "latency_mean_ms"),
        "latency_p50_ms": ("efficiency", "latency_p50_ms"),
        "latency_p95_ms": ("efficiency", "latency_p95_ms"),
        "inference_vram_gib": ("efficiency", "inference_vram_gib"),
        "training_time_seconds": ("training", "training_time_seconds"),
        "peak_vram_gib": ("training", "peak_vram_gib"),
        "checkpoint_size_bytes": ("checkpoint", "size_bytes"),
    }
    result: dict[str, Any] = {}
    for metric, path in paths.items():
        values = {
            quadrant: float(run[path[0]][path[1]]) for quadrant, run in runs.items()
        }
        result[metric] = {
            "direction": METRIC_DIRECTIONS[metric],
            "quadrant_values": values,
            "effects": _metric_effect(values),
        }
    return result


def build_summary(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairing = _pairing_audit(runs)
    clean = {
        "DG_only_Y01_vs_B1_Y00": clean_tolerance(runs["Y00"], runs["Y01"]),
        "Mask_only_Y10_vs_B1_Y00": clean_tolerance(runs["Y00"], runs["Y10"]),
        "RU_Y11_vs_B1_Y00": clean_tolerance(runs["Y00"], runs["Y11"]),
    }
    label = attribution_label(
        clean["DG_only_Y01_vs_B1_Y00"],
        clean["Mask_only_Y10_vs_B1_Y00"],
        clean["RU_Y11_vs_B1_Y00"],
    )
    direction = {
        "GARDEN_MASK_DOMINANT": (
            "下一版优先研究静态区域保护、Mask置信度和masked loss；暂不先改拓扑窗口。"
        ),
        "GARDEN_TOPOLOGY_DOMINANT": (
            "下一版优先研究fine监督与增密重叠、增密窗口和细节容量；暂不先改Mask公式。"
        ),
        "GARDEN_NEGATIVE_INTERACTION": (
            "下一版需联合处理Mask保护和拓扑时序，但仍应分阶段消融。"
        ),
        "GARDEN_BOTH_CONTRIBUTE": (
            "下一版需同时修正两个子系统，但不得一次加入无关模块。"
        ),
        "GARDEN_CAUSAL_INCONCLUSIVE": (
            "证据不足以选择新版本；只补最小缺失证据，不自动扩展实验。"
        ),
    }[label]
    public_runs = {
        quadrant: {key: value for key, value in run.items() if key != "per_image"}
        for quadrant, run in runs.items()
    }
    return {
        "protocol": PROTOCOL,
        "causal_layout": QUADRANTS,
        "pairing_audit": pairing,
        "runs": public_runs,
        "clean_tolerance": clean,
        "per_image_paired_effects": _per_image_effects(runs),
        "scalar_effects_and_interactions": _scalar_effects(runs),
        "representative_images": _representative_images(runs),
        "attribution": {
            "label": label,
            "next_version_direction_only": direction,
            "new_algorithm_code_written": False,
        },
        "efficiency_protocol": {
            "new_checkpoint_measurements_per_method": 1,
            "three_by_three_executed": False,
            "reason": "当前目标是质量因果归因，不是重复测量固定checkpoint。",
        },
        "phase_U": {"implemented": False, "started": False},
    }


def _short_steps(events: list[int]) -> str:
    if not events:
        return "[]"
    if len(events) <= 4:
        return str(events)
    return f"[{events[0]}, {events[1]}, ..., {events[-2]}, {events[-1]}]"


def render_markdown(summary: dict[str, Any]) -> str:
    runs = summary["runs"]
    lines = [
        "# Phase R：Garden 2×2 因果试验报告",
        "",
        f"唯一归因标签：`{summary['attribution']['label']}`",
        "",
        summary["attribution"]["next_version_direction_only"],
        "",
        "## 四象限、配置与 commit",
        "",
        "| 象限 | 方法 | M | T | 训练 commit | 评测 commit | 配置文件记录 |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for quadrant in ("Y00", "Y01", "Y10", "Y11"):
        run = runs[quadrant]
        contract = run["contract"]
        lines.append(
            f"| {quadrant} | {run['method']} | {contract['M']} | {contract['T']} | "
            f"`{run['git_commit']}` | `{run['evaluation_git_commit']}` | "
            f"`{run['path']}/config.yaml` |"
        )
    lines.extend(
        [
            "",
            "24 张测试图的名称和顺序：`PASS`；四组均为 161 张训练图、24 张测试图。",
            "",
            "测试图：" + ", ".join(summary["pairing_audit"]["test_names"]),
            "",
            "## 拓扑事件与 Mask 更新",
            "",
            "| 象限 | refine | split/clone/prune | stats window | reset | post-backward 顺序 | Mask update/pause |",
            "| --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for quadrant in ("Y00", "Y01", "Y10", "Y11"):
        run = runs[quadrant]
        events = run["contract"]["topology_events"]
        mask = run["mask"]
        mask_text = "N/A"
        if mask is not None:
            mask_text = (
                f"{mask['mask_update_count']} updates / {mask['mask_pause_count']} pauses; "
                f"segments={mask['mask_pause_segments']}"
            )
        lines.append(
            f"| {quadrant} | {events['refine_event_count']} "
            f"({_short_steps(events['refine_steps'])}) | "
            f"split={events['split_event_count']}, clone={events['clone_event_count']}, "
            f"prune={events['prune_event_count']} | {events['statistics_window_inclusive']} | "
            f"{events['reset_steps']} | {events['post_backward_order']} | {mask_text} |"
        )
    lines.extend(
        [
            "",
            "DG-only 的拓扑事件与 RU 完全一致；Mask-only 的拓扑事件与实际 B1 完全一致。"
            "B1 固定版本的 opacity-reset 条件没有产生 reset，因此 Mask-only 没有静默加入暂停。",
            "",
            "## 四象限观测量",
            "",
            "| 象限 | PSNR↑ | SSIM↑ | LPIPS↓ | Gaussian | 训练时间(s)↓ | 训练显存(GiB)↓ | FPS↑ | p50/p95(ms)↓ | 推理显存(GiB)↓ | ckpt bytes↓ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for quadrant in ("Y00", "Y01", "Y10", "Y11"):
        run = runs[quadrant]
        metrics, training, efficiency, checkpoint = (
            run["metrics"],
            run["training"],
            run["efficiency"],
            run["checkpoint"],
        )
        lines.append(
            f"| {quadrant} | {metrics['psnr']:.6f} | {metrics['ssim']:.6f} | "
            f"{metrics['lpips']:.6f} | {efficiency['gaussian_count']} | "
            f"{training['training_time_seconds']:.3f} | {training['peak_vram_gib']:.3f} | "
            f"{efficiency['render_fps']:.3f} | {efficiency['latency_p50_ms']:.3f}/"
            f"{efficiency['latency_p95_ms']:.3f} | {efficiency['inference_vram_gib']:.3f} | "
            f"{checkpoint['size_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Garden clean tolerance",
            "",
            "| 对比 | 结果 | ΔPSNR | ΔSSIM | ΔLPIPS（越低越好） | 最差逐图 ΔPSNR |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, gate in summary["clean_tolerance"].items():
        lines.append(
            f"| {name} | {'PASS' if gate['pass'] else 'FAIL'} | "
            f"{gate['delta_psnr']:.6f} | {gate['delta_ssim']:.6f} | "
            f"{gate['delta_lpips']:.6f} | {gate['worst_per_image_psnr_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 条件效应与交互项（标量）",
            "",
            "所有值均为原始差；LPIPS、延迟、显存、训练时间和 checkpoint 大小越低越好。",
            "",
            "| 指标 | M@T0 | M@T1 | T@M0 | T@M1 | M×T interaction | 方向 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for metric, record in summary["scalar_effects_and_interactions"].items():
        effect = record["effects"]
        lines.append(
            f"| {metric} | {effect['mask_at_T0_Y10_minus_Y00']:.6g} | "
            f"{effect['mask_at_T1_Y11_minus_Y01']:.6g} | "
            f"{effect['topology_at_M0_Y01_minus_Y00']:.6g} | "
            f"{effect['topology_at_M1_Y11_minus_Y10']:.6g} | "
            f"{effect['interaction_Y11_minus_Y10_minus_Y01_plus_Y00']:.6g} | "
            f"{record['direction']} |"
        )
    lines.extend(
        [
            "",
            "## 逐图配对统计",
            "",
            "PSNR、SSIM、LPIPS 的四项条件效应及二阶交互均使用同一 24 张图，"
            "固定 seed=42 做 10,000 次 paired bootstrap。表内 CI 是平均原始差的 95% CI。",
            "",
            "| 效应 | 指标 | 均值差 | 中位数差 | 改善图数/24 | bootstrap 95% CI | 最差单图下降 |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for effect_name, metrics in summary["per_image_paired_effects"].items():
        for metric, record in metrics.items():
            ci = record["paired_bootstrap_mean_difference_ci95"]
            lines.append(
                f"| {effect_name} | {metric} | {record['mean_difference']:.6f} | "
                f"{record['median_difference']:.6f} | {record['improvement_count']}/24 | "
                f"[{ci[0]:.6f}, {ci[1]:.6f}] | "
                f"{record['worst_single_image_decline']:.6f} |"
            )
    representatives = summary["representative_images"]
    lines.extend(
        [
            "",
            "## 6 张代表图",
            "",
            representatives["selection_rule"],
            "",
            "| 排序位置(0-based) | 测试索引(0-based) | 图像名 | RU−B1 PSNR |",
            "| ---: | ---: | --- | ---: |",
        ]
    )
    for row in representatives["images"]:
        lines.append(
            f"| {row['rank_zero_based_after_sort']} | {row['test_index_zero_based']} | "
            f"{row['image_name']} | {row['RU_minus_B1_psnr']:.6f} |"
        )
    lines.extend(["", "## 四象限完整配置", ""])
    for quadrant in ("Y00", "Y01", "Y10", "Y11"):
        lines.extend(
            [
                f"### {quadrant} {runs[quadrant]['method']}",
                "",
                "```json",
                json.dumps(runs[quadrant]["config"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 强制停止声明",
            "",
            "本次没有执行 3×3；原因是当前目标是质量因果归因，而非重复测量固定 checkpoint。",
            "阶段 U 未实现、未启动。本报告只给下一版方向建议，没有编写新算法代码、修改配置或启动训练。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--dg-only", type=Path, required=True)
    parser.add_argument("--mask-only", type=Path, required=True)
    parser.add_argument("--ru", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    markdown_path, json_path = ensure_output_targets_absent(output_dir)
    roots = {
        "Y00": args.b1,
        "Y01": args.dg_only,
        "Y10": args.mask_only,
        "Y11": args.ru,
    }
    runs = {
        quadrant: _load_run(path.expanduser().resolve(), quadrant)
        for quadrant, path in roots.items()
    }
    summary = build_summary(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"GARDEN-CAUSAL-2X2-SUMMARY-PASS {summary['attribution']['label']}")
    print(f"markdown={markdown_path}")
    print(f"json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
