"""Pure configuration, metric, and decision logic for PURI-GS-ST Phase 4A."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPECTED_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "profile": "static_transient_gate",
    "gsplat_version": "1.5.3",
    "seed": 42,
    "input_contract": {
        "scene": "room",
        "data_factor": 4,
        "test_every": 8,
        "normalize_world_space": True,
        "expected_frame_count": 16,
        "expected_clean_count": 8,
        "expected_patched_count": 8,
        "expected_checkpoint_step": 9999,
        "expected_checkpoint_sha256": "a1edd558de7434381a719b3b0a9629b53fb268fa2f1eb72235609448e3589d7d",
        "expected_derived_manifest_sha256": "14461238a297fcadad6395c2486a4563e334f0ae8f64a649e07be3a70912c7a9",
        "expected_derived_tree_sha256": "9a6ccf435c29342dddff198cd90bc9f5d727b962f47da9bb16372750d68de6eb",
    },
    "static_render": {
        "sh_degree": 3,
        "near_plane": 0.01,
        "far_plane": 1e10,
        "packed": False,
        "absgrad": True,
        "sparse_grad": False,
        "rasterize_mode": "classic",
        "camera_model": "pinhole",
        "with_ut": False,
        "with_eval3d": False,
    },
    "transient": {
        "grid_stride": 16,
        "plane_z": 1.0,
        "initial_opacity": 0.01,
        "sh_degree": 0,
        "scale_z_fraction": 0.01,
        "alpha_weight": 0.02,
        "opacity_lr": 0.05,
        "color_lr": 0.01,
        "fit_steps_per_frame": 500,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "adam_epsilon": 1e-8,
        "ssim_lambda": 0.2,
    },
    "evaluation": {
        "alpha_threshold": 0.5,
        "projection_p95_max_pixels": 0.1,
        "coverage_inner_border_pixels": 8,
        "coverage_alpha_hole_threshold": 0.01,
        "coverage_alpha_hole_ratio_max": 0.01,
        "precision_min": 0.8,
        "recall_min": 0.6,
        "f1_min": 0.68,
        "iou_min": 0.52,
        "per_frame_f1_min": 0.6,
        "per_frame_f1_required_count": 6,
        "clean_fpr_max": 0.01,
        "clean_mean_alpha_average_max": 0.01,
        "clean_mean_alpha_per_frame_max": 0.03,
        "clean_composite_static_mae_max": 0.003,
        "patch_relative_mae_reduction_min": 0.5,
        "patched_background_change_l1_max": 0.005,
        "static_render_max_difference": 1e-6,
        "static_fps_relative_difference_max": 0.01,
        "borderline_recall_floor": 0.5,
        "borderline_f1_floor": 0.6,
    },
}


def load_gate_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Phase 4A config {config_path}: {error}") from error
    if config != EXPECTED_CONFIG:
        raise ValueError("Phase 4A configuration differs from the frozen protocol")
    return config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: str | Path) -> tuple[str, int]:
    root_path = Path(root)
    files = sorted(path for path in root_path.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root_path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def ensure_output_absent(path: str | Path) -> None:
    output = Path(path)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite Phase 4A output: {output}")


def confusion_counts(predicted: np.ndarray, target: np.ndarray) -> dict[str, int]:
    predicted_array = np.asarray(predicted, dtype=bool)
    target_array = np.asarray(target, dtype=bool)
    if predicted_array.shape != target_array.shape:
        raise ValueError("predicted and target masks must have equal shape")
    return {
        "tp": int(np.count_nonzero(predicted_array & target_array)),
        "fp": int(np.count_nonzero(predicted_array & ~target_array)),
        "tn": int(np.count_nonzero(~predicted_array & ~target_array)),
        "fn": int(np.count_nonzero(~predicted_array & target_array)),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, tn, fn = (int(counts[key]) for key in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    predicted_positive = tp + fp
    target_positive = tp + fn
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / target_positive if target_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "fpr": fpr,
        "predicted_mask_ratio": predicted_positive / total if total else 0.0,
        "ground_truth_mask_ratio": target_positive / total if total else 0.0,
    }


def psnr(predicted: np.ndarray, target: np.ndarray) -> float:
    difference = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    mse = float(np.mean(difference * difference))
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def aggregate_summary(rows: list[dict[str, Any]], engineering: dict[str, Any]) -> dict[str, Any]:
    if len(rows) != 16:
        raise ValueError("Phase 4A summary requires exactly 16 frames")
    clean = [row for row in rows if row["kind"] == "clean"]
    patched = [row for row in rows if row["kind"] == "transient"]
    if len(clean) != 8 or len(patched) != 8:
        raise ValueError("Phase 4A summary requires 8 clean and 8 patched frames")
    patched_counts = {
        key: sum(int(row[key]) for row in patched) for key in ("tp", "fp", "tn", "fn")
    }
    patched_micro = metrics_from_counts(patched_counts)
    clean_counts = {
        key: sum(int(row[key]) for row in clean) for key in ("tp", "fp", "tn", "fn")
    }
    clean_micro = metrics_from_counts(clean_counts)
    return {
        "frame_count": len(rows),
        "clean_frame_count": len(clean),
        "patched_frame_count": len(patched),
        "patched_micro": patched_micro,
        "patched_frames_f1_at_least_0_60": sum(float(row["f1"]) >= 0.60 for row in patched),
        "patched_mean_predicted_mask_ratio": float(np.mean([row["predicted_mask_ratio"] for row in patched])),
        "patched_mean_ground_truth_mask_ratio": float(np.mean([row["ground_truth_mask_ratio"] for row in patched])),
        "patched_mean_alpha_inside_gt": float(np.mean([row["mean_alpha_inside_gt"] for row in patched])),
        "patched_mean_alpha_outside_gt": float(np.mean([row["mean_alpha_outside_gt"] for row in patched])),
        "patched_mean_relative_patch_mae_reduction": float(np.mean([row["relative_patch_mae_reduction"] for row in patched])),
        "patched_mean_background_change_l1": float(np.mean([row["background_change_l1"] for row in patched])),
        "clean_fpr": float(clean_micro["fpr"]),
        "clean_mean_predicted_mask_ratio": float(np.mean([row["predicted_mask_ratio"] for row in clean])),
        "clean_mean_transient_alpha": float(np.mean([row["mean_transient_alpha"] for row in clean])),
        "clean_max_frame_mean_transient_alpha": float(max(row["mean_transient_alpha"] for row in clean)),
        "clean_mean_p95_transient_alpha": float(np.mean([row["p95_transient_alpha"] for row in clean])),
        "clean_mean_abs_composite_static": float(np.mean([row["mean_abs_composite_static"] for row in clean])),
        "clean_mean_psnr_composite_target": float(np.mean([row["psnr_composite_target"] for row in clean])),
        "clean_mean_psnr_static_target": float(np.mean([row["psnr_static_target"] for row in clean])),
        "engineering": engineering,
    }


def decide_gate(summary: dict[str, Any], hard_gates: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    limits = config["evaluation"]
    required_hard = {
        "PASS_EXECUTION": True,
        "STATIC_CHECKPOINT_UNCHANGED": True,
        "STATIC_GRADIENT_LEAK": False,
        "TRANSIENT_GEOMETRY_FROZEN": True,
        "PREMULTIPLIED_COMPOSITING_PASS": True,
        "GRID_PROJECTION_PASS": True,
        "GRID_COVERAGE_PASS": True,
    }
    hard_pass = all(hard_gates.get(key) == expected for key, expected in required_hard.items())
    if not hard_pass:
        return {
            "decision": "ST_GATE_FAIL",
            "hard_gate_pass": False,
            "hard_gates": hard_gates,
            "numerical_gates_evaluated": False,
            "selectivity_gate_pass": False,
            "selectivity_checks": {},
            "clean_gate_pass": False,
            "clean_checks": {},
            "reconstruction_gate_pass": False,
            "reconstruction_checks": {},
            "static_inference_gate_pass": False,
            "static_inference_checks": {},
        }
    patched = summary["patched_micro"]
    selectivity_checks = {
        "precision": patched["precision"] >= limits["precision_min"],
        "recall": patched["recall"] >= limits["recall_min"],
        "f1": patched["f1"] >= limits["f1_min"],
        "iou": patched["iou"] >= limits["iou_min"],
        "per_frame_f1_count": summary["patched_frames_f1_at_least_0_60"] >= limits["per_frame_f1_required_count"],
    }
    clean_checks = {
        "fpr": summary["clean_fpr"] <= limits["clean_fpr_max"],
        "mean_alpha_average": summary["clean_mean_transient_alpha"] <= limits["clean_mean_alpha_average_max"],
        "each_frame_mean_alpha": summary["clean_max_frame_mean_transient_alpha"] <= limits["clean_mean_alpha_per_frame_max"],
        "composite_static_mae": summary["clean_mean_abs_composite_static"] <= limits["clean_composite_static_mae_max"],
    }
    reconstruction_checks = {
        "patch_relative_mae_reduction": summary["patched_mean_relative_patch_mae_reduction"] >= limits["patch_relative_mae_reduction_min"],
        "patched_background_change_l1": summary["patched_mean_background_change_l1"] <= limits["patched_background_change_l1_max"],
    }
    engineering = summary["engineering"]
    static_checks = {
        "render_identity": engineering["static_render_max_abs_difference"] <= limits["static_render_max_difference"],
        "fps": engineering["static_fps_relative_difference"] <= limits["static_fps_relative_difference_max"],
    }
    all_pass = hard_pass and all(selectivity_checks.values()) and all(clean_checks.values()) and all(reconstruction_checks.values()) and all(static_checks.values())
    if all_pass:
        decision = "ST_GATE_PASS"
    else:
        borderline = (
            hard_pass
            and selectivity_checks["precision"]
            and clean_checks["fpr"]
            and clean_checks["each_frame_mean_alpha"]
            and static_checks["render_identity"]
            and static_checks["fps"]
            and summary["patched_mean_relative_patch_mae_reduction"] > 0.0
            and patched["recall"] >= limits["borderline_recall_floor"]
            and patched["f1"] >= limits["borderline_f1_floor"]
            and (not selectivity_checks["recall"] or not selectivity_checks["f1"] or not selectivity_checks["iou"])
        )
        decision = "ST_GATE_BORDERLINE" if borderline else "ST_GATE_FAIL"
    return {
        "decision": decision,
        "hard_gate_pass": hard_pass,
        "hard_gates": hard_gates,
        "numerical_gates_evaluated": True,
        "selectivity_gate_pass": all(selectivity_checks.values()),
        "selectivity_checks": selectivity_checks,
        "clean_gate_pass": all(clean_checks.values()),
        "clean_checks": clean_checks,
        "reconstruction_gate_pass": all(reconstruction_checks.values()),
        "reconstruction_checks": reconstruction_checks,
        "static_inference_gate_pass": all(static_checks.values()),
        "static_inference_checks": static_checks,
    }


def json_safe(value: Any) -> Any:
    """Convert non-finite floats to explicit strings before strict JSON output."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sum_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    materialized = list(rows)
    return {key: sum(int(row[key]) for row in materialized) for key in ("tp", "fp", "tn", "fn")}
