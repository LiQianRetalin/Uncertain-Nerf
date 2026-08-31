#!/usr/bin/env python3
"""Zero-training failure attribution for the frozen Room CVTR mask experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from puri_gs.config import load_experiment_config
from puri_gs.cvtr import (
    CVTR_STAGE_NAMES,
    CVTRConfig,
    CVTRStageTrace,
    ViewEvidence,
    bilinear_sample,
    compute_cross_view_transient,
    compute_cvtr_stage_trace,
    load_binary_mask,
    project_camera_z,
    scene_residual_threshold,
    select_camera_neighbors,
    spatial_filter_and_cap,
    unproject_camera_z,
)
from tools.build_cvtr_masks import (
    _cvtr_config,
    _load_cache,
    _load_colmap_classes,
    repository_commit,
    require_empty_output,
    sha256_file,
)
from tools.validate_cvtr_synthetic import _load_derived_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
EXPECTED_CONFIG = CVTRConfig()
ROUNDTRIP_P95_LIMIT = 0.05
METRIC_REPRODUCTION_TOLERANCE = 1e-5
DERIVED_PIXEL_TOLERANCE = 1.0 / 255.0 + 1e-6
STAGE_COUNT = len(CVTR_STAGE_NAMES)


class ContractFailure(RuntimeError):
    """Raised after a frozen execution contract fails."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractFailure("INPUT_JSON_FAIL", f"cannot read {path}: {error}") from error


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ContractFailure("GIT_PROVENANCE_FAIL", result.stderr.strip())
    return result.stdout.strip()


def _tree_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def ensure_outputs_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ContractFailure(
            "OUTPUT_OVERWRITE_ATTEMPT",
            f"refusing to overwrite existing attribution outputs: {existing}",
        )


def confusion_counts(predicted: np.ndarray, target: np.ndarray) -> dict[str, int]:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target masks must have equal shape")
    return {
        "tp": int(np.count_nonzero(predicted & target)),
        "fp": int(np.count_nonzero(predicted & ~target)),
        "tn": int(np.count_nonzero(~predicted & ~target)),
        "fn": int(np.count_nonzero(~predicted & target)),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int | None]:
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    predicted_positive = tp + fp
    target_positive = tp + fn
    precision = tp / predicted_positive if predicted_positive else None
    recall = tp / target_positive if target_positive else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    union = tp + fp + fn
    iou = tp / union if union else None
    total = tp + fp + tn + fn
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "predicted_mask_ratio": predicted_positive / total if total else 0.0,
        "ground_truth_mask_ratio": target_positive / total if total else 0.0,
    }


def compare_reproduced_metrics(
    reproduced: dict[str, float],
    existing: dict[str, float],
    tolerance: float = METRIC_REPRODUCTION_TOLERANCE,
) -> tuple[bool, dict[str, float]]:
    differences = {
        key: abs(float(value) - float(existing[key]))
        for key, value in reproduced.items()
    }
    return all(value <= tolerance for value in differences.values()), differences


def _finite_summary(values: Sequence[float | None]) -> dict[str, float | None]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def stage_metric_tables(
    stage_masks: np.ndarray,
    targets: np.ndarray,
    kinds: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return stage aggregates, flat stage rows, and per-frame rows."""

    masks = np.asarray(stage_masks, dtype=bool)
    targets = np.asarray(targets, dtype=bool)
    if masks.ndim != 4 or masks.shape[1] != STAGE_COUNT:
        raise ValueError("stage_masks must have shape [frames, 8, H, W]")
    if masks.shape[0] != targets.shape[0] or masks.shape[2:] != targets.shape[1:]:
        raise ValueError("stage masks and targets do not align")
    if len(kinds) != masks.shape[0]:
        raise ValueError("kinds length does not match stage masks")
    patched_indices = [index for index, kind in enumerate(kinds) if kind == "transient"]
    clean_indices = [index for index, kind in enumerate(kinds) if kind == "clean"]
    if len(patched_indices) != 8 or len(clean_indices) != 8:
        raise ValueError("attribution requires exactly 8 clean and 8 transient frames")

    output: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = [
        {"frame_index": index, "kind": kinds[index]} for index in range(len(kinds))
    ]
    for stage_index, stage_name in enumerate(CVTR_STAGE_NAMES):
        patched_predictions = masks[patched_indices, stage_index]
        patched_targets = targets[patched_indices]
        micro_counts = confusion_counts(patched_predictions, patched_targets)
        micro = metrics_from_counts(micro_counts)
        frame_metrics: list[dict[str, Any]] = []
        for frame_index in patched_indices:
            item = metrics_from_counts(
                confusion_counts(masks[frame_index, stage_index], targets[frame_index])
            )
            frame_metrics.append(item)
            for key in ("tp", "fp", "fn"):
                per_frame_rows[frame_index][f"{stage_name}_{key}"] = item[key]

        clean_predictions = masks[clean_indices, stage_index]
        clean_positive = int(np.count_nonzero(clean_predictions))
        clean_pixels = int(clean_predictions.size)
        clean_fprs = [
            float(masks[index, stage_index].mean()) for index in clean_indices
        ]
        for frame_index, clean_fpr in zip(clean_indices, clean_fprs):
            per_frame_rows[frame_index][f"{stage_name}_fp"] = int(
                masks[frame_index, stage_index].sum()
            )
            per_frame_rows[frame_index][f"{stage_name}_clean_fpr"] = clean_fpr
        clean = {
            "micro_fpr": clean_positive / clean_pixels,
            "predicted_positive_pixels": clean_positive,
            "per_frame_fpr": clean_fprs,
            "macro": _finite_summary(clean_fprs),
        }
        macro = {
            key: _finite_summary([item[key] for item in frame_metrics])
            for key in ("precision", "recall", "f1", "iou")
        }
        output[stage_name] = {
            "patched_micro": micro,
            "patched_macro": macro,
            "clean": clean,
        }
        row: dict[str, Any] = {
            "stage_index": stage_index,
            "stage": stage_name,
            **{f"micro_{key}": value for key, value in micro.items()},
            "clean_micro_fpr": clean["micro_fpr"],
            "clean_macro_mean_fpr": clean["macro"]["mean"],
            "clean_macro_median_fpr": clean["macro"]["median"],
            "clean_max_fpr": clean["macro"]["max"],
        }
        for metric_name, summary in macro.items():
            for summary_name, value in summary.items():
                row[f"macro_{metric_name}_{summary_name}"] = value
        flat_rows.append(row)
    return output, flat_rows, per_frame_rows


def stage_transition_statistics(stage_metrics: dict[str, Any]) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    epsilon = EXPECTED_CONFIG.epsilon
    for index in range(1, STAGE_COUNT):
        previous_name = CVTR_STAGE_NAMES[index - 1]
        current_name = CVTR_STAGE_NAMES[index]
        previous = stage_metrics[previous_name]["patched_micro"]
        current = stage_metrics[current_name]["patched_micro"]
        previous_precision = previous["precision"]
        current_precision = current["precision"]
        previous_recall = previous["recall"]
        current_recall = current["recall"]
        transitions.append(
            {
                "from_stage": previous_name,
                "to_stage": current_name,
                "tp_retention": current["tp"] / (previous["tp"] + epsilon),
                "fp_retention": current["fp"] / (previous["fp"] + epsilon),
                "delta_tp_removed": previous["tp"] - current["tp"],
                "delta_fp_removed": previous["fp"] - current["fp"],
                "delta_precision": (
                    None
                    if previous_precision is None or current_precision is None
                    else current_precision - previous_precision
                ),
                "delta_recall": (
                    None
                    if previous_recall is None or current_recall is None
                    else current_recall - previous_recall
                ),
            }
        )

    def largest(field: str) -> str:
        eligible = [row for row in transitions if row[field] is not None]
        return max(eligible, key=lambda row: row[field])["to_stage"]

    clean_fprs = {
        name: stage_metrics[name]["clean"]["micro_fpr"] for name in CVTR_STAGE_NAMES
    }
    return {
        "transitions": transitions,
        "largest_FP_reduction_stage": largest("delta_fp_removed"),
        "largest_TP_loss_stage": largest("delta_tp_removed"),
        "largest_precision_gain_stage": largest("delta_precision"),
        "largest_recall_loss_stage": min(
            (row for row in transitions if row["delta_recall"] is not None),
            key=lambda row: row["delta_recall"],
        )["to_stage"],
        "highest_clean_FPR_stage": max(clean_fprs, key=clean_fprs.get),
    }


def distribution_summary(
    values: np.ndarray,
    *,
    threshold: float | None = None,
    percentiles: Sequence[int] = (10, 25, 50, 75, 90, 95, 99),
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        output: dict[str, Any] = {
            "count": 0,
            "mean": None,
            "std": None,
            "max": None,
        }
        output.update({f"p{p}": None for p in percentiles})
        if threshold is not None:
            output["fraction_above_threshold"] = None
        return output
    output = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "max": float(array.max()),
    }
    output.update(
        {f"p{p}": float(np.percentile(array, p)) for p in percentiles}
    )
    if threshold is not None:
        output["fraction_above_threshold"] = float(np.mean(array > threshold))
    return output


def auroc_rank(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute AUROC using average ranks, with no external statistics package."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    finite = np.isfinite(scores)
    scores = scores[finite]
    labels = labels[finite]
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (
        positive * negative
    )


def sobel_gradient(image: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Return a replicate-padded Sobel magnitude for RGB, gray, or depth arrays."""

    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[-1] != 3:
            raise ValueError("RGB input must have three channels")
        array = (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        )
    if array.ndim != 2:
        raise ValueError("Sobel input must be [H, W] or [H, W, 3]")
    array = array.astype(np.float32, copy=True)
    valid_mask = np.isfinite(array) if valid is None else np.asarray(valid, dtype=bool)
    if valid_mask.shape != array.shape:
        raise ValueError("valid mask must match Sobel input")
    finite_values = array[valid_mask & np.isfinite(array)]
    fill = float(np.median(finite_values)) if finite_values.size else 0.0
    array[~valid_mask | ~np.isfinite(array)] = fill
    tensor = torch.from_numpy(array)[None, None]
    tensor = F.pad(tensor, (1, 1, 1, 1), mode="replicate")
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )[None, None]
    kernel_y = kernel_x.transpose(-1, -2)
    gradient_x = F.conv2d(tensor, kernel_x)
    gradient_y = F.conv2d(tensor, kernel_y)
    magnitude = torch.sqrt(gradient_x.square() + gradient_y.square())[0, 0]
    result = magnitude.numpy()
    result[~valid_mask] = 0.0
    return result


def top_fraction_mask(
    values: np.ndarray, valid: np.ndarray, fraction: float = 0.10
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.shape != valid.shape or not 0.0 < fraction < 1.0:
        raise ValueError("invalid top-fraction inputs")
    output = np.zeros_like(valid)
    if not valid.any():
        return output
    threshold = float(np.quantile(values[valid], 1.0 - fraction))
    output[valid] = values[valid] >= threshold
    return output


def patch_boundary_and_interior(
    mask: np.ndarray, boundary_width: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or boundary_width <= 0:
        raise ValueError("patch mask must be [H, W] and boundary width positive")
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    inverse = 1.0 - tensor
    dilated_inverse = F.max_pool2d(
        inverse,
        kernel_size=2 * boundary_width + 1,
        stride=1,
        padding=boundary_width,
    )
    interior = (tensor > 0.5) & (dilated_inverse < 0.5)
    interior_array = interior[0, 0].numpy()
    return mask & ~interior_array, interior_array


def first_failure_map(stage_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(stage_masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] != STAGE_COUNT:
        raise ValueError("stage masks must have shape [8, H, W]")
    removal = np.full(masks.shape[1:], STAGE_COUNT, dtype=np.int8)
    removal[~masks[0]] = 0
    unresolved = masks[0].copy()
    for stage_index in range(1, STAGE_COUNT):
        removed_here = unresolved & ~masks[stage_index]
        removal[removed_here] = stage_index
        unresolved &= masks[stage_index]
    removal[unresolved] = STAGE_COUNT
    return removal


def count_first_failures(stage_masks: np.ndarray, final_fn: np.ndarray) -> dict[str, int]:
    removal = first_failure_map(stage_masks)
    final_fn = np.asarray(final_fn, dtype=bool)
    if final_fn.shape != removal.shape:
        raise ValueError("final FN mask does not match stage masks")
    names = list(CVTR_STAGE_NAMES) + ["S8_final_survivor"]
    return {
        names[index]: int(np.count_nonzero(final_fn & (removal == index)))
        for index in range(STAGE_COUNT + 1)
    }


def bilinear_contract_audit() -> dict[str, Any]:
    image = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]
    )
    pixels = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [3.0, 2.0],
            [1.5, 1.0],
            [0.25, 0.75],
            [-2.0, -2.0],
            [5.0, 4.0],
        ]
    )
    expected = torch.tensor([0.0, 3.0, 20.0, 23.0, 11.5, 7.75, 0.0, 0.0])
    actual = bilinear_sample(image, pixels)
    max_error = float((actual - expected).abs().max().item())
    return {
        "status": "PASS" if max_error <= 1e-6 else "SAMPLING_COORDINATE_CONTRACT_FAIL",
        "max_absolute_error": max_error,
        "align_corners": True,
        "padding_mode": "zeros",
        "coordinate_order": "x_y",
        "image_order": "H_W",
    }


def camera_inverse_error(camtoworld: torch.Tensor) -> float:
    identity = torch.eye(4, dtype=torch.float64)
    matrix = camtoworld.detach().to(dtype=torch.float64, device="cpu")
    worldtocamera = torch.linalg.inv(matrix)
    return float((matrix @ worldtocamera - identity).abs().max().item())


def same_view_roundtrip(
    view: ViewEvidence,
    *,
    grid_stride: int = 32,
    alpha_min: float = 0.5,
) -> np.ndarray:
    height, width = view.depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(0, height, grid_stride, device=view.depth.device),
        torch.arange(0, width, grid_stride, device=view.depth.device),
        indexing="ij",
    )
    valid = (
        torch.isfinite(view.depth[yy, xx])
        & (view.depth[yy, xx] > 0)
        & torch.isfinite(view.alpha[yy, xx])
        & (view.alpha[yy, xx] >= alpha_min)
    )
    pixels = torch.stack([xx[valid], yy[valid]], dim=-1).to(view.depth.dtype)
    if pixels.numel() == 0:
        return np.empty(0, dtype=np.float64)
    depths = view.depth[yy[valid], xx[valid]]
    world = unproject_camera_z(pixels, depths, view.K, view.camtoworld)
    projected, _ = project_camera_z(world, view.K, view.camtoworld)
    return torch.linalg.vector_norm(projected - pixels, dim=-1).cpu().numpy()


def run_cpu_self_test() -> dict[str, Any]:
    source = ViewEvidence(
        image_name="source",
        residual=torch.zeros(7, 7),
        depth=torch.full((7, 7), 2.0),
        alpha=torch.ones(7, 7),
        K=torch.tensor([[4.0, 0.0, 3.0], [0.0, 4.0, 3.0], [0.0, 0.0, 1.0]]),
        camtoworld=torch.eye(4),
    )
    source.residual[2:5, 2:5] = 1.0
    neighbors = []
    for index in range(3):
        neighbors.append(
            ViewEvidence(
                image_name=f"neighbor_{index}",
                residual=torch.zeros(7, 7),
                depth=torch.full((7, 7), 2.0),
                alpha=torch.ones(7, 7),
                K=source.K.clone(),
                camtoworld=torch.eye(4),
            )
        )
    with torch.inference_mode():
        trace = compute_cvtr_stage_trace(source, neighbors, 0.5)
        production = compute_cross_view_transient(source, neighbors, 0.5)
        final = spatial_filter_and_cap(
            production.raw_transient_mask, source.residual
        )
    return {
        "status": "PASS" if torch.equal(trace.final_mask, final) else "FAIL",
        "final_positive_pixels": int(trace.final_mask.sum().item()),
        "sampling": bilinear_contract_audit(),
        "auroc": auroc_rank(np.array([0.0, 0.2, 0.8, 1.0]), np.array([0, 0, 1, 1])),
        "optimizer_step_count": 0,
        "backward_call_count": 0,
    }


def checkpoint_audit(path: Path) -> tuple[dict[str, Any], str]:
    before_sha = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ContractFailure("CHECKPOINT_CONTRACT_FAIL", "checkpoint is not a dictionary")
    keys = sorted(checkpoint)
    step = int(checkpoint.get("step", -1))
    splats = checkpoint.get("splats")
    finite = isinstance(splats, dict) and all(
        isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
        for value in splats.values()
    )
    audit = {
        "path": str(path),
        "sha256_before": before_sha,
        "step": step,
        "keys": keys,
        "splat_keys": sorted(splats) if isinstance(splats, dict) else None,
        "all_splat_tensors_finite": finite,
        "status": "PASS",
    }
    if step != 9999 or keys != ["splats", "step"] or not finite:
        audit["status"] = "CHECKPOINT_CONTRACT_FAIL"
    del checkpoint
    return audit, before_sha


def load_existing_evidence(
    existing_output: Path,
) -> tuple[dict[str, ViewEvidence], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    cache_manifest = _read_json(existing_output / "render_cache_manifest.json")
    metadata = cache_manifest.get("views")
    if not isinstance(metadata, list) or not metadata:
        raise ContractFailure("EXISTING_EVIDENCE_FAIL", "render cache manifest is empty")
    records: dict[str, ViewEvidence] = {}
    for item in metadata:
        record = _load_cache(existing_output / item["cache_path"], item)
        if record.image_name in records:
            raise ContractFailure(
                "IMAGE_CAMERA_MAPPING_FAIL", f"duplicate cache image {record.image_name}"
            )
        records[record.image_name] = record
    existing_manifest = _read_json(existing_output / "manifest.json")
    existing_metrics = _read_json(existing_output / "metrics.json")
    return records, metadata, existing_manifest, existing_metrics


def audit_derived_and_mapping(
    *,
    data_dir: Path,
    gsplat_dir: Path,
    derived_dir: Path,
    derived_manifest: dict[str, Any],
    cache_records: dict[str, ViewEvidence],
    cache_metadata: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    entries = derived_manifest.get("entries")
    if not isinstance(entries, list):
        raise ContractFailure("DERIVED_DATA_ALIGNMENT_FAIL", "derived manifest has no entries")
    kinds = [entry.get("kind") for entry in entries]
    file_count = sum(1 for path in derived_dir.rglob("*") if path.is_file())
    derived_audit: dict[str, Any] = {
        "manifest_entry_count": len(entries),
        "clean_count": kinds.count("clean"),
        "transient_count": kinds.count("transient"),
        "file_count": file_count,
        "expected_file_count": 1 + 2 * len(entries),
        "frames": [],
        "status": "PASS",
    }
    failures: list[str] = []
    if len(entries) != 16 or kinds != ["clean"] * 8 + ["transient"] * 8:
        failures.append("manifest must contain 8 clean then 8 transient frames")
    if file_count != 1 + 2 * len(entries):
        failures.append("derived file count differs from manifest contract")

    Parser, Dataset = _load_colmap_classes(gsplat_dir)
    parser = Parser(data_dir=str(data_dir), factor=4, normalize=True, test_every=8)
    dataset = Dataset(parser, split="train", val_every=0)
    parser_name_to_global = {
        name: index for index, name in enumerate(parser.image_names)
    }
    dataset_name_to_local: dict[str, int] = {}
    for local_index, global_value in enumerate(dataset.indices):
        global_index = int(global_value)
        dataset_name_to_local[parser.image_names[global_index]] = local_index
    metadata_by_name = {item["image_name"]: item for item in cache_metadata}
    mapping_rows: list[dict[str, Any]] = []
    frame_data: dict[str, dict[str, Any]] = {}
    mapping_failures: list[str] = []

    for entry in entries:
        name = entry.get("image_name")
        if name not in parser_name_to_global or name not in dataset_name_to_local:
            mapping_failures.append(f"{name}: missing from parser train split")
            continue
        if name not in cache_records or name not in metadata_by_name:
            mapping_failures.append(f"{name}: missing from frozen render cache")
            continue
        local_index = dataset_name_to_local[name]
        global_index = parser_name_to_global[name]
        item = dataset[local_index]
        if int(dataset.indices[local_index]) != global_index:
            mapping_failures.append(f"{name}: dataset/parser index mismatch")
            continue
        parser_rgb = (
            item["image"].clamp(0, 255).to(torch.uint8).cpu().numpy().copy()
        )
        image_path = derived_dir / entry["derived_image"]
        mask_path = derived_dir / entry["ground_truth_mask"]
        with Image.open(image_path) as image:
            derived_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(mask_path) as image:
            mask_values = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        unique_values = sorted(np.unique(mask_values).tolist())
        gt = mask_values == 255
        same_shape = parser_rgb.shape[:2] == derived_rgb.shape[:2] == gt.shape
        max_outside_difference = None
        max_clean_difference = None
        area_ratio = float(gt.mean())
        manifest_ratio = float(entry.get("target_area_ratio", -1.0))
        frame_failures: list[str] = []
        if not same_shape:
            frame_failures.append("image/mask/parser shape mismatch")
        else:
            difference = (
                np.abs(derived_rgb.astype(np.float32) - parser_rgb.astype(np.float32))
                / 255.0
            )
            if entry["kind"] == "clean":
                max_clean_difference = float(difference.max())
                if gt.any():
                    frame_failures.append("clean GT mask is nonzero")
                if max_clean_difference > DERIVED_PIXEL_TOLERANCE:
                    frame_failures.append("clean image differs from parser target")
            else:
                if not gt.any():
                    frame_failures.append("transient GT mask is empty")
                outside = ~gt
                max_outside_difference = (
                    float(difference[outside].max()) if outside.any() else 0.0
                )
                if max_outside_difference > DERIVED_PIXEL_TOLERANCE:
                    frame_failures.append("patched image differs outside GT patch")
        if not set(unique_values).issubset({0, 255}):
            frame_failures.append(f"mask values are not binary: {unique_values}")
        if not math.isclose(area_ratio, manifest_ratio, abs_tol=1e-12):
            frame_failures.append("GT area ratio differs from manifest")

        cache = cache_records[name]
        K = item["K"].to(torch.float64)
        camtoworld = item["camtoworld"].to(torch.float64)
        K_error = float((K - cache.K.to(torch.float64)).abs().max().item())
        pose_error = float(
            (camtoworld - cache.camtoworld.to(torch.float64)).abs().max().item()
        )
        cache_shape = tuple(cache.residual.shape)
        if cache_shape != tuple(parser_rgb.shape[:2]):
            mapping_failures.append(f"{name}: cache/parser shape mismatch")
        if K_error > 1e-5 or pose_error > 1e-5:
            mapping_failures.append(f"{name}: cache camera matrices differ from dataset")
        camera_id = int(parser.camera_ids[global_index])
        mapping_rows.append(
            {
                "image_name": name,
                "kind": entry["kind"],
                "manifest_index": int(entry["index"]),
                "dataset_local_index": local_index,
                "parser_global_index": global_index,
                "camera_id": camera_id,
                "height": int(parser_rgb.shape[0]),
                "width": int(parser_rgb.shape[1]),
                "cache_height": int(cache_shape[0]),
                "cache_width": int(cache_shape[1]),
                "K_max_error": K_error,
                "camtoworld_max_error": pose_error,
                "K": K.tolist(),
                "camtoworld": camtoworld.tolist(),
            }
        )
        frame_audit = {
            "image_name": name,
            "kind": entry["kind"],
            "shape": list(gt.shape),
            "mask_values": unique_values,
            "gt_positive_pixels": int(gt.sum()),
            "area_ratio": area_ratio,
            "manifest_area_ratio": manifest_ratio,
            "max_clean_pixel_difference": max_clean_difference,
            "max_outside_patch_pixel_difference": max_outside_difference,
            "failures": frame_failures,
        }
        derived_audit["frames"].append(frame_audit)
        failures.extend(f"{name}: {message}" for message in frame_failures)
        frame_data[name] = {
            "entry": entry,
            "parser_rgb": parser_rgb,
            "derived_rgb": derived_rgb,
            "gt": gt,
            "K": K,
            "camtoworld": camtoworld,
            "dataset_local_index": local_index,
            "parser_global_index": global_index,
            "camera_id": camera_id,
        }

    if failures:
        derived_audit["status"] = "DERIVED_DATA_ALIGNMENT_FAIL"
        derived_audit["failures"] = failures
    mapping_audit = {
        "status": "PASS" if not mapping_failures and len(mapping_rows) == 16 else "IMAGE_CAMERA_MAPPING_FAIL",
        "frame_count": len(mapping_rows),
        "frames": mapping_rows,
        "failures": mapping_failures,
    }
    return derived_audit, mapping_audit, mapping_rows, frame_data


def audit_camera_and_roundtrip(
    selected_names: Sequence[str], records: dict[str, ViewEvidence]
) -> tuple[dict[str, Any], dict[str, Any]]:
    camera_rows: list[dict[str, Any]] = []
    all_errors: list[np.ndarray] = []
    for name in selected_names:
        view = records[name]
        inverse_error = camera_inverse_error(view.camtoworld)
        camera_rows.append({"image_name": name, "max_inverse_error": inverse_error})
        all_errors.append(same_view_roundtrip(view))
    camera_max = max(row["max_inverse_error"] for row in camera_rows)
    camera = {
        "status": "PASS" if camera_max <= 1e-5 else "CAMERA_TRANSFORM_CONTRACT_FAIL",
        "limit": 1e-5,
        "maximum_error": camera_max,
        "frames": camera_rows,
    }
    errors = np.concatenate(all_errors) if all_errors else np.empty(0)
    if errors.size:
        roundtrip = {
            "sample_count": int(errors.size),
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
            "max": float(errors.max()),
        }
    else:
        roundtrip = {
            "sample_count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    roundtrip["limit_p95_pixels"] = ROUNDTRIP_P95_LIMIT
    roundtrip["status"] = (
        "PASS"
        if errors.size and roundtrip["p95"] <= ROUNDTRIP_P95_LIMIT
        else "PROJECTION_ROUNDTRIP_FAIL"
    )
    return camera, roundtrip


def audit_depth(
    selected_names: Sequence[str], records: dict[str, ViewEvidence]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failure = False
    for name in selected_names:
        depth = records[name].depth.detach().cpu().numpy().astype(np.float64)
        alpha = records[name].alpha.detach().cpu().numpy().astype(np.float64)
        finite = np.isfinite(depth)
        positive = finite & (depth > 0)
        values = depth[positive]
        alpha_valid = np.isfinite(alpha) & (alpha >= EXPECTED_CONFIG.alpha_min)
        alpha_depth_valid = positive[alpha_valid]
        contract_ratio = float(alpha_depth_valid.mean()) if alpha_depth_valid.size else 0.0
        row = {
            "image_name": name,
            "depth_min": float(values.min()) if values.size else None,
            "depth_median": float(np.median(values)) if values.size else None,
            "depth_p95": float(np.percentile(values, 95)) if values.size else None,
            "depth_max": float(values.max()) if values.size else None,
            "finite_ratio": float(finite.mean()),
            "positive_ratio": float(positive.mean()),
            "alpha_valid_positive_finite_ratio": contract_ratio,
        }
        if values.size == 0 or contract_ratio < 1.0:
            failure = True
        rows.append(row)
    return {
        "status": "DEPTH_CONTRACT_FAIL" if failure else "PASS",
        "semantics": "gsplat RGB+ED expected camera-z depth",
        "frames": rows,
    }


@torch.inference_mode()
def single_gaussian_depth_contract(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"status": "NOT_RUN_CPU_DRY_RUN"}
    from gsplat.rendering import rasterization

    expected_z = 2.5
    renders, alpha, _ = rasterization(
        means=torch.tensor([[0.0, 0.0, expected_z]], device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        scales=torch.tensor([[0.15, 0.15, 0.15]], device=device),
        opacities=torch.tensor([0.99], device=device),
        colors=torch.tensor([[0.4, 0.5, 0.6]], device=device),
        viewmats=torch.eye(4, device=device)[None],
        Ks=torch.tensor(
            [[[80.0, 0.0, 16.0], [0.0, 80.0, 16.0], [0.0, 0.0, 1.0]]],
            device=device,
        ),
        width=33,
        height=33,
        packed=False,
        render_mode="RGB+ED",
    )
    valid = alpha[0, ..., 0] > 1e-3
    maximum_error = float(
        (renders[0, ..., 3][valid] - expected_z).abs().max().item()
    )
    return {
        "status": "PASS" if valid.any() and maximum_error <= 1e-4 else "DEPTH_CONTRACT_FAIL",
        "valid_pixel_count": int(valid.sum().item()),
        "expected_camera_z": expected_z,
        "maximum_absolute_error": maximum_error,
        "tolerance": 1e-4,
    }


def reproduce_traces(
    *,
    records: dict[str, ViewEvidence],
    selected_names: Sequence[str],
    existing_output: Path,
    existing_manifest: dict[str, Any],
    existing_metrics: dict[str, Any],
    derived_frame_data: dict[str, dict[str, Any]],
    config: CVTRConfig,
) -> tuple[list[CVTRStageTrace], dict[str, Any], dict[str, Any]]:
    names = list(records)
    residuals = [records[name].residual for name in names]
    alphas = [records[name].alpha for name in names]
    threshold, median, scale, sample_count = scene_residual_threshold(
        residuals, alphas, config
    )
    camtoworlds = torch.stack([records[name].camtoworld for name in names])
    neighbors = select_camera_neighbors(camtoworlds, names, config.neighbor_count)
    existing_neighbors = _read_json(existing_output / "neighbors.json")["neighbors"]
    if neighbors != existing_neighbors:
        raise ContractFailure(
            "IMAGE_CAMERA_MAPPING_FAIL", "recomputed camera neighbors differ from frozen neighbors"
        )
    existing_mask_paths = {
        item["image_name"]: existing_output / item["mask_path"]
        for item in existing_manifest["images"]
    }
    traces: list[CVTRStageTrace] = []
    reproduced_masks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    mask_mismatches: list[str] = []
    trace_mismatches: list[str] = []
    for name in selected_names:
        source = records[name]
        neighbor_records = [records[item] for item in neighbors[name]]
        trace = compute_cvtr_stage_trace(source, neighbor_records, threshold, config)
        production_cross = compute_cross_view_transient(
            source, neighbor_records, threshold, config
        )
        production_final = spatial_filter_and_cap(
            production_cross.raw_transient_mask, source.residual, config
        )
        original = load_binary_mask(existing_mask_paths[name])
        if not torch.equal(trace.final_mask.cpu(), production_final.cpu()):
            trace_mismatches.append(name)
        if not torch.equal(trace.final_mask.cpu(), original.cpu()):
            mask_mismatches.append(name)
        traces.append(trace)
        reproduced_masks.append(trace.final_mask.cpu())
        targets.append(torch.from_numpy(derived_frame_data[name]["gt"]))

    predicted = torch.stack(reproduced_masks)
    target = torch.stack(targets)
    tp = int((predicted & target).sum().item())
    fp = int((predicted & ~target).sum().item())
    fn = int((~predicted & target).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-15)
    iou = tp / max(tp + fp + fn, 1)
    clean_indices = [
        index
        for index, name in enumerate(selected_names)
        if derived_frame_data[name]["entry"]["kind"] == "clean"
    ]
    clean_fpr = float(predicted[clean_indices].float().mean().item())
    reproduced = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "clean_false_positive_rate": clean_fpr,
    }
    metrics_match, differences = compare_reproduced_metrics(
        reproduced,
        existing_metrics,
        METRIC_REPRODUCTION_TOLERANCE,
    )
    reproduction = {
        "status": (
            "PASS"
            if not trace_mismatches and not mask_mismatches and metrics_match
            else (
                "FINAL_MASK_REPRODUCTION_FAIL"
                if trace_mismatches or mask_mismatches
                else "FINAL_METRICS_REPRODUCTION_FAIL"
            )
        ),
        "trace_vs_production_mismatches": trace_mismatches,
        "trace_vs_original_mask_mismatches": mask_mismatches,
        "reproduced_metrics": reproduced,
        "existing_metrics": {
            key: existing_metrics[key] for key in reproduced
        },
        "absolute_differences": differences,
        "tolerance": METRIC_REPRODUCTION_TOLERANCE,
    }
    threshold_evidence = {
        "residual_median": median,
        "residual_scale": scale,
        "residual_threshold": threshold,
        "residual_sample_count_after_alpha_gate": sample_count,
        "residual_sample_stride": config.residual_sample_stride,
        "alpha_min": config.alpha_min,
    }
    return traces, reproduction, threshold_evidence


def _stack_stage_masks(traces: Sequence[CVTRStageTrace]) -> np.ndarray:
    return np.stack(
        [
            np.stack([mask.cpu().numpy().astype(bool) for mask in trace.stage_masks])
            for trace in traces
        ]
    )


def _stack_trace_field(
    traces: Sequence[CVTRStageTrace], field: str, dtype: np.dtype | None = None
) -> np.ndarray:
    array = np.stack(
        [getattr(trace, field).cpu().numpy() for trace in traces]
    )
    return array.astype(dtype, copy=False) if dtype is not None else array


def _histogram_0_to_3(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values)
    return {str(index): int(np.count_nonzero(values == index)) for index in range(4)}


def neighbor_distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "histogram": _histogram_0_to_3(array)}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "histogram": _histogram_0_to_3(array),
    }


def evidence_distribution_analysis(
    *,
    traces: Sequence[CVTRStageTrace],
    targets: np.ndarray,
    kinds: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    stage_masks = _stack_stage_masks(traces)
    residual = _stack_trace_field(traces, "rgb_residual", np.float32)
    inbounds = _stack_trace_field(
        traces, "inbounds_positive_neighbor_count", np.int16
    )
    alpha_counts = _stack_trace_field(
        traces, "neighbor_alpha_valid_count", np.int16
    )
    depth_counts = _stack_trace_field(
        traces, "depth_consistent_neighbor_count", np.int16
    )
    static_support = _stack_trace_field(traces, "static_support_ratio", np.float32)
    spatial_support = _stack_trace_field(traces, "spatial_support_score", np.float32)
    depth_errors = _stack_trace_field(
        traces, "depth_consistency_error_per_neighbor", np.float32
    )
    clean = np.asarray([kind == "clean" for kind in kinds])
    patched = ~clean
    residual_classes = {
        "clean_all_pixels": distribution_summary(
            residual[clean].reshape(-1), threshold=threshold
        ),
        "patched_gt_patch": distribution_summary(
            residual[patched][targets[patched]], threshold=threshold
        ),
        "patched_outside_patch": distribution_summary(
            residual[patched][~targets[patched]], threshold=threshold
        ),
    }
    patched_scores = residual[patched].reshape(-1)
    patched_labels = targets[patched].reshape(-1)
    residual_evidence = {
        "scene_threshold": threshold,
        "classes": residual_classes,
        "patch_vs_background_auroc": auroc_rank(patched_scores, patched_labels),
    }

    final = stage_masks[:, -1]
    true_positive = final & targets
    false_positive = final & ~targets
    false_negative = ~final & targets
    neighbor_evidence: dict[str, Any] = {}
    for label, selector in (
        ("final_tp", true_positive),
        ("final_fp", false_positive),
        ("final_fn", false_negative),
    ):
        neighbor_evidence[label] = {
            "inbounds_positive_neighbor_count": neighbor_distribution(inbounds[selector]),
            "neighbor_alpha_valid_count": neighbor_distribution(alpha_counts[selector]),
            "depth_consistent_neighbor_count": neighbor_distribution(depth_counts[selector]),
        }

    candidate = stage_masks[:, 0]
    depth_evidence: dict[str, Any] = {}
    for label, selector in (
        ("gt_positive", candidate & targets),
        ("gt_negative", candidate & ~targets),
    ):
        expanded = np.broadcast_to(selector[:, None], depth_errors.shape)
        values = depth_errors[expanded]
        summary = distribution_summary(values, percentiles=(50, 75, 90, 95))
        finite_values = values[np.isfinite(values)]
        summary["fraction_le_0_05"] = (
            float(np.mean(finite_values <= EXPECTED_CONFIG.depth_relative_tolerance))
            if finite_values.size
            else None
        )
        depth_evidence[label] = summary

    s4 = stage_masks[:, 4]
    static_scores: list[np.ndarray] = []
    static_labels: list[np.ndarray] = []
    static_evidence: dict[str, Any] = {}
    for label, selector in (
        ("gt_positive", s4 & targets),
        ("gt_negative", s4 & ~targets),
    ):
        values = static_support[selector]
        summary = distribution_summary(values, percentiles=(25, 50, 75))
        summary["fraction_ge_0_5"] = (
            float(np.mean(values >= EXPECTED_CONFIG.static_support_threshold))
            if values.size
            else None
        )
        static_evidence[label] = summary
        static_scores.append(values)
        static_labels.append(np.full(values.shape, label == "gt_positive", dtype=bool))
    static_evidence["auroc"] = auroc_rank(
        np.concatenate(static_scores), np.concatenate(static_labels)
    )

    s5 = stage_masks[:, 5]
    spatial_scores: list[np.ndarray] = []
    spatial_labels: list[np.ndarray] = []
    spatial_evidence: dict[str, Any] = {}
    for label, selector in (
        ("gt_positive", s5 & targets),
        ("gt_negative", s5 & ~targets),
    ):
        values = spatial_support[selector]
        summary = distribution_summary(values, percentiles=(25, 50, 75))
        summary["fraction_ge_5_over_9"] = (
            float(np.mean(values >= EXPECTED_CONFIG.patch_min_support))
            if values.size
            else None
        )
        spatial_evidence[label] = summary
        spatial_scores.append(values)
        spatial_labels.append(np.full(values.shape, label == "gt_positive", dtype=bool))
    spatial_evidence["auroc"] = auroc_rank(
        np.concatenate(spatial_scores), np.concatenate(spatial_labels)
    )
    return {
        "residual": residual_evidence,
        "neighbors": neighbor_evidence,
        "depth_consistency": depth_evidence,
        "static_support": static_evidence,
        "spatial_support": spatial_evidence,
    }


def location_attribution(
    *,
    traces: Sequence[CVTRStageTrace],
    targets: np.ndarray,
    parser_rgbs: np.ndarray,
    depths: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, np.ndarray]]]:
    stage_masks = _stack_stage_masks(traces)
    residual = _stack_trace_field(traces, "rgb_residual", np.float32)
    alpha = _stack_trace_field(traces, "current_alpha", np.float32)
    final = stage_masks[:, -1]
    fp_totals = {key: 0 for key in ("C1_rgb_edges", "C2_depth_edges", "C3_image_border", "C4_high_static_residual")}
    total_fp = 0
    explained_any = 0
    multi_category = 0
    fn_boundary = 0
    fn_interior = 0
    fn_first_stage = {name: 0 for name in CVTR_STAGE_NAMES}
    location_arrays: list[dict[str, np.ndarray]] = []

    for index, trace in enumerate(traces):
        gt = targets[index]
        fp = final[index] & ~gt
        fn = ~final[index] & gt
        valid_depth = np.isfinite(depths[index]) & (depths[index] > 0)
        rgb_gradient = sobel_gradient(parser_rgbs[index].astype(np.float32) / 255.0)
        depth_gradient = sobel_gradient(depths[index], valid_depth)
        c1 = top_fraction_mask(rgb_gradient, np.ones_like(gt, dtype=bool), 0.10)
        c2 = top_fraction_mask(depth_gradient, valid_depth, 0.10)
        height, width = gt.shape
        yy, xx = np.mgrid[:height, :width]
        c3 = (xx <= 8) | (yy <= 8) | (xx >= width - 9) | (yy >= height - 9)
        c4 = ~gt & (residual[index] > threshold)
        categories = [c1, c2, c3, c4]
        category_names = list(fp_totals)
        category_hits = np.stack([fp & category for category in categories])
        for category_name, hits in zip(category_names, category_hits):
            fp_totals[category_name] += int(hits.sum())
        hit_count = category_hits.sum(axis=0)
        total_fp += int(fp.sum())
        explained_any += int(np.count_nonzero(fp & (hit_count > 0)))
        multi_category += int(np.count_nonzero(fp & (hit_count > 1)))

        boundary, interior = patch_boundary_and_interior(gt)
        fn_boundary += int(np.count_nonzero(fn & boundary))
        fn_interior += int(np.count_nonzero(fn & interior))
        first_counts = count_first_failures(stage_masks[index], fn)
        for stage_name in CVTR_STAGE_NAMES:
            fn_first_stage[stage_name] += first_counts[stage_name]
        location_arrays.append(
            {
                "rgb_gradient": rgb_gradient,
                "depth_gradient": depth_gradient,
                "rgb_edges": c1,
                "depth_edges": c2,
                "image_border": c3,
                "high_static_residual": c4,
                "patch_boundary": boundary,
                "patch_interior": interior,
                "final_fp": fp,
                "final_fn": fn,
                "stage_removal": first_failure_map(stage_masks[index]),
                "low_alpha": alpha[index] < EXPECTED_CONFIG.alpha_min,
            }
        )

    fp = {
        "final_false_positive_pixels": total_fp,
        "category_counts": fp_totals,
        "category_coverage": {
            key: value / max(total_fp, 1) for key, value in fp_totals.items()
        },
        "unexplained_ratio": (total_fp - explained_any) / max(total_fp, 1),
        "multi_category_overlap_ratio": multi_category / max(total_fp, 1),
        "categories_are_nonexclusive": True,
    }
    fn_total = fn_boundary + fn_interior
    fn = {
        "final_false_negative_pixels": fn_total,
        "patch_boundary_fn": fn_boundary,
        "patch_interior_fn": fn_interior,
        "patch_boundary_fn_ratio": fn_boundary / max(fn_total, 1),
        "patch_interior_fn_ratio": fn_interior / max(fn_total, 1),
        "first_failure_stage_counts": fn_first_stage,
        "first_failure_stage_ratios": {
            key: value / max(fn_total, 1) for key, value in fn_first_stage.items()
        },
        "first_failure_categories": {
            "residual_threshold_failure": fn_first_stage[CVTR_STAGE_NAMES[0]],
            "low_current_alpha": fn_first_stage[CVTR_STAGE_NAMES[1]],
            "fewer_than_2_reprojectable_neighbors": fn_first_stage[CVTR_STAGE_NAMES[2]],
            "fewer_than_2_neighbor_alpha_valid": fn_first_stage[CVTR_STAGE_NAMES[3]],
            "depth_consistency_failure": fn_first_stage[CVTR_STAGE_NAMES[4]],
            "static_support_failure": fn_first_stage[CVTR_STAGE_NAMES[5]],
            "spatial_support_failure": fn_first_stage[CVTR_STAGE_NAMES[6]],
            "area_cap_deletion": fn_first_stage[CVTR_STAGE_NAMES[7]],
        },
        "each_fn_counted_once": sum(fn_first_stage.values()) == fn_total,
    }
    return fp, fn, location_arrays


def per_frame_analysis(
    *,
    names: Sequence[str],
    kinds: Sequence[str],
    traces: Sequence[CVTRStageTrace],
    targets: np.ndarray,
    threshold: float,
    base_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage_masks = _stack_stage_masks(traces)
    residual = _stack_trace_field(traces, "rgb_residual", np.float32)
    depth_errors = _stack_trace_field(
        traces, "depth_consistency_error_per_neighbor", np.float32
    )
    depth_counts = _stack_trace_field(
        traces, "depth_consistent_neighbor_count", np.int16
    )
    static_support = _stack_trace_field(traces, "static_support_ratio", np.float32)
    rows: list[dict[str, Any]] = []
    for index, (name, kind) in enumerate(zip(names, kinds)):
        final_counts = confusion_counts(stage_masks[index, -1], targets[index])
        final_metrics = metrics_from_counts(final_counts)
        production_candidate = traces[index].production_candidate_mask.cpu().numpy()
        supported = (
            depth_counts[index] >= EXPECTED_CONFIG.minimum_valid_neighbors
        )
        candidate_count = int(production_candidate.sum())
        mean_static_support = (
            float(static_support[index][production_candidate].mean())
            if candidate_count
            else None
        )
        row = dict(base_rows[index])
        row.update(
            {
                "image_name": name,
                "kind": kind,
                "final_precision": final_metrics["precision"] if kind == "transient" else None,
                "final_recall": final_metrics["recall"] if kind == "transient" else None,
                "final_f1": final_metrics["f1"] if kind == "transient" else None,
                "final_iou": final_metrics["iou"] if kind == "transient" else None,
                "clean_fpr": float(stage_masks[index, -1].mean()) if kind == "clean" else None,
                "predicted_mask_ratio": float(stage_masks[index, -1].mean()),
                "ground_truth_mask_ratio": float(targets[index].mean()),
                "valid_neighbor_ratio": (
                    float(supported[production_candidate].mean())
                    if candidate_count
                    else 1.0
                ),
                "mean_depth_consistency_error": (
                    float(np.nanmean(depth_errors[index]))
                    if np.isfinite(depth_errors[index]).any()
                    else None
                ),
                "mean_static_support": mean_static_support,
                "mean_residual": float(residual[index].mean()),
                "p95_residual": float(np.percentile(residual[index], 95)),
                "fraction_above_residual_threshold": float(
                    np.mean(residual[index] > threshold)
                ),
            }
        )
        rows.append(row)

    patched_rows = [row for row in rows if row["kind"] == "transient"]
    clean_rows = [row for row in rows if row["kind"] == "clean"]
    anomalies = {
        "lowest_precision_frames": [
            row["image_name"]
            for row in sorted(patched_rows, key=lambda row: row["final_precision"])[:2]
        ],
        "highest_clean_fpr_frames": [
            row["image_name"]
            for row in sorted(clean_rows, key=lambda row: row["clean_fpr"], reverse=True)[:2]
        ],
        "lowest_recall_frames": [
            row["image_name"]
            for row in sorted(patched_rows, key=lambda row: row["final_recall"])[:2]
        ],
        "largest_predicted_area_frames": [
            row["image_name"]
            for row in sorted(rows, key=lambda row: row["predicted_mask_ratio"], reverse=True)[:2]
        ],
    }
    return rows, anomalies


def _save_stage_outputs(
    *,
    output_dir: Path,
    names: Sequence[str],
    traces: Sequence[CVTRStageTrace],
    save_stage_arrays: bool,
) -> None:
    stage_masks = _stack_stage_masks(traces)
    stage_mask_dir = output_dir / "stage_masks"
    stage_mask_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, name in enumerate(names):
        safe_name = Path(name).name + ".png"
        for stage_index, stage_name in enumerate(CVTR_STAGE_NAMES):
            destination = stage_mask_dir / stage_name / safe_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                stage_masks[frame_index, stage_index].astype(np.uint8) * 255,
                mode="L",
            ).save(destination)
    if not save_stage_arrays:
        return
    arrays_dir = output_dir / "stage_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_dir / "cvtr_stage_trace.npz",
        image_names=np.asarray(names),
        stage_names=np.asarray(CVTR_STAGE_NAMES),
        stage_masks=stage_masks,
        rgb_residual=_stack_trace_field(traces, "rgb_residual", np.float32),
        current_alpha=_stack_trace_field(traces, "current_alpha", np.float32),
        inbounds_positive_neighbor_count=_stack_trace_field(
            traces, "inbounds_positive_neighbor_count", np.int8
        ),
        neighbor_alpha_valid_count=_stack_trace_field(
            traces, "neighbor_alpha_valid_count", np.int8
        ),
        depth_consistent_neighbor_count=_stack_trace_field(
            traces, "depth_consistent_neighbor_count", np.int8
        ),
        static_support_ratio=_stack_trace_field(
            traces, "static_support_ratio", np.float32
        ),
        spatial_support_score=_stack_trace_field(
            traces, "spatial_support_score", np.float32
        ),
        depth_consistency_error_per_neighbor=_stack_trace_field(
            traces, "depth_consistency_error_per_neighbor", np.float32
        ),
    )


def _confusion_overlay(rgb: np.ndarray, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    overlay = np.clip(base * 0.45, 0.0, 1.0)
    tp = predicted & target
    fp = predicted & ~target
    fn = ~predicted & target
    overlay[tp] = np.array([0.0, 1.0, 0.0])
    overlay[fp] = np.array([1.0, 0.0, 0.0])
    overlay[fn] = np.array([0.0, 0.25, 1.0])
    return overlay


def _colored_overlay(
    rgb: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    primary_color: tuple[float, float, float],
    secondary_color: tuple[float, float, float],
) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    output = np.clip(base * 0.55, 0.0, 1.0)
    output[secondary] = secondary_color
    output[primary] = primary_color
    return output


def _continuous_colormap(
    values: np.ndarray, *, vmin: float, vmax: float
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    normalized = np.nan_to_num(
        (array - vmin) / max(vmax - vmin, 1e-12), nan=0.0, posinf=1.0, neginf=0.0
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    stops = np.array(
        [
            [0.02, 0.02, 0.08],
            [0.18, 0.05, 0.45],
            [0.70, 0.10, 0.35],
            [0.95, 0.45, 0.08],
            [1.00, 0.95, 0.25],
        ],
        dtype=np.float32,
    )
    scaled = normalized * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, len(stops) - 1)
    weight = (scaled - lower)[..., None]
    rgb = stops[lower] * (1.0 - weight) + stops[upper] * weight
    return (rgb * 255.0).round().astype(np.uint8)


def _categorical_colormap(values: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [60, 60, 60],
            [31, 119, 180],
            [255, 127, 14],
            [44, 160, 44],
            [214, 39, 40],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [188, 189, 34],
        ],
        dtype=np.uint8,
    )
    indices = np.clip(np.asarray(values, dtype=np.int64), 0, len(palette) - 1)
    return palette[indices]


def _visual_panel(
    value: np.ndarray,
    title: str,
    *,
    kind: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    width: int = 360,
    image_height: int = 240,
) -> Image.Image:
    from PIL import ImageDraw

    title_height = 24
    colorbar_height = 28 if kind in {"continuous", "categorical"} else 8
    panel = Image.new("RGB", (width, title_height + image_height + colorbar_height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((5, 5), title, fill="black")
    if kind == "rgb":
        array = np.asarray(value, dtype=np.float32)
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        display = np.clip(array, 0, 255).astype(np.uint8)
        resample = Image.Resampling.BILINEAR
    elif kind == "binary":
        display = np.asarray(value, dtype=bool).astype(np.uint8) * 255
        display = np.repeat(display[..., None], 3, axis=-1)
        resample = Image.Resampling.NEAREST
    elif kind == "categorical":
        display = _categorical_colormap(value)
        resample = Image.Resampling.NEAREST
    else:
        display = _continuous_colormap(value, vmin=vmin, vmax=vmax)
        resample = Image.Resampling.BILINEAR
    image = Image.fromarray(display, mode="RGB").resize(
        (width, image_height), resample=resample
    )
    panel.paste(image, (0, title_height))
    if kind == "continuous":
        bar_values = np.linspace(vmin, vmax, width, dtype=np.float32)[None]
        bar = Image.fromarray(
            _continuous_colormap(bar_values, vmin=vmin, vmax=vmax), mode="RGB"
        ).resize((width, 12), Image.Resampling.BILINEAR)
        panel.paste(bar, (0, title_height + image_height))
        draw.text((2, title_height + image_height + 13), f"{vmin:.4g}", fill="black")
        right_label = f"{vmax:.4g}"
        draw.text((width - 45, title_height + image_height + 13), right_label, fill="black")
    elif kind == "categorical":
        bar_values = np.arange(9, dtype=np.int8)[None]
        bar = Image.fromarray(_categorical_colormap(bar_values), mode="RGB").resize(
            (width, 12), Image.Resampling.NEAREST
        )
        panel.paste(bar, (0, title_height + image_height))
        draw.text((2, title_height + image_height + 13), "0", fill="black")
        draw.text((width - 15, title_height + image_height + 13), "8", fill="black")
    return panel


def _compose_panel_grid(
    panels: Sequence[Image.Image], columns: int, background: str = "white"
) -> Image.Image:
    if not panels or columns <= 0:
        raise ValueError("panel grid requires panels and positive columns")
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * width, rows * height), background)
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % columns) * width, (index // columns) * height))
    return canvas


def save_visualizations(
    *,
    output_dir: Path,
    names: Sequence[str],
    traces: Sequence[CVTRStageTrace],
    targets: np.ndarray,
    parser_rgbs: np.ndarray,
    rendered_rgbs: np.ndarray,
    depths: np.ndarray,
    location_arrays: Sequence[dict[str, np.ndarray]],
) -> dict[str, float]:
    stage_masks = _stack_stage_masks(traces)
    residual = _stack_trace_field(traces, "rgb_residual", np.float32)
    alpha = _stack_trace_field(traces, "current_alpha", np.float32)
    valid_counts = _stack_trace_field(
        traces, "depth_consistent_neighbor_count", np.int16
    )
    static_support = _stack_trace_field(traces, "static_support_ratio", np.float32)
    spatial_support = _stack_trace_field(traces, "spatial_support_score", np.float32)
    rgb_gradients = np.stack([item["rgb_gradient"] for item in location_arrays])
    depth_gradients = np.stack([item["depth_gradient"] for item in location_arrays])
    rgb_gradient_vmax = float(np.percentile(rgb_gradients, 99))
    finite_depth_gradient = depth_gradients[np.isfinite(depth_gradients)]
    depth_gradient_vmax = (
        float(np.percentile(finite_depth_gradient, 99))
        if finite_depth_gradient.size
        else 1.0
    )
    rgb_gradient_vmax = max(rgb_gradient_vmax, 1e-6)
    depth_gradient_vmax = max(depth_gradient_vmax, 1e-6)
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, name in enumerate(names):
        safe_name = Path(name).name + ".png"
        panels = [
            _visual_panel(parser_rgbs[frame_index], "parser-exact input", kind="rgb"),
            _visual_panel(rendered_rgbs[frame_index], "B1 render", kind="rgb"),
            _visual_panel(
                residual[frame_index], "absolute RGB residual", kind="continuous", vmin=0, vmax=1
            ),
            _visual_panel(targets[frame_index], "GT transient mask", kind="binary"),
        ]
        panels.extend(
            _visual_panel(stage_masks[frame_index, offset], CVTR_STAGE_NAMES[offset], kind="binary")
            for offset in range(8)
        )
        panels.extend(
            [
                _visual_panel(
                    _confusion_overlay(
                        parser_rgbs[frame_index],
                        stage_masks[frame_index, -1],
                        targets[frame_index],
                    ),
                    "final TP green / FP red / FN blue",
                    kind="rgb",
                ),
                _visual_panel(
                    rgb_gradients[frame_index],
                    "RGB gradient",
                    kind="continuous",
                    vmin=0,
                    vmax=rgb_gradient_vmax,
                ),
                _visual_panel(
                    depth_gradients[frame_index],
                    "depth gradient",
                    kind="continuous",
                    vmin=0,
                    vmax=depth_gradient_vmax,
                ),
                _visual_panel(
                    valid_counts[frame_index],
                    "depth-consistent neighbor count",
                    kind="continuous",
                    vmin=0,
                    vmax=EXPECTED_CONFIG.neighbor_count,
                ),
                _visual_panel(
                    static_support[frame_index],
                    "static-support score",
                    kind="continuous",
                    vmin=0,
                    vmax=1,
                ),
                _visual_panel(
                    spatial_support[frame_index],
                    "spatial-support score",
                    kind="continuous",
                    vmin=0,
                    vmax=1,
                ),
                _visual_panel(
                    alpha[frame_index],
                    "current alpha",
                    kind="continuous",
                    vmin=0,
                    vmax=1,
                ),
                _visual_panel(
                    location_arrays[frame_index]["stage_removal"],
                    "first stage-removal map (8=survivor)",
                    kind="categorical",
                ),
            ]
        )
        _compose_panel_grid(panels, columns=4).save(visual_dir / safe_name)

        fp = location_arrays[frame_index]["final_fp"]
        fn = location_arrays[frame_index]["final_fn"]
        focus_panels = [
            _visual_panel(
                _colored_overlay(
                    parser_rgbs[frame_index],
                    fp,
                    location_arrays[frame_index]["rgb_edges"],
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                ),
                "final FP red + RGB edges yellow",
                kind="rgb",
            ),
            _visual_panel(
                _colored_overlay(
                    parser_rgbs[frame_index],
                    fp,
                    location_arrays[frame_index]["depth_edges"],
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                ),
                "final FP red + depth edges yellow",
                kind="rgb",
            ),
            _visual_panel(
                _colored_overlay(
                    parser_rgbs[frame_index],
                    fn & location_arrays[frame_index]["patch_boundary"],
                    fn & location_arrays[frame_index]["patch_interior"],
                    (0.0, 0.25, 1.0),
                    (0.0, 1.0, 1.0),
                ),
                "FN boundary blue + interior cyan",
                kind="rgb",
            ),
            _visual_panel(
                location_arrays[frame_index]["stage_removal"],
                "stage-removal map",
                kind="categorical",
            ),
        ]
        _compose_panel_grid(focus_panels, columns=4).save(
            visual_dir / f"focus_{safe_name}"
        )
    return {
        "rgb_gradient_vmin": 0.0,
        "rgb_gradient_vmax": rgb_gradient_vmax,
        "depth_gradient_vmin": 0.0,
        "depth_gradient_vmax": depth_gradient_vmax,
    }


def _dominant_items(values: dict[str, float | int], count: int = 2) -> list[str]:
    return [
        key
        for key, _ in sorted(values.items(), key=lambda item: item[1], reverse=True)[:count]
    ]


def write_reports(
    *,
    markdown_path: Path,
    json_path: Path,
    execution_status: str,
    contract_status: str,
    summary: dict[str, Any],
    stage_metrics: dict[str, Any] | None,
    contract_audit: dict[str, Any],
) -> None:
    payload = {
        "execution_status": execution_status,
        "contract_status": contract_status,
        "facts": summary,
        "contract_audit": contract_audit,
    }
    _write_json(json_path, payload)
    lines = [
        "# PURI-GS Phase 3A-F：CVTR 零训练失败归因",
        "",
        f"执行状态：`{execution_status}`",
        "",
        f"契约状态：`{contract_status}`",
        "",
        "## 零训练边界",
        "",
        "```text",
        "analysis_only = true",
        "optimizer_step_count = 0",
        "backward_call_count = 0",
        "checkpoint_write_count = 0",
        "continuation_run_count = 0",
        "```",
        "",
        "## 事实摘要",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: `{value}`")
    if stage_metrics is not None:
        lines.extend(
            [
                "",
                "## 分阶段 patched micro 与 clean FPR",
                "",
                "| 阶段 | Precision | Recall | F1 | Clean FPR |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for stage_name in CVTR_STAGE_NAMES:
            item = stage_metrics[stage_name]
            micro = item["patched_micro"]
            clean_fpr = item["clean"]["micro_fpr"]

            def fmt(value: float | None) -> str:
                return "null" if value is None else f"{value:.8f}"

            lines.append(
                f"| {stage_name} | {fmt(micro['precision'])} | "
                f"{fmt(micro['recall'])} | {fmt(micro['f1'])} | {clean_fpr:.8f} |"
            )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本报告只提供执行契约和事实归因，不作是否重跑、概念失败或下一算法路线判断。",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def package_evidence(
    *,
    output_dir: Path,
    report_paths: Sequence[Path],
    archive_path: Path,
    checksum_path: Path,
) -> str:
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name)
        for report_path in report_paths:
            archive.add(report_path, arcname=f"reports/{report_path.name}")
    digest = sha256_file(archive_path)
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    return digest


def _key_evidence_hashes(existing_output: Path) -> dict[str, str]:
    names = (
        "metrics.json",
        "manifest.json",
        "neighbors.json",
        "render_cache_manifest.json",
        "scene_statistics.json",
        "per_frame.csv",
    )
    return {
        name: sha256_file(existing_output / name)
        for name in names
        if (existing_output / name).is_file()
    }


def _finalize_contract_stop(
    *,
    output_dir: Path,
    provenance: dict[str, Any],
    contract_audit: dict[str, Any],
    markdown_report: Path,
    json_report: Path,
    archive_path: Path,
    checksum_path: Path,
) -> int:
    provenance["execution_status"] = "FAIL_EXECUTION"
    provenance["contract_status"] = "POTENTIAL_IMPLEMENTATION_ERROR"
    _write_json(output_dir / "provenance.json", provenance)
    _write_json(output_dir / "contract" / "contract_audit.json", contract_audit)
    summary = {
        "failure_statuses": contract_audit.get("failure_statuses", []),
        "analysis_stopped_before_attribution": True,
    }
    write_reports(
        markdown_path=markdown_report,
        json_path=json_report,
        execution_status="FAIL_EXECUTION",
        contract_status="POTENTIAL_IMPLEMENTATION_ERROR",
        summary=summary,
        stage_metrics=None,
        contract_audit=contract_audit,
    )
    package_evidence(
        output_dir=output_dir,
        report_paths=[markdown_report, json_report],
        archive_path=archive_path,
        checksum_path=checksum_path,
    )
    print(json.dumps(summary, indent=2))
    return 2


def run_analysis(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.expanduser().resolve()
    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    derived_dir = args.derived_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    existing_output = args.existing_output.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    device = torch.device(args.device)

    current_branch = _git_output("branch", "--show-current")
    current_commit = repository_commit()
    commit_short = current_commit[:7]
    expected_output_name = f"cvtr_failure_attribution_{commit_short}"
    if output_dir.name != expected_output_name:
        raise ContractFailure(
            "OUTPUT_CONTRACT_FAIL",
            f"output directory must be named {expected_output_name}",
        )
    markdown_report = PROJECT_ROOT / "reports" / "PHASE_3A_CVTR_FAILURE_ATTRIBUTION.md"
    json_report = PROJECT_ROOT / "reports" / "phase_3a_cvtr_failure_attribution.json"
    archive_path = output_dir.parent / f"puri_gs_cvtr_failure_attribution_{commit_short}.tar.gz"
    checksum_path = output_dir.parent / f"puri_gs_cvtr_failure_attribution_{commit_short}.sha256"
    ensure_outputs_absent(
        [output_dir, markdown_report, json_report, archive_path, checksum_path]
    )
    required_files = [
        checkpoint_path,
        config_path,
        derived_dir / "manifest.json",
        existing_output / "metrics.json",
        existing_output / "manifest.json",
        existing_output / "render_cache_manifest.json",
        existing_output / "neighbors.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ContractFailure("INPUT_PATH_CONTRACT_FAIL", f"missing inputs: {missing}")
    source_image_dir = data_dir / "images"
    parser_factor_dir = data_dir / "images_4_png"
    source_image_count = sum(1 for path in source_image_dir.rglob("*") if path.is_file())
    factor_image_count = sum(1 for path in parser_factor_dir.rglob("*") if path.is_file())
    if source_image_count == 0 or factor_image_count != source_image_count:
        raise ContractFailure(
            "DERIVED_DATA_ALIGNMENT_FAIL",
            "parser-exact images_4_png must already exist and match the source image count",
        )
    require_empty_output(output_dir)
    for name in ("contract", "metrics", "evidence", "stage_masks", "stage_arrays", "visualizations"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    command_text = shlex.join([sys.executable, *sys.argv]) + "\n"
    (output_dir / "run_command.txt").write_text(command_text, encoding="utf-8")

    checkpoint_info, checkpoint_sha_before = checkpoint_audit(checkpoint_path)
    derived_sha_before, derived_file_count_before = _tree_digest(derived_dir)
    existing_hashes_before = _key_evidence_hashes(existing_output)
    derived_manifest_path = derived_dir / "manifest.json"
    existing_metrics_path = existing_output / "metrics.json"
    existing_manifest_path = existing_output / "manifest.json"
    config_sha = sha256_file(config_path)
    derived_manifest_sha = sha256_file(derived_manifest_path)
    existing_metrics_sha = sha256_file(existing_metrics_path)
    existing_manifest_sha = sha256_file(existing_manifest_path)
    profile = load_experiment_config(config_path)
    config = _cvtr_config(profile)
    config_status = "PASS" if config.to_dict() == EXPECTED_CONFIG.to_dict() else "CONFIG_FROZEN_CONTRACT_FAIL"
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ContractFailure("DEPTH_CONTRACT_FAIL", "CUDA device was requested but is unavailable")

    records, cache_metadata, existing_manifest, existing_metrics = load_existing_evidence(
        existing_output
    )
    source_commit = existing_manifest.get("source_commit")
    source_sha = existing_manifest.get("source_checkpoint_sha256")
    checkpoint_source_status = (
        "PASS" if source_sha == checkpoint_sha_before else "CHECKPOINT_CONTRACT_FAIL"
    )
    derived_manifest = _load_derived_manifest(derived_dir)
    selected_names = [entry["image_name"] for entry in derived_manifest["entries"]]
    kinds = [entry["kind"] for entry in derived_manifest["entries"]]

    provenance: dict[str, Any] = {
        "current_branch": current_branch,
        "current_commit": current_commit,
        "source_commit": source_commit,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha_before,
        "checkpoint_step": checkpoint_info["step"],
        "checkpoint_keys": checkpoint_info["keys"],
        "gsplat_version": importlib.metadata.version("gsplat"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "derived_dir": str(derived_dir),
        "derived_manifest_sha256": derived_manifest_sha,
        "existing_metrics_path": str(existing_metrics_path),
        "existing_metrics_sha256": existing_metrics_sha,
        "existing_manifest_sha256": existing_manifest_sha,
        "existing_tool_commit": existing_manifest.get("tool_commit"),
        "analysis_only": True,
        "optimizer_step_count": 0,
        "backward_call_count": 0,
        "checkpoint_write_count": 0,
        "continuation_run_count": 0,
        "cvtr_config": config.to_dict(),
        "run_command": command_text.strip(),
    }

    with torch.inference_mode():
        derived_audit, mapping_audit, mapping_rows, frame_data = (
            audit_derived_and_mapping(
                data_dir=data_dir,
                gsplat_dir=gsplat_dir,
                derived_dir=derived_dir,
                derived_manifest=derived_manifest,
                cache_records=records,
                cache_metadata=cache_metadata,
            )
        )
        camera_audit, roundtrip_audit = audit_camera_and_roundtrip(
            selected_names, records
        )
        sampling_audit = bilinear_contract_audit()
        depth_audit = audit_depth(selected_names, records)
        gaussian_depth_audit = single_gaussian_depth_contract(device)

    _write_json(output_dir / "contract" / "derived_data_audit.json", derived_audit)
    _write_json(
        output_dir / "contract" / "image_camera_manifest_audit.json",
        mapping_audit,
    )
    _write_json(output_dir / "contract" / "camera_transform_audit.json", camera_audit)
    _write_json(output_dir / "contract" / "projection_roundtrip_audit.json", roundtrip_audit)
    _write_json(output_dir / "contract" / "bilinear_sampling_audit.json", sampling_audit)
    _write_json(output_dir / "contract" / "depth_audit.json", depth_audit)
    _write_json(output_dir / "contract" / "single_gaussian_depth_audit.json", gaussian_depth_audit)

    contract_components = {
        "branch": "PASS" if current_branch == "dev" else "BRANCH_CONTRACT_FAIL",
        "config": config_status,
        "checkpoint": checkpoint_info["status"],
        "checkpoint_source_sha": checkpoint_source_status,
        "derived_data": derived_audit["status"],
        "image_camera_mapping": mapping_audit["status"],
        "camera_transform": camera_audit["status"],
        "projection_roundtrip": roundtrip_audit["status"],
        "sampling_coordinates": sampling_audit["status"],
        "depth_arrays": depth_audit["status"],
        "single_gaussian_depth": gaussian_depth_audit["status"],
    }
    failure_statuses = [status for status in contract_components.values() if status != "PASS"]
    contract_audit: dict[str, Any] = {
        "components": contract_components,
        "failure_statuses": failure_statuses,
        "status": "IMPLEMENTATION_CONTRACT_PASS" if not failure_statuses else "POTENTIAL_IMPLEMENTATION_ERROR",
    }
    _write_json(output_dir / "contract" / "contract_audit.json", contract_audit)
    if failure_statuses:
        checkpoint_sha_after = sha256_file(checkpoint_path)
        provenance["checkpoint_sha256_after"] = checkpoint_sha_after
        provenance["checkpoint_unchanged"] = checkpoint_sha_after == checkpoint_sha_before
        return _finalize_contract_stop(
            output_dir=output_dir,
            provenance=provenance,
            contract_audit=contract_audit,
            markdown_report=markdown_report,
            json_report=json_report,
            archive_path=archive_path,
            checksum_path=checksum_path,
        )

    with torch.inference_mode():
        traces, reproduction, threshold_evidence = reproduce_traces(
            records=records,
            selected_names=selected_names,
            existing_output=existing_output,
            existing_manifest=existing_manifest,
            existing_metrics=existing_metrics,
            derived_frame_data=frame_data,
            config=config,
        )
    _write_json(output_dir / "contract" / "final_reproduction_audit.json", reproduction)
    contract_audit["components"]["final_reproduction"] = reproduction["status"]
    if reproduction["status"] != "PASS":
        contract_audit["failure_statuses"].append(reproduction["status"])
        contract_audit["status"] = "POTENTIAL_IMPLEMENTATION_ERROR"
        _write_json(output_dir / "contract" / "contract_audit.json", contract_audit)
        checkpoint_sha_after = sha256_file(checkpoint_path)
        provenance["checkpoint_sha256_after"] = checkpoint_sha_after
        provenance["checkpoint_unchanged"] = checkpoint_sha_after == checkpoint_sha_before
        return _finalize_contract_stop(
            output_dir=output_dir,
            provenance=provenance,
            contract_audit=contract_audit,
            markdown_report=markdown_report,
            json_report=json_report,
            archive_path=archive_path,
            checksum_path=checksum_path,
        )

    targets = np.stack([frame_data[name]["gt"] for name in selected_names])
    parser_rgbs = np.stack([frame_data[name]["parser_rgb"] for name in selected_names])
    depths = np.stack([records[name].depth.cpu().numpy() for name in selected_names])
    stage_masks = _stack_stage_masks(traces)
    stage_metrics, stage_rows, base_per_frame_rows = stage_metric_tables(
        stage_masks, targets, kinds
    )
    transitions = stage_transition_statistics(stage_metrics)
    threshold = float(threshold_evidence["residual_threshold"])
    stride = config.residual_sample_stride
    threshold_evidence["residual_sample_count_before_alpha_gate"] = int(
        sum(
            torch.isfinite(record.residual[::stride, ::stride]).sum().item()
            for record in records.values()
        )
    )
    distributions = evidence_distribution_analysis(
        traces=traces,
        targets=targets,
        kinds=kinds,
        threshold=threshold,
    )
    fp_attribution, fn_attribution, location_arrays = location_attribution(
        traces=traces,
        targets=targets,
        parser_rgbs=parser_rgbs,
        depths=depths,
        threshold=threshold,
    )
    per_frame_rows, anomalies = per_frame_analysis(
        names=selected_names,
        kinds=kinds,
        traces=traces,
        targets=targets,
        threshold=threshold,
        base_rows=base_per_frame_rows,
    )

    _write_json(output_dir / "metrics" / "stage_metrics.json", stage_metrics)
    _write_csv(output_dir / "metrics" / "stage_metrics.csv", stage_rows)
    _write_json(
        output_dir / "metrics" / "per_frame_metrics.json",
        {"frames": per_frame_rows, "anomalies": anomalies},
    )
    _write_csv(output_dir / "metrics" / "per_frame_metrics.csv", per_frame_rows)
    _write_json(output_dir / "metrics" / "stage_transition.json", transitions)
    _write_json(output_dir / "evidence" / "scene_threshold.json", threshold_evidence)
    _write_json(
        output_dir / "evidence" / "residual_distributions.json",
        distributions["residual"],
    )
    _write_json(
        output_dir / "evidence" / "neighbor_distributions.json",
        distributions["neighbors"],
    )
    _write_json(
        output_dir / "evidence" / "depth_consistency_distributions.json",
        distributions["depth_consistency"],
    )
    _write_json(
        output_dir / "evidence" / "static_support_distributions.json",
        distributions["static_support"],
    )
    _write_json(
        output_dir / "evidence" / "spatial_support_distributions.json",
        distributions["spatial_support"],
    )
    _write_json(output_dir / "evidence" / "fp_attribution.json", fp_attribution)
    _write_json(output_dir / "evidence" / "fn_attribution.json", fn_attribution)
    _write_json(output_dir / "evidence" / "per_frame_anomalies.json", anomalies)
    _save_stage_outputs(
        output_dir=output_dir,
        names=selected_names,
        traces=traces,
        save_stage_arrays=args.save_stage_arrays,
    )

    metadata_by_name = {item["image_name"]: item for item in cache_metadata}
    rendered_rgbs: list[np.ndarray] = []
    for name in selected_names:
        with np.load(
            existing_output / metadata_by_name[name]["cache_path"], allow_pickle=False
        ) as cache:
            rendered_rgbs.append(cache["rendered_rgb"].astype(np.float32))
    visualization_scales = None
    if args.save_visualizations:
        visualization_scales = save_visualizations(
            output_dir=output_dir,
            names=selected_names,
            traces=traces,
            targets=targets,
            parser_rgbs=parser_rgbs,
            rendered_rgbs=np.stack(rendered_rgbs),
            depths=depths,
            location_arrays=location_arrays,
        )
        _write_json(
            output_dir / "evidence" / "visualization_scales.json",
            visualization_scales,
        )

    checkpoint_sha_after = sha256_file(checkpoint_path)
    derived_sha_after, derived_file_count_after = _tree_digest(derived_dir)
    existing_hashes_after = _key_evidence_hashes(existing_output)
    unchanged_components = {
        "checkpoint": checkpoint_sha_after == checkpoint_sha_before,
        "derived_tree": derived_sha_after == derived_sha_before
        and derived_file_count_after == derived_file_count_before,
        "existing_key_evidence": existing_hashes_after == existing_hashes_before,
    }
    if not all(unchanged_components.values()):
        contract_audit["status"] = "POTENTIAL_IMPLEMENTATION_ERROR"
        if not unchanged_components["checkpoint"]:
            contract_audit["failure_statuses"].append("CHECKPOINT_MODIFICATION_DETECTED")
        if not unchanged_components["derived_tree"]:
            contract_audit["failure_statuses"].append("DERIVED_DATA_ALIGNMENT_FAIL")
        if not unchanged_components["existing_key_evidence"]:
            contract_audit["failure_statuses"].append("READ_ONLY_EVIDENCE_MODIFICATION_DETECTED")
        contract_audit["read_only_components"] = unchanged_components
        _write_json(output_dir / "contract" / "contract_audit.json", contract_audit)
        provenance["checkpoint_sha256_after"] = checkpoint_sha_after
        provenance["checkpoint_unchanged"] = unchanged_components["checkpoint"]
        return _finalize_contract_stop(
            output_dir=output_dir,
            provenance=provenance,
            contract_audit=contract_audit,
            markdown_report=markdown_report,
            json_report=json_report,
            archive_path=archive_path,
            checksum_path=checksum_path,
        )
    contract_audit["read_only_components"] = unchanged_components
    contract_audit["status"] = "IMPLEMENTATION_CONTRACT_PASS"
    _write_json(output_dir / "contract" / "contract_audit.json", contract_audit)

    raw = stage_metrics[CVTR_STAGE_NAMES[0]]["patched_micro"]
    final_patched = stage_metrics[CVTR_STAGE_NAMES[-1]]["patched_micro"]
    final_reproduced = reproduction["reproduced_metrics"]
    final_clean_fpr = stage_metrics[CVTR_STAGE_NAMES[-1]]["clean"]["micro_fpr"]
    summary = {
        "raw_candidate_precision": raw["precision"],
        "raw_candidate_recall": raw["recall"],
        "final_precision": final_reproduced["precision"],
        "final_recall": final_reproduced["recall"],
        "final_clean_FPR": final_clean_fpr,
        "final_patched_precision": final_patched["precision"],
        "final_patched_recall": final_patched["recall"],
        "largest_FP_reduction_stage": transitions["largest_FP_reduction_stage"],
        "largest_TP_loss_stage": transitions["largest_TP_loss_stage"],
        "largest_precision_gain_stage": transitions["largest_precision_gain_stage"],
        "largest_recall_loss_stage": transitions["largest_recall_loss_stage"],
        "dominant_FP_categories": _dominant_items(fp_attribution["category_coverage"]),
        "dominant_FN_first_failure_stages": _dominant_items(
            fn_attribution["first_failure_stage_counts"]
        ),
        "residual_patch_vs_background_auroc": distributions["residual"]["patch_vs_background_auroc"],
        "static_support_auroc": distributions["static_support"]["auroc"],
        "spatial_support_auroc": distributions["spatial_support"]["auroc"],
        "per_frame_anomalies": anomalies,
    }
    provenance.update(
        {
            "checkpoint_sha256_after": checkpoint_sha_after,
            "checkpoint_unchanged": True,
            "derived_tree_sha256_before": derived_sha_before,
            "derived_tree_sha256_after": derived_sha_after,
            "existing_evidence_hashes_before": existing_hashes_before,
            "existing_evidence_hashes_after": existing_hashes_after,
            "execution_status": "PASS_EXECUTION",
            "contract_status": "IMPLEMENTATION_CONTRACT_PASS",
        }
    )
    _write_json(output_dir / "provenance.json", provenance)
    write_reports(
        markdown_path=markdown_report,
        json_path=json_report,
        execution_status="PASS_EXECUTION",
        contract_status="IMPLEMENTATION_CONTRACT_PASS",
        summary=summary,
        stage_metrics=stage_metrics,
        contract_audit=contract_audit,
    )
    archive_sha = package_evidence(
        output_dir=output_dir,
        report_paths=[markdown_report, json_report],
        archive_path=archive_path,
        checksum_path=checksum_path,
    )
    result = {
        "execution_status": "PASS_EXECUTION",
        "contract_status": "IMPLEMENTATION_CONTRACT_PASS",
        "output_dir": str(output_dir),
        "markdown_report": str(markdown_report),
        "json_report": str(json_report),
        "evidence_archive": str(archive_path),
        "evidence_archive_sha256": archive_sha,
        "summary": summary,
        "optimizer_step_count": 0,
        "backward_call_count": 0,
        "checkpoint_write_count": 0,
        "continuation_run_count": 0,
    }
    print(json.dumps(result, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--gsplat-dir", type=Path)
    parser.add_argument("--derived-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--existing-output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-stage-arrays", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--self-test-cpu", action="store_true")
    args = parser.parse_args()
    if not args.self_test_cpu:
        missing = [
            name
            for name in (
                "data_dir",
                "gsplat_dir",
                "derived_dir",
                "checkpoint",
                "config",
                "existing_output",
                "output_dir",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"analysis requires: {', '.join(missing)}")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test_cpu:
        result = run_cpu_self_test()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    try:
        return run_analysis(args)
    except ContractFailure as error:
        print(json.dumps({"status": error.status, "error": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
