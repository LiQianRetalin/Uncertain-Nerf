#!/usr/bin/env python3
"""Compare short B1 runs by fixed eval metrics and exact Gaussian count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict:
    value = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or "step" not in value or "splats" not in value:
        raise ValueError(f"not a standard Gaussian checkpoint: {path}")
    return value


def _load_metrics(path: Path) -> dict[str, float]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"B1 metrics must be a JSON object: {path}")
    required = ("psnr", "ssim", "lpips")
    if any(name not in value for name in required):
        raise ValueError(f"B1 metrics miss one of {required}: {path}")
    return {name: float(value[name]) for name in required}


def compare(
    reference: dict,
    candidate: dict,
    *,
    reference_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> dict:
    if reference["step"] != candidate["step"]:
        raise ValueError("B1 checkpoint steps differ")
    reference_splats = reference["splats"]
    candidate_splats = candidate["splats"]
    if set(reference_splats) != set(candidate_splats):
        raise ValueError("B1 Gaussian state keys differ")

    maximum_absolute_difference = 0.0
    tensor_differences = []
    for name in sorted(reference_splats):
        left = reference_splats[name]
        right = candidate_splats[name]
        if left.shape != right.shape:
            raise ValueError(f"B1 Gaussian tensor shape differs: {name}")
            continue
        if left.numel():
            difference = float((left - right).abs().max().item())
            maximum_absolute_difference = max(maximum_absolute_difference, difference)
        if not torch.equal(left, right):
            tensor_differences.append(name)

    gaussian_count = int(reference_splats["means"].shape[0])
    candidate_count = int(candidate_splats["means"].shape[0])
    tolerances = {"psnr": 0.05, "ssim": 0.001, "lpips": 0.002}
    metric_deltas = {
        name: candidate_metrics[name] - reference_metrics[name]
        for name in tolerances
    }
    metric_pass = {
        name: abs(metric_deltas[name]) <= tolerance
        for name, tolerance in tolerances.items()
    }
    passed = gaussian_count == candidate_count and all(metric_pass.values())
    return {
        "status": "B1_REGRESSION_PASS" if passed else "B1_REGRESSION_FAIL",
        "step": int(reference["step"]),
        "reference_gaussian_count": gaussian_count,
        "candidate_gaussian_count": candidate_count,
        "metric_tolerances": tolerances,
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": metric_deltas,
        "metric_pass": metric_pass,
        "maximum_absolute_difference": maximum_absolute_difference,
        "tensor_differences_diagnostic_only": tensor_differences,
        "tensor_equality_required": False,
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite comparison output: {output}")
    result = compare(
        _load(args.reference),
        _load(args.candidate),
        reference_metrics=_load_metrics(args.reference_metrics),
        candidate_metrics=_load_metrics(args.candidate_metrics),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "B1_REGRESSION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
