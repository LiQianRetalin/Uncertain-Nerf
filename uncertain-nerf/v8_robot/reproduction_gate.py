"""Acceptance gate for the pinned gsplat v1.5.3 Garden reproduction."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _check(
    checks: list[dict[str, Any]],
    name: str,
    actual: float,
    operator: str,
    target: float,
) -> None:
    if operator == ">=":
        passed = actual >= target
    elif operator == "<=":
        passed = actual <= target
    else:
        raise ValueError(f"unsupported operator: {operator}")
    checks.append(
        {
            "name": name,
            "actual": actual,
            "operator": operator,
            "target": target,
            "passed": passed,
        }
    )


def evaluate_reproduction(
    stats: Mapping[str, Any], limits: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare one official trainer stats JSON with the Garden reference band."""

    if limits.get("protocol") != "puri-gsplat153-garden-reproduction-1":
        raise ValueError("unsupported reproduction protocol")
    acceptance = limits.get("acceptance")
    reference = limits.get("reference")
    if not isinstance(acceptance, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("limits must contain acceptance and reference objects")

    psnr = _number(stats, "psnr")
    ssim = _number(stats, "ssim")
    lpips = _number(stats, "lpips")
    num_gs = _number(stats, "num_GS")
    ellipse_time = _number(stats, "ellipse_time")
    if not num_gs.is_integer():
        raise ValueError("num_GS must be an integer")
    if ellipse_time < 0:
        raise ValueError("ellipse_time must be non-negative")

    checks: list[dict[str, Any]] = []
    _check(checks, "psnr", psnr, ">=", _number(acceptance, "psnr_min"))
    _check(checks, "ssim", ssim, ">=", _number(acceptance, "ssim_min"))
    _check(checks, "lpips", lpips, "<=", _number(acceptance, "lpips_max"))
    _check(checks, "num_GS_min", num_gs, ">=", _number(acceptance, "num_GS_min"))
    _check(checks, "num_GS_max", num_gs, "<=", _number(acceptance, "num_GS_max"))
    failed = [check["name"] for check in checks if not check["passed"]]

    return {
        "protocol": limits["protocol"],
        "candidate": limits.get("candidate"),
        "dataset": limits.get("dataset"),
        "seed": limits.get("seed"),
        "steps": limits.get("steps"),
        "decision": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "checks": checks,
        "reference": dict(reference),
        "measured": {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "num_GS": int(num_gs),
            "ellipse_time": ellipse_time,
        },
    }
