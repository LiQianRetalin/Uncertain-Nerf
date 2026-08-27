"""Small, dependency-free admission gate for the robot Gaussian baseline.

The gate deliberately covers only the short screen.  Geometry/localization and
uncertainty calibration are full-evaluation gates after a backbone is admitted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def _nested(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Missing required metric: {path}")
        value = value[key]
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Metric {path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric {path} must be finite")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Metric {path} must be a boolean")
    return value


def _integer(value: Any, path: str) -> int:
    number = _finite_number(value, path)
    if not number.is_integer():
        raise ValueError(f"Metric {path} must be an integer")
    return int(number)


def _check(
    checks: list[dict[str, Any]],
    name: str,
    actual: Any,
    operator: str,
    target: Any,
) -> None:
    if operator == ">=":
        passed = actual >= target
    elif operator == "<=":
        passed = actual <= target
    elif operator == "==":
        passed = actual == target
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    checks.append(
        {
            "name": name,
            "actual": actual,
            "operator": operator,
            "target": target,
            "passed": bool(passed),
        }
    )


def evaluate_candidate(
    metrics: Mapping[str, Any], limits: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate one candidate and return a machine-readable decision."""

    if _nested(metrics, "protocol") != _nested(limits, "protocol"):
        raise ValueError("Metrics and limits use different protocol versions")

    checks: list[dict[str, Any]] = []
    _check(checks, "input_mode", _nested(metrics, "input_mode"), "==", "rgb_only")
    _check(
        checks,
        "external_pose_api",
        _boolean(
            _nested(metrics, "integration.external_pose_api"),
            "integration.external_pose_api",
        ),
        "==",
        True,
    )
    _check(
        checks,
        "single_raster_pass",
        _boolean(
            _nested(metrics, "integration.single_raster_pass"),
            "integration.single_raster_pass",
        ),
        "==",
        True,
    )
    _check(
        checks,
        "rendering_mlp",
        _boolean(
            _nested(metrics, "integration.rendering_mlp"),
            "integration.rendering_mlp",
        ),
        "==",
        False,
    )

    numeric_checks = (
        ("clean.psnr_db", ">=", "clean_psnr_db_min"),
        ("robust.static_psnr_db", ">=", "robust_static_psnr_db_min"),
        ("robust.floater_pixel_rate", "<=", "robust_floater_pixel_rate_max"),
        ("runtime.render_fps", ">=", "render_fps_min"),
        ("runtime.render_p95_ms", "<=", "render_p95_ms_max"),
        ("runtime.map_update_p95_ms", "<=", "map_update_p95_ms_max"),
        ("runtime.peak_vram_mb", "<=", "peak_vram_mb_max"),
        ("runtime.model_size_mb", "<=", "model_size_mb_max"),
        ("integration.sh_degree", "<=", "sh_degree_max"),
    )
    for metric_path, operator, limit_name in numeric_checks:
        actual = _finite_number(_nested(metrics, metric_path), metric_path)
        target = _finite_number(_nested(limits, limit_name), limit_name)
        _check(checks, metric_path, actual, operator, target)

    width = _integer(_nested(metrics, "runtime.width"), "runtime.width")
    height = _integer(_nested(metrics, "runtime.height"), "runtime.height")
    _check(checks, "runtime.width", width, "==", int(_nested(limits, "runtime_width")))
    _check(checks, "runtime.height", height, "==", int(_nested(limits, "runtime_height")))

    # SSIM and LPIPS are mandatory records even though the first admission gate
    # does not tune thresholds around Fern alone.
    for metric_path in ("clean.ssim", "clean.lpips"):
        _finite_number(_nested(metrics, metric_path), metric_path)

    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "protocol": _nested(metrics, "protocol"),
        "candidate": _nested(metrics, "candidate"),
        "decision": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "checks": checks,
    }


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value
