"""Dependency-free loading and validation for PURI-GS experiment profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_RESPONSIBILITY_FIELDS = {
    "enabled",
    "start_step",
    "threshold",
    "min_weight",
    "pool_size",
    "epsilon",
}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML profile without adding a YAML dependency."""

    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load experiment config {config_path}: {error}") from error
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if config.get("profile") not in {"b0", "b1", "a1"}:
        raise ValueError("profile must be one of b0, b1, a1")
    if config.get("gsplat_version") != "1.5.3":
        raise ValueError("gsplat_version must remain pinned to 1.5.3")
    if config.get("seed") != 42:
        raise ValueError("the pinned gsplat trainer seed must be 42")

    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping")
    for field in ("data_factor", "test_every", "sh_degree", "ssim_lambda"):
        if field not in training:
            raise ValueError(f"training.{field} is required")
    if training["data_factor"] <= 0 or training["test_every"] <= 0:
        raise ValueError("data_factor and test_every must be positive")
    if not 0 <= training["ssim_lambda"] <= 1:
        raise ValueError("ssim_lambda must be in [0, 1]")

    strategy = config.get("strategy")
    if not isinstance(strategy, dict) or strategy.get("type") != "default":
        raise ValueError("strategy.type must be default")
    if not isinstance(strategy.get("absgrad"), bool):
        raise ValueError("strategy.absgrad must be boolean")
    if not isinstance(strategy.get("grow_grad2d"), (int, float)):
        raise ValueError("strategy.grow_grad2d must be numeric")

    responsibility = config.get("responsibility")
    if not isinstance(responsibility, dict):
        raise ValueError("responsibility must be a mapping")
    missing = REQUIRED_RESPONSIBILITY_FIELDS.difference(responsibility)
    if missing:
        raise ValueError(f"missing responsibility fields: {sorted(missing)}")
    if config["profile"] in {"b0", "b1"} and responsibility["enabled"]:
        raise ValueError("B0/B1 must disable responsibility")
    if config["profile"] == "a1" and not responsibility["enabled"]:
        raise ValueError("A1 must enable responsibility")


def trainer_method_args(config: dict[str, Any]) -> list[str]:
    """Translate only the method-specific profile fields to gsplat CLI flags."""

    strategy = config["strategy"]
    responsibility = config["responsibility"]
    args = ["--strategy.grow_grad2d", str(strategy["grow_grad2d"])]
    if strategy["absgrad"]:
        args.append("--strategy.absgrad")
    if responsibility["enabled"]:
        args.extend(
            [
                "--responsibility_enabled",
                "--responsibility_start_step",
                str(responsibility["start_step"]),
                "--responsibility_threshold",
                str(responsibility["threshold"]),
                "--responsibility_min_weight",
                str(responsibility["min_weight"]),
                "--responsibility_pool_size",
                str(responsibility["pool_size"]),
                "--responsibility_epsilon",
                str(responsibility["epsilon"]),
            ]
        )
    return args
