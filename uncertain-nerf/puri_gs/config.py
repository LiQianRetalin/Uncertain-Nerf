"""Dependency-free loading and validation for PURI-GS experiment profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from puri_gs.cvtr import CVTRConfig


REQUIRED_RESPONSIBILITY_FIELDS = {
    "enabled",
    "start_step",
    "threshold",
    "min_weight",
    "pool_size",
    "epsilon",
}

REQUIRED_CVTR_FIELDS = set(CVTRConfig.__dataclass_fields__) | {"enabled"}

CAUSAL_PROFILE = "ru_causal"
RU_PART_PROFILE = "ru_part"
RU_PART_MODES = ("parent", "noop", "current")

PAPER_CONTROLS = ("ru_align", "ru_tar")
TAR_REFINE_WINDOWS = [
    {"start": 10000, "stop": 20000, "every": 100},
    {"start": 20000, "stop": 24000, "every": 200},
]


def parse_refine_windows(value: Any, total_steps: int = 30000) -> tuple:
    """Resolve optional ordered, non-overlapping [start, stop) windows."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError("refine_windows must be a non-empty list")
    windows = []
    for window in value:
        if not isinstance(window, dict) or set(window) != {"start", "stop", "every"}:
            raise ValueError("refine window requires start, stop and every")
        start, stop, every = (window[key] for key in ("start", "stop", "every"))
        if any(type(v) is not int for v in (start, stop, every)):
            raise ValueError("refine window values must be integers")
        if not 0 <= start < stop <= total_steps or every <= 0:
            raise ValueError("invalid refine window or window beyond total_steps")
        if windows and start < windows[-1][1]:
            raise ValueError("refine windows overlap or are out of order")
        windows.append((start, stop, every))
    return tuple(windows)

CAUSAL_COMMON_FIXED_FIELDS = {
    "method": "puri_gs_ru_causal_2x2",
    "total_steps": 30000,
    "absgrad": True,
    "grow_grad2d": 0.0006,
    "ssim_lambda": 0.2,
    "seed": 42,
    "sh_degree": 3,
}

MASK_FIXED_FIELDS = {
    "mask_begin_step": 500,
    "mask_threshold": 0.25,
    "mask_erode_kernel": 7,
    "mask_learning_rate": 0.001,
    "mask_hidden_dim": 16,
    "dino_model": "dinov2_vits14_reg",
    "dino_feature_dim": 384,
    "dino_coarse_grid": 16,
    "dino_fine_grid": 36,
    "dino_coarse_input": 224,
    "dino_fine_input": 504,
    "residual_hist_bins": 10000,
    "residual_hist_momentum": 0.95,
    "residual_lower_quantile": 0.60,
    "residual_upper_quantile": 0.80,
    "mask_cos_weight": 0.5,
    "mask_residual_weight": 0.5,
    "mask_static_prior_weight": 2.0,
    "mask_static_prior_decay": 2000,
    "bootstrap_switch_step": 20000,
    "mask_pause_after_reset": 300,
}

DELAYED_TOPOLOGY_FIXED_FIELDS = {
    "densify_start_step": 10000,
    "densify_stop_step": 20000,
    "densify_every": 100,
    "opacity_reset_start_step": 15000,
    "opacity_reset_every": 3000,
}

RU_FIXED_FIELDS = {
    "method": "puri_gs_ru",
    "total_steps": 30000,
    "absgrad": True,
    "grow_grad2d": 0.0006,
    "mask_enabled": True,
    "mask_begin_step": 500,
    "mask_threshold": 0.25,
    "mask_erode_kernel": 7,
    "mask_learning_rate": 0.001,
    "mask_hidden_dim": 16,
    "dino_model": "dinov2_vits14_reg",
    "dino_feature_dim": 384,
    "dino_coarse_grid": 16,
    "dino_fine_grid": 36,
    "dino_coarse_input": 224,
    "dino_fine_input": 504,
    "residual_hist_bins": 10000,
    "residual_hist_momentum": 0.95,
    "residual_lower_quantile": 0.60,
    "residual_upper_quantile": 0.80,
    "mask_cos_weight": 0.5,
    "mask_residual_weight": 0.5,
    "mask_static_prior_weight": 2.0,
    "mask_static_prior_decay": 2000,
    "bootstrap_switch_step": 20000,
    "densify_start_step": 10000,
    "densify_stop_step": 20000,
    "densify_every": 100,
    "opacity_reset_start_step": 15000,
    "opacity_reset_every": 3000,
    "mask_pause_after_reset": 300,
    "ssim_lambda": 0.2,
    "seed": 42,
    "sh_degree": 3,
}

RU_PART_FIXED_FIELDS = {
    **RU_FIXED_FIELDS,
    "method": "ru_part",
    "rho0": 0.5,
    "gaussian_hard_cap": 2312002,
    "track_pose_neighbors": 4,
    "track_minimum_cameras": 3,
    "track_tolerance_patches": 1.0,
    "birth_radius_patches": 0.5,
    "birth_initial_opacity": 0.1,
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
    if "v3_screening" in config:
        mode = config["v3_screening"]
        expected = {
            **RU_FIXED_FIELDS, "schema_version": 1, "profile": "ru",
            "base_profile": "b1", "gsplat_version": "1.5.3",
            "delayed_densification": True,
            "training": {"data_factor": 4, "test_every": 8},
            "v3_screening": mode,
        }
        if mode not in ("parent", "v3") or config != expected:
            raise ValueError("V3 screening requires the exact standard RU configuration")
        return
    control = config.get("paper_control")
    if "paper_control" in config and control not in PAPER_CONTROLS:
        raise ValueError("paper_control must be ru_align or ru_tar")
    if control is not None:
        expected = {
            **RU_FIXED_FIELDS,
            "schema_version": 1, "profile": "ru", "base_profile": "b1",
            "gsplat_version": "1.5.3", "delayed_densification": True,
            "training": {"data_factor": 4, "test_every": 8},
            "bootstrap_switch_step": 10000, "paper_control": control,
        }
        if control == "ru_tar":
            parse_refine_windows(config.get("refine_windows"), config.get("total_steps", 0))
            expected["refine_windows"] = TAR_REFINE_WINDOWS
        if config != expected:
            raise ValueError("paper control differs from its fixed single-factor configuration")
        return
    if "refine_windows" in config:
        raise ValueError("refine_windows is disabled for legacy profiles")
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if config.get("profile") not in {
        "b0",
        "b1",
        "a1",
        "b1c",
        "cvtr",
        "ru",
        RU_PART_PROFILE,
        CAUSAL_PROFILE,
    }:
        raise ValueError(
            "profile must be one of b0, b1, a1, b1c, cvtr, ru, ru_part, ru_causal"
        )
    if config.get("gsplat_version") != "1.5.3":
        raise ValueError("gsplat_version must remain pinned to 1.5.3")
    if config.get("seed") != 42:
        raise ValueError("the pinned gsplat trainer seed must be 42")

    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping")
    required_training = (
        ("data_factor", "test_every")
        if config["profile"] in {"ru", RU_PART_PROFILE, CAUSAL_PROFILE}
        else ("data_factor", "test_every", "sh_degree", "ssim_lambda")
    )
    for field in required_training:
        if field not in training:
            raise ValueError(f"training.{field} is required")
    if training["data_factor"] <= 0 or training["test_every"] <= 0:
        raise ValueError("data_factor and test_every must be positive")
    if "ssim_lambda" in training and not 0 <= training["ssim_lambda"] <= 1:
        raise ValueError("ssim_lambda must be in [0, 1]")

    if config["profile"] == "ru":
        mismatches = {
            field: (config.get(field), expected)
            for field, expected in RU_FIXED_FIELDS.items()
            if config.get(field) != expected
        }
        if mismatches:
            raise ValueError(f"PURI-GS-RU fixed fields differ: {mismatches}")
        if config.get("delayed_densification") is not True:
            raise ValueError("PURI-GS-RU must enable delayed_densification")
        if "strategy" in config or "responsibility" in config or "cvtr" in config:
            raise ValueError(
                "PURI-GS-RU keeps its fixed fields flat and may not mix legacy methods"
            )
        return

    if config["profile"] == RU_PART_PROFILE:
        mode = config.get("intervention_mode")
        if mode not in RU_PART_MODES:
            raise ValueError(
                f"RU-PART intervention_mode must be one of {RU_PART_MODES}"
            )
        mismatches = {
            field: (config.get(field), expected)
            for field, expected in RU_PART_FIXED_FIELDS.items()
            if config.get(field) != expected
        }
        if mismatches:
            raise ValueError(f"RU-PART fixed fields differ: {mismatches}")
        required = {
            **RU_PART_FIXED_FIELDS,
            "schema_version": 1,
            "profile": RU_PART_PROFILE,
            "base_profile": "ru",
            "gsplat_version": "1.5.3",
            "delayed_densification": True,
            "training": {"data_factor": 4, "test_every": 8},
            "intervention_mode": mode,
        }
        if config != required:
            raise ValueError("RU-PART contains fields outside its fixed preregistration")
        return

    if config["profile"] == CAUSAL_PROFILE:
        common_mismatches = {
            field: (config.get(field), expected)
            for field, expected in CAUSAL_COMMON_FIXED_FIELDS.items()
            if config.get(field) != expected
        }
        if common_mismatches:
            raise ValueError(
                f"Garden causal common fields differ: {common_mismatches}"
            )
        mask_enabled = config.get("mask_enabled")
        delayed_topology = config.get("delayed_densification")
        if not isinstance(mask_enabled, bool) or not isinstance(
            delayed_topology, bool
        ):
            raise ValueError(
                "ru_causal requires boolean mask_enabled and delayed_densification"
            )
        allowed = {
            (False, True): "dg_only",
            (True, False): "mask_only",
        }
        factors = (mask_enabled, delayed_topology)
        expected_variant = allowed.get(factors)
        if expected_variant is None:
            raise ValueError(
                "ru_causal permits only the missing (M,T) quadrants (0,1) and (1,0)"
            )
        if config.get("causal_variant") != expected_variant:
            raise ValueError(
                f"causal_variant must be {expected_variant!r} for factors {factors}"
            )

        active_groups = []
        if mask_enabled:
            active_groups.append(MASK_FIXED_FIELDS)
        if delayed_topology:
            active_groups.append(DELAYED_TOPOLOGY_FIXED_FIELDS)
        for group in active_groups:
            mismatches = {
                field: (config.get(field), expected)
                for field, expected in group.items()
                if config.get(field) != expected
            }
            if mismatches:
                raise ValueError(f"Garden causal fixed fields differ: {mismatches}")

        inactive_fields = set()
        if not mask_enabled:
            inactive_fields.update(MASK_FIXED_FIELDS)
        if not delayed_topology:
            inactive_fields.update(DELAYED_TOPOLOGY_FIXED_FIELDS)
        present_inactive = sorted(inactive_fields.intersection(config))
        if present_inactive:
            raise ValueError(
                f"inactive causal subsystem fields must be absent: {present_inactive}"
            )
        if "strategy" in config or "responsibility" in config or "cvtr" in config:
            raise ValueError(
                "ru_causal keeps fixed factors flat and may not mix legacy methods"
            )
        return

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
    if config["profile"] != "a1" and responsibility["enabled"]:
        raise ValueError("only A1 may enable pixel-wise responsibility")

    cvtr = config.get("cvtr")
    if config["profile"] in {"b1c", "cvtr"}:
        if not isinstance(cvtr, dict):
            raise ValueError("continuation profiles require a cvtr mapping")
        missing = REQUIRED_CVTR_FIELDS.difference(cvtr)
        if missing:
            raise ValueError(f"missing cvtr fields: {sorted(missing)}")
        expected = CVTRConfig()
        actual = CVTRConfig(
            **{
                field: cvtr[field]
                for field in CVTRConfig.__dataclass_fields__
            }
        )
        if actual != expected:
            raise ValueError("Phase 3 CVTR hyperparameters are frozen")
        if config["profile"] == "cvtr" and not cvtr["enabled"]:
            raise ValueError("CVTR profile must enable fixed CVTR masks")
        if config["profile"] == "b1c" and cvtr["enabled"]:
            raise ValueError("B1 continuation control must disable CVTR masks")
        continuation = config.get("continuation")
        if not isinstance(continuation, dict):
            raise ValueError("continuation profiles require continuation metadata")
        if continuation != {
            "enabled": True,
            "source_step": 9999,
            "target_step": 14999,
            "additional_steps": 5000,
        }:
            raise ValueError("Phase 3 continuation is frozen to step 9999 -> 14999")
    elif cvtr is not None:
        if not isinstance(cvtr, dict) or cvtr.get("enabled"):
            raise ValueError("non-continuation profiles may not enable CVTR")


def trainer_method_args(config: dict[str, Any]) -> list[str]:
    """Translate only the method-specific profile fields to gsplat CLI flags."""

    if config["profile"] in {"ru", RU_PART_PROFILE}:
        args = [
            "--puri_gs_ru_enabled",
            "--strategy.absgrad",
            "--strategy.grow_grad2d",
            str(config["grow_grad2d"]),
        ]
        fields = (
            "mask_begin_step",
            "mask_threshold",
            "mask_erode_kernel",
            "mask_learning_rate",
            "mask_hidden_dim",
            "dino_model",
            "dino_feature_dim",
            "dino_coarse_grid",
            "dino_fine_grid",
            "dino_coarse_input",
            "dino_fine_input",
            "residual_hist_bins",
            "residual_hist_momentum",
            "residual_lower_quantile",
            "residual_upper_quantile",
            "mask_cos_weight",
            "mask_residual_weight",
            "mask_static_prior_weight",
            "mask_static_prior_decay",
            "bootstrap_switch_step",
            "densify_start_step",
            "densify_stop_step",
            "densify_every",
            "opacity_reset_start_step",
            "opacity_reset_every",
            "mask_pause_after_reset",
        )
        for field in fields:
            args.extend([f"--{field}", str(config[field])])
        if config["profile"] == RU_PART_PROFILE:
            args.append("--puri_gs_ru_part_enabled")
            args.extend(["--ru_part_mode", config["intervention_mode"]])
        if config.get("v3_screening"):
            args.extend(["--ru_v3_mode", config["v3_screening"]])
        if config.get("paper_control"):
            args.extend(["--puri_gs_paper_control", config["paper_control"]])
            if "refine_windows" in config:
                args.extend(["--refine_windows", json.dumps(config["refine_windows"])])
        return args

    if config["profile"] == CAUSAL_PROFILE:
        args = [
            "--strategy.absgrad",
            "--strategy.grow_grad2d",
            str(config["grow_grad2d"]),
        ]
        if config["mask_enabled"]:
            args.append("--puri_gs_mask_enabled")
            for field in MASK_FIXED_FIELDS:
                args.extend([f"--{field}", str(config[field])])
        if config["delayed_densification"]:
            args.append("--puri_gs_delayed_topology_enabled")
            for field in DELAYED_TOPOLOGY_FIXED_FIELDS:
                args.extend([f"--{field}", str(config[field])])
        return args

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
    cvtr = config.get("cvtr")
    if isinstance(cvtr, dict) and cvtr.get("enabled"):
        args.extend(
            [
                "--cvtr_enabled",
                "--cvtr_transient_weight",
                str(cvtr["transient_weight"]),
            ]
        )
    return args


def causal_factors(config: dict[str, Any]) -> tuple[bool, bool] | None:
    """Return semantic-mask/topology factors for the four audited profiles."""

    profile = config.get("profile")
    if profile == "b1":
        return False, False
    if profile in {"ru", RU_PART_PROFILE}:
        return True, True
    if profile == CAUSAL_PROFILE:
        return bool(config["mask_enabled"]), bool(config["delayed_densification"])
    return None
