#!/usr/bin/env python3
"""Validated PURI-GS launcher for the pinned gsplat v1.5.3 trainer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from puri_gs.config import (
    CAUSAL_PROFILE,
    RU_PART_PROFILE,
    causal_factors,
    load_experiment_config,
    parse_refine_windows,
    trainer_method_args,
)
from puri_gs.delayed_absgrad import DelayedAbsGradSchedule, topology_event_summary


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
PATCH_PATH = PROJECT_ROOT / "patches" / "gsplat_v1.5.3_robot_screen.patch"
CVTR_PATCH_PATH = PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_cvtr.patch"
RU_PATCH_PATH = PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_ru.patch"
CAUSAL_PATCH_PATH = (
    PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_garden_causal.patch"
)
PAPER_CONTROL_PATCH_PATH = PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_paper_controls.patch"
EFFICIENCY_AUDIT_PATCH_PATH = (
    PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_efficiency_audit.patch"
)
ONTOGO_PATCH_PATH = PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_ontogo.patch"
RU_PART_PATCH_PATH = PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_ru_part.patch"
RU_PART_DIAGNOSTIC_PATCH_PATH = (
    PROJECT_ROOT / "patches" / "gsplat_v1.5.3_ru_part_mechanism_diagnostic.patch"
)
EXPECTED_GSPLAT_COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _verify_applied_patch(
    gsplat_dir: Path, patch_path: Path, *, unidiff_zero: bool = False
) -> None:
    zero_context = ["--unidiff-zero"] if unidiff_zero else []
    check = subprocess.run(
        [
            "git",
            "apply",
            "--reverse",
            "--check",
            *zero_context,
            "--ignore-whitespace",
            str(patch_path),
        ],
        cwd=gsplat_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(f"required trainer patch is not applied: {patch_path.name}")


def _verify_gsplat(
    gsplat_dir: Path,
    *,
    require_cvtr: bool = False,
    require_ru: bool = False,
    require_causal: bool = False,
    require_efficiency_audit: bool = False,
    require_ontogo: bool = False,
    require_paper_control: bool = False,
    require_ru_part: bool = False,
) -> None:
    if _git("rev-parse", "HEAD", cwd=gsplat_dir) != EXPECTED_GSPLAT_COMMIT:
        raise RuntimeError("GSPLAT_DIR is not the pinned v1.5.3 checkout")
    source = (gsplat_dir / "examples" / "simple_trainer.py").read_text(encoding="utf-8")
    if require_ru_part:
        _verify_applied_patch(gsplat_dir, RU_PART_DIAGNOSTIC_PATCH_PATH)
        required = (
            "puri_gs_ru_part_enabled",
            "ru_part_mode",
            "ru_part_track_cache",
            "ru_part_replay_ckpt",
            "ru_part_diagnostic_stop_after_step",
            "RUPARTStrategy",
        )
        if any(marker not in source for marker in required):
            raise RuntimeError("RU-PART trainer stack is incomplete")
        return
    if require_paper_control or "puri_gs_paper_control: Optional[str]" in source:
        _verify_applied_patch(gsplat_dir, PAPER_CONTROL_PATCH_PATH)
        required = ("puri_gs_ru_enabled", "puri_gs_mask_enabled",
                    "puri_gs_delayed_topology_enabled", "PURI-GS-FACTORS",
                    "train_keyword", "ru_validation.json", "per_image_metrics.csv")
        if any(marker not in source for marker in required):
            raise RuntimeError("paper-control trainer stack is incomplete")
        if require_cvtr:
            _verify_applied_patch(gsplat_dir, CVTR_PATCH_PATH)
        if require_efficiency_audit:
            _verify_applied_patch(gsplat_dir, EFFICIENCY_AUDIT_PATCH_PATH)
        if require_ontogo:
            _verify_applied_patch(gsplat_dir, ONTOGO_PATCH_PATH)
        return
    def verify_causal_stack() -> None:
        _verify_applied_patch(gsplat_dir, CAUSAL_PATCH_PATH)
        trainer_source = (gsplat_dir / "examples" / "simple_trainer.py").read_text(
            encoding="utf-8"
        )
        required_markers = (
            "puri_gs_ru_enabled",
            "puri_gs_mask_enabled",
            "puri_gs_delayed_topology_enabled",
            "PURI-GS-FACTORS",
        )
        if any(marker not in trainer_source for marker in required_markers):
            raise RuntimeError("Garden causal trainer patch stack is incomplete")

    def verify_efficiency_stack() -> None:
        # The audit patch is stacked on the RU superset and changes some of the
        # same evaluation hunks.  Once stacked, reverse-applying the older RU
        # patch is no longer a valid state check, so validate the top patch and
        # the required lower-layer source markers together.
        _verify_applied_patch(gsplat_dir, EFFICIENCY_AUDIT_PATCH_PATH)
        trainer_source = (gsplat_dir / "examples" / "simple_trainer.py").read_text(
            encoding="utf-8"
        )
        dataset_source = (
            gsplat_dir / "examples" / "datasets" / "colmap.py"
        ).read_text(encoding="utf-8")
        required_trainer_markers = (
            "puri_gs_ru_enabled",
            "train_keyword",
            "eval_warmup_renders",
            "eval_disable_image_save",
            "per_image_latency.csv",
        )
        if any(marker not in trainer_source for marker in required_trainer_markers):
            raise RuntimeError("efficiency-audit trainer patch stack is incomplete")
        if "_is_png_file" not in dataset_source:
            raise RuntimeError("efficiency-audit dataset patch stack is incomplete")

    if require_ontogo:
        _verify_applied_patch(gsplat_dir, ONTOGO_PATCH_PATH)
        trainer_source = (gsplat_dir / "examples" / "simple_trainer.py").read_text(
            encoding="utf-8"
        )
        dataset_source = (
            gsplat_dir / "examples" / "datasets" / "colmap.py"
        ).read_text(encoding="utf-8")
        required_markers = (
            "puri_gs_ru_enabled",
            "eval_warmup_renders",
            "eval_disable_image_save",
            "ontogo-patio-high",
            "OnTheGoPatioHighParser",
        )
        if any(marker not in trainer_source for marker in required_markers):
            raise RuntimeError("On-the-go trainer patch stack is incomplete")
        if "_is_png_file" not in dataset_source:
            raise RuntimeError("On-the-go dataset patch stack is incomplete")
        return
    if require_causal:
        verify_causal_stack()
        return
    if require_efficiency_audit:
        verify_efficiency_stack()
        return
    if not require_cvtr:
        try:
            verify_efficiency_stack()
            return
        except RuntimeError:
            pass
    if require_cvtr:
        _verify_applied_patch(gsplat_dir, CVTR_PATCH_PATH)
    elif require_ru:
        try:
            verify_causal_stack()
        except RuntimeError:
            _verify_applied_patch(gsplat_dir, RU_PATCH_PATH)
        trainer_source = (gsplat_dir / "examples" / "simple_trainer.py").read_text(
            encoding="utf-8"
        )
        dataset_source = (
            gsplat_dir / "examples" / "datasets" / "colmap.py"
        ).read_text(encoding="utf-8")
        if "train_keyword" not in trainer_source or "_is_png_file" not in dataset_source:
            raise RuntimeError("PURI-GS base trainer/data patch is incomplete")
    else:
        try:
            _verify_applied_patch(gsplat_dir, PATCH_PATH)
        except RuntimeError:
            try:
                verify_causal_stack()
            except RuntimeError:
                # The RU patch is a strict, default-disabled superset of the base
                # trainer. It is therefore also the source used for matched B1 runs.
                _verify_applied_patch(gsplat_dir, RU_PATCH_PATH)


def _verify_dataset(
    data_dir: Path,
    data_factor: int,
    train_keyword: str | None,
    test_keyword: str | None,
    dataset_format: str = "colmap",
) -> None:
    if (train_keyword is None) != (test_keyword is None):
        raise ValueError("train_keyword and test_keyword must be supplied together")
    if dataset_format == "ontogo-patio-high":
        if train_keyword != "clutter" or test_keyword != "extra":
            raise ValueError(
                "Patio-High requires --train-keyword clutter --test-keyword extra"
            )
        from puri_gs.ontogo import validate_prepared_patio_high

        validate_prepared_patio_high(data_dir, factor=data_factor, load_points=False)
        return
    if dataset_format != "colmap":
        raise ValueError(f"unsupported dataset format: {dataset_format}")
    image_dir = data_dir / f"images_{data_factor}"
    sparse_dir = data_dir / "sparse" / "0"
    missing = [
        path
        for path in (
            image_dir,
            sparse_dir / "cameras.bin",
            sparse_dir / "images.bin",
            sparse_dir / "points3D.bin",
        )
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"dataset inputs are missing: {[str(path) for path in missing]}")
    if train_keyword is not None:
        names = [path.name.casefold() for path in image_dir.iterdir() if path.is_file()]
        if not any(train_keyword.casefold() in name for name in names):
            raise RuntimeError(f"no images match train keyword {train_keyword!r}")
        if not any(test_keyword.casefold() in name for name in names):
            raise RuntimeError(f"no images match test keyword {test_keyword!r}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _continuation_source_inventory(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("continuation checkpoint must be a mapping")
    keys = sorted(checkpoint)
    state_names = {
        "optimizer": any(name in checkpoint for name in ("optimizer", "optimizers")),
        "strategy": any(name in checkpoint for name in ("strategy", "strategy_state")),
        "rng": any(name in checkpoint for name in ("rng", "rng_state", "random_state")),
    }
    if checkpoint.get("step") != 9999 or "splats" not in checkpoint:
        raise ValueError("continuation checkpoint must contain step=9999 and splats")
    if any(state_names.values()):
        raise ValueError("Phase 3 source unexpectedly contains optimizer/strategy/RNG state")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "step": checkpoint["step"],
        "keys": keys,
        "contains_optimizer_state": state_names["optimizer"],
        "contains_strategy_state": state_names["strategy"],
        "contains_rng_state": state_names["rng"],
        "continuation_policy": "fresh optimizer, strategy, scheduler, and RNG initialization for both branches",
    }


def _runtime_environment(gsplat_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch

        torch_info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        torch_info = {"version": None, "cuda_available": False}
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch_info,
        "gsplat": {
            "version": config["gsplat_version"],
            "commit": _git("rev-parse", "HEAD", cwd=gsplat_dir),
            "source": str(gsplat_dir),
        },
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "Pillow", "tyro", "torchmetrics", "fused-ssim")
        },
        "repository_commit": _git("rev-parse", "HEAD", cwd=REPOSITORY_ROOT),
    }


def _apply_runtime_overrides(
    config: dict[str, Any], responsibility_start_step: int | None
) -> dict[str, Any]:
    """Return the effective profile while keeping the checked-in config immutable."""

    effective = copy.deepcopy(config)
    if responsibility_start_step is None:
        return effective
    if effective["profile"] != "a1":
        raise ValueError("responsibility_start_step can only override the A1 profile")
    if responsibility_start_step < 0:
        raise ValueError("responsibility_start_step must be non-negative")
    effective["responsibility"]["start_step"] = responsibility_start_step
    return effective


def _causal_contract(config: dict[str, Any]) -> dict[str, Any] | None:
    factors = causal_factors(config)
    if factors is None:
        return None
    mask_enabled, delayed_topology = factors
    schedule = None
    if delayed_topology:
        schedule = DelayedAbsGradSchedule(
            densify_start_step=config.get("densify_start_step", 10_000),
            densify_stop_step=config.get("densify_stop_step", 20_000),
            densify_every=config.get("densify_every", 100),
            opacity_reset_start_step=config.get("opacity_reset_start_step", 15_000),
            opacity_reset_every=config.get("opacity_reset_every", 3_000),
            mask_pause_after_reset=config.get("mask_pause_after_reset", 300),
            refine_windows=parse_refine_windows(config.get("refine_windows")),
        )
    pause = config.get("mask_pause_after_reset", 0) if mask_enabled else 0
    topology = topology_event_summary(
        delayed_topology=delayed_topology,
        total_steps=config["total_steps"],
        mask_pause_after_reset=pause,
        delayed_schedule=schedule,
    )
    return {
        "protocol": "puri-gs-ru-garden-causal-2x2-v1",
        "semantic_mask_enabled": mask_enabled,
        "delayed_topology_enabled": delayed_topology,
        "M": int(mask_enabled),
        "T": int(delayed_topology),
        "topology_events": topology,
        "expected_mask_update_count": (
            config["total_steps"] - topology["mask_pause_step_count"]
            if mask_enabled
            else 0
        ),
    }


def _build_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    gsplat_dir: Path,
    data_dir: Path,
    result_dir: Path,
) -> list[str]:
    training = config["training"]
    data_factor = args.data_factor or training["data_factor"]
    dataset_format = getattr(args, "dataset_format", "colmap")
    sh_degree = config.get("sh_degree", training.get("sh_degree"))
    ssim_lambda = config.get("ssim_lambda", training.get("ssim_lambda"))
    command = [
        sys.executable,
        "simple_trainer.py",
        "default",
        "--disable_viewer",
        "--disable_video",
        "--data_dir",
        str(data_dir),
        "--data_factor",
        str(data_factor),
        "--result_dir",
        str(result_dir),
        "--test_every",
        str(training["test_every"]),
        "--val_every",
        "0",
        "--eval_split",
        "test",
        "--sh_degree",
        str(sh_degree),
        "--ssim_lambda",
        str(ssim_lambda),
        "--tb_every",
        "0",
    ]
    if dataset_format != "colmap":
        command.extend(["--dataset_format", dataset_format])
    if args.train_keyword is not None:
        command.extend(["--train_keyword", args.train_keyword])
        command.extend(["--test_keyword", args.test_keyword])
    if args.checkpoint is None:
        if args.resume_checkpoint is not None:
            continuation = config.get("continuation")
            if not isinstance(continuation, dict) or not continuation.get("enabled"):
                raise ValueError("--resume-checkpoint requires a continuation profile")
            max_steps = continuation["target_step"] + 1
        else:
            max_steps = args.max_steps or config.get("total_steps", 10_000)
        command.extend(
            [
                "--max_steps",
                str(max_steps),
                "--eval_steps",
                "-1",
                "--save_steps",
                str(max_steps),
            ]
        )
        command.extend(trainer_method_args(config))
        if config["profile"] == RU_PART_PROFILE:
            if config["intervention_mode"] != "parent":
                if args.track_cache is None:
                    raise ValueError("RU-PART noop/current training requires --track-cache")
                command.extend(["--ru_part_track_cache", str(args.track_cache.resolve())])
            if args.non_scientific_smoke:
                command.append("--ru_part_non_scientific_smoke")
            replay_checkpoint = getattr(args, "replay_checkpoint", None)
            if replay_checkpoint is not None:
                command.extend(["--ru_part_replay_ckpt", str(replay_checkpoint.resolve())])
            if args.diagnostic_stage is not None:
                command.extend(["--ru_part_diagnostic_stage", args.diagnostic_stage])
                if args.diagnostic_stop_after_step is not None:
                    command.extend([
                        "--ru_part_diagnostic_stop_after_step",
                        str(args.diagnostic_stop_after_step),
                    ])
                if args.diagnostic_roi_dir is not None:
                    command.extend([
                        "--ru_part_diagnostic_roi_dir",
                        str(args.diagnostic_roi_dir.resolve()),
                    ])
                if args.diagnostic_roi_confirmed:
                    command.append("--ru_part_diagnostic_roi_confirmed")
        factors = causal_factors(config)
        uses_mask = factors is not None and factors[0]
        if uses_mask:
            for value, flag in (
                (args.dino_repo_dir, "--dino_repo_dir"),
                (args.dino_weight_path, "--dino_weight_path"),
                (args.feature_cache_dir, "--feature_cache_dir"),
            ):
                if value is None:
                    raise ValueError(f"semantic-mask training requires {flag}")
                command.extend([flag, str(value.resolve())])
        if args.resume_checkpoint is not None:
            command.extend(["--resume_ckpt", str(args.resume_checkpoint.resolve())])
        cvtr = config.get("cvtr")
        if isinstance(cvtr, dict) and cvtr.get("enabled"):
            if args.cvtr_mask_dir is None:
                raise ValueError("CVTR continuation requires --cvtr-mask-dir")
            command.extend(["--cvtr_mask_dir", str(args.cvtr_mask_dir.resolve())])
        elif args.cvtr_mask_dir is not None:
            raise ValueError("--cvtr-mask-dir is valid only for the CVTR profile")
    else:
        command.extend(["--ckpt", str(args.checkpoint.resolve())])
        eval_warmup_renders = getattr(args, "eval_warmup_renders", 0)
        if eval_warmup_renders:
            command.extend(
                ["--eval_warmup_renders", str(eval_warmup_renders)]
            )
        if getattr(args, "eval_disable_image_save", False):
            command.append("--eval_disable_image_save")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--data-factor", type=int)
    parser.add_argument(
        "--dataset-format",
        choices=("colmap", "ontogo-patio-high"),
        default="colmap",
    )
    parser.add_argument("--train-keyword")
    parser.add_argument("--test-keyword")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--eval-warmup-renders",
        type=int,
        default=0,
        help="Checkpoint-evaluation warmup renders excluded from latency metrics.",
    )
    parser.add_argument(
        "--eval-disable-image-save",
        action="store_true",
        help="Do not save evaluation canvases; metrics and latency CSV are unchanged.",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--cvtr-mask-dir", type=Path)
    parser.add_argument("--dino-repo-dir", type=Path)
    parser.add_argument("--dino-weight-path", type=Path)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--track-cache", type=Path)
    parser.add_argument("--replay-checkpoint", type=Path)
    parser.add_argument(
        "--diagnostic-stage", choices=("SMOKE", "U", "R1", "R2", "VJP", "P1", "P2", "O")
    )
    parser.add_argument("--diagnostic-stop-after-step", type=int)
    parser.add_argument("--diagnostic-roi-dir", type=Path)
    parser.add_argument("--diagnostic-roi-confirmed", action="store_true")
    parser.add_argument(
        "--non-scientific-smoke",
        action="store_true",
        help="RU-PART only: stop after the real step-10000 event; never reusable as a result.",
    )
    parser.add_argument(
        "--responsibility-start-step",
        type=int,
        help="A1-only smoke override; the effective value is recorded in config.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu < 0:
        raise ValueError("gpu must be non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.eval_warmup_renders < 0:
        raise ValueError("eval_warmup_renders must be non-negative")
    if args.eval_warmup_renders and args.checkpoint is None:
        raise ValueError("--eval-warmup-renders requires --checkpoint")
    if args.eval_disable_image_save and args.checkpoint is None:
        raise ValueError("--eval-disable-image-save requires --checkpoint")
    config_path = args.config.expanduser().resolve()
    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    config = _apply_runtime_overrides(
        load_experiment_config(config_path), args.responsibility_start_step
    )
    is_continuation = config["profile"] in {"b1c", "cvtr"}
    is_ru = config["profile"] == "ru"
    is_ru_part = config["profile"] == RU_PART_PROFILE
    is_causal = config["profile"] == CAUSAL_PROFILE
    factors = causal_factors(config)
    uses_mask = factors is not None and factors[0]
    is_paper_control = bool(config.get("paper_control"))
    if is_ru_part:
        if _git("branch", "--show-current", cwd=REPOSITORY_ROOT) != "ru-part":
            raise RuntimeError("RU-PART must run from the approved ru-part branch")
        if result_dir.exists():
            raise FileExistsError(f"RU-PART refuses to overwrite any existing output: {result_dir}")
        if (
            args.data_factor not in (None, 4)
            or args.train_keyword is not None
            or args.test_keyword is not None
            or args.dataset_format != "colmap"
            or data_dir.name.casefold() != "garden"
        ):
            raise ValueError("RU-PART first run is fixed to Garden factor4 and the every-eighth split")
        if args.checkpoint is None and config["intervention_mode"] != "parent":
            if args.track_cache is None or not args.track_cache.expanduser().is_file():
                raise RuntimeError("RU-PART noop/current static-track cache is missing")
            from puri_gs.static_tracks import load_static_track_cache

            load_static_track_cache(args.track_cache.expanduser().resolve())
        elif args.checkpoint is None and args.track_cache is not None:
            raise ValueError("RU-PART parent must not receive --track-cache")
        if args.non_scientific_smoke:
            if args.checkpoint is not None or args.max_steps != 10001:
                raise ValueError("RU-PART smoke must train from step0 through the real step10000 event")
            if "NON_SCIENTIFIC_SMOKE" not in result_dir.name:
                raise ValueError("RU-PART smoke output name must contain NON_SCIENTIFIC_SMOKE")
        elif args.max_steps not in (None, config["total_steps"]):
            raise ValueError("RU-PART Full is fixed to 30000 steps")
        if args.replay_checkpoint is not None:
            args.replay_checkpoint = args.replay_checkpoint.expanduser().resolve()
            if not args.replay_checkpoint.is_file():
                raise RuntimeError(
                    f"RU-PART replay checkpoint is missing: {args.replay_checkpoint}"
                )
        if args.diagnostic_stage is not None:
            if args.checkpoint is not None or args.non_scientific_smoke:
                raise ValueError("mechanism diagnostic cannot evaluate or use legacy smoke")
            if args.diagnostic_stage == "VJP":
                if args.diagnostic_stop_after_step is not None:
                    raise ValueError("VJP does not execute training updates")
                if args.replay_checkpoint is None or args.diagnostic_roi_dir is None:
                    raise ValueError("VJP requires step9999 replay and a fixed ROI directory")
                args.diagnostic_roi_dir = args.diagnostic_roi_dir.expanduser().resolve()
                if not (args.diagnostic_roi_dir / "roi_evidence.json").is_file():
                    raise RuntimeError("VJP ROI directory is incomplete")
            elif args.diagnostic_stage in {"P1", "P2", "O"}:
                if args.diagnostic_stop_after_step != 10_399:
                    raise ValueError("P1/P2/O terminal boundary is fixed at step10399")
                if args.replay_checkpoint is None or args.diagnostic_roi_dir is None:
                    raise ValueError("P1/P2/O require step9999 replay and fixed ROI directory")
                args.diagnostic_roi_dir = args.diagnostic_roi_dir.expanduser().resolve()
                if not (args.diagnostic_roi_dir / "static_confirmation.json").is_file():
                    raise RuntimeError("P1/P2/O require explicit static ROI confirmation")
                decision_path = args.diagnostic_roi_dir / "vjp_decision.json"
                if not decision_path.is_file() or json.loads(
                    decision_path.read_text(encoding="utf-8")
                ).get("status") != "LOCAL_CONTROLLABILITY_PRESENT":
                    raise RuntimeError("P1/P2/O require the positive formal VJP gate")
                if not args.diagnostic_roi_confirmed:
                    raise ValueError("P1/P2/O require the explicit confirmation flag")
            elif args.diagnostic_stage == "SMOKE":
                if args.replay_checkpoint is not None or args.diagnostic_stop_after_step not in range(100):
                    raise ValueError("trainer smoke must start at step0 and stop within 100 updates")
                if args.diagnostic_roi_dir is not None or args.diagnostic_roi_confirmed:
                    raise ValueError("trainer smoke cannot receive ROI arguments")
            else:
                if args.diagnostic_stop_after_step != 10_199:
                    raise ValueError("U/R terminal boundary is fixed at step10199")
                if args.diagnostic_stage == "U" and args.replay_checkpoint is not None:
                    raise ValueError("U must be uninterrupted and cannot restore a replay checkpoint")
                if args.diagnostic_stage in {"R1", "R2"} and args.replay_checkpoint is None:
                    raise ValueError("R1/R2 require the registered step9999 replay checkpoint")
                if args.diagnostic_roi_dir is not None or args.diagnostic_roi_confirmed:
                    raise ValueError("ROI arguments are valid only for VJP")
        elif (
            args.diagnostic_stop_after_step is not None
            or args.diagnostic_roi_dir is not None
            or args.diagnostic_roi_confirmed
        ):
            raise ValueError("diagnostic options require --diagnostic-stage")
    elif (
        args.track_cache is not None
        or args.replay_checkpoint is not None
        or args.non_scientific_smoke
        or args.diagnostic_stage is not None
        or args.diagnostic_stop_after_step is not None
        or args.diagnostic_roi_dir is not None
        or args.diagnostic_roi_confirmed
    ):
        raise ValueError("RU-PART-only arguments were supplied to another profile")
    if is_paper_control:
        if _git("branch", "--show-current", cwd=REPOSITORY_ROOT) != "dev":
            raise RuntimeError("paper controls must run on dev")
        if result_dir.exists():
            raise FileExistsError(f"paper-control output already exists: {result_dir}")
        if (args.data_factor not in (None, 4) or args.train_keyword is not None
                or args.test_keyword is not None or args.dataset_format != "colmap"
                or args.eval_warmup_renders or args.eval_disable_image_save):
            raise ValueError("paper controls require the fixed COLMAP split and evaluation protocol")
    _verify_gsplat(
        gsplat_dir,
        require_cvtr=is_continuation,
        require_ru=is_ru,
        require_causal=is_causal,
        require_efficiency_audit=bool(args.eval_warmup_renders),
        require_ontogo=args.dataset_format == "ontogo-patio-high",
        require_paper_control=is_paper_control,
        require_ru_part=is_ru_part,
    )
    if (is_ru or is_causal) and args.max_steps is not None:
        if args.max_steps != config["total_steps"] and not 1 <= args.max_steps <= 100:
            raise ValueError(
                "PURI-GS-RU profiles allow only the fixed 30000-step run or a <=100-step smoke"
            )
    data_factor = args.data_factor or config["training"]["data_factor"]
    _verify_dataset(
        data_dir,
        data_factor,
        args.train_keyword,
        args.test_keyword,
        args.dataset_format,
    )
    if args.checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("--checkpoint and --resume-checkpoint are mutually exclusive")
    if args.checkpoint is not None and not args.checkpoint.expanduser().is_file():
        raise RuntimeError(f"checkpoint does not exist: {args.checkpoint}")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.expanduser().resolve()
        if not args.resume_checkpoint.is_file():
            raise RuntimeError(f"resume checkpoint does not exist: {args.resume_checkpoint}")
        if not is_continuation:
            raise ValueError("--resume-checkpoint requires a continuation profile")
    elif is_continuation and args.checkpoint is None:
        raise ValueError("continuation training requires --resume-checkpoint")
    if args.cvtr_mask_dir is not None:
        args.cvtr_mask_dir = args.cvtr_mask_dir.expanduser().resolve()
        if not (args.cvtr_mask_dir / "manifest.json").is_file():
            raise RuntimeError(
                f"CVTR mask manifest does not exist: {args.cvtr_mask_dir / 'manifest.json'}"
            )
    if uses_mask and args.checkpoint is None:
        for path, label, expected in (
            (args.dino_repo_dir, "DINOv2 repository", "hubconf.py"),
            (args.dino_weight_path, "DINOv2 weight", None),
            (args.feature_cache_dir, "feature cache", "manifest.json"),
        ):
            if path is None:
                raise ValueError(f"{label} path is required for semantic-mask training")
            resolved = path.expanduser().resolve()
            required = resolved / expected if expected is not None else resolved
            if not required.exists():
                raise RuntimeError(f"{label} is missing: {required}")
    command = _build_command(args, config, gsplat_dir, data_dir, result_dir)
    printable = f"CUDA_VISIBLE_DEVICES={args.gpu} {shlex.join(command)}"
    contract = _causal_contract(config)
    if contract is not None:
        print(
            "PURI-GS-FACTORS "
            f"M={contract['M']} T={contract['T']} "
            f"resets={contract['topology_events']['reset_steps']}"
        )
    print(printable)
    if args.dry_run:
        return 0

    continuation_inventory = (
        _continuation_source_inventory(args.resume_checkpoint)
        if args.resume_checkpoint is not None
        else None
    )

    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(f"result directory is not empty: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=not is_paper_control)
    (result_dir / "config.yaml").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (result_dir / "run_command.txt").write_text(printable + "\n", encoding="utf-8")
    (result_dir / "git_commit.txt").write_text(
        _git("rev-parse", "HEAD", cwd=REPOSITORY_ROOT) + "\n", encoding="utf-8"
    )
    (result_dir / "environment.json").write_text(
        json.dumps(_runtime_environment(gsplat_dir, config), indent=2) + "\n",
        encoding="utf-8",
    )
    if is_paper_control:
        from puri_gs.paper_controls import check_control_diffs, resolved_schedule, write_json_new

        profiles = [load_experiment_config(PROJECT_ROOT / "configs" / name) for name in (
            "puri_gs_ru_full30k.yaml", "puri_gs_ru_align_full30k.yaml", "puri_gs_ru_tar_full30k.yaml",
        )]
        write_json_new(result_dir / "resolved_config_diff.json", check_control_diffs(*profiles))
        if args.checkpoint is not None:
            write_json_new(result_dir / "resolved_schedule.json", resolved_schedule(config))
    if is_causal and contract is not None:
        (result_dir / "causal_contract.json").write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )
    if args.dataset_format == "ontogo-patio-high":
        from puri_gs.ontogo import PROTOCOL_FILE, sha256_file

        protocol_path = data_dir / PROTOCOL_FILE
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol_record = {
            "prepared_data_dir": str(data_dir),
            "protocol_file": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "protocol": protocol,
        }
        (result_dir / "dataset_protocol.json").write_text(
            json.dumps(protocol_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if continuation_inventory is not None:
        (result_dir / "continuation_source.json").write_text(
            json.dumps(continuation_inventory, indent=2) + "\n",
            encoding="utf-8",
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Run the patched example file from the source checkout, but import gsplat itself
    # from the pinned installed wheel. Adding gsplat_dir here shadows the wheel and
    # incorrectly triggers the source checkout's JIT-extension fallback.
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        command,
        cwd=gsplat_dir / "examples",
        env=env,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
