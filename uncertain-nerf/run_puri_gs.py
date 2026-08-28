#!/usr/bin/env python3
"""Validated B0/B1/A1 launcher for the pinned gsplat v1.5.3 trainer."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from puri_gs.config import load_experiment_config, trainer_method_args


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
PATCH_PATH = PROJECT_ROOT / "patches" / "gsplat_v1.5.3_robot_screen.patch"
EXPECTED_GSPLAT_COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _verify_gsplat(gsplat_dir: Path) -> None:
    if _git("rev-parse", "HEAD", cwd=gsplat_dir) != EXPECTED_GSPLAT_COMMIT:
        raise RuntimeError("GSPLAT_DIR is not the pinned v1.5.3 checkout")
    check = subprocess.run(
        [
            "git",
            "apply",
            "--reverse",
            "--check",
            "--ignore-whitespace",
            str(PATCH_PATH),
        ],
        cwd=gsplat_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "required PURI-GS trainer patch is not applied to the pinned checkout"
        )


def _verify_dataset(
    data_dir: Path,
    data_factor: int,
    train_keyword: str | None,
    test_keyword: str | None,
) -> None:
    if (train_keyword is None) != (test_keyword is None):
        raise ValueError("train_keyword and test_keyword must be supplied together")
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


def _build_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    gsplat_dir: Path,
    data_dir: Path,
    result_dir: Path,
) -> list[str]:
    training = config["training"]
    data_factor = args.data_factor or training["data_factor"]
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
        str(training["sh_degree"]),
        "--ssim_lambda",
        str(training["ssim_lambda"]),
        "--tb_every",
        "0",
    ]
    if args.train_keyword is not None:
        command.extend(["--train_keyword", args.train_keyword])
        command.extend(["--test_keyword", args.test_keyword])
    if args.checkpoint is None:
        command.extend(
            [
                "--max_steps",
                str(args.max_steps),
                "--eval_steps",
                "-1",
                "--save_steps",
                str(args.max_steps),
            ]
        )
        command.extend(trainer_method_args(config))
    else:
        command.extend(["--ckpt", str(args.checkpoint.resolve())])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--data-factor", type=int)
    parser.add_argument("--train-keyword")
    parser.add_argument("--test-keyword")
    parser.add_argument("--checkpoint", type=Path)
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
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    config_path = args.config.expanduser().resolve()
    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    config = _apply_runtime_overrides(
        load_experiment_config(config_path), args.responsibility_start_step
    )
    _verify_gsplat(gsplat_dir)
    data_factor = args.data_factor or config["training"]["data_factor"]
    _verify_dataset(
        data_dir, data_factor, args.train_keyword, args.test_keyword
    )
    if args.checkpoint is not None and not args.checkpoint.expanduser().is_file():
        raise RuntimeError(f"checkpoint does not exist: {args.checkpoint}")
    command = _build_command(args, config, gsplat_dir, data_dir, result_dir)
    printable = f"CUDA_VISIBLE_DEVICES={args.gpu} {shlex.join(command)}"
    print(printable)
    if args.dry_run:
        return 0

    if result_dir.exists() and any(result_dir.iterdir()):
        raise RuntimeError(f"result directory is not empty: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    python_path = [str(PROJECT_ROOT), str(gsplat_dir)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        command,
        cwd=gsplat_dir / "examples",
        env=env,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
