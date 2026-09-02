#!/usr/bin/env python3
"""Strict read-only pairing audit for the frozen Android B1/RU 30k runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED_TRAIN_COUNT = 122
EXPECTED_TEST_COUNT = 19
EXPECTED_CHECKPOINT_STEP = 29_999
EXPECTED_SPLIT_PROTOCOL = "filename-keyword"
EXPECTED_TRAIN_KEYWORD = "clutter"
EXPECTED_TEST_KEYWORD = "extra"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required pairing artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required pairing artifact is missing: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"required pairing artifact is empty: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("config.training must be an object")
    profile = config.get("profile")
    if profile == "b1":
        strategy = config.get("strategy")
        if not isinstance(strategy, dict):
            raise ValueError("B1 config.strategy must be an object")
        absgrad = strategy.get("absgrad")
        grow_grad2d = strategy.get("grow_grad2d")
    elif profile == "ru":
        absgrad = config.get("absgrad")
        grow_grad2d = config.get("grow_grad2d")
    else:
        raise ValueError(f"unexpected profile in Android pairing audit: {profile!r}")
    return {
        "seed": config.get("seed"),
        "total_steps": config.get("total_steps"),
        "data_factor": training.get("data_factor"),
        "test_every": training.get("test_every"),
        "sh_degree": config.get("sh_degree", training.get("sh_degree")),
        "ssim_lambda": config.get("ssim_lambda", training.get("ssim_lambda")),
        "gsplat_version": config.get("gsplat_version"),
        "strategy_type": "default",
        "absgrad": absgrad,
        "grow_grad2d": grow_grad2d,
    }


def _checkpoint_inventory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"standard checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a mapping: {path}")
    keys = sorted(str(key) for key in checkpoint)
    splats = checkpoint.get("splats")
    if not isinstance(splats, dict) or "means" not in splats:
        raise ValueError(f"checkpoint has no splats.means tensor: {path}")
    means = splats["means"]
    if not isinstance(means, torch.Tensor) or means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"checkpoint splats.means must have shape [N,3]: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "step": checkpoint.get("step"),
        "keys": keys,
        "gaussian_count": int(means.shape[0]),
    }


def _metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"per-image metrics are missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != EXPECTED_TEST_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TEST_COUNT} per-image rows, got {len(source)}: {path}"
        )
    rows = []
    for row in source:
        values = {metric: float(row[metric]) for metric in ("psnr", "ssim", "lpips")}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite per-image metric: {path}")
        rows.append({"image_name": row["image_name"], **values})
    return rows


def _environment_fingerprint(environment: dict[str, Any]) -> dict[str, Any]:
    torch_info = environment.get("torch")
    gsplat = environment.get("gsplat")
    packages = environment.get("packages")
    if not all(isinstance(item, dict) for item in (torch_info, gsplat, packages)):
        raise ValueError("environment.json lacks torch/gsplat/packages objects")
    return {
        "repository_commit": environment.get("repository_commit"),
        "gpu": torch_info.get("gpu"),
        "torch_version": torch_info.get("version"),
        "cuda_runtime": torch_info.get("cuda_runtime"),
        "gsplat_version": gsplat.get("version"),
        "gsplat_commit": gsplat.get("commit"),
        "torchmetrics_version": packages.get("torchmetrics"),
        "fused_ssim_version": packages.get("fused-ssim"),
    }


def _load_run(path: Path, expected_profile: str) -> dict[str, Any]:
    config = _read_json(path / "config.yaml")
    if config.get("profile") != expected_profile:
        raise ValueError(f"expected {expected_profile} profile: {path}")
    split = _read_json(path / "dataset_split.json")
    train_names = split.get("train")
    test_names = split.get("test")
    if not isinstance(train_names, list) or not isinstance(test_names, list):
        raise ValueError(f"dataset split must contain train/test lists: {path}")
    if len(train_names) != EXPECTED_TRAIN_COUNT or len(test_names) != EXPECTED_TEST_COUNT:
        raise ValueError(f"unexpected Android split counts: {path}")
    if len(set(train_names)) != len(train_names) or len(set(test_names)) != len(test_names):
        raise ValueError(f"dataset split contains duplicate names: {path}")
    if set(train_names).intersection(test_names):
        raise ValueError(f"Android train/test split overlaps: {path}")
    metrics = _metric_rows(path / "independent_eval" / "per_image_metrics.csv")
    validation = _read_json(path / "independent_eval" / "ru_validation.json")
    if not (
        validation.get("standard_checkpoint_load_pass") is True
        and validation.get("evaluation_imported_dino") is False
        and validation.get("evaluation_loaded_mask_head") is False
        and validation.get("evaluation_rasterization_count_ratio") == 1.0
    ):
        raise ValueError(f"independent evaluation is not the standard inference path: {path}")
    command = _read_text(path / "run_command.txt")
    return {
        "path": str(path),
        "git_commit": _read_text(path / "git_commit.txt"),
        "config": config,
        "protocol_config": _protocol_config(config),
        "split": split,
        "train_names": train_names,
        "test_names": test_names,
        "metric_names": [row["image_name"] for row in metrics],
        "training_environment": _environment_fingerprint(
            _read_json(path / "environment.json")
        ),
        "evaluation_environment": _environment_fingerprint(
            _read_json(path / "independent_eval" / "environment.json")
        ),
        "checkpoint": _checkpoint_inventory(
            path / "ckpts" / "ckpt_29999_rank0.pt"
        ),
        "run_command": command,
    }


def decide(b1: dict[str, Any], ru: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "same_training_commit": b1["git_commit"] == ru["git_commit"],
        "same_shared_config": b1["protocol_config"] == ru["protocol_config"],
        "same_train_split_order": b1["train_names"] == ru["train_names"],
        "same_test_split_order": b1["test_names"] == ru["test_names"],
        "metric_rows_follow_test_order_b1": b1["metric_names"] == b1["test_names"],
        "metric_rows_follow_test_order_ru": ru["metric_names"] == ru["test_names"],
        "same_training_environment": (
            b1["training_environment"] == ru["training_environment"]
        ),
        "same_evaluation_environment": (
            b1["evaluation_environment"] == ru["evaluation_environment"]
        ),
        "checkpoint_steps_are_29999": (
            b1["checkpoint"]["step"] == EXPECTED_CHECKPOINT_STEP
            and ru["checkpoint"]["step"] == EXPECTED_CHECKPOINT_STEP
        ),
        "standard_checkpoint_keys": (
            b1["checkpoint"]["keys"] == ["splats", "step"]
            and ru["checkpoint"]["keys"] == ["splats", "step"]
        ),
        "filename_keyword_protocol": (
            b1["split"].get("protocol") == EXPECTED_SPLIT_PROTOCOL
            and ru["split"].get("protocol") == EXPECTED_SPLIT_PROTOCOL
            and b1["split"].get("train_keyword") == EXPECTED_TRAIN_KEYWORD
            and ru["split"].get("train_keyword") == EXPECTED_TRAIN_KEYWORD
            and b1["split"].get("test_keyword") == EXPECTED_TEST_KEYWORD
            and ru["split"].get("test_keyword") == EXPECTED_TEST_KEYWORD
        ),
        "training_commands_request_same_split": all(
            "--train_keyword clutter" in run["run_command"]
            and "--test_keyword extra" in run["run_command"]
            for run in (b1, ru)
        ),
    }
    mismatches = [name for name, passed in checks.items() if not passed]
    decision = "ANDROID_PAIRING_VALID" if not mismatches else "ANDROID_PAIRING_INVALID"
    return {
        "protocol": "puri-gs-ru-android-pairing-audit-1",
        "decision": decision,
        "checks": checks,
        "mismatches": mismatches,
        "b1": {
            key: b1[key]
            for key in (
                "path",
                "git_commit",
                "protocol_config",
                "training_environment",
                "evaluation_environment",
                "checkpoint",
            )
        },
        "ru": {
            key: ru[key]
            for key in (
                "path",
                "git_commit",
                "protocol_config",
                "training_environment",
                "evaluation_environment",
                "checkpoint",
            )
        },
        "split": {
            "protocol": b1["split"].get("protocol"),
            "train_keyword": b1["split"].get("train_keyword"),
            "test_keyword": b1["split"].get("test_keyword"),
            "train_count": len(b1["train_names"]),
            "test_count": len(b1["test_names"]),
            "train_names": b1["train_names"],
            "test_names": b1["test_names"],
        },
        "historical_b1_protocol_difference": {
            "historical_psnr": 24.31206703186035,
            "current_psnr": 23.479188919067383,
            "historical_training_commit": "32a7d787d6886d847061a853c66f7de90f2eebe4",
            "historical_config": "configs/puri_gs_b1_absgrad.yaml",
            "historical_step_budget": 10000,
            "historical_checkpoint_step": 9999,
            "current_config": "configs/puri_gs_b1_full30k.yaml",
            "current_step_budget": 30000,
            "current_checkpoint_step": 29999,
            "known_same_fields": [
                "seed=42",
                "data_factor=4",
                "SH degree=3",
                "AbsGrad=true",
                "grow_grad2d=0.0006",
                "filename clutter/extra split (122/19)",
            ],
            "interpretation": "The historical 10k B1 is a different protocol and must not replace the paired current 30k B1.",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    rows = [
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items()
    ]
    mismatch_text = (
        "无。" if not report["mismatches"] else ", ".join(report["mismatches"])
    )
    return "\n".join(
        [
            "# Android B1/RU 严格配对协议审计",
            "",
            f"最终状态：`{report['decision']}`",
            "",
            "| 检查 | 结果 |",
            "| --- | --- |",
            *rows,
            "",
            f"不一致项：{mismatch_text}",
            "",
            "## 当前 30k 协议",
            "",
            f"- B1 commit：`{report['b1']['git_commit']}`",
            f"- RU commit：`{report['ru']['git_commit']}`",
            f"- 训练/测试图：{report['split']['train_count']}/{report['split']['test_count']}",
            "- 共同字段：seed 42、30k、factor 4、SH 3、SSIM 0.2、AbsGrad、grow_grad2d 0.0006。",
            "- checkpoint 已真实只读加载，并核对 step、标准键和 Gaussian 数。",
            "- 独立评测必须是 splats-only 标准路径，且逐图 CSV 顺序与 test split 完全一致。",
            "",
            "## 历史 B1 差异",
            "",
            "历史 Android B1 PSNR 24.312067 来自 10k 短筛：训练 commit `32a7d787...`、",
            "`puri_gs_b1_absgrad.yaml`、checkpoint step 9999。当前 B1 PSNR 23.479189 来自",
            "固定 30k 配置 `puri_gs_b1_full30k.yaml`、checkpoint step 29999。两者共同保持",
            "seed 42、factor 4、SH 3、AbsGrad 0.0006 和 clutter/extra 122/19 split；实际不同",
            "项是训练 commit、配置入口、step budget 与 checkpoint step。因此历史 10k B1 只能标记为",
            "不同协议，不能替换当前配对 B1。",
            "",
            "如果最终状态为 `ANDROID_PAIRING_INVALID`，不得继续 Android 3+3；只根据 JSON 的",
            "`mismatches` 决定最小重跑，不自动启动训练。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1-run", type=Path, required=True)
    parser.add_argument("--ru-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    json_path = output_dir / "android_ru_pairing_audit.json"
    markdown_path = output_dir / "ANDROID_RU_PAIRING_AUDIT.md"
    for target in (json_path, markdown_path):
        if target.exists():
            raise RuntimeError(f"refusing to overwrite an existing pairing report: {target}")
    b1 = _load_run(args.b1_run.expanduser().resolve(), "b1")
    ru = _load_run(args.ru_run.expanduser().resolve(), "ru")
    report = decide(b1, ru)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(report["decision"])
    if report["mismatches"]:
        print("MISMATCHES=" + ",".join(report["mismatches"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
