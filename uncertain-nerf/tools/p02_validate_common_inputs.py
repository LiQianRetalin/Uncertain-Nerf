#!/usr/bin/env python3
"""Validate P02 common inputs and pinned external source contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image


EXPECTED_COMMITS = {
    "robustsplat": "a130281d6d0c004032a9a57e8d6a14962d9836d3",
    "sls_mlp": "0caae3cc45bb1fddf86bd47e4a521888f5c49889",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--android-root", type=Path, required=True)
    parser.add_argument("--patio-root", type=Path, required=True)
    parser.add_argument("--robust-source", type=Path)
    parser.add_argument("--spotless-source", type=Path)
    parser.add_argument(
        "--execution-scope", choices=("local", "server"), default="local"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_dataset(
    root: Path, protocol_entry: dict[str, Any], scene: str
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    protocol_path = root / "common_input_protocol.json"
    validation_path = root / "common_input_validation.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    expected_decision = (
        "ANDROID_COMMON_COLMAP_VALID"
        if scene == "android"
        else "PATIO_HIGH_COMMON_COLMAP_VALID"
    )
    if validation["decision"] != expected_decision:
        raise RuntimeError(f"{scene}: validation decision mismatch")
    if sha256_file(protocol_path) != protocol_entry["common_input_protocol_sha256"]:
        raise RuntimeError(f"{scene}: protocol SHA-256 mismatch")
    if sha256_file(validation_path) != protocol_entry["common_input_validation_sha256"]:
        raise RuntimeError(f"{scene}: validation SHA-256 mismatch")
    for key in (
        "camera_intrinsics_verified",
        "poses_verified",
        "points_verified",
        "projection_matrices_verified",
        "image_hashes_verified",
    ):
        if not validation.get(key):
            raise RuntimeError(f"{scene}: prior validation flag is false: {key}")

    records = protocol["images"]
    expected_names = [record["name"] for record in records]
    actual_files = sorted(
        path for path in (root / "images").iterdir() if path.is_file()
    )
    actual_names = [path.name for path in actual_files]
    if actual_names != sorted(expected_names):
        raise RuntimeError(f"{scene}: actual image list differs from protocol")
    if protocol_entry["ordered_image_names"] != expected_names:
        raise RuntimeError(f"{scene}: P01 ordered image list differs from protocol")
    expected_content_sha = sha256_json(
        [[record["name"], record["sha256"]] for record in records]
    )
    if protocol_entry["image_content_manifest_sha256"] != expected_content_sha:
        raise RuntimeError(f"{scene}: P01 image-content manifest mismatch")

    for index, record in enumerate(records, 1):
        image_path = root / "images" / record["name"]
        if sha256_file(image_path) != record["sha256"]:
            raise RuntimeError(f"{scene}: image hash mismatch: {record['name']}")
        with Image.open(image_path) as image:
            if image.size != (protocol["image_width"], protocol["image_height"]):
                raise RuntimeError(f"{scene}: image size mismatch: {record['name']}")
        if index == 1 or index % 50 == 0 or index == len(records):
            print(f"P02_HASHED_{scene.upper()}={index}/{len(records)}", flush=True)

    sparse = root / "sparse" / "0"
    for name, field in (
        ("cameras.bin", "cameras_bin_sha256"),
        ("images.bin", "images_bin_sha256"),
        ("points3D.bin", "points3D_bin_sha256"),
    ):
        if sha256_file(sparse / name) != validation[field]:
            raise RuntimeError(f"{scene}: sparse file hash mismatch: {name}")

    train = [name for name in expected_names if "clutter" in name.casefold()]
    test = [name for name in expected_names if "extra" in name.casefold()]
    excluded = [name for name in expected_names if name not in set(train + test)]
    if (len(train), len(test), len(excluded)) != (
        protocol["train_count"],
        protocol["test_count"],
        protocol.get("excluded_clean_count", protocol.get("excluded_count")),
    ):
        raise RuntimeError(f"{scene}: keyword split count mismatch")
    return {
        "scene": scene,
        "status": "LOCAL_COMMON_INPUT_VALID",
        "root": str(root),
        "protocol_sha256": sha256_file(protocol_path),
        "validation_sha256": sha256_file(validation_path),
        "image_count": len(expected_names),
        "train_count": len(train),
        "test_count": len(test),
        "excluded_count": len(excluded),
        "image_size": [protocol["image_width"], protocol["image_height"]],
        "initial_point_count": protocol["initial_point_count"],
        "train_names": train,
        "test_names": test,
        "excluded_names": excluded,
        "sls_feature_names": [str(Path(name).with_suffix(".npy")) for name in train],
    }


def _git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=source, text=True, encoding="utf-8"
    ).strip()


def _validate_source(source: Path, method: str) -> dict[str, Any]:
    source = source.expanduser().resolve()
    commit = _git(source, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMITS[method]:
        raise RuntimeError(f"{method}: source commit mismatch: {commit}")
    required = (
        ["train.py", "render.py", "scene/dataset_readers.py", "utils/mask_utils.py"]
        if method == "robustsplat"
        else ["examples/spotless_trainer.py", "examples/datasets/colmap.py", "examples/requirements.txt"]
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"{method}: missing source files: {missing}")
    loader_path = source / (
        "scene/dataset_readers.py"
        if method == "robustsplat"
        else "examples/datasets/colmap.py"
    )
    loader_text = loader_path.read_text(encoding="utf-8")
    if "clutter" not in loader_text or "extra" not in loader_text:
        raise RuntimeError(f"{method}: keyword split contract absent")
    if method == "sls_mlp" and "load_keyword" not in loader_text:
        raise RuntimeError("sls_mlp: train-only feature load contract absent")
    return {
        "method": method,
        "status": "PINNED_SOURCE_STATIC_CONTRACT_VALID",
        "source": str(source),
        "commit": commit,
        "working_tree_status": _git(source, "status", "--short"),
        "required_file_sha256": {
            name: sha256_file(source / name) for name in required
        },
        "keyword_split_present": True,
        "train_only_feature_load_present": method == "sls_mlp",
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    protocol_by_id = {
        entry["protocol_id"]: entry for entry in manifest["protocols"]
    }
    datasets = [
        _validate_dataset(
            args.android_root,
            protocol_by_id["android-colmap-common-factor4-v1"],
            "android",
        ),
        _validate_dataset(
            args.patio_root,
            protocol_by_id["patio-high-colmap-common-factor4-v1"],
            "patio_high",
        ),
    ]
    sources = []
    if args.robust_source:
        sources.append(_validate_source(args.robust_source, "robustsplat"))
    if args.spotless_source:
        sources.append(_validate_source(args.spotless_source, "sls_mlp"))

    import csv

    with args.run_plan.open("r", encoding="utf-8-sig", newline="") as stream:
        run_plan = list(csv.DictReader(stream))
    if len(run_plan) != 4 or len({row["run_id"] for row in run_plan}) != 4:
        raise RuntimeError("P02 run plan must contain four unique identities")
    dataset_by_scene = {entry["scene"]: entry for entry in datasets}
    for row in run_plan:
        if row["protocol_id"] not in protocol_by_id:
            raise RuntimeError(f"unknown protocol in run plan: {row['protocol_id']}")
        if row["loader_data_factor"] != "1" or row["seed"] != "42":
            raise RuntimeError(f"factor/seed mismatch: {row['run_id']}")
        if row["train_keyword"] != "clutter" or row["test_keyword"] != "extra":
            raise RuntimeError(f"keyword mismatch: {row['run_id']}")
        if not dataset_by_scene[row["scene"]]["train_names"]:
            raise RuntimeError(f"empty train split: {row['run_id']}")

    is_server = args.execution_scope == "server"
    runtime_repository = None
    if args.repository_root:
        repository_root = args.repository_root.expanduser().resolve()
        runtime_repository = {
            "root": str(repository_root),
            "branch": _git(repository_root, "branch", "--show-current"),
            "head": _git(repository_root, "rev-parse", "HEAD"),
            "origin_ru_part": _git(repository_root, "rev-parse", "origin/ru-part"),
            "working_tree_status": _git(repository_root, "status", "--short"),
        }
    result = {
        "schema": "puri-gs-p02-local-preflight-v1",
        "status": (
            "SERVER_COMMON_PREFLIGHT_VALID"
            if is_server
            else "LOCAL_PREPARATION_VALID_SERVER_AUTH_BLOCKED"
        ),
        "p01_repository": manifest["repository"],
        "runtime_repository": runtime_repository,
        "datasets": datasets,
        "sources": sources,
        "run_plan_count": len(run_plan),
        "gpu_priority": manifest["gpu_priority"],
        "server_checks": (
            {
                "authentication": "PASS_RUNNING_ON_SERVER",
                "repository": "CHECKED_BY_PIPELINE",
                "datasets": "COMMON_INPUT_VALID",
                "gpu_inventory": "PENDING",
                "environments": "PENDING",
                "smoke": "NOT_RUN",
                "formal_training": "NOT_RUN",
            }
            if is_server
            else {
                "ssh_target": "chenglong@172.16.55.2",
                "authentication": "BLOCKED_PERMISSION_DENIED_PUBLICKEY_PASSWORD",
                "repository": "NOT_CHECKED",
                "datasets": "NOT_CHECKED_OR_TRANSFERRED",
                "gpu_inventory": "NOT_CHECKED",
                "environments": "NOT_CREATED",
                "smoke": "NOT_RUN",
                "formal_training": "NOT_RUN",
            }
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "P02_SERVER_COMMON_PREFLIGHT=PASS"
        if is_server
        else "P02_LOCAL_PREFLIGHT=PASS_SERVER_AUTH_BLOCKED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
