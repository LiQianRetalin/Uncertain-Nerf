#!/usr/bin/env python3
"""Validate and freeze the complete P03 Corner input identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.ontogo import PROTOCOL_FILE, sha256_file
from puri_gs.ontogo_corner import (
    EXPECTED_FRAME_COUNT,
    EXPECTED_TEST_COUNT,
    EXPECTED_TRAIN_COUNT,
    EXPECTED_UNASSIGNED,
    OnTheGoCornerParser,
    validate_prepared_corner,
)
from tools.export_ontogo_corner_colmap_common import validate_common_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Original Corner release. Omit only for a server recheck of frozen transferred data.",
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--spotless-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=source, text=True, encoding="utf-8"
    ).strip()


def git_blob(source: Path, revision_path: str) -> bytes:
    return subprocess.check_output(["git", "show", revision_path], cwd=source)


def main() -> int:
    args = parse_args()
    raw = args.raw_dir.expanduser().resolve() if args.raw_dir else None
    prepared = args.prepared_dir.expanduser().resolve()
    common = args.common_dir.expanduser().resolve()
    spotless = args.spotless_source.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    contract = validate_prepared_corner(prepared, verify_image_hashes=True)
    common_validation = validate_common_input(prepared, common)
    common_protocol_path = common / "common_input_protocol.json"
    common_protocol = json.loads(common_protocol_path.read_text(encoding="utf-8"))
    if raw is not None:
        raw_transforms = raw / "transforms.json"
        raw_split = raw / "split.json"
        if sha256_file(raw_transforms) != contract["protocol"]["transforms_sha256"]:
            raise RuntimeError("raw/prepared transforms hashes differ")
        if sha256_file(raw_split) != contract["protocol"]["split_sha256"]:
            raise RuntimeError("raw/prepared split hashes differ")

    parser = OnTheGoCornerParser(str(prepared), normalize=False)
    camera_id = 1
    mapx = parser.mapx_dict.get(camera_id)
    mapy = parser.mapy_dict.get(camera_id)
    width, height = parser.imsize_dict[camera_id]
    roi = parser.roi_undist_dict.get(camera_id, (0, 0, width, height))
    raw_records = []
    max_pixel_abs_diff = 0
    for index, record in enumerate(contract["protocol"]["images"]):
        prepared_path = prepared / "images_4" / record["file"]
        common_path = common / "images" / record["file"]
        raw_size = [record["source_width"], record["source_height"]]
        if raw is not None:
            raw_path = raw / "images" / record["source_name"]
            if sha256_file(raw_path) != record["source_sha256"]:
                raise RuntimeError(f"raw image hash differs: {raw_path.name}")
            with Image.open(raw_path) as raw_image:
                raw_size = list(raw_image.size)
            if raw_size != [record["source_width"], record["source_height"]]:
                raise RuntimeError(f"raw image dimensions differ: {raw_path.name}")
        native = cv2.imread(str(prepared_path), cv2.IMREAD_COLOR)
        external = cv2.imread(str(common_path), cv2.IMREAD_COLOR)
        if native is None or external is None:
            raise RuntimeError(f"cannot decode prepared/common image {record['file']}")
        if mapx is not None and mapy is not None:
            native = cv2.remap(native, mapx, mapy, cv2.INTER_LINEAR)
        x, y, roi_width, roi_height = roi
        native = native[y : y + roi_height, x : x + roi_width]
        if native.shape != external.shape:
            raise RuntimeError(f"native/common image shape differs: {record['file']}")
        difference = int(
            np.max(np.abs(native.astype(np.int16) - external.astype(np.int16)))
        )
        max_pixel_abs_diff = max(max_pixel_abs_diff, difference)
        if difference != 0:
            raise RuntimeError(f"native/common pixels differ: {record['file']}")
        raw_records.append(
            {
                "frame_index": index,
                "source_name": record["source_name"],
                "source_sha256": record["source_sha256"],
                "source_size": raw_size,
                "prepared_name": record["file"],
                "prepared_sha256": record["sha256"],
                "common_sha256": sha256_file(common_path),
                "common_size": [external.shape[1], external.shape[0]],
            }
        )

    benchmark_path = spotless / "examples" / "sls_benchmark.sh"
    benchmark_bytes = git_blob(spotless, "HEAD:examples/sls_benchmark.sh")
    benchmark_lines = benchmark_bytes.decode("utf-8").splitlines()
    corner_line = next(
        index + 1
        for index, line in enumerate(benchmark_lines)
        if "corner" in line and "for SCENE" in line
    )
    default_path = spotless / "examples" / "spotless_trainer.py"
    default_bytes = git_blob(spotless, "HEAD:examples/spotless_trainer.py")
    default_lines = default_bytes.decode("utf-8").splitlines()
    lower_line = next(
        index + 1 for index, line in enumerate(default_lines) if "lower_bound: float" in line
    )
    upper_line = next(
        index + 1 for index, line in enumerate(default_lines) if "upper_bound: float" in line
    )
    spotless_commit = git(spotless, "rev-parse", "HEAD")
    if spotless_commit != "0caae3cc45bb1fddf86bd47e4a521888f5c49889":
        raise RuntimeError("SpotLessSplats source is not the frozen P03 commit")

    train_names = [
        record["name"]
        for record in common_protocol["images"]
        if record["split"] == "clutter"
    ]
    test_names = [
        record["name"]
        for record in common_protocol["images"]
        if record["split"] == "extra"
    ]
    excluded_names = [
        record["name"]
        for record in common_protocol["images"]
        if record["split"] == "excluded"
    ]
    if (
        len(raw_records),
        len(train_names),
        len(test_names),
        len(excluded_names),
    ) != (
        EXPECTED_FRAME_COUNT,
        EXPECTED_TRAIN_COUNT,
        EXPECTED_TEST_COUNT,
        len(EXPECTED_UNASSIGNED),
    ):
        raise RuntimeError("Corner counts differ from the frozen contract")

    validation = {
        "schema": "puri-gs-p03-input-validation-v1",
        "status": "PASS",
        "raw_image_count": len(raw_records),
        "raw_images_verified": raw is not None,
        "prepared_images_verified": True,
        "native_common_pixels_exact": True,
        "native_common_max_pixel_abs_diff": max_pixel_abs_diff,
        "camera_intrinsics_verified": common_validation["camera_intrinsics_verified"],
        "poses_verified": common_validation["poses_verified"],
        "points_verified": common_validation["points_verified"],
        "projection_matrices_verified": common_validation[
            "projection_matrices_verified"
        ],
        "common_validation": common_validation,
        "source_to_common_mapping": raw_records,
    }
    (output / "input_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "puri-gs-p03-protocol-manifest-v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "scene": "corner",
        "physical_data_factor": 4,
        "external_loader_data_factor": 1,
        "frame_count": len(raw_records),
        "train_count": len(train_names),
        "test_count": len(test_names),
        "excluded_count": len(excluded_names),
        "train_names": train_names,
        "test_names": test_names,
        "excluded_names": excluded_names,
        "representative_test_names": [
            test_names[0], test_names[len(test_names) // 2], test_names[-1]
        ],
        "prepared_protocol_file": PROTOCOL_FILE,
        "prepared_protocol_sha256": sha256_file(prepared / PROTOCOL_FILE),
        "common_protocol_file": common_protocol_path.name,
        "common_protocol_sha256": sha256_file(common_protocol_path),
        "common_validation_file": "common_input_validation.json",
        "common_validation_sha256": sha256_file(
            common / "common_input_validation.json"
        ),
        "image_size": [common_protocol["image_width"], common_protocol["image_height"]],
        "initial_point_count": common_protocol["initial_point_count"],
        "ordered_image_content_sha256": json_hash(
            [[row["prepared_name"], row["common_sha256"]] for row in raw_records]
        ),
        "split_policy": "official Corner split.json; clutter=train, extra=test, frame 121 excluded",
        "test_gt_use": "ordinary post-training independent evaluation only",
        "sls_recipe": {
            "loss_type": "robust",
            "semantics": True,
            "cluster": False,
            "ubp": False,
            "lower_bound": 0.5,
            "upper_bound": 0.9,
            "source_commit": spotless_commit,
            "benchmark_file": "examples/sls_benchmark.sh",
            "benchmark_file_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
            "corner_general_recipe_line": corner_line,
            "defaults_file": "examples/spotless_trainer.py",
            "defaults_file_sha256": hashlib.sha256(default_bytes).hexdigest(),
            "lower_bound_line": lower_line,
            "upper_bound_line": upper_line,
            "basis": "Corner is in the general SLS-mlp loop; only patio-high and spot receive 0.3/0.8 overrides",
        },
    }
    (output / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("P03_INPUT_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
