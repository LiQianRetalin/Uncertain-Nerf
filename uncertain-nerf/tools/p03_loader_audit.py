#!/usr/bin/env python3
"""Combine the three native Corner loader identities into one P03 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.ontogo_corner import OnTheGoCornerParser, validate_prepared_corner  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--ru-feature-validation", type=Path, required=True)
    parser.add_argument("--robust-audit", type=Path, required=True)
    parser.add_argument("--sls-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared = args.prepared_dir.expanduser().resolve()
    common = args.common_dir.expanduser().resolve()
    gsplat_examples = args.gsplat_dir.expanduser().resolve() / "examples"
    normalize_module = gsplat_examples / "datasets" / "normalize.py"
    if not normalize_module.is_file():
        raise FileNotFoundError(f"missing gsplat normalization module: {normalize_module}")
    if str(gsplat_examples) not in sys.path:
        sys.path.insert(0, str(gsplat_examples))
    contract = validate_prepared_corner(prepared, verify_image_hashes=True)
    native = OnTheGoCornerParser(str(prepared), factor=4, normalize=True)
    common_protocol_path = common / "common_input_protocol.json"
    common_protocol = json.loads(common_protocol_path.read_text(encoding="utf-8"))
    expected_train = [row["name"] for row in common_protocol["images"] if row["split"] == "clutter"]
    expected_test = [row["name"] for row in common_protocol["images"] if row["split"] == "extra"]
    expected_excluded = [row["name"] for row in common_protocol["images"] if row["split"] == "excluded"]
    native_train = [native.image_names[index] for index in contract["train_indices"]]
    native_test = [native.image_names[index] for index in contract["test_indices"]]
    native_excluded = [native.image_names[index] for index in range(len(native.image_names)) if index not in set(contract["train_indices"] + contract["test_indices"])]
    feature = json.loads(args.ru_feature_validation.read_text(encoding="utf-8"))
    robust = json.loads(args.robust_audit.read_text(encoding="utf-8"))
    sls = json.loads(args.sls_audit.read_text(encoding="utf-8"))
    internal_checks = {
        "train_names_exact": native_train == expected_train,
        "test_names_exact": native_test == expected_test,
        "excluded_names_exact": native_excluded == expected_excluded,
        "feature_fit_names_exact": [row["image_name"] for row in feature["records"]] == expected_train,
        "image_size_exact": list(native.imsize_dict[1]) == [common_protocol["image_width"], common_protocol["image_height"]],
        "initial_point_count_exact": len(native.points) == common_protocol["initial_point_count"],
        "feature_count_exact": feature["feature_count"] == 101,
    }
    status = "PASS" if all(internal_checks.values()) and robust["status"] == "PASS" and sls["status"] == "PASS" else "FAIL"
    result = {
        "schema": "puri-gs-p03-loader-audit-v1",
        "status": status,
        "protocol_sha256": sha256_file(common_protocol_path),
        "expected_counts": {"all": 122, "train": 101, "test": 20, "excluded": 1},
        "internal": {
            "loader_class": "puri_gs.ontogo_corner.OnTheGoCornerParser + gsplat Dataset",
            "physical_factor": 4,
            "loader_factor": 4,
            "native_normalization": "similarity_from_cameras + align_principal_axes",
            "train_names": native_train,
            "test_names": native_test,
            "excluded_names": native_excluded,
            "feature_fit_names": [row["image_name"] for row in feature["records"]],
            "image_size": list(native.imsize_dict[1]),
            "initial_point_count": len(native.points),
            "checks": internal_checks,
        },
        "robustsplat": robust,
        "sls_mlp": sls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"P03_LOADER_AUDIT={status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
