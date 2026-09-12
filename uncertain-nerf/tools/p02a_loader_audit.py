#!/usr/bin/env python3
"""Instantiate the frozen P02 loaders and audit split/feature identities without training."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("robustsplat", "sls-mlp"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def protocol_lists(protocol):
    names = [record["name"] for record in protocol["images"]]
    train = [name for name in names if "clutter" in name.casefold()]
    test = [name for name in names if "extra" in name.casefold()]
    excluded = [name for name in names if name not in set(train + test)]
    return names, train, test, excluded


def audit_robust(source, data_dir, expected):
    sys.path.insert(0, str(source))
    from scene.dataset_readers import readColmapSceneInfo

    scene = readColmapSceneInfo(
        str(data_dir), images=None, depths="", eval=True,
        train_test_exp=False, llffhold=8,
    )
    train = [camera.image_name for camera in scene.train_cameras]
    test = [camera.image_name for camera in scene.test_cameras]
    loaded = train + test
    initial_points = None if scene.point_cloud is None else int(len(scene.point_cloud.points))
    image_sizes = sorted({
        (int(camera.width), int(camera.height))
        for camera in scene.train_cameras + scene.test_cameras
    })
    return {
        "loader_class": "scene.dataset_readers.readColmapSceneInfo",
        "loader_factor": 1,
        "native_normalization": "getNerfppNorm(train_cam_infos)",
        "train_names": train,
        "test_names": test,
        "excluded_names": [name for name in expected[0] if name not in set(loaded)],
        "feature_fit_names": train,
        "image_sizes": [list(item) for item in image_sizes],
        "initial_point_count": initial_points,
        "feature_files": [],
    }


def audit_sls(source, data_dir, expected):
    sys.path.insert(0, str(source / "examples"))
    sys.path.insert(0, str(source))
    from datasets.colmap import ClutterDataset, SemanticParser

    parser = SemanticParser(
        data_dir=str(data_dir), factor=1, normalize=True,
        load_keyword="clutter", semantic_dir="SD", cluster=False,
    )
    trainset = ClutterDataset(
        parser, split="train", train_keyword="clutter",
        test_keyword="extra", semantics=True,
    )
    testset = ClutterDataset(
        parser, split="test", train_keyword="clutter",
        test_keyword="extra", semantics=False,
    )
    train = [parser.image_names[int(index)] for index in trainset.indices]
    test = [parser.image_names[int(index)] for index in testset.indices]
    loaded = train + test
    feature_rows = []
    for name in train:
        path = data_dir / "SD" / Path(name).with_suffix(".npy").name
        array = np.load(str(path), mmap_mode="r", allow_pickle=False)
        feature_rows.append({
            "image_name": name,
            "image_sha256": sha256_file(data_dir / "images" / name),
            "feature_file": path.name,
            "feature_sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": bool(np.isfinite(array).all()),
            "bytes": int(path.stat().st_size),
        })
    image_sizes = sorted({
        tuple(int(value) for value in parser.imsize_dict[camera_id])
        for camera_id in parser.imsize_dict
    })
    return {
        "loader_class": "datasets.colmap.SemanticParser+ClutterDataset",
        "loader_factor": 1,
        "native_normalization": "SemanticParser(normalize=True)",
        "train_names": train,
        "test_names": test,
        "excluded_names": [name for name in expected[0] if name not in set(loaded)],
        "feature_fit_names": [row["image_name"] for row in feature_rows],
        "image_sizes": [list(item) for item in image_sizes],
        "initial_point_count": int(len(parser.points)),
        "feature_files": feature_rows,
    }


def main():
    args = parse_args()
    source = args.source.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    protocol_path = data_dir / "common_input_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol_lists(protocol)
    if args.method == "robustsplat":
        result = audit_robust(source, data_dir, expected)
    else:
        result = audit_sls(source, data_dir, expected)

    expected_all, expected_train, expected_test, expected_excluded = expected
    checks = {
        "train_names_exact": result["train_names"] == expected_train,
        "test_names_exact": result["test_names"] == expected_test,
        "excluded_names_exact": result["excluded_names"] == expected_excluded,
        "feature_fit_names_exact": result["feature_fit_names"] == expected_train,
        "image_size_exact": result["image_sizes"] == [[
            int(protocol["image_width"]), int(protocol["image_height"])
        ]],
        "initial_point_count_exact": result["initial_point_count"] == int(protocol["initial_point_count"]),
        "feature_count_exact": (
            len(result["feature_files"]) == len(expected_train)
            if args.method == "sls-mlp" else True
        ),
        "feature_shape_dtype_finite": (
            all(
                row["shape"] == [1280, 50, 50]
                and row["dtype"] == "float32"
                and row["finite"]
                for row in result["feature_files"]
            ) if args.method == "sls-mlp" else True
        ),
    }
    output = {
        "schema": "puri-gs-p02a-loader-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": args.method,
        "source": str(source),
        "source_file_sha256": {
            "robust_dataset_readers": (
                sha256_file(source / "scene" / "dataset_readers.py")
                if args.method == "robustsplat" else None
            ),
            "sls_colmap": (
                sha256_file(source / "examples" / "datasets" / "colmap.py")
                if args.method == "sls-mlp" else None
            ),
        },
        "data_dir": str(data_dir),
        "protocol_sha256": sha256_file(protocol_path),
        "expected_counts": {
            "all": len(expected_all), "train": len(expected_train),
            "test": len(expected_test), "excluded": len(expected_excluded),
        },
        "checks": checks,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("P02A_LOADER_AUDIT_{}_{}".format(args.method.upper().replace("-", "_"), output["status"]))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
