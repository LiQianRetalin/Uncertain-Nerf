#!/usr/bin/env python3
"""Audit frozen Corner float predictions without computing quality metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-count", type=int, default=20)
    args = parser.parse_args()
    data = args.data_dir.expanduser().resolve()
    predictions = args.prediction_dir.expanduser().resolve()
    protocol_path = data / "common_input_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    test = [row for row in protocol["images"] if row["split"] == "extra"]
    if len(test) != args.required_count:
        raise RuntimeError("frozen Corner test count mismatch")
    expected = [Path(row["name"]).with_suffix(".npy").name for row in test]
    actual = sorted(path.name for path in predictions.glob("*.npy"))
    if actual != sorted(expected):
        raise RuntimeError("prediction list differs from frozen Corner test list")
    records = []
    for row in test:
        gt_path = data / "images" / row["name"]
        if sha256_file(gt_path) != row["sha256"]:
            raise RuntimeError(f"ground-truth hash mismatch: {row['name']}")
        with Image.open(gt_path) as image:
            width, height = image.size
        path = predictions / Path(row["name"]).with_suffix(".npy").name
        value = np.load(path, allow_pickle=False)
        if value.shape != (height, width, 3):
            raise RuntimeError(f"prediction shape mismatch: {path.name}")
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError(f"prediction dtype/finiteness mismatch: {path.name}")
        if float(value.min()) < -1e-6 or float(value.max()) > 1.000001:
            raise RuntimeError(f"prediction range mismatch: {path.name}")
        records.append(
            {
                "image_name": row["name"],
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "minimum": float(value.min()),
                "maximum": float(value.max()),
                "sha256": sha256_file(path),
            }
        )
    result = {
        "schema": "puri-gs-p03-prediction-audit-v1",
        "status": "PASS",
        "run_id": args.run_id,
        "test_count": len(records),
        "protocol_sha256": sha256_file(protocol_path),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"P03_PREDICTION_AUDIT_PASS={args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
