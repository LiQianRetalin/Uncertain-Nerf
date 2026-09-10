#!/usr/bin/env python3
"""Evaluate native float predictions with the frozen P01 metric contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    protocol_path = data_dir / "common_input_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    test_records = [
        record for record in protocol["images"] if "extra" in record["name"].casefold()
    ]
    expected_predictions = [Path(record["name"]).with_suffix(".npy").name for record in test_records]
    actual_predictions = sorted(path.name for path in prediction_dir.glob("*.npy"))
    if actual_predictions != sorted(expected_predictions):
        raise RuntimeError("prediction file list differs from the frozen test list")

    device = torch.device(args.device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)
    rows = []
    for index, record in enumerate(test_records):
        gt_path = data_dir / "images" / record["name"]
        if sha256_file(gt_path) != record["sha256"]:
            raise RuntimeError(f"ground-truth image hash mismatch: {record['name']}")
        gt_array = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        pred_path = prediction_dir / Path(record["name"]).with_suffix(".npy").name
        pred_array = np.load(pred_path, allow_pickle=False)
        if pred_array.shape != gt_array.shape:
            raise RuntimeError(
                f"prediction shape mismatch for {record['name']}: {pred_array.shape}"
            )
        if not np.isfinite(pred_array).all():
            raise RuntimeError(f"non-finite prediction: {record['name']}")
        if pred_array.min() < -1e-6 or pred_array.max() > 1.000001:
            raise RuntimeError(f"prediction is not clamped to [0,1]: {record['name']}")
        gt = torch.from_numpy(gt_array).permute(2, 0, 1).unsqueeze(0).to(device)
        pred = torch.from_numpy(pred_array).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            values = {
                "psnr": float(psnr(pred, gt).item()),
                "ssim": float(ssim(pred, gt).item()),
                "lpips": float(lpips(pred, gt).item()),
            }
        psnr.reset()
        ssim.reset()
        lpips.reset()
        rows.append(
            {
                "run_id": args.run_id,
                "scene": args.scene,
                "method": args.method,
                "seed": 42,
                "image_index": index,
                "image_name": record["name"],
                **values,
                "prediction_sha256": sha256_file(pred_path),
                "checkpoint_sha256": args.checkpoint_sha256,
                "evaluator_id": "puri-gs-gsplat153-independent-eval-v1",
                "source": "Reproduced/common-input",
            }
        )
        print(f"P02_EVALUATED={index + 1}/{len(test_records)}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "per_image_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "puri-gs-p02-independent-evaluation-v1",
        "status": "VALID",
        "run_id": args.run_id,
        "scene": args.scene,
        "method": args.method,
        "seed": 42,
        "test_image_count": len(rows),
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "lpips": float(np.mean([row["lpips"] for row in rows])),
        "checkpoint_sha256": args.checkpoint_sha256,
        "protocol_sha256": sha256_file(protocol_path),
        "evaluator_id": "puri-gs-gsplat153-independent-eval-v1",
        "source": "Reproduced/common-input",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("P02_INDEPENDENT_EVALUATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
