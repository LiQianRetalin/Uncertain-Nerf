#!/usr/bin/env python3
"""Independently evaluate full-resolution Corner float predictions for P03."""

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
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    data = args.data_dir.expanduser().resolve()
    predictions = args.prediction_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    protocol_path = data / "common_input_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    test = [row for row in protocol["images"] if row["split"] == "extra"]
    if len(test) != 20:
        raise RuntimeError("P03 requires exactly 20 Corner test images")
    expected = sorted(Path(row["name"]).with_suffix(".npy").name for row in test)
    if sorted(path.name for path in predictions.glob("*.npy")) != expected:
        raise RuntimeError("prediction list differs from frozen Corner test list")

    device = torch.device(args.device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    rows = []
    for index, record in enumerate(test):
        gt_path = data / "images" / record["name"]
        if sha256_file(gt_path) != record["sha256"]:
            raise RuntimeError(f"ground-truth hash mismatch: {record['name']}")
        gt_array = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        pred_path = predictions / Path(record["name"]).with_suffix(".npy").name
        pred_array = np.load(pred_path, allow_pickle=False)
        if pred_array.shape != gt_array.shape or pred_array.dtype != np.float32:
            raise RuntimeError(f"prediction shape/dtype mismatch: {pred_path.name}")
        if not np.isfinite(pred_array).all() or pred_array.min() < -1e-6 or pred_array.max() > 1.000001:
            raise RuntimeError(f"prediction is non-finite or unclamped: {pred_path.name}")
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
                "scene": "corner",
                "method": args.method,
                "seed": 42,
                "image_index": index,
                "image_name": record["name"],
                **values,
                "prediction_sha256": sha256_file(pred_path),
                "checkpoint_sha256": args.checkpoint_sha256,
                "evaluator_id": "puri-gs-gsplat153-independent-eval-p03-v1",
                "lpips_network": "alex",
                "lpips_normalize": True,
                "source": "Reproduced/common-input",
            }
        )
        print(f"P03_EVALUATED={index + 1}/20", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "puri-gs-p03-independent-evaluation-v1",
        "status": "VALID",
        "run_id": args.run_id,
        "scene": "corner",
        "method": args.method,
        "seed": 42,
        "test_image_count": len(rows),
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "lpips": float(np.mean([row["lpips"] for row in rows])),
        "checkpoint_sha256": args.checkpoint_sha256,
        "protocol_sha256": sha256_file(protocol_path),
        "evaluator_id": "puri-gs-gsplat153-independent-eval-p03-v1",
        "lpips_network": "alex",
        "lpips_normalize": True,
        "source": "Reproduced/common-input",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("P03_INDEPENDENT_EVALUATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
