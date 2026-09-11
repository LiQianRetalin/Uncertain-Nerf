#!/usr/bin/env python3
"""Export the fixed P02-A common-view comparison panels from existing predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCENES = {
    "android": {
        "fixed": ["1extra000.png", "1extra009.png", "1extra018.png"],
        "diagnostic": [],
        "p01_group": "ru-generalization-rerun-9e292309",
    },
    "patio_high": {
        "fixed": [
            "extra_0000_IMG_8467.png",
            "extra_0022_IMG_8489.png",
            "extra_0044_IMG_8511.png",
        ],
        "diagnostic": ["extra_0034_IMG_8501.png", "extra_0035_IMG_8502.png"],
        "p01_group": "ru-generalization-ontogo-749d584b",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_internal_canvas(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    canvas = load_rgb(path)
    if canvas.shape[0] != expected_shape[0] or canvas.shape[1] != expected_shape[1] * 2:
        raise RuntimeError(f"unexpected P01 canvas shape {canvas.shape}: {path}")
    return canvas[:, expected_shape[1] :, :]


def load_float_prediction(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise RuntimeError(f"invalid float prediction {array.shape}: {path}")
    if float(array.min()) < -1e-6 or float(array.max()) > 1.000001:
        raise RuntimeError(f"prediction outside [0,1]: {path}")
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def error_image(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    error = np.abs(gt.astype(np.float32) - pred.astype(np.float32)).mean(axis=2) / 255.0
    scaled = np.clip(error / 0.25, 0.0, 1.0)
    red = np.rint(255.0 * scaled).astype(np.uint8)
    green = np.rint(255.0 * np.maximum(0.0, 1.0 - np.abs(scaled - 0.5) * 2.0)).astype(np.uint8)
    blue = np.rint(255.0 * (1.0 - scaled)).astype(np.uint8)
    return np.stack([red, green, blue], axis=2)


def tile(array: np.ndarray, label: str, width: int = 503, height: int = 378) -> Image.Image:
    image = Image.fromarray(array, mode="RGB").resize((width, height), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (width, height + 28), "white")
    result.paste(image, (0, 28))
    ImageDraw.Draw(result).text((8, 7), label, fill="black")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--android-data", type=Path, required=True)
    parser.add_argument("--patio-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    code_root = args.repo_root.resolve() / "uncertain-nerf"
    work_root = args.work_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_roots = {"android": args.android_data.resolve(), "patio_high": args.patio_data.resolve()}
    manifest_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for scene, spec in SCENES.items():
        data_root = data_roots[scene]
        protocol = json.loads((data_root / "common_input_protocol.json").read_text(encoding="utf-8"))
        test_names = [row["name"] for row in protocol["images"] if "extra" in row["name"].casefold()]
        fixed_expected = [test_names[0], test_names[len(test_names) // 2], test_names[-1]]
        if fixed_expected != spec["fixed"]:
            raise RuntimeError(f"frozen representative rule mismatch for {scene}: {fixed_expected}")

        internal_base = code_root / "logs-puri" / spec["p01_group"]
        robust_base = work_root / "outputs" / f"P02-{scene}-robustsplat" / "test" / "ours_30000" / "float_predictions"
        sls_base = work_root / "outputs" / f"P02-{scene}-sls-mlp" / "renders" / "float_step29999"
        for role, name in [("common", item) for item in spec["fixed"]] + [("diagnostic", item) for item in spec["diagnostic"]]:
            index = test_names.index(name)
            gt_path = data_root / "images" / name
            gt = load_rgb(gt_path)
            predictions = {
                "B1": load_internal_canvas(
                    internal_base / f"{scene}_b1_30k" / "independent_eval" / "renders" / f"test_step29999_{index:04d}.png",
                    gt.shape,
                ),
                "RU": load_internal_canvas(
                    internal_base / f"{scene}_ru_30k" / "independent_eval" / "renders" / f"test_step29999_{index:04d}.png",
                    gt.shape,
                ),
                "RobustSplat": load_float_prediction(robust_base / Path(name).with_suffix(".npy").name, gt.shape),
                "SLS-MLP": load_float_prediction(sls_base / Path(name).with_suffix(".npy").name, gt.shape),
            }
            top = [tile(gt, f"GT | {name}")] + [tile(predictions[key], key) for key in predictions]
            blank = np.full_like(gt, 255)
            bottom = [tile(blank, "Error scale: 0 (blue) to >=0.25 (red)")] + [
                tile(error_image(gt, predictions[key]), f"|GT-{key}|") for key in predictions
            ]
            panel = Image.new("RGB", (sum(item.width for item in top), top[0].height + bottom[0].height), "white")
            x = 0
            for item in top:
                panel.paste(item, (x, 0)); x += item.width
            x = 0
            for item in bottom:
                panel.paste(item, (x, top[0].height)); x += item.width
            panel_path = output / f"{scene}-{role}-{index + 1:02d}-{Path(name).stem}.png"
            panel.save(panel_path, optimize=True)
            manifest_rows.append({
                "scene": scene, "selection_role": role, "test_index_1based": index + 1,
                "image_name": name, "panel_file": panel_path.name,
                "panel_sha256": sha256_file(panel_path), "width": gt.shape[1], "height": gt.shape[0],
                "gt_sha256": sha256_file(gt_path),
            })
            if role == "diagnostic":
                for method, pred in predictions.items():
                    diagnostic_rows.append({
                        "scene": scene, "image_name": name, "test_index_1based": index + 1,
                        "method": method,
                        "mean_absolute_rgb_error": float(np.abs(gt.astype(np.float32) - pred.astype(np.float32)).mean() / 255.0),
                        "mapping_shape_pass": pred.shape == gt.shape,
                    })

    with (output / "representatives_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader(); writer.writerows(manifest_rows)
    with (output / "patio_fixed_diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0]))
        writer.writeheader(); writer.writerows(diagnostic_rows)
    print(f"P02A_REPRESENTATIVE_PANELS_PASS={len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
