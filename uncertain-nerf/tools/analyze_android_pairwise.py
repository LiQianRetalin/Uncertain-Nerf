#!/usr/bin/env python3
"""Paired per-image analysis for existing PURI-GS Android 10k results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


PROFILES = ("b0", "b1", "a1")
METRICS = ("psnr", "ssim", "lpips")
BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 10_000


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _ensure_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite existing analysis outputs: {existing}")


def split_eval_canvas(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the embedded GT and render as float RGB arrays in [0, 1]."""

    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if image.shape[1] % 2:
        raise RuntimeError(f"evaluation canvas width is not even: {path}")
    middle = image.shape[1] // 2
    return image[:, :middle], image[:, middle:]


def build_file_manifest(
    result_root: Path, *, step: int = 9999, expected_count: int = 19
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate profile splits, render names, ordering, and embedded GT pixels."""

    split_by_profile: dict[str, dict[str, Any]] = {}
    files_by_profile: dict[str, list[Path]] = {}
    for profile in PROFILES:
        train_split = _read_json(result_root / profile / "dataset_split.json")
        eval_split = _read_json(result_root / f"{profile}_eval" / "dataset_split.json")
        if train_split.get("test") != eval_split.get("test"):
            raise RuntimeError(f"{profile} train/eval test lists differ")
        split_by_profile[profile] = train_split
        render_dir = result_root / f"{profile}_eval" / "renders"
        files = sorted(render_dir.glob(f"test_step{step}_*.png"))
        if len(files) != expected_count:
            raise RuntimeError(
                f"expected {expected_count} {profile} renders, got {len(files)}"
            )
        files_by_profile[profile] = files

    reference_test = split_by_profile["b0"].get("test", [])
    if len(reference_test) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} ground-truth names, got {len(reference_test)}"
        )
    for profile in PROFILES[1:]:
        if split_by_profile[profile].get("test") != reference_test:
            raise RuntimeError(f"test image ordering differs for {profile}")

    reference_output_names = [path.name for path in files_by_profile["b0"]]
    for profile in PROFILES[1:]:
        output_names = [path.name for path in files_by_profile[profile]]
        if output_names != reference_output_names:
            raise RuntimeError(f"test render filename set/order differs for {profile}")

    ordering: list[dict[str, str]] = []
    for index, image_name in enumerate(reference_test):
        output_name = reference_output_names[index]
        reference_gt, _ = split_eval_canvas(files_by_profile["b0"][index])
        for profile in PROFILES[1:]:
            candidate_gt, _ = split_eval_canvas(files_by_profile[profile][index])
            if not np.array_equal(reference_gt, candidate_gt):
                raise RuntimeError(
                    f"embedded ground truth differs for {profile}: {output_name}"
                )
        ordering.append(
            {
                "eval_index": str(index),
                "image_name": image_name,
                "evaluation_canvas": output_name,
            }
        )

    manifest = {
        "expected_count": expected_count,
        "ground_truth_source": "left half of each official evaluation canvas",
        "ground_truth_files": reference_test,
        "B0_render_files": reference_output_names,
        "B1_render_files": [path.name for path in files_by_profile["b1"]],
        "A1_render_files": [path.name for path in files_by_profile["a1"]],
        "missing_files": {profile: [] for profile in PROFILES},
        "extra_files": {profile: [] for profile in PROFILES},
        "ordering": ordering,
        "embedded_ground_truth_identical": True,
    }
    rows = [
        {
            "image_name": item["image_name"],
            "eval_file": item["evaluation_canvas"],
        }
        for item in ordering
    ]
    return manifest, rows


def compute_per_image_metrics(
    result_root: Path,
    rows: list[dict[str, str]],
    *,
    device: str,
    step: int = 9999,
) -> list[dict[str, Any]]:
    """Use the same TorchMetrics implementations and LPIPS settings as gsplat."""

    import torch
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric_modules: dict[str, tuple[Any, Any, Any]] = {}
    for profile in PROFILES:
        metric_modules[profile] = (
            PeakSignalNoiseRatio(data_range=1.0).to(device),
            StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
            LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(device),
        )

    output: list[dict[str, Any]] = []
    with torch.no_grad():
        for input_row in rows:
            row: dict[str, Any] = dict(input_row)
            for profile in PROFILES:
                path = (
                    result_root
                    / f"{profile}_eval"
                    / "renders"
                    / input_row["eval_file"]
                )
                gt, render = split_eval_canvas(path)
                gt_tensor = (
                    torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
                )
                render_tensor = (
                    torch.from_numpy(render).permute(2, 0, 1).unsqueeze(0).to(device)
                )
                psnr, ssim, lpips = metric_modules[profile]
                row[f"{profile.upper()}_psnr"] = float(
                    psnr(render_tensor, gt_tensor).item()
                )
                row[f"{profile.upper()}_ssim"] = float(
                    ssim(render_tensor, gt_tensor).item()
                )
                row[f"{profile.upper()}_lpips"] = float(
                    lpips(render_tensor, gt_tensor).item()
                )
            for baseline in ("B0", "B1"):
                for metric in METRICS:
                    row[f"A1_minus_{baseline}_{metric}"] = (
                        row[f"A1_{metric}"] - row[f"{baseline}_{metric}"]
                    )
            output.append(row)
    return output


def validate_metric_consistency(
    result_root: Path, rows: list[dict[str, Any]], *, step: int = 9999
) -> dict[str, Any]:
    """Compare recomputed 8-bit-canvas means with official pre-save metrics."""

    tolerances = {"psnr": 0.05, "ssim": 0.002, "lpips": 0.002}
    result: dict[str, Any] = {}
    for profile in PROFILES:
        official = _read_json(
            result_root / f"{profile}_eval" / "stats" / f"test_step{step}.json"
        )
        result[profile] = {}
        for metric in METRICS:
            recomputed = float(
                np.mean([row[f"{profile.upper()}_{metric}"] for row in rows])
            )
            difference = abs(recomputed - float(official[metric]))
            result[profile][metric] = {
                "recomputed_from_saved_canvas": recomputed,
                "official_before_png_quantization": float(official[metric]),
                "absolute_difference": difference,
                "tolerance": tolerances[metric],
            }
            if difference > tolerances[metric]:
                raise RuntimeError(
                    f"metric consistency failed for {profile}/{metric}: "
                    f"difference={difference} tolerance={tolerances[metric]}"
                )
    return result


def summarize_differences(
    values: Iterable[float], *, improvement_when_negative: bool = False
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("difference array must be non-empty and finite")
    improved = array < 0 if improvement_when_negative else array > 0
    declined = array > 0 if improvement_when_negative else array < 0
    positive = np.maximum(array, 0.0)
    positive_total = float(positive.sum())
    top3_share = (
        float(np.sort(positive)[-3:].sum() / positive_total)
        if positive_total > 0
        else 0.0
    )
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "improved_count": int(improved.sum()),
        "improved_ratio": float(improved.mean()),
        "declined_count": int(declined.sum()),
        "declined_ratio": float(declined.mean()),
        "unchanged_count": int((~improved & ~declined).sum()),
        "top3_positive_contribution_share": top3_share,
    }


def paired_bootstrap(
    values: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or samples <= 0:
        raise ValueError("bootstrap inputs must be non-empty and samples positive")
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, array.size, size=(samples, array.size))
    sampled_means = array[sampled_indices].mean(axis=1)
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return {
        "seed": seed,
        "samples": samples,
        "mean": float(array.mean()),
        "ci95_percentile": [float(low), float(high)],
    }


def decide_pairwise(
    pairwise_summary: dict[str, Any], bootstrap_summary: dict[str, Any]
) -> str:
    psnr = pairwise_summary["A1_minus_B1"]["psnr"]
    ssim = pairwise_summary["A1_minus_B1"]["ssim"]
    lpips = pairwise_summary["A1_minus_B1"]["lpips"]
    psnr_ci = bootstrap_summary["A1_minus_B1"]["psnr"]["ci95_percentile"]
    stable = (
        psnr["improved_ratio"] >= 0.70
        and psnr["mean"] > 0
        and psnr_ci[0] > 0
        and ssim["mean"] >= 0
        and lpips["mean"] <= 0
    )
    if stable:
        return "PAIRWISE_STABLE"

    dominated_by_three = (
        psnr["top3_positive_contribution_share"] >= 0.80
        and psnr["median"] <= 0
    )
    clearly_conflicting = ssim["mean"] < -0.005 or lpips["mean"] > 0.01
    unstable = (
        psnr["improved_ratio"] < 0.50
        or psnr["mean"] <= 0
        or psnr_ci[1] <= 0
        or dominated_by_three
        or clearly_conflicting
    )
    if unstable:
        return "PAIRWISE_UNSTABLE"
    return "PAIRWISE_BORDERLINE"


def build_summaries(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    pairwise: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    for baseline in ("B0", "B1"):
        label = f"A1_minus_{baseline}"
        pairwise[label] = {}
        bootstrap[label] = {}
        for metric in METRICS:
            values = [row[f"{label}_{metric}"] for row in rows]
            pairwise[label][metric] = summarize_differences(
                values, improvement_when_negative=metric == "lpips"
            )
            bootstrap[label][metric] = paired_bootstrap(values)
    decision = decide_pairwise(pairwise, bootstrap)
    pairwise["decision"] = decision
    bootstrap["decision"] = decision
    return pairwise, bootstrap


def _draw_pairwise_plot(rows: list[dict[str, Any]], metric: str, path: Path) -> None:
    width, height = 1400, 560
    left, right, top, bottom = 90, 30, 45, 190
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    values_b1 = np.asarray([row[f"A1_minus_B1_{metric}"] for row in rows])
    values_b0 = np.asarray([row[f"A1_minus_B0_{metric}"] for row in rows])
    maximum = max(float(np.abs(values_b1).max()), float(np.abs(values_b0).max()), 1e-8)
    limit = maximum * 1.15

    def y_of(value: float) -> int:
        return int(top + (limit - value) / (2 * limit) * plot_h)

    zero_y = y_of(0.0)
    draw.line((left, zero_y, left + plot_w, zero_y), fill=(70, 70, 70), width=2)
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(0, 0, 0), width=1)
    count = len(rows)
    x_positions = [left + int((i + 0.5) * plot_w / count) for i in range(count)]
    series = ((values_b1, (30, 100, 210), "A1-B1"), (values_b0, (230, 120, 30), "A1-B0"))
    for values, color, _ in series:
        points = [(x_positions[i], y_of(float(value))) for i, value in enumerate(values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    draw.text((left, 12), f"Android per-image {metric.upper()} difference", fill=(0, 0, 0))
    draw.text((left + 420, 12), "A1-B1", fill=(30, 100, 210))
    draw.text((left + 500, 12), "A1-B0", fill=(230, 120, 30))
    draw.text((8, top - 5), f"+{limit:.4g}", fill=(0, 0, 0))
    draw.text((8, zero_y - 5), "0", fill=(0, 0, 0))
    draw.text((8, top + plot_h - 5), f"-{limit:.4g}", fill=(0, 0, 0))
    for x, row in zip(x_positions, rows):
        draw.line((x, top + plot_h, x, top + plot_h + 5), fill=(0, 0, 0))
        label = Image.new("RGB", (150, 14), "white")
        ImageDraw.Draw(label).text((0, 1), str(row["image_name"]), fill=(0, 0, 0))
        label = label.rotate(90, expand=True)
        image.paste(label, (x - label.width // 2, top + plot_h + 8))
    image.save(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step", type=int, default=9999)
    parser.add_argument("--expected-count", type=int, default=19)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    targets = [
        output_dir / "android_test_file_manifest.json",
        output_dir / "android_per_image_metrics.csv",
        output_dir / "android_per_image_metrics.json",
        output_dir / "android_pairwise_summary.json",
        output_dir / "android_bootstrap_summary.json",
        output_dir / "android_pairwise_psnr.png",
        output_dir / "android_pairwise_ssim.png",
        output_dir / "android_pairwise_lpips.png",
    ]
    _ensure_absent(targets)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, base_rows = build_file_manifest(
        result_root, step=args.step, expected_count=args.expected_count
    )
    rows = compute_per_image_metrics(
        result_root, base_rows, device=args.device, step=args.step
    )
    manifest["metric_consistency"] = validate_metric_consistency(
        result_root, rows, step=args.step
    )
    pairwise, bootstrap = build_summaries(rows)

    _write_json(targets[0], manifest)
    _write_csv(targets[1], rows)
    _write_json(targets[2], rows)
    _write_json(targets[3], pairwise)
    _write_json(targets[4], bootstrap)
    for metric, path in zip(METRICS, targets[5:]):
        _draw_pairwise_plot(rows, metric, path)
    print(json.dumps({"decision": pairwise["decision"], "outputs": [str(p) for p in targets]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
