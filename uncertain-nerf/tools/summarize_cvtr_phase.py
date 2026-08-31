#!/usr/bin/env python3
"""Evaluate six Phase 3 continuations and make the frozen CVTR decision."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENES = ("android", "garden", "room")
METHODS = ("b1c", "cvtr")
EXPECTED_TEST_COUNTS = {"android": 19, "garden": 24, "room": 39}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def ensure_outputs_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite Phase 3 outputs: {existing}")


def load_metrics(device: str):
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    return (
        PeakSignalNoiseRatio(data_range=1.0).to(device),
        StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(
            device
        ),
    )


def render(splats: dict, data: dict, device: str):
    import torch
    from gsplat.rendering import rasterization

    target = data["image"].unsqueeze(0).to(device) / 255.0
    camtoworld = data["camtoworld"].unsqueeze(0).to(device)
    K = data["K"].unsqueeze(0).to(device)
    height, width = target.shape[1:3]
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    predicted, _, _ = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv(camtoworld),
        Ks=K,
        width=width,
        height=height,
        packed=False,
        absgrad=True,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        with_ut=False,
        with_eval3d=False,
        sh_degree=3,
        near_plane=0.01,
        far_plane=1e10,
    )
    if "mask" in data:
        predicted[~data["mask"].unsqueeze(0).to(device)] = 0
    return predicted.clamp(0.0, 1.0), target


def evaluate_run(
    *,
    scene: str,
    method: str,
    run_dir: Path,
    eval_dir: Path,
    dataset: Any,
    parser: Any,
    device: str,
    warmup: int,
) -> tuple[list[dict], dict]:
    import torch

    checkpoint_path = run_dir / "ckpts" / "ckpt_14999_rank0.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("step") != 14999:
        raise RuntimeError(f"unexpected continuation checkpoint: {checkpoint_path}")
    splats = {
        name: tensor.detach().to(device).requires_grad_(False)
        for name, tensor in checkpoint["splats"].items()
    }
    train_split = read_json(run_dir / "dataset_split.json")
    eval_split = read_json(eval_dir / "dataset_split.json")
    if train_split != eval_split:
        raise RuntimeError(f"train/eval split mismatch for {scene}/{method}")
    expected_names = [parser.image_names[int(index)] for index in dataset.indices]
    if train_split.get("test") != expected_names:
        raise RuntimeError(f"dataset split mismatch for {scene}/{method}")

    with torch.no_grad():
        first = dataset[0]
        for _ in range(warmup):
            render(splats, first, device)
        torch.cuda.synchronize()
    psnr, ssim, lpips = load_metrics(device)
    rows: list[dict] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            data = dataset[index]
            torch.cuda.synchronize()
            start = time.perf_counter()
            predicted, target = render(splats, data, device)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0
            predicted_nchw = predicted.permute(0, 3, 1, 2)
            target_nchw = target.permute(0, 3, 1, 2)
            rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "image_name": expected_names[index],
                    "psnr": float(psnr(predicted_nchw, target_nchw).item()),
                    "ssim": float(ssim(predicted_nchw, target_nchw).item()),
                    "lpips": float(lpips(predicted_nchw, target_nchw).item()),
                    "latency_ms": latency_ms,
                }
            )
    official = read_json(eval_dir / "stats" / "test_step14999.json")
    means = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in ("psnr", "ssim", "lpips")
    }
    tolerances = {"psnr": 0.02, "ssim": 0.002, "lpips": 0.002}
    consistency = {
        metric: abs(means[metric] - float(official[metric]))
        for metric in means
    }
    if any(consistency[key] > tolerances[key] for key in consistency):
        raise RuntimeError(f"official metric mismatch for {scene}/{method}")
    train_stats = read_json(run_dir / "stats" / "train_step14999_rank0.json")
    source = read_json(run_dir / "continuation_source.json")
    latencies = np.asarray([row["latency_ms"] for row in rows])
    summary = {
        **means,
        "num_GS": int(official["num_GS"]),
        "peak_vram_gib": float(train_stats["mem"]),
        "continuation_train_seconds": float(train_stats["ellipse_time"]),
        "fps": float(1000.0 / latencies.mean()),
        "latency_p50_ms": float(np.quantile(latencies, 0.50)),
        "latency_p95_ms": float(np.quantile(latencies, 0.95)),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "source": source,
        "official_metric_absolute_difference": consistency,
    }
    del checkpoint, splats
    torch.cuda.empty_cache()
    return rows, summary


def paired_bootstrap(
    deltas: np.ndarray, samples: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    sampled_means = deltas[indices].mean(axis=1)
    return {
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "positive_fraction": float(np.mean(deltas > 0)),
        "ci95_low": float(np.quantile(sampled_means, 0.025)),
        "ci95_high": float(np.quantile(sampled_means, 0.975)),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def nearest_train_mask_areas(parser: Any, trainset: Any, testset: Any, manifest: dict):
    areas = {item["image_name"]: item["mask_area_ratio"] for item in manifest["images"]}
    train_indices = np.asarray(trainset.indices, dtype=int)
    train_centers = parser.camtoworlds[train_indices, :3, 3]
    train_names = [parser.image_names[index] for index in train_indices]
    output = []
    for test_index in testset.indices:
        center = parser.camtoworlds[int(test_index), :3, 3]
        distances = np.linalg.norm(train_centers - center[None], axis=1)
        ranked = sorted(
            zip(distances.tolist(), train_names), key=lambda item: (item[0], item[1])
        )
        output.append(float(areas[ranked[0][1]]))
    return output


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def decide(summary: dict, paired: dict, masks: dict, synthetic: dict) -> dict:
    mask_gate = synthetic.get("gate") == "PASS" and all(
        masks[scene]["clean_mask_guard"] == "PASS" for scene in ("garden", "room")
    )
    android = {
        metric: summary["android"]["cvtr"][metric]
        - summary["android"]["b1c"][metric]
        for metric in ("psnr", "ssim", "lpips")
    }
    dynamic_gate = (
        android["psnr"] >= 0.30
        and paired["android"]["psnr"]["positive_fraction"] >= 0.70
        and android["ssim"] >= 0
        and android["lpips"] <= 0
        and paired["android"]["psnr"]["ci95_low"] > 0
    )
    clean_deltas = {
        scene: {
            metric: summary[scene]["cvtr"][metric]
            - summary[scene]["b1c"][metric]
            for metric in ("psnr", "ssim", "lpips")
        }
        for scene in ("garden", "room")
    }
    clean_gate = (
        np.mean([clean_deltas[s]["psnr"] for s in clean_deltas]) >= -0.10
        and min(clean_deltas[s]["psnr"] for s in clean_deltas) >= -0.20
        and np.mean([clean_deltas[s]["ssim"] for s in clean_deltas]) >= -0.003
        and np.mean([clean_deltas[s]["lpips"] for s in clean_deltas]) <= 0.005
        and paired["room"]["psnr"]["minimum"] >= -1.0
    )
    efficiency_details = {}
    for scene in SCENES:
        cvtr = summary[scene]["cvtr"]
        control = summary[scene]["b1c"]
        efficiency_details[scene] = {
            "fps_ratio": cvtr["fps"] / control["fps"],
            "gaussian_ratio": cvtr["num_GS"] / control["num_GS"],
            "vram_ratio": cvtr["peak_vram_gib"] / control["peak_vram_gib"],
            "train_time_ratio": cvtr["continuation_train_seconds"]
            / control["continuation_train_seconds"],
        }
    efficiency_gate = all(
        item["fps_ratio"] >= 0.98
        and item["gaussian_ratio"] <= 1.10
        and item["vram_ratio"] <= 1.10
        and item["train_time_ratio"] <= 1.15
        for item in efficiency_details.values()
    )
    if mask_gate and clean_gate and dynamic_gate and efficiency_gate:
        decision = "CVTR_PASS"
    elif (
        mask_gate
        and clean_gate
        and efficiency_gate
        and 0 < android["psnr"] < 0.30
        and paired["android"]["psnr"]["positive_fraction"] > 0.5
        and android["ssim"] >= 0
        and android["lpips"] <= 0
    ):
        decision = "CVTR_BORDERLINE"
    else:
        decision = "CVTR_REJECT"
    return {
        "decision": decision,
        "mask_gate": mask_gate,
        "dynamic_gate": dynamic_gate,
        "clean_gate": clean_gate,
        "efficiency_gate": efficiency_gate,
        "android_deltas": android,
        "clean_deltas": clean_deltas,
        "efficiency": efficiency_details,
    }


def write_reports(summary_document: dict, decision: dict) -> None:
    continuation_report = PROJECT_ROOT / "reports" / "PHASE_3B_CVTR_CONTINUATION.md"
    lines = [
        "# PURI-GS Phase 3B：CVTR continuation",
        "",
        f"执行结论：`{decision['decision']}`",
        "",
        "| Scene | Method | PSNR | SSIM | LPIPS | GS | FPS | p50 ms | p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene in SCENES:
        for method in METHODS:
            item = summary_document["scenes"][scene][method]
            lines.append(
                f"| {scene} | {method} | {item['psnr']:.6f} | {item['ssim']:.6f} | "
                f"{item['lpips']:.6f} | {item['num_GS']} | {item['fps']:.3f} | "
                f"{item['latency_p50_ms']:.3f} | {item['latency_p95_ms']:.3f} |"
            )
    lines.extend(["", "完整逐图、bootstrap、mask 相关性和门禁输入见 analysis JSON/CSV。", ""])
    continuation_report.write_text("\n".join(lines), encoding="utf-8")
    decision_report = PROJECT_ROOT / "reports" / "PHASE_3_CVTR_DECISION.md"
    decision_report.write_text(
        "\n".join(
            [
                "# PURI-GS Phase 3：CVTR 决策",
                "",
                f"正式结论：`{decision['decision']}`",
                "",
                f"- Synthetic/Clean mask gate: `{decision['mask_gate']}`",
                f"- Android dynamic gate: `{decision['dynamic_gate']}`",
                f"- Garden/Room clean gate: `{decision['clean_gate']}`",
                f"- Efficiency gate: `{decision['efficiency_gate']}`",
                "",
                "本结论由预注册门槛直接生成，不触发自动调参。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(PROJECT_ROOT / "reports" / "phase_3_cvtr_decision.json", decision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--commit-short", required=True)
    parser.add_argument("--android-data", type=Path, required=True)
    parser.add_argument("--garden-data", type=Path, required=True)
    parser.add_argument("--room-data", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--synthetic-metrics", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "cvtr_continuation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples_dir = args.gsplat_dir.expanduser().resolve() / "examples"
    sys.path.insert(0, str(examples_dir))
    from datasets.colmap import Dataset, Parser

    import gsplat

    runtime = Path(gsplat.__file__).resolve()
    if runtime.is_relative_to(args.gsplat_dir.expanduser().resolve()):
        raise RuntimeError("gsplat source checkout shadows the pinned wheel")
    output_dir = args.output_dir.expanduser().resolve()
    output_paths = [
        output_dir / "per_image_metrics.csv",
        output_dir / "summary.json",
        output_dir / "bootstrap.json",
        PROJECT_ROOT / "reports" / "PHASE_3B_CVTR_CONTINUATION.md",
        PROJECT_ROOT / "reports" / "PHASE_3_CVTR_DECISION.md",
        PROJECT_ROOT / "reports" / "phase_3_cvtr_decision.json",
    ]
    ensure_outputs_absent(output_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dirs = {
        "android": args.android_data.expanduser().resolve(),
        "garden": args.garden_data.expanduser().resolve(),
        "room": args.room_data.expanduser().resolve(),
    }
    runs_root = args.runs_root.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    all_rows: list[dict] = []
    scene_summary: dict[str, dict] = {}
    bootstrap: dict[str, dict] = {}
    masks = {
        scene: read_json(mask_root / scene / "scene_statistics.json")
        for scene in SCENES
    }
    mask_manifests = {
        scene: read_json(mask_root / scene / "manifest.json") for scene in SCENES
    }
    for scene in SCENES:
        parser = Parser(
            data_dir=str(data_dirs[scene]), factor=4, normalize=True, test_every=8
        )
        split_args = (
            {"train_keyword": "clutter", "test_keyword": "extra"}
            if scene == "android"
            else {}
        )
        testset = Dataset(parser, split="test", val_every=0, **split_args)
        trainset = Dataset(parser, split="train", val_every=0, **split_args)
        if len(testset) != EXPECTED_TEST_COUNTS[scene]:
            raise RuntimeError(f"unexpected {scene} test count: {len(testset)}")
        scene_summary[scene] = {}
        method_rows = {}
        source_sha = None
        for method in METHODS:
            name = f"{scene}_{method}_10kto15k_seed42_{args.commit_short}"
            rows, item = evaluate_run(
                scene=scene,
                method=method,
                run_dir=runs_root / name,
                eval_dir=runs_root / f"{name}_eval",
                dataset=testset,
                parser=parser,
                device=args.device,
                warmup=args.warmup,
            )
            if source_sha is None:
                source_sha = item["source"]["sha256"]
            elif item["source"]["sha256"] != source_sha:
                raise RuntimeError(f"continuation branches use different sources: {scene}")
            if method == "cvtr":
                mask_manifest = mask_manifests[scene]
                if mask_manifest["source_checkpoint_sha256"] != source_sha:
                    raise RuntimeError(f"CVTR masks use a different B1 source: {scene}")
            method_rows[method] = rows
            all_rows.extend(rows)
            scene_summary[scene][method] = item
        nearest_areas = nearest_train_mask_areas(
            parser, trainset, testset, mask_manifests[scene]
        )
        bootstrap[scene] = {}
        for metric in ("psnr", "ssim", "lpips"):
            delta = np.asarray(
                [row[metric] for row in method_rows["cvtr"]]
            ) - np.asarray([row[metric] for row in method_rows["b1c"]])
            bootstrap[scene][metric] = {
                **paired_bootstrap(delta, args.bootstrap_samples, 42),
                "minimum": float(delta.min()),
                "maximum": float(delta.max()),
                "nearest_train_mask_area_correlation": correlation(
                    nearest_areas, delta.tolist()
                ),
            }
        for index, row in enumerate(method_rows["cvtr"]):
            row["nearest_train_mask_area"] = nearest_areas[index]
            for metric in ("psnr", "ssim", "lpips"):
                row[f"delta_{metric}_vs_b1c"] = (
                    row[metric] - method_rows["b1c"][index][metric]
                )
        for index, row in enumerate(method_rows["b1c"]):
            row["nearest_train_mask_area"] = nearest_areas[index]
            for metric in ("psnr", "ssim", "lpips"):
                row[f"delta_{metric}_vs_b1c"] = 0.0

    synthetic = read_json(args.synthetic_metrics.expanduser().resolve())
    decision = decide(scene_summary, bootstrap, masks, synthetic)
    summary_document = {
        "execution": "PASS_EXECUTION",
        "scenes": scene_summary,
        "mask_statistics": masks,
        "android_mask_area_distribution": [
            item["mask_area_ratio"] for item in mask_manifests["android"]["images"]
        ],
        "synthetic_metrics": synthetic,
        "decision": decision,
        "protocol": {
            "source_step": 9999,
            "target_step": 14999,
            "additional_steps": 5000,
            "seed": 42,
            "warmup": args.warmup,
            "gsplat_runtime": str(runtime),
        },
    }
    with output_paths[0].open("w", newline="", encoding="utf-8") as stream:
        fields = sorted(set().union(*(row.keys() for row in all_rows)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(output_paths[1], summary_document)
    write_json(output_paths[2], bootstrap)
    write_reports(summary_document, decision)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
