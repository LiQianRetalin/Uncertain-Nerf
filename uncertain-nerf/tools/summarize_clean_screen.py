#!/usr/bin/env python3
"""Re-evaluate and summarize garden/room B0/B1/A1 clean 10k screens."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCENES = ("garden", "room")
PROFILES = ("b0", "b1", "a1")
EXPECTED_TEST_COUNTS = {"garden": 24, "room": 39}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _ensure_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite existing analysis outputs: {existing}")


def choose_clean_baseline(scene_summary: dict[str, Any]) -> dict[str, Any]:
    psnr_delta = float(
        np.mean(
            [
                scene_summary[scene]["b1"]["psnr"]
                - scene_summary[scene]["b0"]["psnr"]
                for scene in SCENES
            ]
        )
    )
    lpips_delta = float(
        np.mean(
            [
                scene_summary[scene]["b1"]["lpips"]
                - scene_summary[scene]["b0"]["lpips"]
                for scene in SCENES
            ]
        )
    )
    gaussian_ratio = float(
        np.mean(
            [
                scene_summary[scene]["b1"]["num_GS"]
                / scene_summary[scene]["b0"]["num_GS"]
                for scene in SCENES
            ]
        )
    )
    train_time_ratio = float(
        np.mean(
            [
                scene_summary[scene]["b1"]["train_seconds"]
                / scene_summary[scene]["b0"]["train_seconds"]
                for scene in SCENES
            ]
        )
    )
    efficiency_better = gaussian_ratio <= 0.95 or train_time_ratio <= 0.95
    b1_selected = psnr_delta >= -0.10 and lpips_delta <= 0.005 and efficiency_better
    return {
        "primary_baseline": "b1" if b1_selected else "b0",
        "b1_label": "efficiency-strong baseline" if b1_selected else "efficiency reference",
        "mean_B1_minus_B0_psnr": psnr_delta,
        "mean_B1_minus_B0_lpips": lpips_delta,
        "mean_B1_over_B0_gaussian_ratio": gaussian_ratio,
        "mean_B1_over_B0_train_time_ratio": train_time_ratio,
        "efficiency_better": efficiency_better,
    }


def decide_clean_numeric(
    scene_summary: dict[str, Any], baseline_selection: dict[str, Any]
) -> dict[str, Any]:
    def mean_delta(baseline: str, metric: str) -> float:
        return float(
            np.mean(
                [
                    scene_summary[scene]["a1"][metric]
                    - scene_summary[scene][baseline][metric]
                    for scene in SCENES
                ]
            )
        )

    deltas: dict[str, Any] = {}
    for baseline in ("b0", "b1"):
        deltas[f"A1_minus_{baseline.upper()}"] = {
            metric: mean_delta(baseline, metric)
            for metric in ("psnr", "ssim", "lpips")
        }
    primary = baseline_selection["primary_baseline"]
    per_scene_psnr = {
        scene: scene_summary[scene]["a1"]["psnr"]
        - scene_summary[scene][primary]["psnr"]
        for scene in SCENES
    }
    fps_ratios = {
        scene: scene_summary[scene]["a1"]["render_fps"]
        / scene_summary[scene][primary]["render_fps"]
        for scene in SCENES
    }
    primary_delta = deltas[f"A1_minus_{primary.upper()}"]
    preferred = (
        deltas["A1_minus_B0"]["psnr"] >= 0.10
        and deltas["A1_minus_B1"]["psnr"] >= 0.0
        and deltas["A1_minus_B0"]["ssim"] >= 0.0
        and deltas["A1_minus_B1"]["ssim"] >= 0.0
        and deltas["A1_minus_B0"]["lpips"] <= 0.0
        and deltas["A1_minus_B1"]["lpips"] <= 0.0
        and min(fps_ratios.values()) >= 0.90
    )
    noninferior = (
        primary_delta["psnr"] >= -0.15
        and min(per_scene_psnr.values()) >= -0.30
        and primary_delta["ssim"] >= -0.005
        and primary_delta["lpips"] <= 0.01
        and min(fps_ratios.values()) >= 0.90
    )
    if preferred:
        numeric_decision = "CLEAN_PREFERRED_PASS"
    elif noninferior:
        numeric_decision = "CLEAN_NONINFERIOR"
    else:
        numeric_decision = "CLEAN_FAIL"
    return {
        "numeric_decision": numeric_decision,
        "final_decision": (
            "CLEAN_FAIL"
            if numeric_decision == "CLEAN_FAIL"
            else "PENDING_CLEAN_EDGE_AND_TEXTURE_VISUAL_REVIEW"
        ),
        "primary_baseline": primary,
        "mean_deltas": deltas,
        "per_scene_A1_minus_primary_psnr": per_scene_psnr,
        "per_scene_A1_over_primary_fps_ratio": fps_ratios,
    }


def decide_phase2(
    *, pairwise_decision: str, clean_decision: str, edge_classification: str,
    metrics_stably_improved: int, efficiency_pass: bool, gaussian_or_vram_ok: bool
) -> str:
    if (
        pairwise_decision == "PAIRWISE_UNSTABLE"
        or clean_decision == "CLEAN_FAIL"
        or edge_classification == "HIGH_EDGE_SENSITIVITY"
        or not efficiency_pass
    ):
        return "A1_REJECT"
    if (
        pairwise_decision == "PAIRWISE_STABLE"
        and clean_decision in {"CLEAN_PREFERRED_PASS", "CLEAN_NONINFERIOR"}
        and edge_classification != "HIGH_EDGE_SENSITIVITY"
        and metrics_stably_improved >= 2
        and gaussian_or_vram_ok
    ):
        return "A1_CONFIRMED_CANDIDATE"
    if (
        pairwise_decision == "PAIRWISE_BORDERLINE"
        and clean_decision in {"CLEAN_PREFERRED_PASS", "CLEAN_NONINFERIOR"}
        and edge_classification
        in {"LOW_EDGE_SENSITIVITY", "MODERATE_EDGE_SENSITIVITY"}
        and efficiency_pass
    ):
        return "A1_BORDERLINE_RETAIN"
    return "A1_REJECT"


def _load_eval_modules(device: str) -> tuple[Any, Any, Any]:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    return (
        PeakSignalNoiseRatio(data_range=1.0).to(device),
        StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device),
    )


def _render(
    *, splats: dict[str, Any], data: dict[str, Any], device: str, absgrad: bool
) -> tuple[Any, Any]:
    import torch
    from gsplat.rendering import rasterization

    pixels = data["image"].unsqueeze(0).to(device) / 255.0
    camtoworlds = data["camtoworld"].unsqueeze(0).to(device)
    Ks = data["K"].unsqueeze(0).to(device)
    masks = data["mask"].unsqueeze(0).to(device) if "mask" in data else None
    height, width = pixels.shape[1:3]
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    rendered, _, _ = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=width,
        height=height,
        packed=False,
        absgrad=absgrad,
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
    rendered = rendered.clamp(0.0, 1.0)
    if masks is not None:
        rendered[~masks] = 0
    return rendered, pixels


def evaluate_profile(
    *,
    scene: str,
    profile: str,
    run_dir: Path,
    eval_dir: Path,
    dataset: Any,
    parser: Any,
    device: str,
    warmup: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    checkpoint_path = run_dir / "ckpts" / "ckpt_9999_rank0.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("step") != 9999:
        raise RuntimeError(f"unexpected checkpoint step: {checkpoint_path}")
    splats = {
        name: tensor.detach().to(device).requires_grad_(False)
        for name, tensor in checkpoint["splats"].items()
    }
    train_split = _read_json(run_dir / "dataset_split.json")
    eval_split = _read_json(eval_dir / "dataset_split.json")
    if train_split != eval_split:
        raise RuntimeError(f"train/eval split files differ for {scene}/{profile}")
    expected_names = [parser.image_names[int(index)] for index in dataset.indices]
    if train_split.get("test") != expected_names:
        raise RuntimeError(f"checkpoint split differs from current data for {scene}/{profile}")

    with torch.no_grad():
        first = dataset[0]
        for _ in range(warmup):
            _render(
                splats=splats,
                data=first,
                device=device,
                absgrad=profile != "b0",
            )
        torch.cuda.synchronize()

    psnr_metric, ssim_metric, lpips_metric = _load_eval_modules(device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            data = dataset[index]
            torch.cuda.synchronize()
            start = time.perf_counter()
            rendered, pixels = _render(
                splats=splats,
                data=data,
                device=device,
                absgrad=profile != "b0",
            )
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            render_tensor = rendered.permute(0, 3, 1, 2)
            pixel_tensor = pixels.permute(0, 3, 1, 2)
            rows.append(
                {
                    "scene": scene,
                    "profile": profile,
                    "image_name": expected_names[index],
                    "psnr": float(psnr_metric(render_tensor, pixel_tensor).item()),
                    "ssim": float(ssim_metric(render_tensor, pixel_tensor).item()),
                    "lpips": float(lpips_metric(render_tensor, pixel_tensor).item()),
                    "latency_ms": float(latency * 1000.0),
                }
            )

    official = _read_json(eval_dir / "stats" / "test_step9999.json")
    means = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in ("psnr", "ssim", "lpips")
    }
    tolerances = {"psnr": 0.02, "ssim": 0.002, "lpips": 0.002}
    consistency = {
        metric: {
            "recomputed": means[metric],
            "official": official[metric],
            "absolute_difference": abs(means[metric] - official[metric]),
            "tolerance": tolerances[metric],
        }
        for metric in means
    }
    if any(item["absolute_difference"] > item["tolerance"] for item in consistency.values()):
        raise RuntimeError(f"metric consistency check failed for {scene}/{profile}: {consistency}")

    train_metrics = _read_json(run_dir / "stats" / "train_step9999_rank0.json")
    latencies = np.asarray([row["latency_ms"] for row in rows], dtype=np.float64)
    summary = {
        **means,
        "num_GS": int(official["num_GS"]),
        "peak_vram_gib": float(train_metrics["mem"]),
        "train_seconds": float(train_metrics["ellipse_time"]),
        "render_fps": float(1000.0 / latencies.mean()),
        "latency_mean_ms": float(latencies.mean()),
        "latency_p50_ms": float(np.quantile(latencies, 0.50)),
        "latency_p95_ms": float(np.quantile(latencies, 0.95)),
        "warmup_frames": warmup,
        "measured_frames": len(rows),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "metric_consistency": consistency,
    }
    del splats, checkpoint
    torch.cuda.empty_cache()
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--commit-short", required=True)
    parser.add_argument("--garden-data", type=Path, required=True)
    parser.add_argument("--room-data", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup <= 0:
        raise ValueError("warmup must be positive")
    project_root = Path(__file__).resolve().parents[1]
    examples_dir = args.gsplat_dir.expanduser().resolve() / "examples"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    from datasets.colmap import Dataset, Parser

    import gsplat

    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    runtime_path = Path(gsplat.__file__).resolve()
    if runtime_path.is_relative_to(gsplat_dir):
        raise RuntimeError("gsplat source checkout shadows the installed fixed wheel")

    output_dir = args.output_dir.expanduser().resolve()
    csv_path = output_dir / "mipnerf360_clean_summary.csv"
    json_path = output_dir / "mipnerf360_clean_summary.json"
    per_image_csv = output_dir / "mipnerf360_clean_per_image.csv"
    per_image_json = output_dir / "mipnerf360_clean_per_image.json"
    _ensure_absent((csv_path, json_path, per_image_csv, per_image_json))
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_root = args.runs_root.expanduser().resolve()
    data_dirs = {
        "garden": args.garden_data.expanduser().resolve(),
        "room": args.room_data.expanduser().resolve(),
    }
    all_rows: list[dict[str, Any]] = []
    scene_summary: dict[str, Any] = {}
    reference_splits: dict[str, Any] = {}
    for scene in SCENES:
        parser = Parser(
            data_dir=str(data_dirs[scene]), factor=4, normalize=True, test_every=8
        )
        dataset = Dataset(parser, split="test", val_every=0)
        if len(dataset) != EXPECTED_TEST_COUNTS[scene]:
            raise RuntimeError(
                f"expected {EXPECTED_TEST_COUNTS[scene]} {scene} test images, got {len(dataset)}"
            )
        scene_summary[scene] = {}
        for profile in PROFILES:
            name = f"{scene}_{profile}_short10k_seed42_{args.commit_short}"
            run_dir = runs_root / name
            eval_dir = runs_root / f"{name}_eval"
            rows, summary = evaluate_profile(
                scene=scene,
                profile=profile,
                run_dir=run_dir,
                eval_dir=eval_dir,
                dataset=dataset,
                parser=parser,
                device=args.device,
                warmup=args.warmup,
            )
            split = _read_json(run_dir / "dataset_split.json")
            signature = (split.get("train"), split.get("test"))
            if scene not in reference_splits:
                reference_splits[scene] = signature
            elif signature != reference_splits[scene]:
                raise RuntimeError(f"profile split differs within {scene}: {profile}")
            all_rows.extend(rows)
            scene_summary[scene][profile] = summary

    baseline = choose_clean_baseline(scene_summary)
    clean = decide_clean_numeric(scene_summary, baseline)
    summary_document = {
        "execution_decision": "PASS_EXECUTION",
        "clean_decision": clean,
        "baseline_selection": baseline,
        "scenes": scene_summary,
        "protocol": {
            "steps": 10000,
            "seed": 42,
            "data_factor": 4,
            "sh_degree": 3,
            "gpu": "NVIDIA L20",
            "warmup_frames": args.warmup,
            "metric_implementation": "gsplat v1.5.3 TorchMetrics PSNR/SSIM and Alex LPIPS",
            "gsplat_runtime_path": str(runtime_path),
        },
        "phase2_final_decision": "PENDING_ANDROID_PAIRWISE_RESPONSIBILITY_AND_CLEAN_VISUAL_REVIEW",
    }
    summary_rows = [
        {"scene": scene, "profile": profile, **scene_summary[scene][profile]}
        for scene in SCENES
        for profile in PROFILES
    ]
    _write_csv(csv_path, summary_rows)
    _write_json(json_path, summary_document)
    _write_csv(per_image_csv, all_rows)
    _write_json(per_image_json, all_rows)
    print(json.dumps({"decision": clean["final_decision"], "outputs": [str(csv_path), str(json_path), str(per_image_csv), str(per_image_json)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
