#!/usr/bin/env python3
"""Read-only Garden/Room A1 responsibility post-mortem after Phase 2 rejection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


SCENES = ("garden", "room")
EXPECTED_COUNTS = {
    "garden": {"train": 161, "test": 24},
    "room": {"train": 272, "test": 39},
}
EXPECTED_A1 = {
    "enabled": True,
    "start_step": 3000,
    "threshold": 1.5,
    "min_weight": 0.2,
    "pool_size": 3,
    "epsilon": 1e-6,
}
HISTOGRAM_EDGES = np.linspace(0.2, 1.0, 81, dtype=np.float64)
SUMMARY_FIELDS = (
    "q_mean",
    "q_median",
    "q_p10",
    "q_p90",
    "q_lt_0_8_ratio",
    "q_lt_0_5_ratio",
    "q_eq_0_2_ratio",
    "edge_overlap_q_0_8",
    "edge_overlap_q_0_5",
    "neff_pixels",
    "neff_ratio",
    "residual_mean",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ensure_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(
            f"refusing to overwrite existing post-mortem outputs: {existing}"
        )


def effective_sample_size(
    responsibility: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    epsilon: float = 1e-6,
) -> tuple[float, float]:
    """Return raw effective pixels and the valid-pixel-normalized ratio."""

    q = np.asarray(responsibility, dtype=np.float64)
    valid_mask = np.ones_like(q, dtype=bool) if valid is None else np.asarray(valid, bool)
    if q.shape != valid_mask.shape or not valid_mask.any():
        raise ValueError("responsibility and non-empty valid mask must have equal shape")
    values = q[valid_mask]
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("valid responsibility values must be finite and positive")
    neff = float(values.sum() ** 2 / (np.square(values).sum() + epsilon))
    return neff, float(neff / values.size)


def responsibility_image_statistics(
    responsibility: np.ndarray,
    gradient: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    min_weight: float = 0.2,
    epsilon: float = 1e-6,
) -> dict[str, float | int]:
    """Compute all requested diagnostics for one training image."""

    q = np.asarray(responsibility, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    e = np.asarray(residual, dtype=np.float64)
    valid_mask = np.ones_like(q, dtype=bool) if valid is None else np.asarray(valid, bool)
    if not (q.shape == g.shape == e.shape == valid_mask.shape):
        raise ValueError("responsibility, gradient, residual, and valid must match")
    if not valid_mask.any():
        raise ValueError("at least one valid pixel is required")
    values = q[valid_mask]
    neff, neff_ratio = effective_sample_size(q, valid_mask, epsilon=epsilon)
    high_gradient_threshold = float(np.quantile(g[valid_mask], 0.90))
    high_gradient = (g >= high_gradient_threshold) & valid_mask
    low08 = (q < 0.8) & valid_mask
    low05 = (q < 0.5) & valid_mask
    return {
        "valid_pixel_count": int(values.size),
        "q_mean": float(values.mean()),
        "q_median": float(np.median(values)),
        "q_p10": float(np.quantile(values, 0.10)),
        "q_p90": float(np.quantile(values, 0.90)),
        "q_lt_0_8_ratio": float((values < 0.8).mean()),
        "q_lt_0_5_ratio": float((values < 0.5).mean()),
        "q_eq_0_2_ratio": float(
            np.isclose(values, min_weight, atol=1e-6, rtol=0.0).mean()
        ),
        "high_gradient_threshold": high_gradient_threshold,
        "edge_overlap_q_0_8": float(
            (low08 & high_gradient).sum() / (low08.sum() + epsilon)
        ),
        "edge_overlap_q_0_5": float(
            (low05 & high_gradient).sum() / (low05.sum() + epsilon)
        ),
        "neff_pixels": neff,
        "neff_ratio": neff_ratio,
        "residual_mean": float(e[valid_mask].mean()),
    }


def rank_worst_test_views(
    metric_rows: list[dict[str, Any]],
    *,
    scene: str = "room",
    count: int = 5,
) -> list[dict[str, Any]]:
    """Rank test views by A1-B1 PSNR, worst first."""

    if count <= 0:
        raise ValueError("count must be positive")
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in metric_rows:
        if row.get("scene") == scene and row.get("profile") in {"b1", "a1"}:
            grouped.setdefault(str(row["image_name"]), {})[str(row["profile"])] = row
    ranked: list[dict[str, Any]] = []
    for image_name, profiles in grouped.items():
        if set(profiles) != {"b1", "a1"}:
            raise RuntimeError(f"missing B1/A1 clean metrics for {scene}/{image_name}")
        b1, a1 = profiles["b1"], profiles["a1"]
        ranked.append(
            {
                "image_name": image_name,
                "b1_psnr": float(b1["psnr"]),
                "a1_psnr": float(a1["psnr"]),
                "a1_minus_b1_psnr": float(a1["psnr"] - b1["psnr"]),
                "a1_minus_b1_ssim": float(a1["ssim"] - b1["ssim"]),
                "a1_minus_b1_lpips": float(a1["lpips"] - b1["lpips"]),
            }
        )
    if len(ranked) < count:
        raise RuntimeError(f"only {len(ranked)} complete {scene} test views were found")
    ranked.sort(key=lambda row: (row["a1_minus_b1_psnr"], row["image_name"]))
    for rank, row in enumerate(ranked[:count], start=1):
        row["worst_rank"] = rank
    return ranked[:count]


def pose_neighbor_scores(
    test_pose: np.ndarray,
    train_poses: np.ndarray,
    *,
    scene_scale: float,
    angular_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine normalized camera-center and view-direction distance."""

    test = np.asarray(test_pose, dtype=np.float64)
    trains = np.asarray(train_poses, dtype=np.float64)
    if test.shape != (4, 4) or trains.ndim != 3 or trains.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [4,4] and [N,4,4]")
    if scene_scale <= 0 or angular_weight < 0:
        raise ValueError("scene_scale must be positive and angular_weight non-negative")
    position = np.linalg.norm(trains[:, :3, 3] - test[:3, 3], axis=1) / scene_scale
    test_forward = test[:3, 2] / np.linalg.norm(test[:3, 2])
    train_forward = trains[:, :3, 2].copy()
    train_forward /= np.linalg.norm(train_forward, axis=1, keepdims=True)
    cosine = np.clip(train_forward @ test_forward, -1.0, 1.0)
    angle = np.arccos(cosine)
    normalized_angle = angle / math.pi
    score = position + angular_weight * normalized_angle
    return score, position, angle


def nearest_training_views(
    *,
    test_name: str,
    test_pose: np.ndarray,
    train_names: list[str],
    train_poses: np.ndarray,
    scene_scale: float,
    count: int = 3,
    angular_weight: float = 0.25,
) -> list[dict[str, Any]]:
    if len(train_names) != len(train_poses) or count <= 0 or len(train_names) < count:
        raise ValueError("training names/poses must match and contain at least count views")
    score, position, angle = pose_neighbor_scores(
        test_pose,
        train_poses,
        scene_scale=scene_scale,
        angular_weight=angular_weight,
    )
    order = sorted(range(len(train_names)), key=lambda i: (float(score[i]), train_names[i]))
    return [
        {
            "neighbor_rank": rank,
            "test_image_name": test_name,
            "train_image_name": train_names[index],
            "pose_score": float(score[index]),
            "normalized_center_distance": float(position[index]),
            "view_angle_degrees": float(np.degrees(angle[index])),
        }
        for rank, index in enumerate(order[:count], start=1)
    ]


def summarize_scene(
    rows: list[dict[str, Any]], histogram_counts: np.ndarray
) -> dict[str, Any]:
    if not rows or histogram_counts.shape != (len(HISTOGRAM_EDGES) - 1,):
        raise ValueError("scene rows and fixed histogram counts are required")
    statistics: dict[str, Any] = {}
    for field in SUMMARY_FIELDS:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        statistics[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
        }
    total = int(histogram_counts.sum())
    if total <= 0:
        raise ValueError("histogram must contain pixels")
    return {
        "train_image_count": len(rows),
        "valid_pixel_count": int(sum(int(row["valid_pixel_count"]) for row in rows)),
        "image_weighted_statistics": statistics,
        "pixel_weighted_q_histogram": {
            "bin_edges": HISTOGRAM_EDGES.tolist(),
            "counts": histogram_counts.astype(int).tolist(),
            "probabilities": (histogram_counts / total).tolist(),
        },
    }


def compare_scenes(scene_summary: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for field in SUMMARY_FIELDS:
        garden = scene_summary["garden"]["image_weighted_statistics"][field]["mean"]
        room = scene_summary["room"]["image_weighted_statistics"][field]["mean"]
        comparison[field] = {
            "garden_mean": garden,
            "room_mean": room,
            "room_minus_garden": float(room - garden),
        }
    return comparison


def draw_distribution_plot(
    path: Path, scene_rows: dict[str, list[dict[str, Any]]], scene_summary: dict[str, Any]
) -> None:
    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = {"garden": (30, 130, 70), "room": (190, 70, 55)}
    draw.text((40, 20), "A1 clean responsibility post-mortem", fill=(0, 0, 0))

    def draw_panel(top: int, bottom: int, title: str, series: dict[str, np.ndarray], low: float, high: float) -> None:
        left, right = 100, width - 60
        draw.text((left, top - 25), title, fill=(0, 0, 0))
        draw.rectangle((left, top, right, bottom), outline=(0, 0, 0), width=1)
        draw.line((left, bottom, right, bottom), fill=(0, 0, 0), width=1)
        for scene, values in series.items():
            values = np.asarray(values, dtype=np.float64)
            points = []
            for index, value in enumerate(values):
                x = left + int(index * (right - left) / max(len(values) - 1, 1))
                y = bottom - int(np.clip((value - low) / max(high - low, 1e-12), 0, 1) * (bottom - top))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=colors[scene], width=3)
            draw.text((right - 180, top + (0 if scene == "garden" else 20)), scene, fill=colors[scene])
        draw.text((25, top), f"{high:.3f}", fill=(0, 0, 0))
        draw.text((25, bottom - 12), f"{low:.3f}", fill=(0, 0, 0))

    histogram_series = {
        scene: np.asarray(
            scene_summary[scene]["pixel_weighted_q_histogram"]["probabilities"],
            dtype=np.float64,
        )
        for scene in SCENES
    }
    hist_max = max(float(values.max()) for values in histogram_series.values())
    draw_panel(80, 310, "Pixel-weighted q histogram (q=0.2 to 1.0)", histogram_series, 0.0, hist_max * 1.05)

    q_mean_series = {
        scene: np.sort(np.asarray([row["q_mean"] for row in scene_rows[scene]]))
        for scene in SCENES
    }
    q_low = min(float(values.min()) for values in q_mean_series.values())
    q_high = max(float(values.max()) for values in q_mean_series.values())
    draw_panel(380, 610, "Sorted per-training-image q mean", q_mean_series, q_low, q_high)

    neff_series = {
        scene: np.sort(np.asarray([row["neff_ratio"] for row in scene_rows[scene]]))
        for scene in SCENES
    }
    neff_low = min(float(values.min()) for values in neff_series.values())
    neff_high = max(float(values.max()) for values in neff_series.values())
    draw_panel(680, 870, "Sorted per-training-image Neff / valid pixels", neff_series, neff_low, neff_high)
    image.save(path)


def _render(
    *, splats: dict[str, Any], data: dict[str, Any], device: str
) -> tuple[Any, Any, Any | None]:
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
    rendered = rendered.clamp(0.0, 1.0)
    if masks is not None:
        rendered[~masks] = 0
    return rendered, pixels, masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-commit-short", default="128535b")
    parser.add_argument("--garden-data", type=Path, required=True)
    parser.add_argument("--room-data", type=Path, required=True)
    parser.add_argument("--clean-per-image-json", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worst-test-count", type=int, default=5)
    parser.add_argument("--neighbors-per-test", type=int, default=3)
    parser.add_argument("--pose-angular-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    tools_dir = Path(__file__).resolve().parent
    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    for import_path in (project_root, tools_dir, gsplat_dir / "examples"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    from analyze_responsibility_selectivity import save_selectivity_figure, sobel_gradient
    from datasets.colmap import Dataset, Parser
    from puri_gs.responsibility import ResponsibilityConfig, compute_responsibility
    import gsplat

    if args.worst_test_count <= 0 or args.neighbors_per_test <= 0:
        raise ValueError("worst-test-count and neighbors-per-test must be positive")
    if args.pose_angular_weight < 0:
        raise ValueError("pose-angular-weight must be non-negative")
    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    runtime_path = Path(gsplat.__file__).resolve()
    if runtime_path.is_relative_to(gsplat_dir):
        raise RuntimeError("gsplat source checkout shadows the installed fixed wheel")

    output_dir = args.output_dir.expanduser().resolve()
    csv_path = output_dir / "clean_responsibility_per_train_image.csv"
    json_path = output_dir / "clean_responsibility_per_train_image.json"
    summary_path = output_dir / "clean_responsibility_scene_summary.json"
    plot_path = output_dir / "clean_responsibility_distribution.png"
    neighbor_manifest_path = output_dir / "room_worst_test_neighbors.json"
    neighbor_visual_dir = output_dir / "room_worst_test_neighbors"
    _ensure_absent(
        (csv_path, json_path, summary_path, plot_path, neighbor_manifest_path, neighbor_visual_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    neighbor_visual_dir.mkdir(parents=False, exist_ok=False)

    metric_rows = _read_json(args.clean_per_image_json.expanduser().resolve())
    worst_tests = rank_worst_test_views(
        metric_rows, scene="room", count=args.worst_test_count
    )
    runs_root = args.runs_root.expanduser().resolve()
    data_dirs = {
        "garden": args.garden_data.expanduser().resolve(),
        "room": args.room_data.expanduser().resolve(),
    }
    scene_rows: dict[str, list[dict[str, Any]]] = {}
    scene_summary: dict[str, Any] = {}
    neighbor_manifest: dict[str, Any] = {
        "selection": {
            "worst_test_count": args.worst_test_count,
            "neighbors_per_test": args.neighbors_per_test,
            "pose_score": "center_distance/scene_scale + angular_weight*(view_angle/pi)",
            "pose_angular_weight": args.pose_angular_weight,
        },
        "worst_room_test_views": worst_tests,
        "neighbors": [],
    }

    for scene in SCENES:
        print(f"[{scene}] loading parser and fixed A1 checkpoint", flush=True)
        parser = Parser(
            data_dir=str(data_dirs[scene]), factor=4, normalize=True, test_every=8
        )
        trainset = Dataset(parser, split="train", val_every=0)
        testset = Dataset(parser, split="test", val_every=0)
        expected = EXPECTED_COUNTS[scene]
        if len(trainset) != expected["train"] or len(testset) != expected["test"]:
            raise RuntimeError(
                f"unexpected {scene} split: train={len(trainset)} test={len(testset)}"
            )

        run_name = f"{scene}_a1_short10k_seed42_{args.run_commit_short}"
        run_dir = runs_root / run_name
        config = _read_json(run_dir / "config.yaml")
        if config.get("responsibility") != EXPECTED_A1:
            raise RuntimeError(f"fixed A1 responsibility config differs for {scene}")
        split = _read_json(run_dir / "dataset_split.json")
        train_names = [parser.image_names[int(index)] for index in trainset.indices]
        test_names = [parser.image_names[int(index)] for index in testset.indices]
        if split.get("train") != train_names or split.get("test") != test_names:
            raise RuntimeError(f"checkpoint split differs from current {scene} data")

        selected_neighbor_names: set[str] = set()
        if scene == "room":
            parser_index_by_name = {
                name: index for index, name in enumerate(parser.image_names)
            }
            train_parser_indices = np.asarray(trainset.indices, dtype=int)
            train_poses = parser.camtoworlds[train_parser_indices]
            for worst in worst_tests:
                test_name = str(worst["image_name"])
                if test_name not in test_names:
                    raise RuntimeError(f"worst test image is not in room split: {test_name}")
                test_pose = parser.camtoworlds[parser_index_by_name[test_name]]
                neighbors = nearest_training_views(
                    test_name=test_name,
                    test_pose=test_pose,
                    train_names=train_names,
                    train_poses=train_poses,
                    scene_scale=float(parser.scene_scale),
                    count=args.neighbors_per_test,
                    angular_weight=args.pose_angular_weight,
                )
                for neighbor in neighbors:
                    selected_neighbor_names.add(str(neighbor["train_image_name"]))
                    neighbor["worst_test_rank"] = worst["worst_rank"]
                    neighbor["a1_minus_b1_psnr"] = worst["a1_minus_b1_psnr"]
                    neighbor["figure"] = (
                        f"room_worst_test_neighbors/{Path(str(neighbor['train_image_name'])).stem}.png"
                    )
                    neighbor_manifest["neighbors"].append(neighbor)

        checkpoint_path = run_dir / "ckpts" / "ckpt_9999_rank0.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location=args.device, weights_only=True
        )
        if checkpoint.get("step") != 9999:
            raise RuntimeError(f"unexpected checkpoint step for {scene}")
        splats = {
            name: tensor.detach().to(args.device).requires_grad_(False)
            for name, tensor in checkpoint["splats"].items()
        }
        responsibility_config = ResponsibilityConfig(**EXPECTED_A1)
        rows: list[dict[str, Any]] = []
        histogram_counts = np.zeros(len(HISTOGRAM_EDGES) - 1, dtype=np.int64)
        with torch.no_grad():
            for dataset_index in range(len(trainset)):
                data = trainset[dataset_index]
                image_name = train_names[dataset_index]
                rendered, pixels, masks = _render(
                    splats=splats, data=data, device=args.device
                )
                residual = (rendered - pixels).abs().mean(dim=-1)
                q = compute_responsibility(
                    residual.detach(), responsibility_config, valid_mask=masks
                )
                gradient = sobel_gradient(pixels)
                valid = (
                    torch.ones_like(q, dtype=torch.bool) if masks is None else masks
                )
                q_np = q[0].cpu().numpy()
                residual_np = residual[0].cpu().numpy()
                gradient_np = gradient[0].cpu().numpy()
                valid_np = valid[0].cpu().numpy()
                stats = responsibility_image_statistics(
                    q_np,
                    gradient_np,
                    residual_np,
                    valid_np,
                    min_weight=responsibility_config.min_weight,
                    epsilon=responsibility_config.epsilon,
                )
                rows.append(
                    {
                        "scene": scene,
                        "train_index": dataset_index,
                        "parser_index": int(trainset.indices[dataset_index]),
                        "image_name": image_name,
                        **stats,
                    }
                )
                histogram_counts += np.histogram(
                    q_np[valid_np], bins=HISTOGRAM_EDGES
                )[0]
                if scene == "room" and image_name in selected_neighbor_names:
                    save_selectivity_figure(
                        neighbor_visual_dir / f"{Path(image_name).stem}.png",
                        ground_truth=pixels[0].cpu().numpy(),
                        render=rendered[0].cpu().numpy(),
                        residual=residual_np,
                        responsibility=q_np,
                        gradient=gradient_np,
                        valid=valid_np,
                        title=f"Room A1 nearby training view: {image_name}",
                    )
                if (dataset_index + 1) % 20 == 0 or dataset_index + 1 == len(trainset):
                    print(
                        f"[{scene}] analyzed {dataset_index + 1}/{len(trainset)} training views",
                        flush=True,
                    )

        if len(rows) != expected["train"]:
            raise RuntimeError(f"incomplete {scene} training-image analysis")
        scene_rows[scene] = rows
        scene_summary[scene] = summarize_scene(rows, histogram_counts)
        scene_summary[scene].update(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": 9999,
                "frozen_parameters": True,
                "backward_calls": 0,
                "rasterizations_per_train_image": 1,
            }
        )
        del splats, checkpoint
        torch.cuda.empty_cache()

    unique_neighbor_figures = {
        neighbor["figure"] for neighbor in neighbor_manifest["neighbors"]
    }
    actual_figures = {
        str(path.relative_to(output_dir)).replace("\\", "/")
        for path in neighbor_visual_dir.glob("*.png")
    }
    if actual_figures != unique_neighbor_figures:
        raise RuntimeError(
            f"neighbor figure mismatch: expected={unique_neighbor_figures} actual={actual_figures}"
        )

    all_rows = scene_rows["garden"] + scene_rows["room"]
    summary_document = {
        "execution_decision": "PASS_EXECUTION",
        "purpose": "diagnose whether Room degradation coincides with broader down-weighting of static difficult regions",
        "protocol": {
            "run_commit_short": args.run_commit_short,
            "data_factor": 4,
            "sh_degree": 3,
            "device": args.device,
            "gsplat_runtime_path": str(runtime_path),
            "responsibility_config": EXPECTED_A1,
            "q_min_equality_tolerance": 1e-6,
            "high_gradient_definition": "top 10% normalized Sobel magnitude within each image",
            "neff_definition": "sum(q)^2 / (sum(q^2) + epsilon)",
            "no_training": True,
            "no_backward": True,
        },
        "scenes": scene_summary,
        "room_minus_garden": compare_scenes(scene_summary),
        "room_worst_test_neighbor_manifest": str(neighbor_manifest_path),
        "interpretation": "PENDING_DISTRIBUTION_AND_NEIGHBOR_VISUAL_REVIEW",
    }
    _write_csv(csv_path, all_rows)
    _write_json(json_path, all_rows)
    _write_json(summary_path, summary_document)
    _write_json(neighbor_manifest_path, neighbor_manifest)
    draw_distribution_plot(plot_path, scene_rows, scene_summary)
    print(
        json.dumps(
            {
                "decision": summary_document["interpretation"],
                "garden_train_images": len(scene_rows["garden"]),
                "room_train_images": len(scene_rows["room"]),
                "neighbor_figures": len(actual_figures),
                "outputs": [
                    str(csv_path),
                    str(json_path),
                    str(summary_path),
                    str(plot_path),
                    str(neighbor_manifest_path),
                    str(neighbor_visual_dir),
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
