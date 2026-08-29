#!/usr/bin/env python3
"""Diagnose A1 low-responsibility selectivity without changing or training the model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


EXPECTED_A1 = {
    "enabled": True,
    "start_step": 3000,
    "threshold": 1.5,
    "min_weight": 0.2,
    "pool_size": 3,
    "epsilon": 1e-6,
}


def deterministic_uniform_indices(length: int, count: int = 8) -> list[int]:
    if length < count or count <= 0:
        raise ValueError("length must be at least count and count must be positive")
    return np.rint(np.linspace(0, length - 1, count)).astype(int).tolist()


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-aware one-based average ranks implemented with NumPy only."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("rank inputs must be non-empty and finite")
    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    )
    ends = np.r_[starts[1:], flat.size]
    group_ids = np.cumsum(
        np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    ) - 1
    average = (starts + ends - 1) / 2.0 + 1.0
    ranked_sorted = average[group_ids]
    ranks = np.empty_like(ranked_sorted)
    ranks[order] = ranked_sorted
    return ranks


def spearman_numpy(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size:
        raise ValueError("Spearman inputs must have equal size")
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    return float(np.dot(left_centered, right_centered) / denominator) if denominator else 0.0


def sobel_gradient(rgb: "Any") -> "Any":
    """Compute normalized, q99-clipped Sobel magnitude with fixed kernels."""

    import torch
    import torch.nn.functional as F

    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape [B, H, W, 3]")
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray = gray.unsqueeze(1)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=rgb.device,
        dtype=rgb.dtype,
    ).reshape(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel_x)
    gy = F.conv2d(padded, kernel_y)
    magnitude = torch.sqrt(gx.square() + gy.square()).squeeze(1)
    normalized: list[Any] = []
    for image in magnitude:
        q99 = torch.quantile(image, 0.99).clamp_min(1e-8)
        normalized.append((image / q99).clamp(0.0, 1.0))
    return torch.stack(normalized)


def edge_overlap_statistics(
    responsibility: np.ndarray,
    gradient: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, float]:
    if not (responsibility.shape == gradient.shape == residual.shape):
        raise ValueError("responsibility, gradient, and residual shapes must match")
    valid_mask = np.ones_like(responsibility, dtype=bool) if valid is None else valid.astype(bool)
    if not valid_mask.any():
        raise ValueError("at least one valid pixel is required")
    q = responsibility[valid_mask]
    g = gradient[valid_mask]
    r = (1.0 - responsibility)[valid_mask]
    e = residual[valid_mask]
    high_gradient_threshold = float(np.quantile(g, 0.9))
    high_gradient = gradient >= high_gradient_threshold
    low08 = (responsibility < 0.8) & valid_mask
    low05 = (responsibility < 0.5) & valid_mask
    epsilon = 1e-6
    return {
        "spearman_degradation_vs_gradient": spearman_numpy(r, g),
        "spearman_degradation_vs_residual": spearman_numpy(r, e),
        "low_q_0_8_ratio": float(low08.sum() / valid_mask.sum()),
        "low_q_0_5_ratio": float(low05.sum() / valid_mask.sum()),
        "high_gradient_threshold": high_gradient_threshold,
        "edge_overlap_q_0_8": float(
            (low08 & high_gradient).sum() / (low08.sum() + epsilon)
        ),
        "edge_overlap_q_0_5": float(
            (low05 & high_gradient).sum() / (low05.sum() + epsilon)
        ),
        "responsibility_mean": float(q.mean()),
        "residual_mean": float(e.mean()),
    }


def summarize_frames(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "selected_index"
    ]
    summary: dict[str, Any] = {"frame_count": len(rows), "statistics": {}}
    for key in numeric_fields:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary["statistics"][key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    summary["classification"] = "PENDING_EIGHT_FRAME_VISUAL_REVIEW"
    return summary


def _uint8_rgb(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255).astype(np.uint8), "RGB")


def _uint8_scalar(array: np.ndarray, low: float, high: float) -> Image.Image:
    scaled = np.clip((array - low) / max(high - low, 1e-8), 0.0, 1.0)
    rgb = np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=-1)
    return Image.fromarray(rgb, "RGB")


def _overlap_image(q: np.ndarray, gradient: np.ndarray, valid: np.ndarray) -> Image.Image:
    threshold = np.quantile(gradient[valid], 0.9)
    low = (q < 0.8) & valid
    edge = (gradient >= threshold) & valid
    rgb = np.zeros((*q.shape, 3), dtype=np.uint8)
    rgb[low] = (40, 100, 230)
    rgb[edge] = (220, 55, 55)
    rgb[low & edge] = (255, 220, 20)
    rgb[~valid] = (120, 120, 120)
    return Image.fromarray(rgb, "RGB")


def _scalar_bar(width: int, low: float, high: float) -> Image.Image:
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (10, 1))
    rgb = np.repeat(gradient[..., None], 3, axis=-1)
    bar = Image.fromarray(rgb, "RGB")
    canvas = Image.new("RGB", (width, 28), "white")
    canvas.paste(bar, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 13), f"{low:.2f}", fill=(0, 0, 0))
    label = f"{high:.2f}"
    draw.text((max(width - 35, 0), 13), label, fill=(0, 0, 0))
    return canvas


def save_selectivity_figure(
    path: Path,
    *,
    ground_truth: np.ndarray,
    render: np.ndarray,
    residual: np.ndarray,
    responsibility: np.ndarray,
    gradient: np.ndarray,
    valid: np.ndarray,
    title: str,
) -> None:
    response = 1.0 - responsibility
    panels = [
        ("ground truth", _uint8_rgb(ground_truth), None),
        ("A1 render", _uint8_rgb(render), None),
        ("absolute residual", _uint8_scalar(residual, 0.0, 1.0), (0.0, 1.0)),
        ("responsibility q", _uint8_scalar(responsibility, 0.2, 1.0), (0.2, 1.0)),
        ("degradation response 1-q", _uint8_scalar(response, 0.0, 0.8), (0.0, 0.8)),
        ("Sobel gradient", _uint8_scalar(gradient, 0.0, 1.0), (0.0, 1.0)),
        ("low-q / high-gradient overlap", _overlap_image(responsibility, gradient, valid), None),
    ]
    panel_width = 320
    image_height = int(ground_truth.shape[0] * panel_width / ground_truth.shape[1])
    header, label_height, bar_height = 32, 24, 30
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), header + label_height + image_height + bar_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for index, (label, panel, limits) in enumerate(panels):
        x = index * panel_width
        draw.text((x + 4, header + 5), label, fill=(0, 0, 0))
        resized = panel.resize((panel_width, image_height), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x, header + label_height))
        if limits is not None:
            canvas.paste(
                _scalar_bar(panel_width, limits[0], limits[1]),
                (x, header + label_height + image_height),
            )
        elif index == 6:
            draw.text(
                (x + 4, header + label_height + image_height + 8),
                "blue=low-q red=edge yellow=overlap",
                fill=(0, 0, 0),
            )
    canvas.save(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _ensure_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite existing analysis outputs: {existing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-result-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    import torch

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from puri_gs.responsibility import ResponsibilityConfig, compute_responsibility

    args = parse_args()
    a1_dir = args.a1_result_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_path = a1_dir / "ckpts" / "ckpt_9999_rank0.pt"
    config = json.loads((a1_dir / "config.yaml").read_text(encoding="utf-8"))
    if config.get("responsibility") != EXPECTED_A1:
        raise RuntimeError("A1 responsibility configuration differs from the fixed protocol")
    if not checkpoint_path.is_file():
        raise RuntimeError(f"A1 checkpoint is missing: {checkpoint_path}")

    frame_list_path = output_dir / "android_responsibility_frame_list.json"
    json_path = output_dir / "android_responsibility_edge_summary.json"
    csv_path = output_dir / "android_responsibility_edge_summary.csv"
    visual_dir = output_dir / "android_responsibility_selectivity"
    _ensure_absent((frame_list_path, json_path, csv_path, visual_dir))

    examples_dir = gsplat_dir / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    from datasets.colmap import Dataset, Parser
    from gsplat.rendering import rasterization

    import gsplat

    runtime_path = Path(gsplat.__file__).resolve()
    if runtime_path.is_relative_to(gsplat_dir):
        raise RuntimeError("gsplat source checkout shadows the installed fixed wheel")

    parser = Parser(
        data_dir=str(data_dir), factor=4, normalize=True, test_every=8
    )
    trainset = Dataset(
        parser,
        split="train",
        val_every=0,
        train_keyword="clutter",
        test_keyword="extra",
    )
    if len(trainset) != 122:
        raise RuntimeError(f"expected 122 Android training images, got {len(trainset)}")
    ordered_items = sorted(
        (
            parser.image_names[int(parser_index)],
            dataset_index,
        )
        for dataset_index, parser_index in enumerate(trainset.indices)
    )
    selected_positions = deterministic_uniform_indices(len(ordered_items), 8)
    selected = [ordered_items[index] for index in selected_positions]

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    if checkpoint.get("step") != 9999:
        raise RuntimeError("A1 checkpoint is not the expected step 9999 checkpoint")
    splats = {
        name: tensor.detach().to(args.device).requires_grad_(False)
        for name, tensor in checkpoint["splats"].items()
    }
    responsibility_config = ResponsibilityConfig(**EXPECTED_A1)
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=False, exist_ok=False)
    frame_list = {
        "selection": "round(linspace(0, 121, 8)) over sorted clutter filenames",
        "sorted_train_count": len(ordered_items),
        "selected_positions": selected_positions,
        "files": [name for name, _ in selected],
    }
    _write_json(frame_list_path, frame_list)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for selected_index, (image_name, dataset_index) in enumerate(selected):
            data = trainset[dataset_index]
            pixels = data["image"].unsqueeze(0).to(args.device) / 255.0
            camtoworlds = data["camtoworld"].unsqueeze(0).to(args.device)
            Ks = data["K"].unsqueeze(0).to(args.device)
            masks = (
                data["mask"].unsqueeze(0).to(args.device)
                if "mask" in data
                else None
            )
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
            stats = edge_overlap_statistics(
                q_np, gradient_np, residual_np, valid_np
            )
            row: dict[str, Any] = {
                "selected_index": selected_index,
                "sorted_position": selected_positions[selected_index],
                "image_name": image_name,
                **stats,
            }
            rows.append(row)
            save_selectivity_figure(
                visual_dir / f"{selected_index:02d}_{Path(image_name).stem}.png",
                ground_truth=pixels[0].cpu().numpy(),
                render=rendered[0].cpu().numpy(),
                residual=residual_np,
                responsibility=q_np,
                gradient=gradient_np,
                valid=valid_np,
                title=f"Android A1 selectivity: {image_name}",
            )

    summary = summarize_frames(rows)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": 9999,
            "frozen_parameters": True,
            "backward_calls": 0,
            "rasterizations_per_frame": 1,
            "responsibility_config": EXPECTED_A1,
            "gsplat_runtime_path": str(runtime_path),
            "frames": rows,
        }
    )
    _write_json(json_path, summary)
    _write_csv(csv_path, rows)
    print(json.dumps({"decision": summary["classification"], "outputs": [str(frame_list_path), str(json_path), str(csv_path), str(visual_dir)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
