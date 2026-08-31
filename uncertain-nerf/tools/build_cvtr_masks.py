#!/usr/bin/env python3
"""Render a frozen B1 checkpoint once and build fixed CVTR masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image

from puri_gs.config import load_experiment_config
from puri_gs.cvtr import (
    CVTRConfig,
    ViewEvidence,
    compute_cross_view_transient,
    effective_sample_size_ratio,
    mask_to_responsibility,
    save_binary_mask,
    scene_residual_threshold,
    select_camera_neighbors,
    spatial_filter_and_cap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "cannot resolve repository commit")
    return result.stdout.strip()


def require_empty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_colmap_classes(gsplat_dir: Path):
    examples_dir = gsplat_dir / "examples"
    colmap_path = examples_dir / "datasets" / "colmap.py"
    if not colmap_path.is_file():
        raise RuntimeError(f"gsplat COLMAP loader is missing: {colmap_path}")
    sys.path.insert(0, str(examples_dir))
    try:
        from datasets.colmap import Dataset, Parser
    except ImportError as error:
        raise RuntimeError("cannot import the pinned gsplat COLMAP loader") from error
    return Parser, Dataset


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "step" not in checkpoint or "splats" not in checkpoint:
        raise ValueError("checkpoint must contain step and splats")
    if int(checkpoint["step"]) != 9999:
        raise ValueError(f"CVTR source checkpoint must be step 9999, got {checkpoint['step']}")
    splats = checkpoint["splats"]
    required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    missing = required.difference(splats)
    if missing:
        raise ValueError(f"checkpoint is missing splat tensors: {sorted(missing)}")
    for name in required:
        if not isinstance(splats[name], torch.Tensor) or not torch.isfinite(splats[name]).all():
            raise ValueError(f"checkpoint tensor {name!r} is missing or non-finite")
    return checkpoint


def _cvtr_config(profile: dict) -> CVTRConfig:
    values = profile.get("cvtr")
    if not isinstance(values, dict):
        raise ValueError("configuration must contain cvtr parameters")
    allowed = set(CVTRConfig.__dataclass_fields__)
    unknown = set(values).difference(allowed | {"enabled", "mask_dir"})
    if unknown:
        raise ValueError(f"unknown CVTR configuration fields: {sorted(unknown)}")
    return CVTRConfig(**{key: value for key, value in values.items() if key in allowed})


def _mask_filename(image_name: str) -> str:
    return Path(image_name).name + ".png"


def _save_cache(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_cache(path: Path, metadata: dict) -> ViewEvidence:
    with np.load(path, allow_pickle=False) as data:
        return ViewEvidence(
            image_name=metadata["image_name"],
            residual=torch.from_numpy(data["residual"].copy()),
            depth=torch.from_numpy(data["depth"].copy()),
            alpha=torch.from_numpy(data["alpha"].copy()),
            K=torch.from_numpy(data["K"].copy()),
            camtoworld=torch.from_numpy(data["camtoworld"].copy()),
        )


def _save_visualization(cache_path: Path, mask: torch.Tensor, output_path: Path) -> None:
    with np.load(cache_path, allow_pickle=False) as data:
        rendered = np.clip(data["rendered_rgb"], 0.0, 1.0)
        target = data["target_rgb"].astype(np.float32) / 255.0
        residual = data["residual"]
    heat = np.zeros_like(rendered)
    normalized = residual / max(float(np.quantile(residual, 0.99)), 1e-6)
    heat[..., 0] = np.clip(normalized, 0.0, 1.0)
    overlay = target.copy()
    binary = mask.cpu().numpy().astype(bool)
    overlay[binary] = 0.35 * overlay[binary] + 0.65 * np.array([1.0, 0.0, 1.0])
    canvas = np.concatenate([target, rendered, heat, overlay], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)).save(output_path)


@torch.inference_mode()
def render_frozen_views(
    *,
    scene: str,
    data_dir: Path,
    checkpoint_path: Path,
    gsplat_dir: Path,
    output_dir: Path,
    data_factor: int,
    test_every: int,
    sh_degree: int,
    train_keyword: str | None,
    test_keyword: str | None,
    device: torch.device,
    target_overrides: Mapping[str, Path] | None = None,
) -> list[dict]:
    """Rasterize every training view exactly once and persist detached evidence."""

    Parser, Dataset = _load_colmap_classes(gsplat_dir)
    parser = Parser(
        data_dir=str(data_dir),
        factor=data_factor,
        normalize=True,
        test_every=test_every,
    )
    dataset = Dataset(
        parser,
        split="train",
        val_every=0,
        train_keyword=train_keyword,
        test_keyword=test_keyword,
    )
    checkpoint = _load_checkpoint(checkpoint_path, device)
    splats = checkpoint["splats"]
    from gsplat.rendering import rasterization

    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"].to(device))
    opacities = torch.sigmoid(splats["opacities"].to(device))
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
    metadata: list[dict] = []
    seen_names: set[str] = set()
    target_overrides = target_overrides or {}

    for local_index in range(len(dataset)):
        item = dataset[local_index]
        global_index = int(dataset.indices[local_index])
        image_name = parser.image_names[global_index]
        if image_name in seen_names:
            raise RuntimeError(f"duplicate training image name: {image_name}")
        seen_names.add(image_name)
        target = item["image"].to(device=device, dtype=torch.float32) / 255.0
        if image_name in target_overrides:
            with Image.open(target_overrides[image_name]) as override_image:
                override = np.asarray(override_image.convert("RGB"), dtype=np.uint8).copy()
            if tuple(override.shape[:2]) != tuple(target.shape[:2]):
                raise ValueError(f"synthetic override shape mismatch for {image_name}")
            target = torch.from_numpy(override).to(device=device, dtype=torch.float32) / 255.0
        K = item["K"].to(device=device, dtype=torch.float32)
        camtoworld = item["camtoworld"].to(device=device, dtype=torch.float32)
        height, width = target.shape[:2]
        renders, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworld[None]),
            Ks=K[None],
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="classic",
            distributed=False,
            camera_model="pinhole",
            sh_degree=sh_degree,
            render_mode="RGB+ED",
        )
        rendered_rgb = renders[0, ..., :3].clamp(0.0, 1.0)
        depth = renders[0, ..., 3]
        alpha = alphas[0, ..., 0]
        residual = (target - rendered_rgb).abs().mean(dim=-1)
        cache_relative = Path("render_cache") / f"{local_index:04d}.npz"
        _save_cache(
            output_dir / cache_relative,
            rendered_rgb=rendered_rgb.detach().cpu().numpy().astype(np.float32),
            target_rgb=(target.detach().cpu().numpy() * 255.0).round().astype(np.uint8),
            residual=residual.detach().cpu().numpy().astype(np.float32),
            depth=depth.detach().cpu().numpy().astype(np.float32),
            alpha=alpha.detach().cpu().numpy().astype(np.float32),
            K=K.detach().cpu().numpy().astype(np.float32),
            camtoworld=camtoworld.detach().cpu().numpy().astype(np.float32),
        )
        metadata.append(
            {
                "scene": scene,
                "image_name": image_name,
                "cache_path": cache_relative.as_posix(),
                "height": height,
                "width": width,
                "local_train_index": local_index,
                "global_parser_index": global_index,
            }
        )
        print(f"rendered {scene}: {local_index + 1}/{len(dataset)} {image_name}", flush=True)
    (output_dir / "render_cache_manifest.json").write_text(
        json.dumps({"scene": scene, "views": metadata}, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_masks_from_cache(
    *,
    scene: str,
    checkpoint_path: Path,
    output_dir: Path,
    view_metadata: list[dict],
    config: CVTRConfig,
    source_commit: str,
    tool_commit: str,
    source_sha256: str,
) -> dict:
    records: dict[str, ViewEvidence] = {}
    residuals: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    names: list[str] = []
    camtoworlds: list[torch.Tensor] = []
    metadata_by_name = {item["image_name"]: item for item in view_metadata}
    for item in view_metadata:
        record = _load_cache(output_dir / item["cache_path"], item)
        records[record.image_name] = record
        residuals.append(record.residual)
        alphas.append(record.alpha)
        names.append(record.image_name)
        camtoworlds.append(record.camtoworld)

    threshold, median, scale, sample_count = scene_residual_threshold(
        residuals, alphas, config
    )
    neighbors = select_camera_neighbors(
        torch.stack(camtoworlds), names, config.neighbor_count
    )
    neighbor_payload = {
        "scene": scene,
        "rule": "euclidean_camera_center_then_filename",
        "neighbor_count": config.neighbor_count,
        "neighbors": neighbors,
    }
    neighbor_text = json.dumps(neighbor_payload, indent=2) + "\n"
    (output_dir / "neighbors.json").write_text(neighbor_text, encoding="utf-8")
    (output_dir / "cvtr_neighbors.json").write_text(neighbor_text, encoding="utf-8")

    mask_filenames = [_mask_filename(name) for name in names]
    if len(set(mask_filenames)) != len(mask_filenames):
        raise RuntimeError("training image basenames collide after CVTR mask naming")

    image_entries: list[dict] = []
    for image_name, mask_filename in zip(names, mask_filenames):
        source = records[image_name]
        neighbor_records = [records[name] for name in neighbors[image_name]]
        cross_view = compute_cross_view_transient(
            source, neighbor_records, threshold, config
        )
        mask = spatial_filter_and_cap(
            cross_view.raw_transient_mask, source.residual, config
        )
        mask_area_ratio = float(mask.float().mean().item())
        if mask_area_ratio > config.max_transient_area + 1e-8:
            raise RuntimeError(f"area cap failed for {image_name}: {mask_area_ratio}")
        q = mask_to_responsibility(mask, config.transient_weight)
        neff_ratio = effective_sample_size_ratio(q, config.epsilon)
        if neff_ratio < config.minimum_neff_ratio:
            raise RuntimeError(
                f"Neff guard failed for {image_name}: {neff_ratio:.8f}"
            )
        mask_relative = Path("masks") / mask_filename
        save_binary_mask(output_dir / mask_relative, mask)
        _save_visualization(
            output_dir / metadata_by_name[image_name]["cache_path"],
            mask,
            output_dir / "visualizations" / mask_filename,
        )
        entry = {
            "scene": scene,
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_checkpoint_sha256": source_sha256,
            "source_commit": source_commit,
            "image_name": image_name,
            "mask_path": mask_relative.as_posix(),
            "mask_area_ratio": mask_area_ratio,
            "valid_neighbor_ratio": cross_view.valid_neighbor_ratio(
                config.minimum_valid_neighbors
            ),
            "candidate_ratio": cross_view.candidate_ratio,
            "neff_ratio": neff_ratio,
        }
        image_entries.append(entry)
        print(f"masked {scene}: {len(image_entries)}/{len(names)} {image_name}", flush=True)

    candidate_ratios = np.array([item["candidate_ratio"] for item in image_entries])
    mask_ratios = np.array([item["mask_area_ratio"] for item in image_entries])
    valid_neighbor_ratios = np.array(
        [item["valid_neighbor_ratio"] for item in image_entries]
    )
    neff_ratios = np.array([item["neff_ratio"] for item in image_entries])
    clean_guard_applies = scene.casefold() in {"garden", "room"}
    clean_guard_pass = bool(
        not clean_guard_applies
        or (
            mask_ratios.mean() <= 0.05
            and np.quantile(mask_ratios, 0.95) <= 0.12
            and neff_ratios.min() >= config.minimum_neff_ratio
        )
    )
    statistics = {
        "scene": scene,
        "scene_residual_median": median,
        "scene_residual_scale": scale,
        "scene_residual_threshold": threshold,
        "scene_residual_sample_count": sample_count,
        "mean_candidate_ratio": float(candidate_ratios.mean()),
        "mean_transient_mask_ratio": float(mask_ratios.mean()),
        "p95_transient_mask_ratio": float(np.quantile(mask_ratios, 0.95)),
        "mean_valid_neighbor_ratio": float(valid_neighbor_ratios.mean()),
        "mean_neff_ratio": float(neff_ratios.mean()),
        "minimum_neff_ratio": float(neff_ratios.min()),
        "clean_mask_guard_applies": clean_guard_applies,
        "clean_mask_guard": "PASS" if clean_guard_pass else "CLEAN_MASK_GUARD_FAIL",
        "config": config.to_dict(),
    }
    manifest = {
        "schema_version": 1,
        "scene": scene,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "source_commit": source_commit,
        "tool_commit": tool_commit,
        "depth_semantics": "gsplat RGB+ED expected camera-z depth",
        "images": image_entries,
    }
    (output_dir / "scene_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return statistics


def run_pipeline(
    args: argparse.Namespace,
    *,
    target_overrides: Mapping[str, Path] | None = None,
) -> dict:
    output_dir = args.output_dir.expanduser().resolve()
    require_empty_output(output_dir)
    profile = load_experiment_config(args.config.expanduser().resolve())
    config = _cvtr_config(profile)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen gsplat rasterization")
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"B1 checkpoint is missing: {checkpoint_path}")
    source_sha256 = sha256_file(checkpoint_path)
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.source_commit):
        raise ValueError("--source-commit must be the B1 training commit hash")
    source_commit = args.source_commit.lower()
    tool_commit = repository_commit()
    start = time.perf_counter()
    render_start = start
    views = render_frozen_views(
        scene=args.scene,
        data_dir=args.data_dir.expanduser().resolve(),
        checkpoint_path=checkpoint_path,
        gsplat_dir=args.gsplat_dir.expanduser().resolve(),
        output_dir=output_dir,
        data_factor=profile["training"]["data_factor"],
        test_every=profile["training"]["test_every"],
        sh_degree=profile["training"]["sh_degree"],
        train_keyword=args.train_keyword,
        test_keyword=args.test_keyword,
        device=device,
        target_overrides=target_overrides,
    )
    render_seconds = time.perf_counter() - render_start
    mask_start = time.perf_counter()
    statistics = build_masks_from_cache(
        scene=args.scene,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        view_metadata=views,
        config=config,
        source_commit=source_commit,
        tool_commit=tool_commit,
        source_sha256=source_sha256,
    )
    statistics.update(
        {
            "render_seconds": render_seconds,
            "mask_build_seconds": time.perf_counter() - mask_start,
            "total_generation_seconds": time.perf_counter() - start,
        }
    )
    (output_dir / "scene_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=("android", "garden", "room"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-keyword")
    parser.add_argument("--test-keyword")
    args = parser.parse_args()
    if (args.train_keyword is None) != (args.test_keyword is None):
        parser.error("--train-keyword and --test-keyword must be supplied together")
    if args.scene == "android" and args.train_keyword is None:
        parser.error("Android requires --train-keyword clutter --test-keyword extra")
    if args.scene != "android" and args.train_keyword is not None:
        parser.error("filename-keyword split is only used for Android")
    return args


def main() -> int:
    args = parse_args()
    statistics = run_pipeline(args)
    print(json.dumps(statistics, indent=2))
    return 0 if statistics["clean_mask_guard"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
