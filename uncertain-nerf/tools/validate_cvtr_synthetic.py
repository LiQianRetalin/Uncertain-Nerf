#!/usr/bin/env python3
"""Prepare and validate the fixed 16-frame Room CVTR synthetic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from puri_gs.cvtr import binary_mask_metrics, load_binary_mask, save_binary_mask
from tools.build_cvtr_masks import _load_colmap_classes, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def uniformly_select(items: list[str], count: int) -> list[str]:
    if len(items) < count:
        raise ValueError(f"need at least {count} Room training images, got {len(items)}")
    indices = np.rint(np.linspace(0, len(items) - 1, count)).astype(int)
    if len(set(indices.tolist())) != count:
        raise RuntimeError("uniform frame selection produced duplicate indices")
    return [items[index] for index in indices]


def _safe_name(image_name: str) -> str:
    return Path(image_name).name + ".png"


def _load_room_training_images(data_dir: Path, gsplat_dir: Path) -> dict[str, np.ndarray]:
    """Load the exact post-resize/post-undistortion targets used by the trainer."""

    Parser, Dataset = _load_colmap_classes(gsplat_dir)
    parser = Parser(
        data_dir=str(data_dir),
        factor=4,
        normalize=True,
        test_every=8,
    )
    dataset = Dataset(parser, split="train", val_every=0)
    images: dict[str, np.ndarray] = {}
    for local_index in range(len(dataset)):
        item = dataset[local_index]
        global_index = int(dataset.indices[local_index])
        image_name = parser.image_names[global_index]
        image = item["image"]
        if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"unexpected pinned target shape for {image_name}")
        images[image_name] = image.clamp(0, 255).to(torch.uint8).cpu().numpy().copy()
    return images


def prepare_synthetic_dataset(
    *,
    data_dir: Path,
    gsplat_dir: Path,
    derived_dir: Path,
    frame_list_path: Path,
) -> dict:
    if derived_dir.exists() and any(derived_dir.iterdir()):
        raise RuntimeError(f"derived output is not empty: {derived_dir}")
    derived_dir.mkdir(parents=True, exist_ok=True)
    images = _load_room_training_images(data_dir, gsplat_dir)
    sorted_names = sorted(images)
    selected = uniformly_select(sorted_names, 16)
    entries: list[dict] = []
    for selected_index, image_name in enumerate(selected):
        target = images[image_name].copy()
        height, width = target.shape[:2]
        ground_truth = np.zeros((height, width), dtype=bool)
        source_name: str | None = None
        destination: dict[str, int] | None = None
        if selected_index >= 8:
            source_name = selected[(selected_index + 5) % len(selected)]
            source = images[source_name]
            if source.shape != target.shape:
                raise RuntimeError("Room synthetic source and target dimensions differ")
            side_fraction = math.sqrt(0.08)
            patch_height = max(1, round(height * side_fraction))
            patch_width = max(1, round(width * side_fraction))
            source_x = (selected_index * 29) % (width - patch_width + 1)
            source_y = (selected_index * 31) % (height - patch_height + 1)
            target_x = ((selected_index - 8) * 71) % (width - patch_width + 1)
            target_y = ((selected_index - 8) * 47) % (height - patch_height + 1)
            patch = source[
                source_y : source_y + patch_height,
                source_x : source_x + patch_width,
            ]
            target[
                target_y : target_y + patch_height,
                target_x : target_x + patch_width,
            ] = patch
            ground_truth[
                target_y : target_y + patch_height,
                target_x : target_x + patch_width,
            ] = True
            destination = {
                "x": target_x,
                "y": target_y,
                "width": patch_width,
                "height": patch_height,
            }
        derived_image = Path("images") / _safe_name(image_name)
        ground_truth_path = Path("ground_truth_masks") / _safe_name(image_name)
        (derived_dir / derived_image).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(target, mode="RGB").save(derived_dir / derived_image)
        save_binary_mask(derived_dir / ground_truth_path, ground_truth)
        entries.append(
            {
                "index": selected_index,
                "image_name": image_name,
                "kind": "clean" if selected_index < 8 else "transient",
                "source_image_name": source_name,
                "derived_image": derived_image.as_posix(),
                "ground_truth_mask": ground_truth_path.as_posix(),
                "destination": destination,
                "target_area_ratio": float(ground_truth.mean()),
            }
        )

    manifest = {
        "schema_version": 1,
        "scene": "room",
        "target_contract": "pinned gsplat Parser/Dataset post-resize post-undistortion",
        "data_factor": 4,
        "test_every": 8,
        "selection": "round(linspace(0, train_count-1, 16)) over sorted train names",
        "clean_count": 8,
        "transient_count": 8,
        "entries": entries,
    }
    text = json.dumps(manifest, indent=2) + "\n"
    (derived_dir / "manifest.json").write_text(text, encoding="utf-8")
    frame_list_path.parent.mkdir(parents=True, exist_ok=True)
    frame_list_path.write_text(text, encoding="utf-8")
    return manifest


def _load_derived_manifest(derived_dir: Path) -> dict:
    path = derived_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load derived Room manifest {path}: {error}") from error
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 16:
        raise ValueError("derived Room manifest must contain exactly 16 entries")
    if [entry.get("kind") for entry in entries] != ["clean"] * 8 + ["transient"] * 8:
        raise ValueError("derived Room manifest must list 8 clean then 8 transient frames")
    return manifest


def evaluate_masks(
    *, output_dir: Path, derived_dir: Path, manifest: dict
) -> tuple[dict, list[dict]]:
    cvtr_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    predicted_paths = {
        item["image_name"]: output_dir / item["mask_path"]
        for item in cvtr_manifest["images"]
    }
    all_predicted: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    rows: list[dict] = []
    clean_false_positive = 0
    clean_pixel_count = 0
    for entry in manifest["entries"]:
        name = entry["image_name"]
        predicted = load_binary_mask(predicted_paths[name])
        target = load_binary_mask(derived_dir / entry["ground_truth_mask"])
        per_frame = binary_mask_metrics(predicted, target)
        row = {
            "image_name": name,
            "kind": entry["kind"],
            **per_frame,
            "predicted_area_ratio": float(predicted.float().mean().item()),
            "target_area_ratio": float(target.float().mean().item()),
        }
        rows.append(row)
        all_predicted.append(predicted.flatten())
        all_target.append(target.flatten())
        if entry["kind"] == "clean":
            clean_false_positive += int(predicted.sum().item())
            clean_pixel_count += predicted.numel()
    aggregate = binary_mask_metrics(
        torch.cat(all_predicted), torch.cat(all_target)
    )
    aggregate.update(
        {
            "clean_false_positive_rate": clean_false_positive
            / max(clean_pixel_count, 1),
            "mean_mask_area": float(
                np.mean([row["predicted_area_ratio"] for row in rows])
            ),
        }
    )
    aggregate["gate"] = (
        "PASS"
        if (
            aggregate["precision"] >= 0.80
            and aggregate["recall"] >= 0.50
            and aggregate["f1"] >= 0.60
            and aggregate["clean_false_positive_rate"] <= 0.02
        )
        else "MASK_VALIDATION_FAIL"
    )
    return aggregate, rows


def write_outputs(output_dir: Path, metrics: dict, rows: list[dict]) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_frame.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = PROJECT_ROOT / "reports" / "PHASE_3A_CVTR_MASK_VALIDATION.md"
    report_path.write_text(
        "\n".join(
            [
                "# PURI-GS Phase 3A：CVTR 合成掩膜验证",
                "",
                f"结论：`{metrics['gate']}`",
                "",
                "| 指标 | 数值 | 门槛 |",
                "|---|---:|---:|",
                f"| Precision | {metrics['precision']:.6f} | >= 0.80 |",
                f"| Recall | {metrics['recall']:.6f} | >= 0.50 |",
                f"| F1 | {metrics['f1']:.6f} | >= 0.60 |",
                f"| IoU | {metrics['iou']:.6f} | 记录 |",
                f"| Clean FPR | {metrics['clean_false_positive_rate']:.6f} | <= 0.02 |",
                f"| Mean mask area | {metrics['mean_mask_area']:.6f} | 记录 |",
                "",
                "该验证只用于 Phase 3 mask 门，不作为论文最终结果。失败时不得启动 continuation。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument(
        "--frame-list",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "cvtr_synthetic_frame_list.json",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "cvtr_synthetic",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.prepare_only and (
        args.checkpoint is None or args.config is None or args.source_commit is None
    ):
        parser.error("validation requires --checkpoint, --source-commit, and --config")
    return args


def main() -> int:
    args = parse_args()
    derived_dir = args.derived_dir.expanduser().resolve()
    if args.prepare_only:
        manifest = prepare_synthetic_dataset(
            data_dir=args.data_dir.expanduser().resolve(),
            gsplat_dir=args.gsplat_dir.expanduser().resolve(),
            derived_dir=derived_dir,
            frame_list_path=args.frame_list.expanduser().resolve(),
        )
        print(json.dumps(manifest, indent=2))
        return 0

    manifest = _load_derived_manifest(derived_dir)
    overrides = {
        entry["image_name"]: derived_dir / entry["derived_image"]
        for entry in manifest["entries"]
    }
    build_args = argparse.Namespace(
        scene="room",
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        source_commit=args.source_commit,
        gsplat_dir=args.gsplat_dir,
        config=args.config,
        output_dir=args.output_dir,
        device=args.device,
        train_keyword=None,
        test_keyword=None,
    )
    output_dir = args.output_dir.expanduser().resolve()
    run_pipeline(build_args, target_overrides=overrides)
    metrics, rows = evaluate_masks(
        output_dir=output_dir,
        derived_dir=derived_dir,
        manifest=manifest,
    )
    write_outputs(output_dir, metrics, rows)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
