#!/usr/bin/env python3
"""Build the frozen Work Package 1 ledgers from local immutable evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ARCHIVE_SHA256 = "ac7376ef0dc89df871635ec2ffe1867a7e721b5979cd84c44029cfe793ed51e3"
ROOM_VISUALS_SHA256 = "360ef84c11c62fd55f5ea4f7179274ee955bfb46b768e7ef6f4896a7aaeac465"
GSPLAT_COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT.parent, text=True, encoding="utf-8"
    ).strip()


def display_path(path: Path) -> str:
    """Render WSL-mounted local paths as the user's native Windows paths."""
    parts = path.resolve().as_posix().split("/")
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        return parts[2].upper() + ":\\" + "\\".join(parts[3:])
    return str(path.resolve())


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    )


def image_manifest(paths: list[Path], label: str) -> tuple[list[str], str]:
    digest = hashlib.sha256()
    names: list[str] = []
    for index, path in enumerate(paths, 1):
        names.append(path.name)
        digest.update(f"{path.name},{sha256_file(path)}\n".encode("utf-8"))
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(f"HASHED_{label.upper()}_IMAGES={index}/{len(paths)}", flush=True)
    return names, digest.hexdigest()


def effective_pinhole(
    params: list[float], source_size: list[int], stored_size: list[int], factor: int
) -> list[list[float]]:
    nominal_width = source_size[0] // factor
    nominal_height = source_size[1] // factor
    scale_x = stored_size[0] / nominal_width
    scale_y = stored_size[1] / nominal_height
    fx, fy, cx, cy = params
    return [
        [fx / factor * scale_x, 0.0, cx / factor * scale_x],
        [0.0, fy / factor * scale_y, cy / factor * scale_y],
        [0.0, 0.0, 1.0],
    ]


def colmap_protocol(
    *,
    protocol_id: str,
    scene: str,
    root: Path,
    factor: int,
    split_path: Path,
    camera_model: str,
    camera_source_size: list[int],
    camera_params: list[float],
    evaluation_size: list[int] | None = None,
    effective_K: list[list[float]] | None = None,
    status: str = "FROZEN_READY",
    notes: str = "",
) -> dict[str, Any]:
    image_dir = root / f"images_{factor}"
    images = image_files(image_dir)
    names, pixel_manifest_sha = image_manifest(images, protocol_id)
    with Image.open(images[0]) as first:
        stored_size = [first.width, first.height]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    sparse = root / "sparse" / "0"
    if not set(split["train"] + split["test"]).issubset(names):
        raise RuntimeError(f"split contains images outside {protocol_id}")
    if effective_K is None and camera_model == "PINHOLE":
        effective_K = effective_pinhole(camera_params, camera_source_size, stored_size, factor)
    return {
        "protocol_id": protocol_id,
        "status": status,
        "scene": scene,
        "dataset_format": "COLMAP",
        "local_root": display_path(root),
        "physical_data_factor": factor,
        "loader_data_factor": factor,
        "image_directory": display_path(image_dir),
        "registered_image_count": len(names),
        "ordered_image_names": names,
        "ordered_image_names_sha256": hashlib.sha256(
            ("\n".join(names) + "\n").encode("utf-8")
        ).hexdigest(),
        "image_content_manifest_sha256": pixel_manifest_sha,
        "stored_image_size": stored_size,
        "evaluation_image_size": evaluation_size or stored_size,
        "camera": {
            "source_model": camera_model,
            "source_size": camera_source_size,
            "source_params": camera_params,
            "effective_K": effective_K,
        },
        "sparse_model": {
            name: {"sha256": sha256_file(sparse / name), "bytes": (sparse / name).stat().st_size}
            for name in ("cameras.bin", "images.bin", "points3D.bin")
        },
        "normalization": {
            "enabled": True,
            "implementation": "gsplat-1.5.3 examples/datasets/normalize.py: similarity_from_cameras then align_principal_axes",
            "gsplat_commit": GSPLAT_COMMIT,
        },
        "split": {
            **split,
            "train_count": len(split["train"]),
            "test_count": len(split["test"]),
            "manifest_sha256": sha256_file(split_path),
            "canonical_sha256": sha256_json(split),
        },
        "evaluation": "all frozen test names; render clamp [0,1]; per-image then arithmetic mean",
        "notes": notes,
    }


def evidence_dir(source_root: Path, scene: str, method: str) -> Path:
    mapping = {
        "android": "logs-puri/ru-generalization-rerun-9e292309",
        "garden": "logs-puri/ru-generalization-rerun-9e292309",
        "patio_high": "logs-puri/ru-generalization-ontogo-749d584b",
        "room": "logs-puri/phase_r",
    }
    return source_root / mapping[scene] / f"{scene}_{method}_30k" / "independent_eval"


def main() -> int:
    args = parse_args()
    archive_root = args.archive_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    p01 = output / "p01"
    p01.mkdir(parents=True, exist_ok=True)
    branch = git("branch", "--show-current")
    if branch != "ru-part":
        raise RuntimeError(f"Work Package 1 must stay on ru-part, got {branch}")

    archive = archive_root / "server-artifacts" / "puri_gs_ru_artifacts_749d584b.tar"
    room_visuals = archive_root / "server-artifacts" / "puri_gs_ru_room_visuals.tar"
    archive_hash = sha256_file(archive)
    room_visuals_hash = sha256_file(room_visuals)
    if archive_hash != ARCHIVE_SHA256 or room_visuals_hash != ROOM_VISUALS_SHA256:
        raise RuntimeError("frozen server archive SHA-256 mismatch")

    source_root = archive_root / "visualizations" / "source-artifacts"
    android_root = data_root / "nerf数据集" / "nerf_robustnerf" / "robustnerf" / "android"
    garden_root = data_root / "nerf数据集" / "mipnerf360" / "360_v2" / "garden"
    room_root = data_root / "nerf数据集" / "mipnerf360" / "360_v2" / "room"
    patio_root = data_root / "PURI-GS-derived" / "ontogo" / "patio_high_v1"
    patio_common_root = data_root / "PURI-GS-derived" / "ontogo" / "patio_high_colmap_common_v1"
    android_common_root = data_root / "PURI-GS-derived" / "robustnerf" / "android_colmap_common_factor4_v1"

    android_split = evidence_dir(source_root, "android", "b1") / "dataset_split.json"
    garden_split = evidence_dir(source_root, "garden", "b1") / "dataset_split.json"
    room_split = evidence_dir(source_root, "room", "b1") / "dataset_split.json"
    protocols: list[dict[str, Any]] = [
        colmap_protocol(
            protocol_id="android-colmap-factor4-keyword-v1",
            scene="android",
            root=android_root,
            factor=4,
            split_path=android_split,
            camera_model="OPENCV",
            camera_source_size=[4032, 3024],
            camera_params=[3043.54919152632, 3049.49085570768, 2016.0, 1512.0, 0.0864110093169592, -0.125081405580787, 0.00295473872857908, -0.00110085156070557],
            evaluation_size=[1007, 755],
            effective_K=[[771.14471436, 0.0, 502.39539334], [0.0, 773.23706055, 379.34622265], [0.0, 0.0, 1.0]],
            notes="263 registered images exist; only 122 clutter train and 19 extra test images participate; 122 clean images are excluded by keyword split.",
        ),
        colmap_protocol(
            protocol_id="garden-colmap-factor4-every8-v1",
            scene="garden",
            root=garden_root,
            factor=4,
            split_path=garden_split,
            camera_model="PINHOLE",
            camera_source_size=[5187, 3361],
            camera_params=[3844.8987769586, 3852.35617923794, 2593.5, 1680.5],
            notes="Frozen registration only; Garden is not scheduled in P02.",
        ),
        colmap_protocol(
            protocol_id="room-colmap-factor4-every8-historical-v1",
            scene="room",
            root=room_root,
            factor=4,
            split_path=room_split,
            camera_model="PINHOLE",
            camera_source_size=[3114, 2075],
            camera_params=[3172.52943742005, 3173.95302570017, 1557.0, 1037.5],
            notes="Historical protected protocol: train272/test39. It must not populate a factor2 main-table row.",
        ),
        colmap_protocol(
            protocol_id="room-colmap-factor2-formal-main-v1",
            scene="room",
            root=room_root,
            factor=2,
            split_path=room_split,
            camera_model="PINHOLE",
            camera_source_size=[3114, 2075],
            camera_params=[3172.52943742005, 3173.95302570017, 1557.0, 1037.5],
            status="READY_DATA_REQUIRES_NEW_MODELS",
            notes="Formal indoor main-table protocol. Same registration and every-eighth names, but changed training pixels/intrinsics require new model identities.",
        ),
    ]

    patio_protocol = json.loads((patio_root / "puri_gs_protocol.json").read_text(encoding="utf-8"))
    patio_common_protocol = json.loads((patio_common_root / "common_input_protocol.json").read_text(encoding="utf-8"))
    patio_common_validation = json.loads((patio_common_root / "common_input_validation.json").read_text(encoding="utf-8"))
    android_common_protocol = json.loads((android_common_root / "common_input_protocol.json").read_text(encoding="utf-8"))
    android_common_validation = json.loads((android_common_root / "common_input_validation.json").read_text(encoding="utf-8"))
    if patio_common_validation["decision"] != "PATIO_HIGH_COMMON_COLMAP_VALID":
        raise RuntimeError("Patio common COLMAP input is not validated")
    if android_common_validation["decision"] != "ANDROID_COMMON_COLMAP_VALID":
        raise RuntimeError("Android common COLMAP input is not validated")
    protocols.extend(
        [
            {
                "protocol_id": "android-colmap-common-factor4-v1",
                "status": "FROZEN_READY",
                "scene": "android",
                "dataset_format": "COLMAP",
                "local_root": display_path(android_common_root),
                "physical_data_factor": 4,
                "loader_data_factor": 1,
                "common_input_protocol_sha256": sha256_file(android_common_root / "common_input_protocol.json"),
                "common_input_validation_sha256": sha256_file(android_common_root / "common_input_validation.json"),
                "conversion_only": True,
                "sfm_reestimated": False,
                "projection_validation": android_common_validation,
                "protocol": {key: value for key, value in android_common_protocol.items() if key != "images"},
                "ordered_image_names": [record["name"] for record in android_common_protocol["images"]],
                "image_content_manifest_sha256": sha256_json(
                    [[record["name"], record["sha256"]] for record in android_common_protocol["images"]]
                ),
                "normalization": "Each external method retains its declared normalization; raw cameras, points, pixels and split stay common.",
                "evaluation": "19 extra images; one frozen independent evaluator after rendering",
            },
            {
                "protocol_id": "patio-high-internal-factor4-v1",
                "status": "FROZEN_READY",
                "scene": "patio_high",
                "dataset_format": "ontogo-patio-high",
                "local_root": display_path(patio_root),
                "physical_data_factor": 4,
                "loader_data_factor": 4,
                "frame_count": 267,
                "train_count": 221,
                "test_count": 45,
                "excluded_count": 1,
                "stored_image_size": [1008, 756],
                "evaluation_image_size": [1007, 755],
                "camera": {"source": "transforms.json OpenGL c2w + OpenCV distortion", "effective_model": "PINHOLE", "effective_K": patio_common_protocol["K"]},
                "transforms_sha256": patio_protocol["transforms_sha256"],
                "split_sha256": patio_protocol["split_sha256"],
                "initial_point_count": patio_protocol["initial_point_count"],
                "initial_points_sha256": patio_protocol["initial_points_sha256"],
                "initial_colors_sha256": patio_protocol["initial_colors_sha256"],
                "ordered_image_names": [record["file"] for record in patio_protocol["images"]],
                "image_content_manifest_sha256": sha256_json(
                    [[record["file"], record["sha256"]] for record in patio_protocol["images"]]
                ),
                "normalization": {"enabled": True, "implementation": "same pinned gsplat similarity/PCA path", "scene_scale": 1.297191125677561},
                "evaluation": "45 extra images; clamp [0,1]; per-image then arithmetic mean",
            },
            {
                "protocol_id": "patio-high-colmap-common-factor4-v1",
                "status": "FROZEN_READY",
                "scene": "patio_high",
                "dataset_format": "COLMAP",
                "local_root": display_path(patio_common_root),
                "physical_data_factor": 4,
                "loader_data_factor": 1,
                "common_input_protocol_sha256": sha256_file(patio_common_root / "common_input_protocol.json"),
                "common_input_validation_sha256": sha256_file(patio_common_root / "common_input_validation.json"),
                "conversion_only": True,
                "sfm_reestimated": False,
                "projection_validation": patio_common_validation,
                "protocol": {key: value for key, value in patio_common_protocol.items() if key != "images"},
                "ordered_image_names": [record["name"] for record in patio_common_protocol["images"]],
                "image_content_manifest_sha256": sha256_json(
                    [[record["name"], record["sha256"]] for record in patio_common_protocol["images"]]
                ),
                "normalization": "Each external method retains its declared normalization; raw cameras, points, pixels and split stay common.",
                "evaluation": "45 extra images; one frozen independent evaluator after rendering",
            },
        ]
    )
    protocol_manifest = {
        "schema": "puri-gs-work-package-1-protocols-v1",
        "generated_date": "2026-09-10",
        "repository": {"branch": branch, "head": git("rev-parse", "HEAD")},
        "gpu_priority": [6, 7, 0, 1, 2, 3, 4, 5],
        "evaluator": {
            "id": "puri-gs-gsplat153-independent-eval-v1",
            "render_clamp": [0.0, 1.0],
            "psnr": "torchmetrics 1.4.3 PeakSignalNoiseRatio(data_range=1.0)",
            "ssim": "torchmetrics 1.4.3 StructuralSimilarityIndexMeasure(data_range=1.0)",
            "lpips": "torchmetrics 1.4.3 LearnedPerceptualImagePatchSimilarity(net_type=alex, normalize=True)",
            "aggregation": "arithmetic mean over the complete frozen test list",
            "server_environment": "NVIDIA L20; Python 3.10.20; torch 2.4.0+cu121; CUDA 12.1; gsplat 1.5.3",
            "strict_timing": "10 warmup renders, image saving disabled, 3 complete repeats; report raw latency and median FPS",
        },
        "protocols": protocols,
    }
    write_json(output / "protocol_manifest.json", protocol_manifest)

    checkpoint = {
        ("android", "b1"): ("9ea312974862c3cd3b4fdfea07be90b7e3aeb799a8e31ae59902fbc8cfb9db7c", 287546594, 1218407, "9e29230952be700dc5b527008e2f60f05b82717d", 614.4800188541412),
        ("android", "ru"): ("11737109e7742872ca28f9d32591d0252bb6339aa8fc77c41ba314ee3656f73a", 130108706, 551297, "9e29230952be700dc5b527008e2f60f05b82717d", 633.9885854721069),
        ("garden", "b1"): ("06e7911a76285db4c4c545cbfa1bdfccf3690764597fe73671f265f305afd35a", 896449250, 3798503, "9e29230952be700dc5b527008e2f60f05b82717d", 1631.61976480484),
        ("garden", "ru"): ("e198d8af74eaed8c8b7da8d97e75b891102b2f4cbb5b548c7dc23207cae915a9", 536713186, 2274197, "9e29230952be700dc5b527008e2f60f05b82717d", 1347.4549956321716),
        ("patio_high", "b1"): ("31f618c78c6444b69b499d3b616f09382dba0e76e6c96952ccb5806ea3873503", 763625506, 3235690, "749d584bd4741ec9619b158942da93b872e87029", 1109.4157421588898),
        ("patio_high", "ru"): ("b3c9fca6ed4470c8a66aba31e2917bf52baa9fed6d2f12a99589ff4fbddce1ae", 102846370, 435779, "749d584bd4741ec9619b158942da93b872e87029", 551.6290900707245),
        ("room", "b1"): ("", "", 1301078, "da5089cd34bad85780e974f732c2bc722b05c77f", ""),
        ("room", "ru"): ("", "", 708337, "da5089cd34bad85780e974f732c2bc722b05c77f", ""),
    }
    archive_members = {
        ("android", "b1"): "logs-puri/ru-generalization-rerun-9e292309/android_b1_30k/ckpts/ckpt_29999_rank0.pt",
        ("android", "ru"): "logs-puri/ru-generalization-rerun-9e292309/android_ru_30k/ckpts/ckpt_29999_rank0.pt",
        ("garden", "b1"): "logs-puri/ru-generalization-rerun-9e292309/garden_b1_30k/ckpts/ckpt_29999_rank0.pt",
        ("garden", "ru"): "logs-puri/ru-generalization-rerun-9e292309/garden_ru_30k/ckpts/ckpt_29999_rank0.pt",
        ("patio_high", "b1"): "logs-puri/ru-generalization-ontogo-749d584b/patio_high_b1_30k/ckpts/ckpt_29999_rank0.pt",
        ("patio_high", "ru"): "logs-puri/ru-generalization-ontogo-749d584b/patio_high_ru_30k/ckpts/ckpt_29999_rank0.pt",
    }
    server_roots = {
        "android": "ru-generalization-rerun-9e292309",
        "garden": "ru-generalization-rerun-9e292309",
        "patio_high": "ru-generalization-ontogo-749d584b",
        "room": "phase_r",
    }
    asset_rows: list[dict[str, Any]] = []
    for scene, method in checkpoint:
        sha, size, count, commit, training_seconds = checkpoint[(scene, method)]
        protocol_id = {
            "android": "android-colmap-factor4-keyword-v1",
            "garden": "garden-colmap-factor4-every8-v1",
            "patio_high": "patio-high-internal-factor4-v1",
            "room": "room-colmap-factor4-every8-historical-v1",
        }[scene]
        run_name = f"{scene}_{method}_30k"
        asset_rows.append(
            {
                "asset_id": f"{scene}-{method}-30k",
                "scene": scene,
                "method": method.upper(),
                "variant": "gsplat153-absgrad" if method == "b1" else "original-GS-RU",
                "role": "frozen_anchor" if scene != "garden" else "frozen_registration_only",
                "status": "FROZEN_CHECKPOINT_LOCAL" if sha else "FROZEN_EVAL_ONLY_CHECKPOINT_SERVER_UNVERIFIED",
                "training_commit": commit,
                "patch_stack": "B1 default+AbsGrad" if method == "b1" else "PURI-GS-RU original mask+delayed densification",
                "dataset_protocol": protocol_id,
                "seed": 42,
                "checkpoint_step": 29999,
                "checkpoint_sha256": sha,
                "checkpoint_bytes": size,
                "gaussian_count": count,
                "training_seconds": training_seconds,
                "local_checkpoint_container": f"{display_path(archive)}::{archive_members[(scene, method)]}" if sha else "",
                "server_checkpoint_path": f"/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/{server_roots[scene]}/{run_name}/ckpts/ckpt_29999_rank0.pt",
                "evaluation_evidence": display_path(evidence_dir(source_root, scene, method)),
                "notes": "Room checkpoint hash/bytes and absolute training seconds are absent from local evidence; server SSH authentication was unavailable." if scene == "room" else "Standard checkpoint keys: splats, step; frozen archive outer SHA-256 reverified.",
            }
        )
    asset_rows.extend(
        [
            {
                "asset_id": "android-b1-10k-historical",
                "scene": "android",
                "method": "B1",
                "variant": "gsplat153-absgrad-historical-10k",
                "role": "historical_reference_not_p02_anchor",
                "status": "REPORTED_ONLY_CHECKPOINT_NOT_LOCAL",
                "training_commit": "32a7d787d6886d847061a853c66f7de90f2eebe4",
                "patch_stack": "configs/puri_gs_b1_absgrad.yaml",
                "dataset_protocol": "android-colmap-factor4-keyword-v1",
                "seed": 42,
                "checkpoint_step": 9999,
                "checkpoint_sha256": "",
                "checkpoint_bytes": "",
                "gaussian_count": "",
                "training_seconds": "",
                "local_checkpoint_container": "",
                "server_checkpoint_path": "",
                "evaluation_evidence": "android_ru_pairing_audit.json historical_b1_protocol_difference",
                "notes": "Reported PSNR 24.312067; different 10k budget, never substitute for the selected paired 30k B1.",
            },
            {
                "asset_id": "android-phase-r-original-pair",
                "scene": "android",
                "method": "B1/RU",
                "variant": "original-Phase-R-pair",
                "role": "historical_result_not_p02_anchor",
                "status": "REPORTED_ONLY_CHECKPOINT_IDENTITY_INCOMPLETE_LOCAL",
                "training_commit": "",
                "patch_stack": "original Phase-R",
                "dataset_protocol": "android-colmap-factor4-keyword-v1",
                "seed": 42,
                "checkpoint_step": 29999,
                "checkpoint_sha256": "",
                "checkpoint_bytes": "",
                "gaussian_count": "",
                "training_seconds": "",
                "local_checkpoint_container": "",
                "server_checkpoint_path": "/home/chenglong/Uncertain-Nerf/uncertain-nerf/logs-puri/phase_r/android_{b1,ru}_30k/ckpts/ckpt_29999_rank0.pt",
                "evaluation_evidence": "reports/phase_r_puri_gs_ru.json",
                "notes": "Preserve reported delta PSNR +1.206509, but do not mix it with rerun checkpoints or strict timing.",
            },
        ]
    )
    asset_fields = [
        "asset_id", "scene", "method", "variant", "role", "status", "training_commit",
        "patch_stack", "dataset_protocol", "seed", "checkpoint_step", "checkpoint_sha256",
        "checkpoint_bytes", "gaussian_count", "training_seconds", "local_checkpoint_container",
        "server_checkpoint_path", "evaluation_evidence", "notes",
    ]
    write_csv(output / "asset_ledger.csv", asset_rows, asset_fields)

    strict_fps = {
        ("android", "b1"): 255.97039816016343,
        ("android", "ru"): 479.11271433037535,
        ("patio_high", "b1"): 111.0492757579698,
        ("patio_high", "ru"): 498.0464841358637,
        ("room", "b1"): 263.9412538261334,
        ("room", "ru"): 424.24063613587913,
    }
    summary_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    for scene in ("android", "patio_high", "room", "garden"):
        paired_names: list[list[str]] = []
        for method in ("b1", "ru"):
            directory = evidence_dir(source_root, scene, method)
            metrics = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
            sha, size, count, commit, training_seconds = checkpoint[(scene, method)]
            summary_rows.append(
                {
                    "scene": scene,
                    "method": method.upper(),
                    "seed": 42,
                    "protocol_id": {"android": "android-colmap-factor4-keyword-v1", "patio_high": "patio-high-internal-factor4-v1", "room": "room-colmap-factor4-every8-historical-v1", "garden": "garden-colmap-factor4-every8-v1"}[scene],
                    "evaluator_id": "puri-gs-gsplat153-independent-eval-v1",
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "lpips": metrics["lpips"],
                    "test_image_count": {"android": 19, "patio_high": 45, "room": 39, "garden": 24}[scene],
                    "gaussian_count": count,
                    "strict_median_fps": strict_fps.get((scene, method), ""),
                    "training_seconds": training_seconds,
                    "checkpoint_sha256": sha,
                    "evaluation_commit": (directory / "git_commit.txt").read_text(encoding="utf-8").strip(),
                    "source": display_path(directory / "test_metrics.json"),
                }
            )
            names: list[str] = []
            with (directory / "per_image_metrics.csv").open(newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    names.append(row["image_name"])
                    per_image_rows.append(
                        {
                            "scene": scene,
                            "method": method.upper(),
                            "seed": 42,
                            "image_name": row["image_name"],
                            "psnr": row["psnr"],
                            "ssim": row["ssim"],
                            "lpips": row["lpips"],
                            "evaluator_id": "puri-gs-gsplat153-independent-eval-v1",
                            "checkpoint_sha256": sha,
                        }
                    )
            paired_names.append(names)
        if paired_names[0] != paired_names[1]:
            raise RuntimeError(f"B1/RU per-image order mismatch for {scene}")
    summary_fields = [
        "scene", "method", "seed", "protocol_id", "evaluator_id", "psnr", "ssim", "lpips",
        "test_image_count", "gaussian_count", "strict_median_fps", "training_seconds",
        "checkpoint_sha256", "evaluation_commit", "source",
    ]
    write_csv(p01 / "reevaluation_summary.csv", summary_rows, summary_fields)
    write_csv(
        p01 / "reevaluation_per_image.csv",
        per_image_rows,
        ["scene", "method", "seed", "image_name", "psnr", "ssim", "lpips", "evaluator_id", "checkpoint_sha256"],
    )

    run_plan_rows = []
    method_specs = {
        "RobustSplat": ("RobustSplat", "a130281d6d0c004032a9a57e8d6a14962d9836d3", "https://github.com/fcyycf/RobustSplat", "协议不同：官方 checkpoint 已发布，但使用作者重跑 SfM/作者降采样输入", "official full recipe"),
        "SLS-mlp": ("SLS-mlp-no-UBP", "0caae3cc45bb1fddf86bd47e4a521888f5c49889", "https://github.com/lilygoli/SpotLessSplats", "协议不同：公开复现数字和数据/特征不是本项目共同输入；未发现可直接复评的同协议 checkpoint", "--loss_type robust --semantics --no-cluster; do not pass --ubp"),
    }
    for scene in ("android", "patio_high"):
        for method in ("RobustSplat", "SLS-mlp"):
            variant, commit, url, official_status, recipe = method_specs[method]
            input_root = android_common_root if scene == "android" else patio_common_root
            protocol_id = (
                "android-colmap-common-factor4-v1"
                if scene == "android"
                else "patio-high-colmap-common-factor4-v1"
            )
            run_plan_rows.append(
                {
                    "run_id": f"P02-{scene}-{method.lower()}",
                    "run_type": "external_training",
                    "scene": scene,
                    "method": method,
                    "variant": variant,
                    "protocol_id": protocol_id,
                    "status": "须训练",
                    "official_asset_status": official_status,
                    "reason": "共同输入下没有可直接复评的 checkpoint；作者协议数字只作 Reported 参考。",
                    "source_commit": commit,
                    "source_url": url,
                    "common_input_root": display_path(input_root),
                    "server_input_root": "TO_STAGE_AFTER_P02_AUTHORIZATION",
                    "physical_data_factor": 4,
                    "loader_data_factor": 1,
                    "train_keyword": "clutter",
                    "test_keyword": "extra",
                    "seed": 42,
                    "gpu_priority": "6>7>0>1>2>3>4>5",
                    "max_attempts": 2,
                    "checkpoint_policy": "作者 checkpoint 仅作作者协议参考/兼容检查；一次最小兼容 smoke 后，在共同输入上执行官方完整配方训练",
                    "method_specific_preparation": "Stable Diffusion features for the exact common-input images; include preparation time" if method == "SLS-mlp" else "none beyond method environment and verified loader mapping",
                    "recipe_lock": recipe,
                    "evaluation": "render exactly the frozen extra list; one independent evaluator; 10 warmups and 3 timing repeats",
                    "normalization_note": "Method-native world normalization is allowed and must be recorded; cameras/pixels/points/split may not change.",
                }
            )
    run_plan_fields = [
        "run_id", "run_type", "scene", "method", "variant", "protocol_id", "status", "official_asset_status", "reason", "source_commit", "source_url",
        "common_input_root", "server_input_root", "physical_data_factor", "loader_data_factor", "train_keyword", "test_keyword",
        "seed", "gpu_priority", "max_attempts", "checkpoint_policy", "method_specific_preparation",
        "recipe_lock", "evaluation", "normalization_note",
    ]
    write_csv(output / "run_plan_P02.csv", run_plan_rows, run_plan_fields)

    run_ledger_rows: list[dict[str, Any]] = []
    for row in asset_rows[:8]:
        run_ledger_rows.append(
            {
                "identity_id": row["asset_id"], "work_package": "P01", "scene": row["scene"],
                "method": row["method"], "variant": row["variant"], "seed": row["seed"],
                "training_status": "REUSED", "training_identity": row["training_commit"],
                "training_path": row["server_checkpoint_path"], "evaluation_status": "REUSED_VALID",
                "evaluation_identity": "puri-gs-gsplat153-independent-eval-v1",
                "evidence_path": row["evaluation_evidence"], "notes": row["notes"],
            }
        )
    for row in run_plan_rows:
        run_ledger_rows.append(
            {
                "identity_id": row["run_id"], "work_package": "P02", "scene": row["scene"],
                "method": row["method"], "variant": row["variant"], "seed": row["seed"],
                "training_status": "NOT_RUN_REQUIRES_P02_AUTHORIZATION", "training_identity": row["source_commit"],
                "training_path": "", "evaluation_status": "NOT_RUN", "evaluation_identity": "puri-gs-gsplat153-independent-eval-v1",
                "evidence_path": "", "notes": row["reason"],
            }
        )
    ledger_fields = [
        "identity_id", "work_package", "scene", "method", "variant", "seed", "training_status",
        "training_identity", "training_path", "evaluation_status", "evaluation_identity", "evidence_path", "notes",
    ]
    write_csv(p01 / "run_ledger.csv", run_ledger_rows, ledger_fields)

    write_json(
        p01 / "archive_verification.json",
        {
            "schema": "puri-gs-p01-archive-verification-v1",
            "server_artifacts": {"path": display_path(archive), "bytes": archive.stat().st_size, "sha256": archive_hash, "expected_sha256": ARCHIVE_SHA256, "verified": True},
            "room_visuals": {"path": display_path(room_visuals), "bytes": room_visuals.stat().st_size, "sha256": room_visuals_hash, "expected_sha256": ROOM_VISUALS_SHA256, "verified": True},
            "server_live_check": {"attempted": True, "host": "172.16.55.2", "result": "AUTHENTICATION_UNAVAILABLE", "effect": "no live path/hash refresh; frozen local archives remain verified"},
        },
    )
    write_json(
        p01 / "status.json",
        {
            "schema": "puri-gs-work-package-status-v1",
            "work_package": "P01",
            "engineering_status": "COMPLETE_VALID",
            "scientific_status": "COMPLETE_VALID",
            "exit_code": 0,
            "new_training_planned": 0,
            "new_training_completed": 0,
            "historical_checkpoint_rows_audited": 10,
            "paired_models_reevaluated_or_reused": 8,
            "per_image_metric_rows": len(per_image_rows),
            "p02_external_training_identities_planned": 4,
            "p02_external_training_identities_completed": 0,
            "completed": [
                "branch/code/environment/entry-point audit",
                "local dataset and frozen archive identity audit",
                "Android/Patio-High/Room anchor freeze and Garden registration",
                "factor4 historical versus Room factor2 formal protocol separation",
                "Android and Patio-High common COLMAP conversions with image/hash/camera/pose/point/projection readback validation",
                "existing-model summary and complete per-image metric consolidation",
                "four-row P02 external run plan",
            ],
            "unresolved": [
                "live server paths and Room checkpoint hashes could not be refreshed because SSH authentication was unavailable",
                "Room absolute training seconds are absent from local evidence; only the archived RU/B1 ratio 1.025487 is known",
                "external method code/environments/checkpoints/features are intentionally not installed in P01",
                "the full legacy test suite is not locally green because PyYAML is absent and one pre-existing dry-run test requires branch dev; all P01-specific tests pass",
            ],
            "automatic_next_package_started": False,
        },
    )
    (p01 / "NEXT_DECISION.md").write_text(
        """# P01 后续决策（交给方案助手）\n\n1. 是否确认 Android 以 `9e292309` 的 30k B1/RU 配对作为 P02 唯一内部锚点，并把 10k B1 与原 Phase-R Android 行仅保留为历史参考？\n2. 是否确认共同输入口径为 Android 与 Patio-High 两套经回读验证的物理 factor4/loader factor1 PINHOLE COLMAP；作者重跑 SfM 的 Reported 数字只作参考？\n3. 是否按 `run_plan_P02.csv` 开启四个单 seed42 完整训练身份：RobustSplat 与 SLS-mlp（不启用 UBP）各跑 Android、Patio-High？\n\n未得到下一包明确决定前，不自动安装外部方法、不训练、不启动 OAC。\n""",
        encoding="utf-8",
    )
    print(f"P01_PROTOCOLS={len(protocols)}")
    print(f"P01_ASSET_ROWS={len(asset_rows)}")
    print(f"P01_SUMMARY_ROWS={len(summary_rows)}")
    print(f"P01_PER_IMAGE_ROWS={len(per_image_rows)}")
    print(f"P01_P02_RUN_ROWS={len(run_plan_rows)}")
    print("P01_DELIVERABLES=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
