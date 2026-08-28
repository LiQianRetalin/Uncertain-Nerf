"""Read-only local dataset inventory for the PURI-GS experiment sequence."""

from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path(r"E:\7-DataSet\nerf数据集")
DATASET_NAMES = (
    "mipnerf360",
    "nerf_example_data",
    "nerf_llff_data",
    "nerf_on-the-go",
    "nerf_raw",
    "nerf_ref",
    "nerf_ref_real",
    "nerf_robustnerf",
    "nerf_synthetic",
    "SeathruNeRF_dataset",
    "_SeathruNeRF_download_temporary_files",
)
MIPNERF360_SCENES = (
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG header")
    return struct.unpack(">II", header[16:24])


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError("invalid JPEG header")
        while True:
            prefix = handle.read(1)
            if not prefix:
                break
            if prefix != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if marker and marker[0] in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                payload = handle.read(5)
                if len(payload) != 5:
                    break
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            handle.seek(max(length - 2, 0), 1)
    raise ValueError("JPEG dimensions not found")


def _image_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
    if signature == b"\x89PNG\r\n\x1a\n":
        return _png_dimensions(path)
    return _jpeg_dimensions(path)


def _sample_dimensions(paths: list[Path], limit: int = 3) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    if not paths:
        return {"samples": samples, "errors": errors}
    indices = sorted({0, len(paths) // 2, len(paths) - 1})[:limit]
    for index in indices:
        path = paths[index]
        try:
            width, height = _image_dimensions(path)
            samples.append({"file": path.name, "width": width, "height": height})
        except (OSError, ValueError) as error:
            errors.append(f"{path.name}: {error}")
    return {"samples": samples, "errors": errors}


def _colmap_count(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
        if len(prefix) != 8:
            return None
        return int(struct.unpack("<Q", prefix)[0])
    except OSError:
        return None


def _keyword_counts(paths: Iterable[Path]) -> dict[str, int]:
    counts = {"clean": 0, "clutter": 0, "extra": 0, "other": 0}
    for path in paths:
        name = path.stem.casefold()
        matched = False
        for keyword in ("clean", "clutter", "extra"):
            if keyword in name:
                counts[keyword] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1
    return counts


def _audit_colmap_scene(
    scene: Path,
    *,
    data_factor: int = 4,
    require_dynamic_split: bool = False,
) -> dict[str, Any]:
    image_dir = scene / f"images_{data_factor}"
    images = _image_files(image_dir)
    originals = _image_files(scene / "images")
    sparse = scene / "sparse" / "0"
    colmap = {
        name: _colmap_count(sparse / name)
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    missing: list[str] = []
    warnings: list[str] = []
    if not image_dir.is_dir():
        missing.append(f"images_{data_factor}/")
    if not images:
        missing.append(f"images_{data_factor} image files")
    for name, count in colmap.items():
        if count is None:
            missing.append(f"sparse/0/{name}")
    if colmap["images.bin"] is not None and len(images) != colmap["images.bin"]:
        missing.append(
            f"images_{data_factor} count ({len(images)}) != registered cameras "
            f"({colmap['images.bin']})"
        )
    if originals and len(originals) != len(images):
        warnings.append(
            f"original image count ({len(originals)}) differs from images_{data_factor} "
            f"count ({len(images)}); unregistered originals are not loader inputs"
        )

    keyword_counts = _keyword_counts(images)
    if require_dynamic_split:
        if keyword_counts["clutter"] == 0:
            missing.append("clutter training images")
        if keyword_counts["extra"] == 0:
            missing.append("extra clean test images")

    return {
        "scene": scene.name,
        "path": str(scene.resolve()),
        "status": "READY" if not missing else "PARTIAL",
        "reason": (
            "current gsplat COLMAP loader requirements are satisfied"
            if not missing
            else "current gsplat COLMAP loader requirements are incomplete"
        ),
        "missing_items": missing,
        "warnings": warnings,
        "counts": {
            "images": len(originals),
            f"images_{data_factor}": len(images),
            "colmap_cameras": colmap["cameras.bin"],
            "colmap_images": colmap["images.bin"],
            "colmap_points3D": colmap["points3D.bin"],
        },
        "split_keyword_counts": keyword_counts,
        "image_samples": _sample_dimensions(images),
    }


def _dataset_base(directory: Path, nested_name: str) -> Path:
    nested = directory / nested_name
    return nested if nested.is_dir() else directory


def _audit_mipnerf360(directory: Path) -> dict[str, Any]:
    base = _dataset_base(directory, "360_v2")
    scenes: list[dict[str, Any]] = []
    for name in MIPNERF360_SCENES:
        scene = base / name
        if scene.is_dir():
            scenes.append(_audit_colmap_scene(scene, data_factor=4))
        else:
            scenes.append(
                {
                    "scene": name,
                    "path": str(scene.resolve()),
                    "status": "MISSING",
                    "reason": "expected Mip-NeRF 360 scene directory is absent",
                    "missing_items": ["scene directory"],
                }
            )
    missing = [item["scene"] for item in scenes if item["status"] != "READY"]
    return {
        "name": "mipnerf360",
        "path": str(directory.resolve()),
        "status": "READY" if not missing else ("PARTIAL" if directory.exists() else "MISSING"),
        "reason": (
            "all seven official scenes satisfy the current loader"
            if not missing
            else "download/extraction is incomplete; no clean conclusion is allowed"
        ),
        "missing_items": missing,
        "scenes": scenes,
    }


def _audit_robustnerf(directory: Path) -> dict[str, Any]:
    base = _dataset_base(directory, "robustnerf")
    scenes = [
        _audit_colmap_scene(scene, data_factor=4, require_dynamic_split=True)
        for scene in sorted(base.iterdir())
        if scene.is_dir()
    ] if base.is_dir() else []
    ready = [scene for scene in scenes if scene["status"] == "READY"]
    partial = [scene["scene"] for scene in scenes if scene["status"] != "READY"]
    status = "MISSING" if not directory.exists() else (
        "PARTIAL" if not scenes or partial else "READY"
    )
    recommended = next(
        (scene["scene"] for scene in ready if scene["scene"] == "android"),
        ready[0]["scene"] if ready else None,
    )
    return {
        "name": "nerf_robustnerf",
        "path": str(directory.resolve()),
        "status": status,
        "reason": (
            "COLMAP scenes are present; use filename keywords, not every-eighth splits"
            if ready
            else "no scene currently satisfies the dynamic COLMAP protocol"
        ),
        "missing_items": partial,
        "split_protocol": {
            "train_keyword": "clutter",
            "test_keyword": "extra",
            "clean_keyword": "clean",
        },
        "recommended_first_dynamic_scene": recommended,
        "scenes": scenes,
    }


def _audit_transforms_dataset(name: str, directory: Path, nested_name: str) -> dict[str, Any]:
    base = _dataset_base(directory, nested_name)
    scenes: list[dict[str, Any]] = []
    if base.is_dir():
        for scene in sorted(base.iterdir()):
            if not scene.is_dir():
                continue
            transforms = sorted(scene.glob("transforms*.json"))
            images = _image_files(scene / "images")
            scenes.append(
                {
                    "scene": scene.name,
                    "path": str(scene.resolve()),
                    "status": "UNSUPPORTED_FORMAT" if transforms and images else "PARTIAL",
                    "reason": (
                        "poses are stored in transforms JSON; current gsplat entry expects COLMAP"
                        if transforms and images
                        else "images or transforms metadata are incomplete"
                    ),
                    "missing_items": ["COLMAP sparse/0 model"],
                    "counts": {"images": len(images), "transforms_json": len(transforms)},
                    "image_samples": _sample_dimensions(images),
                }
            )
    status = "MISSING" if not directory.exists() else (
        "UNSUPPORTED_FORMAT" if scenes and all(s["status"] == "UNSUPPORTED_FORMAT" for s in scenes) else "PARTIAL"
    )
    return {
        "name": name,
        "path": str(directory.resolve()),
        "status": status,
        "reason": "a minimal transforms-to-COLMAP-compatible adapter is required" if scenes else "dataset is empty",
        "missing_items": ["current-loader-compatible COLMAP model"],
        "scenes": scenes,
    }


def _find_colmap_scenes(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    scenes = {sparse.parent.parent for sparse in directory.rglob("sparse/0") if sparse.is_dir()}
    return sorted(scenes)


def _audit_generic(name: str, directory: Path) -> dict[str, Any]:
    colmap_scenes = _find_colmap_scenes(directory)
    scenes = [_audit_colmap_scene(scene, data_factor=4) for scene in colmap_scenes]
    if scenes:
        status = "READY" if all(scene["status"] == "READY" for scene in scenes) else "PARTIAL"
        reason = "one or more COLMAP scenes were detected"
    elif directory.exists():
        status = "UNSUPPORTED_FORMAT"
        reason = "no current-loader-compatible COLMAP scene was detected"
    else:
        status = "MISSING"
        reason = "dataset directory is absent"
    return {
        "name": name,
        "path": str(directory.resolve()),
        "status": status,
        "reason": reason,
        "missing_items": [] if scenes else ["COLMAP scene with images_4 and sparse/0"],
        "scenes": scenes,
    }


def scan_dataset_root(root: str | Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    """Scan the fixed root without writing, moving, hashing, or decoding full datasets."""

    root = Path(root).expanduser().resolve()
    datasets: list[dict[str, Any]] = []
    for name in DATASET_NAMES:
        directory = root / name
        if name == "mipnerf360":
            result = _audit_mipnerf360(directory)
        elif name == "nerf_robustnerf":
            result = _audit_robustnerf(directory)
        elif name == "nerf_on-the-go":
            result = _audit_transforms_dataset(name, directory, "on-the-go")
        elif name == "nerf_synthetic":
            result = _audit_transforms_dataset(name, directory, "nerf_synthetic")
        else:
            result = _audit_generic(name, directory)
        datasets.append(result)

    robust = next(item for item in datasets if item["name"] == "nerf_robustnerf")
    example = next(item for item in datasets if item["name"] == "nerf_example_data")
    smoke_candidates = [
        scene["path"]
        for scene in example.get("scenes", [])
        if scene["status"] == "READY"
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "read_only": True,
        "allowed_operations": [
            "directory listing",
            "file counting",
            "small image header sampling",
            "COLMAP header checks",
            "README and split metadata checks",
        ],
        "datasets": datasets,
        "recommendations": {
            "smoke_scene": smoke_candidates[0] if smoke_candidates else None,
            "first_dynamic_scene": robust.get("recommended_first_dynamic_scene"),
            "clean_scenes_after_download": ["garden", "room"],
        },
    }


def render_manifest_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# PURI-GS 本地数据清单",
        "",
        f"- 扫描根目录：`{manifest['root']}`",
        f"- 生成时间（UTC）：`{manifest['generated_at_utc']}`",
        "- 扫描方式：只读；未移动、修改、下载、解压或计算大文件哈希。",
        "",
        "## 数据集状态",
        "",
        "| 数据集 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    for dataset in manifest["datasets"]:
        lines.append(
            f"| `{dataset['name']}` | **{dataset['status']}** | {dataset['reason']} |"
        )

    mip = next(item for item in manifest["datasets"] if item["name"] == "mipnerf360")
    lines.extend(
        [
            "",
            "## Mip-NeRF 360",
            "",
            f"- status: **{mip['status']}**",
            f"- missing_items: `{json.dumps(mip['missing_items'], ensure_ascii=False)}`",
            f"- reason: {mip['reason']}",
            "- 在状态变为 READY 前，不进行 B0/B1 clean 正式比较、稀疏视角实验或最终 clean 结论。",
            "",
            "## RobustNeRF 场景",
            "",
            "| 场景 | 状态 | images | images_4 | 注册相机 | clutter(train) | extra(test) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    robust = next(item for item in manifest["datasets"] if item["name"] == "nerf_robustnerf")
    for scene in robust.get("scenes", []):
        counts = scene.get("counts", {})
        split = scene.get("split_keyword_counts", {})
        lines.append(
            f"| `{scene['scene']}` | {scene['status']} | {counts.get('images', 0)} | "
            f"{counts.get('images_4', 0)} | {counts.get('colmap_images')} | "
            f"{split.get('clutter', 0)} | {split.get('extra', 0)} |"
        )

    recommendations = manifest["recommendations"]
    lines.extend(
        [
            "",
            "## 建议",
            "",
            f"- 代码 smoke 场景：`{recommendations['smoke_scene']}`",
            f"- 第一个动态场景：`{recommendations['first_dynamic_scene']}`（格式完整且协议最简单）。",
            "- RobustNeRF 固定划分：文件名含 `clutter` 的图像训练，含 `extra` 的干净图像测试；不采用每 8 张抽一张。",
            "- clean 候选：Mip-NeRF 360 下载完整后使用 `garden` 与 `room`；当前不生效。",
            "- NeRF On-the-go 当前为 transforms JSON，需后续单独批准的最小格式适配；本阶段不转换。",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(
    manifest: dict[str, Any], json_path: str | Path, markdown_path: str | Path
) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_manifest_markdown(manifest), encoding="utf-8")
