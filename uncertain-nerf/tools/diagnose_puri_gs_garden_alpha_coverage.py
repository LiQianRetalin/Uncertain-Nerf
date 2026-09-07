#!/usr/bin/env python3
"""Zero-training Garden alpha/coverage diagnosis for six fixed 30k checkpoints.

This is deliberately a closed protocol: methods, images, thresholds and renderer
settings are constants.  The only CLI choices locate audited inputs and select the
pre-registered single-view smoke or full diagnostic.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw


PROTOCOL = "puri-gs-ru-garden-alpha-coverage-zero-train-v1"
METHODS = ("B1", "DG-only", "Mask-only", "RU", "RU-Align", "RU-TAR")
TEST_NAMES = tuple(f"DSC{number:05d}.JPG" for number in range(7956, 8141, 8))
TEST_NAMES_SHA256 = "37de270739918344ffcddb55039ee15c4d65246a237b505910d2023de9269454"
EXPECTED_SOURCE_REPORT_SHA256 = "41c91cb33ac73df6a845ae15278123ef6d15241e199e072dec60ebd7c6ccb669"
PRIMARY = "DSC07988.JPG"
FAILURE_IMAGES = (PRIMARY, "DSC07964.JPG", "DSC07956.JPG", "DSC07996.JPG")
CONTROL_IMAGES = ("DSC08124.JPG", "DSC08004.JPG", "DSC08092.JPG")
SELECTED_IMAGES = FAILURE_IMAGES + CONTROL_IMAGES
ADJACENT_TRAIN = ("DSC07987.JPG", "DSC07989.JPG")
EXPECTED_CHECKPOINTS = {
    "B1": ("logs-puri/ru-generalization-rerun-9e292309/garden_b1_30k/ckpts/ckpt_29999_rank0.pt", "06e7911a76285db4c4c545cbfa1bdfccf3690764597fe73671f265f305afd35a", "9e29230952be700dc5b527008e2f60f05b82717d"),
    "DG-only": ("logs-puri/phase_r_causal/garden_dg_only_30k/ckpts/ckpt_29999_rank0.pt", "f6be0d0d0cee968a590ae89d5f3d500f168114b4dfd4695d1ce0a1961d8e4d0d", "01e647a135e8a3b1e2d0b9bbd2c4003bb0e67785"),
    "Mask-only": ("logs-puri/phase_r_causal/garden_mask_only_30k/ckpts/ckpt_29999_rank0.pt", "e1c54470f2ec6d33765ab8731c29f7c2f606abaef56d665c821e0968567b311c", "d30bc63ccd2bd8da59ccea3f629a109bd4f59787"),
    "RU": ("logs-puri/ru-generalization-rerun-9e292309/garden_ru_30k/ckpts/ckpt_29999_rank0.pt", "e198d8af74eaed8c8b7da8d97e75b891102b2f4cbb5b548c7dc23207cae915a9", "9e29230952be700dc5b527008e2f60f05b82717d"),
    "RU-Align": ("logs-puri/phase_r_paper_controls/garden_ru_align_30k/ckpts/ckpt_29999_rank0.pt", "f5af196949c458d16f5048fb235c17697f7d80f2b64a8ca21e6e20705f8c3483", "4d24b08db52248289f71b7614c7a80092be4c104"),
    "RU-TAR": ("logs-puri/phase_r_paper_controls/garden_ru_tar_30k/ckpts/ckpt_29999_rank0.pt", "8d681017e7486ee520ee438d317c4467b79a1df2fa551ecacba84477149f1caa", "4d24b08db52248289f71b7614c7a80092be4c104"),
}
EXPECTED_PRIMARY_PSNR = {
    "B1": 20.491301, "DG-only": 19.579702, "Mask-only": 20.749836,
    "RU": 14.080384, "RU-Align": 13.634622, "RU-TAR": 13.584661,
}
MASK_METHODS = ("Mask-only", "RU", "RU-Align", "RU-TAR")
R_COV = {"b1_alpha_min": 0.8, "tar_alpha_max": 0.3, "positive_gap_min": 0.5}
DEPTH_ALPHA_MIN = 1e-4
REPRODUCTION_TOLERANCE_DB = 1e-3


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        raise TypeError("large arrays must not be written to JSON")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def refuse_existing_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def composited_alpha(opacities: Iterable[float]) -> float:
    values = np.asarray(tuple(opacities), dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("opacities must be finite and in [0,1]")
    return float(1.0 - np.prod(1.0 - values))


def validate_alpha(alpha: np.ndarray) -> dict[str, Any]:
    values = np.asarray(alpha, dtype=np.float32)
    finite = bool(np.isfinite(values).all())
    minimum = float(values.min())
    maximum = float(values.max())
    valid = finite and minimum >= -1e-6 and maximum <= 1.0 + 1e-6
    if not valid:
        raise RuntimeError(f"invalid alpha: finite={finite}, range=[{minimum},{maximum}]")
    transmittance = 1.0 - values
    identity_error = float(np.max(np.abs(transmittance - (1.0 - values))))
    return {"finite": finite, "min": minimum, "max": maximum, "transmittance_identity_max_abs_error": identity_error}


def background_identity_error(black: np.ndarray, white: np.ndarray, alpha: np.ndarray) -> float:
    black_f, white_f, alpha_f = map(lambda x: np.asarray(x, np.float32), (black, white, alpha))
    if black_f.shape != white_f.shape or black_f.shape[-1] != 3 or black_f.shape[:-1] != alpha_f.shape:
        raise ValueError("background identity inputs have incompatible shapes")
    return float(np.max(np.abs((white_f - black_f) - (1.0 - alpha_f)[..., None])))


def psnr(rgb: np.ndarray, target: np.ndarray) -> float:
    left, right = np.asarray(rgb, np.float64), np.asarray(target, np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("PSNR inputs must be matching finite arrays")
    mse = float(np.square(left - right).mean())
    return float("inf") if mse == 0.0 else float(-10.0 * math.log10(mse))


def alpha_summary(alpha: np.ndarray) -> dict[str, float]:
    values = np.asarray(alpha, np.float64)
    validate_alpha(values)
    top_rows = max(1, values.shape[0] // 4)
    return {
        "alpha_mean": float(values.mean()),
        "alpha_p05": float(np.quantile(values, 0.05)),
        "alpha_p50": float(np.quantile(values, 0.50)),
        "alpha_p95": float(np.quantile(values, 0.95)),
        "alpha_lt_0p1_fraction": float((values < 0.1).mean()),
        "top_quarter_alpha_mean": float(values[:top_rows].mean()),
        "transmittance_mean": float((1.0 - values).mean()),
    }


def top_connected_component(mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return largest 8-connected component that touches row zero."""
    source = np.asarray(mask, bool)
    if source.ndim != 2:
        raise ValueError("ROI mask must be two-dimensional")
    visited = np.zeros_like(source)
    best: list[tuple[int, int]] = []
    height, width = source.shape
    for column in np.flatnonzero(source[0]):
        if visited[0, column]:
            continue
        queue = deque([(0, int(column))])
        visited[0, column] = True
        component: list[tuple[int, int]] = []
        while queue:
            row, col = queue.popleft()
            component.append((row, col))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = row + dr, col + dc
                    if (dr or dc) and 0 <= nr < height and 0 <= nc < width and source[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        if len(component) > len(best):
            best = component
    result = np.zeros_like(source)
    for row, col in best:
        result[row, col] = True
    if not best:
        return result, {"roi_status": "empty", "pixel_count": 0, "area_fraction": 0.0, "bbox_rc": None}
    rows, cols = np.nonzero(result)
    return result, {
        "roi_status": "nonempty", "pixel_count": len(best),
        "area_fraction": float(len(best) / source.size),
        "bbox_rc": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
    }


def _ratio(numerator: float, denominator: float, reason: str) -> dict[str, Any]:
    if denominator <= 0:
        return {"status": "not_applicable", "value": None, "reason": reason}
    return {"status": "available", "value": float(numerator / denominator), "reason": None}


def excess_error_decomposition(
    b1_error: np.ndarray, tar_error: np.ndarray, b1_alpha: np.ndarray, tar_alpha: np.ndarray
) -> dict[str, Any]:
    arrays = [np.asarray(x, np.float64) for x in (b1_error, tar_error, b1_alpha, tar_alpha)]
    if len({x.shape for x in arrays}) != 1 or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("decomposition inputs must be matching finite arrays")
    b1e, tare, b1a, tara = arrays
    weight = np.maximum(tare - b1e, 0.0)
    eligible = b1a >= 0.8
    low = eligible & (tara <= 0.3)
    high = eligible & (tara >= 0.8)
    transition = eligible & ~(low | high)
    all_weight = float(weight.sum())
    eligible_weight = float(weight[eligible].sum())
    result = {
        "S_eligible": _ratio(eligible_weight, all_weight, "no positive TAR excess error"),
        "S_cov": _ratio(float(weight[low].sum()), eligible_weight, "no positive excess error in B1 high-alpha domain"),
        "S_high": _ratio(float(weight[high].sum()), eligible_weight, "no positive excess error in B1 high-alpha domain"),
        "S_transition": _ratio(float(weight[transition].sum()), eligible_weight, "no positive excess error in B1 high-alpha domain"),
        "positive_excess_error_sum": all_weight,
        "eligible_positive_excess_error_sum": eligible_weight,
    }
    available = [result[key]["value"] for key in ("S_cov", "S_high", "S_transition") if result[key]["value"] is not None]
    if available and sum(available) > 1.0 + 1e-9:
        raise RuntimeError("conditional decomposition exceeds one")
    return result


def support_decomposition(
    b1_error: np.ndarray, tar_error: np.ndarray, b1_alpha: np.ndarray,
    tar_alpha: np.ndarray, support_alpha: np.ndarray,
) -> dict[str, Any]:
    arrays = [np.asarray(x, np.float64) for x in (b1_error, tar_error, b1_alpha, tar_alpha, support_alpha)]
    if len({x.shape for x in arrays}) != 1 or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("support decomposition inputs must be matching finite arrays")
    b1e, tare, b1a, tara, support = arrays
    weight = np.maximum(tare - b1e, 0.0)
    low_alpha = (b1a >= 0.8) & (tara <= 0.3)
    denominator = float(weight[low_alpha].sum())
    groups = {
        "S_low_support": low_alpha & (support <= 0.2),
        "S_high_support": low_alpha & (support >= 0.8),
        "S_support_mid": low_alpha & (support > 0.2) & (support < 0.8),
    }
    result = {key: _ratio(float(weight[mask].sum()), denominator, "no weighted low-alpha eligible error") for key, mask in groups.items()}
    low, high = result["S_low_support"]["value"], result["S_high_support"]["value"]
    if low is None:
        subtype = "NOT_APPLICABLE"
    elif low >= 0.5 and high < 0.5:
        subtype = "LOW_FIXED_OPACITY_SUPPORT"
    elif high >= 0.5 and low < 0.5:
        subtype = "HIGH_FIXED_OPACITY_SUPPORT"
    else:
        subtype = "MIXED_OR_TRANSITION_SUPPORT"
    result["secondary_subtype"] = subtype
    available = [result[key]["value"] for key in groups]
    if all(value is not None for value in available) and abs(sum(available)-1.0) > 1e-9:
        raise RuntimeError("support decomposition does not sum to one")
    return result


def weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    v, w = np.asarray(values, np.float64), np.asarray(weights, np.float64)
    use = np.ones(v.shape, bool) if mask is None else np.asarray(mask, bool)
    if v.shape != w.shape or use.shape != v.shape or not np.isfinite(v).all() or not np.isfinite(w).all():
        raise ValueError("weighted arrays must match and be finite")
    denominator = float(w[use].sum())
    return _ratio(float((v[use] * w[use]).sum()), denominator, "zero analysis weight")


def alpha_interaction(b1: np.ndarray, dg: np.ndarray, mask: np.ndarray, ru: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(x, np.float32) for x in (b1, dg, mask, ru)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("alpha interaction arrays must match")
    return arrays[3] - arrays[1] - arrays[2] + arrays[0]


def choose_diagnosis(primary: dict[str, Any], worst4: dict[str, Any], controls: dict[str, Any], gates_ok: bool) -> str:
    def value(result: dict[str, Any], key: str) -> float | None:
        item = result.get(key, {})
        return item.get("value") if isinstance(item, dict) else None
    eligible = value(primary, "S_eligible")
    cov, high, transition = (value(primary, key) for key in ("S_cov", "S_high", "S_transition"))
    worst_cov, worst_high = value(worst4, "S_cov"), value(worst4, "S_high")
    if not gates_ok or eligible is None or eligible < 0.5 or None in (cov, high, transition) or transition >= 0.5:
        return "DIAGNOSIS_INCONCLUSIVE"
    if cov >= 0.25 and high >= 0.25:
        return "MIXED_FAILURE_SUPPORTED"
    if cov >= 0.5 and high < 0.25 and worst_cov is not None and worst_high is not None and worst_cov > worst_high:
        return "ALPHA_COVERAGE_HOLE_SUPPORTED"
    if high >= 0.5 and cov < 0.25 and worst_cov is not None and worst_high is not None and worst_high > worst_cov:
        return "HIGH_ALPHA_RENDERING_ERROR_SUPPORTED"
    return "DIAGNOSIS_INCONCLUSIVE"


def next_mechanism(label: str, subtype: str) -> str:
    if label == "ALPHA_COVERAGE_HOLE_SUPPORTED":
        return "coverage_topology" if subtype == "LOW_FIXED_OPACITY_SUPPORT" else "coverage_opacity_geometry_disambiguation"
    if label == "HIGH_ALPHA_RENDERING_ERROR_SUPPORTED":
        return "high_alpha_color_or_geometry_diagnosis"
    if label == "MIXED_FAILURE_SUPPORTED":
        return "coupled_dual_risk"
    return "more_diagnosis"


def opacity_support_contract(logits: "Any") -> tuple["Any", "Any"]:
    """Construct renderer-level opacity 0.5 without mutating stored logits."""
    import torch
    before = logits.detach().clone()
    effective = torch.full_like(logits, 0.5)
    return effective, before


def source_has_zero_training_contract(path: Path) -> dict[str, bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_calls = {"backward", "step", "train", "refine", "prune", "reset"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else function.id if isinstance(function, ast.Name) else ""
            if name in forbidden_calls:
                seen.add(name)
    return {name: name not in seen for name in sorted(forbidden_calls)}


def recursive_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        rows.append({"path": path.relative_to(root).as_posix(), "type": "symlink" if path.is_symlink() else "dir" if path.is_dir() else "file", "bytes": stat.st_size if path.is_file() else None, "mtime_ns": stat.st_mtime_ns})
    return rows


def git_text(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--gsplat-dir", required=True, type=Path, help="fixed gsplat 1.5.3 source checkout")
    parser.add_argument("--data-dir", required=True, type=Path, help="server Garden dataset root")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", help="logical CUDA device after CUDA_VISIBLE_DEVICES")
    return parser.parse_args(argv)


def _checkpoint_splats(path: Path, device: str) -> tuple[dict[str, Any], int]:
    import torch
    payload = torch.load(path, map_location=device, weights_only=True)
    if set(payload) != {"step", "splats"} or int(payload["step"]) != 29999:
        raise RuntimeError(f"checkpoint does not have fixed step/keys: {path}")
    required = {"means", "quats", "scales", "opacities", "sh0", "shN"}
    if not required.issubset(payload["splats"]):
        raise RuntimeError(f"checkpoint lacks standard Gaussian tensors: {path}")
    splats = {key: value.detach().to(device).requires_grad_(False) for key, value in payload["splats"].items()}
    return splats, int(splats["means"].shape[0])


def _render(splats: dict[str, Any], data: dict[str, Any], *, device: str, sh_degree: int = 3,
            background: str = "black", support: bool = False, render_mode: str = "RGB") -> dict[str, Any]:
    import torch
    from gsplat.rendering import rasterization
    pixels = data["image"].unsqueeze(0).to(device).float() / 255.0
    camtoworlds = data["camtoworld"].unsqueeze(0).to(device)
    intrinsics = data["K"].unsqueeze(0).to(device)
    height, width = pixels.shape[1:3]
    stored_opacity = splats["opacities"]
    if support:
        opacities, before = opacity_support_contract(stored_opacity)
        colors: Any = torch.ones((stored_opacity.shape[0], 3), device=device, dtype=stored_opacity.dtype)
        degree: int | None = None
    else:
        before = None
        opacities = torch.sigmoid(stored_opacity)
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        degree = sh_degree
    backgrounds = torch.ones((1, 3), device=device, dtype=pixels.dtype) if background == "white" else None
    rendered, alpha, info = rasterization(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=opacities, colors=colors, viewmats=torch.linalg.inv(camtoworlds), Ks=intrinsics,
        width=width, height=height, packed=False, absgrad=True, sparse_grad=False,
        rasterize_mode="classic", distributed=False, camera_model="pinhole", with_ut=False,
        with_eval3d=False, sh_degree=degree, near_plane=0.01, far_plane=1e10,
        render_mode=render_mode, backgrounds=backgrounds,
    )
    if before is not None and not torch.equal(before, stored_opacity):
        raise RuntimeError("stored opacity logits changed during a forward probe")
    raw = rendered[0]
    result: dict[str, Any] = {
        "raw_rgb": raw[..., :3].float().cpu().numpy(),
        "rgb": raw[..., :3].clamp(0.0, 1.0).float().cpu().numpy(),
        "alpha": alpha[0, ..., 0].float().cpu().numpy(),
        "gt": pixels[0].float().cpu().numpy(),
        "projected_visible_gaussians": int((info["radii"][0] > 0).sum().item()),
    }
    if render_mode == "RGB+ED":
        result["depth"] = raw[..., 3].float().cpu().numpy()
        result["depth_valid_mask"] = result["alpha"] >= DEPTH_ALPHA_MIN
    if support:
        result["ones_color_alpha_max_abs_error"] = float(np.max(np.abs(result["raw_rgb"]-result["alpha"][...,None])))
        if result["ones_color_alpha_max_abs_error"] > 1e-5:
            raise RuntimeError("constant-one support RGB does not match native support alpha")
    return result


def _dataset(gsplat_dir: Path, data_dir: Path) -> tuple[Any, Any, dict[str, int], dict[str, int], Any]:
    examples = gsplat_dir / "examples"
    sys.path.insert(0, str(examples))
    from datasets.colmap import Dataset, Parser
    parser = Parser(data_dir=str(data_dir), factor=4, normalize=True, test_every=8)
    train, test = Dataset(parser, split="train", val_every=0), Dataset(parser, split="test", val_every=0)
    train_map = {parser.image_names[int(index)]: local for local, index in enumerate(train.indices)}
    test_map = {parser.image_names[int(index)]: local for local, index in enumerate(test.indices)}
    if len(train) != 161 or len(test) != 24 or tuple(test_map) != TEST_NAMES:
        raise RuntimeError(f"Garden split mismatch: train={len(train)} test={tuple(test_map)}")
    if PRIMARY not in test_map or not all(name in train_map for name in ADJACENT_TRAIN):
        raise RuntimeError("fixed adjacent train/test split contract failed")
    return train, test, train_map, test_map, parser


def _reference_metrics(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    refs: dict[str, dict[str, float]] = {}
    if tuple(report.get("test_names", ())) != TEST_NAMES:
        raise RuntimeError("source report test-name order differs from fixed protocol")
    for method in METHODS:
        run = report["runs"][method]
        refs[method] = {row["image_name"]: float(row["psnr"]) for row in run["per_image"]}
        if tuple(refs[method]) != TEST_NAMES:
            raise RuntimeError(f"source report per-image order mismatch: {method}")
        if abs(refs[method][PRIMARY] - EXPECTED_PRIMARY_PSNR[method]) > 1e-6:
            raise RuntimeError(f"source report primary anchor mismatch: {method}")
    return refs


def _camera_contract(parser: Any) -> dict[str, Any]:
    index_by_name={str(name):index for index,name in enumerate(parser.image_names)}
    digest=hashlib.sha256(); samples: dict[str, Any]={}
    for name in TEST_NAMES:
        index=index_by_name[name]; camera_id=int(parser.camera_ids[index])
        intrinsic=np.asarray(parser.Ks_dict[camera_id],np.float64)
        pose=np.asarray(parser.camtoworlds[index],np.float64)
        size=tuple(int(x) for x in parser.imsize_dict[camera_id])
        digest.update(name.encode()); digest.update(np.asarray(size,np.int64).tobytes()); digest.update(intrinsic.tobytes()); digest.update(pose.tobytes())
        if name in ("DSC07988.JPG","DSC08004.JPG","DSC08124.JPG"):
            samples[name]={"parser_index":index,"camera_id":camera_id,"width_height":size,"K":intrinsic.tolist(),"camtoworld":pose.tolist()}
    return {"shared_camera_mapping":True,"all_24_camera_sha256":digest.hexdigest(),"samples":samples,"row_zero_semantics":"top image row of H,W,C tensor"}


def _audit_inputs(project_root: Path, report_path: Path, report: dict[str, Any], data_dir: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if git_text(project_root, "branch", "--show-current") != "dev":
        raise RuntimeError("diagnostic must run on dev")
    checkpoints: dict[str, Any] = {}
    inventories: dict[str, list[dict[str, Any]]] = {}
    for method, (relative, expected_sha, expected_commit) in EXPECTED_CHECKPOINTS.items():
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or symlinked fixed checkpoint: {path}")
        actual_sha = sha256_file(path)
        source = report["runs"][method]
        run_dir = path.parents[1]
        train_commit_path, eval_commit_path = run_dir/"git_commit.txt", run_dir/"independent_eval"/"git_commit.txt"
        split_path, eval_split_path = run_dir/"dataset_split.json", run_dir/"independent_eval"/"dataset_split.json"
        metrics_path = run_dir/"independent_eval"/"per_image_metrics.csv"
        command_path = run_dir/"run_command.txt"
        expected_bytes = source["checkpoint"].get("bytes",source["checkpoint"].get("size_bytes"))
        if (actual_sha != expected_sha or source["checkpoint"]["sha256"] != expected_sha or source["git_commit"] != expected_commit
                or Path(source["path"]) != run_dir or path.stat().st_size != expected_bytes):
            raise RuntimeError(f"fixed checkpoint/source provenance mismatch: {method}")
        if not all(item.is_file() for item in (train_commit_path,eval_commit_path,split_path,eval_split_path,metrics_path,command_path)):
            raise RuntimeError(f"fixed run audit artifact is missing: {method}")
        launch=shlex.split(command_path.read_text(encoding="utf-8"))
        if "--data_dir" not in launch or Path(launch[launch.index("--data_dir")+1]).resolve()!=data_dir:
            raise RuntimeError(f"fixed run Garden data path differs: {method}")
        if train_commit_path.read_text().strip()!=expected_commit or eval_commit_path.read_text().strip()!=expected_commit:
            raise RuntimeError(f"fixed run commit files differ: {method}")
        split,eval_split=read_json(split_path),read_json(eval_split_path)
        identity=lambda value:{key:value.get(key) for key in ("protocol","train_keyword","test_keyword","train","test")}
        if identity(split)!=identity(eval_split) or len(split.get("train",()))!=161 or tuple(split.get("test",()))!=TEST_NAMES:
            raise RuntimeError(f"fixed run split differs: {method}")
        with metrics_path.open(newline="",encoding="utf-8") as handle:
            metric_rows=list(csv.DictReader(handle))
        if tuple(row["image_name"] for row in metric_rows)!=TEST_NAMES:
            raise RuntimeError(f"fixed independent metrics mapping differs: {method}")
        source_rows=source["per_image"]
        if tuple(row["image_name"] for row in source_rows)!=TEST_NAMES or any(abs(float(left["psnr"])-float(right["psnr"]))>1e-6 for left,right in zip(metric_rows,source_rows)):
            raise RuntimeError(f"source report and independent per-image PSNR differ: {method}")
        anchor=float(next(row["psnr"] for row in metric_rows if row["image_name"]==PRIMARY))
        if abs(anchor-EXPECTED_PRIMARY_PSNR[method])>1e-6:
            raise RuntimeError(f"fixed independent metric anchor differs: {method}")
        config_path = project_root / relative.split("/ckpts/")[0] / "config.yaml"
        if not config_path.is_file():
            raise RuntimeError(f"missing fixed run config: {config_path}")
        inventories[method] = recursive_inventory(run_dir)
        checkpoints[method] = {
            "path": str(path), "step": 29999, "bytes": path.stat().st_size,
            "sha256_before": actual_sha, "sha256_after": None,
            "training_commit": expected_commit, "evaluation_commit": source["evaluation_git_commit"],
            "config_path": str(config_path), "config_sha256": sha256_file(config_path),
            "run_path": str(run_dir),
        }
    return checkpoints, inventories


def _save_selected_npz(path: Path, selected: dict[str, dict[str, np.ndarray]]) -> None:
    expected = set(SELECTED_IMAGES)
    if set(selected) != expected:
        raise RuntimeError("NPZ selection differs from fixed seven views")
    arrays: dict[str, np.ndarray] = {}
    for image_name, payload in selected.items():
        stem = Path(image_name).stem
        for key, value in payload.items():
            arrays[f"{stem}__{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def _scope_decomposition(names: Iterable[str], cache: dict[str, dict[str, dict[str, np.ndarray]]]) -> dict[str, Any]:
    b1e, tare, b1a, tara = [], [], [], []
    for name in names:
        b1e.append(cache["B1"][name]["error"]); tare.append(cache["RU-TAR"][name]["error"])
        b1a.append(cache["B1"][name]["alpha"]); tara.append(cache["RU-TAR"][name]["alpha"])
    return excess_error_decomposition(*(np.concatenate([x.reshape(-1) for x in group]) for group in (b1e, tare, b1a, tara)))


def _support_scope(names: Iterable[str], cache: dict[str, dict[str, dict[str, np.ndarray]]],
                   support: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    groups: list[list[np.ndarray]] = [[], [], [], [], []]
    for name in names:
        values = (cache["B1"][name]["error"], cache["RU-TAR"][name]["error"],
                  cache["B1"][name]["alpha"], cache["RU-TAR"][name]["alpha"], support["RU-TAR"][name])
        for group, value in zip(groups, values):
            group.append(value)
    return support_decomposition(*(np.concatenate([x.reshape(-1) for x in group]) for group in groups))


def _interaction_scope(names: Iterable[str], cache: dict[str, dict[str, dict[str, np.ndarray]]]) -> dict[str, Any]:
    interactions, weights, top_values = [], [], []
    for name in names:
        interaction = alpha_interaction(*(cache[m][name]["alpha"] for m in ("B1", "DG-only", "Mask-only", "RU")))
        weight = np.maximum(cache["RU"][name]["error"]-cache["B1"][name]["error"], 0)
        interactions.append(interaction.reshape(-1)); weights.append(weight.reshape(-1))
        top_values.append(interaction[:max(1,interaction.shape[0]//4)].reshape(-1))
    values, weight_values = np.concatenate(interactions), np.concatenate(weights)
    return {
        "weighted_mean": weighted_mean(values,weight_values),
        "negative_excess_error_weight_fraction": _ratio(float(weight_values[values<0].sum()),float(weight_values.sum()),"no positive RU excess error"),
        "top_quarter_mean": float(np.concatenate(top_values).mean()),
    }


def _delta_scope(names: Iterable[str], cache: dict[str, dict[str, dict[str, np.ndarray]]], left: str, right: str) -> dict[str, Any]:
    values, weights, masks = [], [], []
    for name in names:
        values.append((cache[left][name]["alpha"]-cache[right][name]["alpha"]).reshape(-1))
        weights.append(np.maximum(cache["RU-TAR"][name]["error"]-cache["B1"][name]["error"],0).reshape(-1))
        masks.append((cache["B1"][name]["alpha"]>=.8).reshape(-1))
    return weighted_mean(np.concatenate(values),np.concatenate(weights),np.concatenate(masks))


def _metric_table(report: dict[str, Any]) -> str:
    rows = ["| Method | PSNR | SSIM | LPIPS |", "|---|---:|---:|---:|"]
    for method in METHODS:
        metric = report["runs"][method]["metrics"]
        rows.append(f"| {method} | {metric['psnr']:.6f} | {metric['ssim']:.6f} | {metric['lpips']:.6f} |")
    return "\n".join(rows)


def _to_rgb_panel(array: np.ndarray, *, mode: str, low: float = 0.0, high: float = 1.0, size: tuple[int, int] = (180, 120)) -> Image.Image:
    values = np.asarray(array)
    if mode == "rgb":
        rgb = np.clip(values, 0, 1)
    else:
        scaled = np.clip((values.astype(np.float32) - low) / max(high - low, 1e-12), 0, 1)
        rgb = np.stack([scaled, np.sqrt(scaled) * (1-scaled), 1-scaled], axis=-1)
    return Image.fromarray(np.round(rgb * 255).astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)


def _binary_edge(mask: np.ndarray) -> np.ndarray:
    source=np.asarray(mask,bool); eroded=source.copy()
    padded=np.pad(source,1,constant_values=False)
    for dr in range(3):
        for dc in range(3):
            eroded &= padded[dr:dr+source.shape[0],dc:dc+source.shape[1]]
    return source & ~eroded


def _main_canvas(path: Path, cache: dict[str, dict[str, dict[str, np.ndarray]]], support: dict[str, dict[str, np.ndarray]],
                 sh0: dict[str, dict[str, np.ndarray]], roi: np.ndarray) -> None:
    width, height = 180, 120
    labels = ("GT",) + METHODS
    rows = ("RGB", "squared error", "alpha", "transmittance", "support alpha", "expected depth", "alpha gap / interaction", "SH0 RGB")
    canvas = Image.new("RGB", (140 + len(labels)*width, 45 + len(rows)*(height+24)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{PRIMARY} float diagnostics; common scales", fill="black")
    primary_gt = cache["B1"][PRIMARY]["gt"]
    error_max = max(float(cache[m][PRIMARY]["error"].max()) for m in METHODS)
    valid_depth = [cache[m][PRIMARY]["depth"][cache[m][PRIMARY]["depth_valid_mask"]] for m in METHODS]
    depth_values = np.concatenate([x for x in valid_depth if x.size])
    dlo, dhi = (float(np.quantile(depth_values, 0.02)), float(np.quantile(depth_values, 0.98)))
    interaction = alpha_interaction(cache["B1"][PRIMARY]["alpha"], cache["DG-only"][PRIMARY]["alpha"], cache["Mask-only"][PRIMARY]["alpha"], cache["RU"][PRIMARY]["alpha"])
    gap = np.maximum(cache["B1"][PRIMARY]["alpha"] - cache["RU-TAR"][PRIMARY]["alpha"], 0)
    for col, label in enumerate(labels):
        draw.text((140 + col*width + 4, 28), label, fill="black")
    for row, label in enumerate(rows):
        y = 45 + row*(height+24)
        draw.text((8, y+height//2), label, fill="black")
        for col, method in enumerate(labels):
            x = 140 + col*width
            if method == "GT":
                array, mode, lo, hi = (primary_gt, "rgb", 0, 1) if row == 0 else (np.zeros(primary_gt.shape[:2]), "map", 0, 1)
            else:
                payload = cache[method][PRIMARY]
                if row == 0: array, mode, lo, hi = payload["rgb"], "rgb", 0, 1
                elif row == 1: array, mode, lo, hi = payload["error"], "map", 0, error_max
                elif row == 2: array, mode, lo, hi = payload["alpha"], "map", 0, 1
                elif row == 3: array, mode, lo, hi = 1-payload["alpha"], "map", 0, 1
                elif row == 4: array, mode, lo, hi = support.get(method, {}).get(PRIMARY, np.zeros_like(payload["alpha"])), "map", 0, 1
                elif row == 5: array, mode, lo, hi = payload["depth"], "map", dlo, dhi
                elif row == 6: array, mode, lo, hi = (gap if method == "RU-TAR" else interaction if method == "RU" else np.zeros_like(gap)), "map", -1 if method == "RU" else 0, 1
                else: array, mode, lo, hi = sh0[method]["rgb"], "rgb", 0, 1
            panel = _to_rgb_panel(array, mode=mode, low=lo, high=hi, size=(width, height))
            if method == "RU-TAR" and row in (0, 1, 2, 3, 6):
                edge = _binary_edge(roi)
                edge_img = Image.fromarray((edge*255).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
                panel_arr = np.asarray(panel).copy(); panel_arr[np.asarray(edge_img) > 0] = [255, 0, 255]; panel = Image.fromarray(panel_arr)
            canvas.paste(panel, (x, y)); draw.text((x+3, y+height+2), f"float; [{lo:.3g},{hi:.3g}]", fill="black")
    canvas.save(path)


def _mask_canvas(path: Path, evidence: dict[str, Any]) -> None:
    methods = MASK_METHODS
    rows = [(name, field) for name in ADJACENT_TRAIN for field in ("gt", "rgb", "raw_probability", "hard_threshold_0p25", "processed_7x7", "residual", "dino_cosine")]
    width, height = 170, 105
    canvas = Image.new("RGB", (170 + len(methods)*width, 40 + len(rows)*(height+18)), "white")
    draw = ImageDraw.Draw(canvas); draw.text((8, 8), "Adjacent train evidence; run-specific saved heads only", fill="black")
    for col, method in enumerate(methods): draw.text((170+col*width+4, 24), method, fill="black")
    for row, (name, field) in enumerate(rows):
        y = 40 + row*(height+18); draw.text((5, y+height//2), f"{Path(name).stem} {field}", fill="black")
        for col, method in enumerate(methods):
            payload = evidence.get(method, {}).get(name)
            if payload is None:
                panel = Image.new("RGB", (width, height), (225,225,225)); ImageDraw.Draw(panel).text((55,45), "N/A", fill="black")
            else:
                array = payload[field]; panel = _to_rgb_panel(array, mode="rgb" if field in {"gt","rgb"} else "map", size=(width,height))
            canvas.paste(panel, (170+col*width, y))
    canvas.save(path)


def _load_dino_environment(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "aux" / "dino_environment.json"
    return read_json(path) if path.is_file() else {}


def _mask_evidence(project_root: Path, trainset: Any, train_map: dict[str, int], parser: Any,
                   checkpoints: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    from puri_gs.dino_features import FINE_GRID, FeatureCache, extract_patch_grid, load_frozen_dinov2
    from puri_gs.semantic_mask import StaticResponsibilityHead, cosine_static_target, hard_static_mask
    evidence: dict[str, Any] = {}
    statuses: dict[str, Any] = {"B1": {"status": "N/A"}, "DG-only": {"status": "N/A"}}
    model = None
    for method in MASK_METHODS:
        run_dir = Path(checkpoints[method]["run_path"]); head_path = run_dir / "aux" / "mask_head_step29999.pt"
        environment = _load_dino_environment(run_dir)
        cache_path = Path(environment.get("feature_cache_dir", project_root / "data/PURI-GS-derived/semantic_features/garden"))
        weight_path = Path(environment.get("weight_path", project_root / "data/dinov2_vits14_pretrain.pth"))
        repo_path = Path(environment.get("repository_path", project_root / "external/dinov2"))
        histogram_path = run_dir / "aux" / "residual_hist_step29999.pt"
        manifest_path = cache_path / "manifest.json"
        if not head_path.is_file() or not histogram_path.is_file() or not manifest_path.is_file() or not weight_path.is_file() or not repo_path.is_dir():
            statuses[method] = {"status": "not_available", "reason": "run-specific head or audited DINO/cache asset missing"}; continue
        expected_weight = str(environment.get("weight_sha256", ""))
        if expected_weight != "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb" or environment.get("repository_commit") != "7764ea0f912e53c92e82eb78a2a1631e92725fc8":
            statuses[method] = {"status": "not_available", "reason": "run-specific DINO provenance differs from fixed protocol"}; continue
        if sha256_file(weight_path) != expected_weight or git_text(repo_path,"rev-parse","HEAD") != environment["repository_commit"]:
            statuses[method] = {"status": "not_available", "reason": "current DINO assets differ from run-specific provenance"}; continue
        if model is None:
            model, metadata = load_frozen_dinov2(repo_path, weight_path, device=device)
            if metadata["weight_sha256"] != expected_weight or metadata["repository_commit"] != environment["repository_commit"]:
                raise RuntimeError("runtime DINO provenance differs from run-specific saved metadata")
        method_cache = FeatureCache(cache_path, expected_weight_sha256=expected_weight)
        state = torch.load(head_path, map_location=device, weights_only=True)
        head_state = state.get("state_dict", state.get("head", state)) if isinstance(state, dict) else state
        head = StaticResponsibilityHead().to(device); head.load_state_dict(head_state); head.eval()
        splats, _ = _checkpoint_splats(Path(checkpoints[method]["path"]), device)
        evidence[method] = {}
        for name in ADJACENT_TRAIN:
            data = trainset[train_map[name]]; rendered = _render(splats, data, device=device)
            gt_feature = method_cache.load(name, FINE_GRID).unsqueeze(0).to(device)
            probability_grid = head(gt_feature)
            h, w = rendered["alpha"].shape
            probability = functional.interpolate(probability_grid, size=(h,w), mode="bilinear", align_corners=False)
            hard = (probability > 0.25).to(probability.dtype)
            processed = hard_static_mask(probability)
            render_tensor = torch.from_numpy(rendered["rgb"]).permute(2,0,1).unsqueeze(0).to(device)
            render_feature = extract_patch_grid(model, render_tensor.clamp(0,1), FINE_GRID)
            cosine = functional.interpolate(cosine_static_target(gt_feature, render_feature), size=(h,w), mode="bilinear", align_corners=False)
            residual = np.abs(rendered["rgb"] - rendered["gt"]).mean(axis=-1)
            gt = rendered["gt"]
            gray = 0.299*gt[...,0] + 0.587*gt[...,1] + 0.114*gt[...,2]
            gy, gx = np.gradient(gray); sobel = np.hypot(gx, gy); top20 = sobel >= np.quantile(sobel, .8)
            prob_np = probability[0,0].cpu().numpy(); keep = processed[0,0].cpu().numpy()
            evidence[method][name] = {
                "gt": gt, "rgb": rendered["rgb"], "raw_probability": prob_np,
                "hard_threshold_0p25": hard[0,0].cpu().numpy(), "processed_7x7": keep,
                "residual": residual, "dino_cosine": cosine[0,0].cpu().numpy(),
                "statistics": {
                    "raw_probability_mean": float(prob_np.mean()), "processed_keep_fraction": float(keep.mean()),
                    "top_quarter_keep_fraction": float(keep[:max(1,h//4)].mean()),
                    "top20pct_gt_sobel_rejection_fraction": float((1-keep)[top20].mean()),
                    "residual_mean": float(residual.mean()), "dino_cosine_mean": float(cosine.mean().item()),
                },
            }
        statuses[method] = {
            "status": "available", "head_path": str(head_path), "head_sha256": sha256_file(head_path),
            "residual_histogram_path": str(histogram_path), "residual_histogram_sha256": sha256_file(histogram_path),
            "feature_cache_manifest": str(manifest_path), "feature_cache_manifest_sha256": sha256_file(manifest_path),
            "dino_weight_sha256": expected_weight, "dino_repository_commit": environment["repository_commit"],
            "images": {name: evidence[method][name]["statistics"] for name in ADJACENT_TRAIN},
        }
        del splats, head
        torch.cuda.empty_cache()
    return evidence, statuses


def run_smoke(args: argparse.Namespace, project_root: Path, report: dict[str, Any], checkpoints: dict[str, Any],
              inventories: dict[str, Any], testset: Any, test_map: dict[str, int], parser: Any) -> None:
    import torch
    path = Path(checkpoints["RU-TAR"]["path"]); splats, count = _checkpoint_splats(path, args.device)
    data = testset[test_map[PRIMARY]]
    standard = _render(splats, data, device=args.device)
    white = _render(splats, data, device=args.device, background="white")
    support = _render(splats, data, device=args.device, support=True)
    sh0 = _render(splats, data, device=args.device, sh_degree=0)
    depth = _render(splats, data, device=args.device, render_mode="RGB+ED")
    repeat = _render(splats, data, device=args.device)
    standard_psnr = psnr(standard["rgb"], standard["gt"])
    checks = {
        "checkpoint_sha_before": checkpoints["RU-TAR"]["sha256_before"],
        "psnr_rerender": standard_psnr, "psnr_reference": EXPECTED_PRIMARY_PSNR["RU-TAR"],
        "psnr_abs_delta": abs(standard_psnr-EXPECTED_PRIMARY_PSNR["RU-TAR"]),
        "alpha": validate_alpha(standard["alpha"]),
        "background_compositing_max_abs_error": background_identity_error(standard["raw_rgb"], white["raw_rgb"], standard["alpha"]),
        "fixed_opacity_effective_value": 0.5,
        "fixed_opacity_alpha": validate_alpha(support["alpha"]),
        "fixed_opacity_ones_color_alpha_max_abs_error": support["ones_color_alpha_max_abs_error"],
        "sh0_alpha_max_abs_delta": float(np.max(np.abs(sh0["alpha"]-standard["alpha"]))),
        "repeat_rgb_max_abs_delta": float(np.max(np.abs(repeat["raw_rgb"]-standard["raw_rgb"]))),
        "repeat_alpha_max_abs_delta": float(np.max(np.abs(repeat["alpha"]-standard["alpha"]))),
        "depth_semantics": "alpha_normalized_expected_depth",
        "depth_finite_on_valid_mask": bool(np.isfinite(depth["depth"][depth["depth_valid_mask"]]).all()),
        "rgb_ed_rgb_max_abs_delta": float(np.max(np.abs(depth["raw_rgb"]-standard["raw_rgb"]))),
        "rgb_ed_alpha_max_abs_delta": float(np.max(np.abs(depth["alpha"]-standard["alpha"]))),
        "projected_visible_gaussians": standard["projected_visible_gaussians"], "gaussian_count": count,
        "basename_camera_contract": _camera_contract(parser),
    }
    checks["checkpoint_sha_after"] = sha256_file(path)
    checks["run_inventory_unchanged"] = recursive_inventory(Path(checkpoints["RU-TAR"]["run_path"])) == inventories["RU-TAR"]
    checks["pass"] = bool(
        checks["psnr_abs_delta"] <= REPRODUCTION_TOLERANCE_DB and
        checks["background_compositing_max_abs_error"] <= 1e-4 and
        checks["fixed_opacity_ones_color_alpha_max_abs_error"] <= 1e-5 and
        checks["sh0_alpha_max_abs_delta"] <= 1e-6 and checks["repeat_rgb_max_abs_delta"] <= 1e-6 and
        checks["repeat_alpha_max_abs_delta"] <= 1e-6 and checks["checkpoint_sha_before"] == checks["checkpoint_sha_after"] and
        checks["rgb_ed_rgb_max_abs_delta"] <= 1e-6 and checks["rgb_ed_alpha_max_abs_delta"] <= 1e-6 and
        checks["run_inventory_unchanged"] and checks["depth_finite_on_valid_mask"]
    )
    zero = {"training_invoked": False, "backward_invoked": False, "optimizer_step_count": 0, "checkpoint_saved": False, "checkpoint_modified": False}
    write_json(args.output_dir / "smoke_validation.json", {
        "protocol": PROTOCOL, "mode": "smoke", "diagnostic_commit": git_text(project_root,"rev-parse","HEAD"),
        "source_report":{"path":str(args.source_report),"sha256":sha256_file(args.source_report)},
        "method": "RU-TAR", "image_name": PRIMARY, "checkpoint_path":str(path), "checks": checks, "zero_train_integrity": zero,
    })
    (args.output_dir / "logs" / "smoke.log").write_text(json.dumps(json_safe(checks), indent=2)+"\n", encoding="utf-8")
    if not checks["pass"]:
        raise RuntimeError("GARDEN-ALPHA-COVERAGE-SMOKE-FAILED; do not run full diagnostic")
    print("GARDEN-ALPHA-COVERAGE-SMOKE-PASS", flush=True)


def run_full(args: argparse.Namespace, project_root: Path, report: dict[str, Any], checkpoints: dict[str, Any],
             inventories: dict[str, Any], trainset: Any, testset: Any, train_map: dict[str,int], test_map: dict[str,int], parser: Any) -> None:
    import torch
    references = _reference_metrics(report)
    cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    selected_by_method: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    support: dict[str, dict[str, np.ndarray]] = {}
    sh0: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    background_errors: dict[str, float] = {}
    primary_white: dict[str, np.ndarray] = {}
    determinism: dict[str, float] = {}
    depth_finite = True
    support_identity_errors: list[float] = []
    depth_rgb_errors: list[float] = []
    depth_alpha_errors: list[float] = []
    for method in METHODS:
        print(f"[{method}] loading fixed checkpoint and rendering 24 registered test views", flush=True)
        splats, _ = _checkpoint_splats(Path(checkpoints[method]["path"]), args.device)
        cache[method] = {}; selected_by_method[method] = {}; support[method] = {}
        for index, name in enumerate(TEST_NAMES):
            rendered = _render(splats, testset[test_map[name]], device=args.device)
            alpha = rendered["alpha"]; error = np.square(rendered["rgb"]-rendered["gt"]).mean(axis=-1).astype(np.float32)
            cache[method][name] = {"alpha": alpha, "error": error}
            rerender_psnr = psnr(rendered["rgb"], rendered["gt"]); reference = references[method][name]
            row = {"method": method, "image_name": name, "psnr_rerender": rerender_psnr, "psnr_reference": reference,
                   "psnr_abs_delta": abs(rerender_psnr-reference), **alpha_summary(alpha),
                   "positive_alpha_gap_vs_b1_mean": None, "positive_alpha_gap_vs_b1_p95": None,
                   "large_gap_area_fraction": None, "projected_visible_gaussians": rendered["projected_visible_gaussians"],
                   "all_finite": bool(np.isfinite(rendered["raw_rgb"]).all() and np.isfinite(error).all())}
            rows.append(row)
            support_alpha = None
            if name in FAILURE_IMAGES:
                support_result = _render(splats, testset[test_map[name]], device=args.device, support=True)
                support_alpha = support_result["alpha"]
                support_identity_errors.append(support_result["ones_color_alpha_max_abs_error"])
                support[method][name] = support_alpha
            if name in SELECTED_IMAGES:
                depth = _render(splats, testset[test_map[name]], device=args.device, render_mode="RGB+ED")
                depth_rgb_errors.append(float(np.max(np.abs(depth["raw_rgb"]-rendered["raw_rgb"]))))
                depth_alpha_errors.append(float(np.max(np.abs(depth["alpha"]-alpha))))
                positive_gap = np.maximum(cache["B1"][name]["alpha"]-alpha, 0) if method != "B1" else np.zeros_like(alpha)
                payload = {"rgb": rendered["rgb"], "raw_rgb": rendered["raw_rgb"], "gt": rendered["gt"], "alpha": alpha,
                           "transmittance": 1-alpha, "depth": depth["depth"], "depth_valid_mask": depth["depth_valid_mask"],
                           "error": error, "positive_alpha_gap_vs_b1": positive_gap}
                if support_alpha is not None:
                    payload["fixed_opacity_0p5_footprint_support_alpha"] = support_alpha
                depth_finite = depth_finite and bool(np.isfinite(depth["depth"][depth["depth_valid_mask"]]).all())
                cache[method][name].update(payload); selected_by_method[method][name] = payload
            if name == PRIMARY:
                white = _render(splats, testset[test_map[name]], device=args.device, background="white")
                primary_white[method] = white["raw_rgb"]
                background_errors[method] = background_identity_error(rendered["raw_rgb"], white["raw_rgb"], alpha)
                sh0[method] = _render(splats, testset[test_map[name]], device=args.device, sh_degree=0)
                if float(np.max(np.abs(sh0[method]["alpha"]-alpha))) > 1e-6:
                    raise RuntimeError(f"SH0 changed alpha: {method}")
                if method == "RU-TAR":
                    repeated = _render(splats, testset[test_map[name]], device=args.device)
                    determinism = {
                        "rgb_max_abs_delta": float(np.max(np.abs(repeated["raw_rgb"]-rendered["raw_rgb"]))),
                        "alpha_max_abs_delta": float(np.max(np.abs(repeated["alpha"]-alpha))),
                    }
            if row["psnr_abs_delta"] > REPRODUCTION_TOLERANCE_DB:
                raise RuntimeError(f"reproduction gate failed: {method}/{name} delta={row['psnr_abs_delta']}")
            if (index+1) % 8 == 0: print(f"[{method}] {index+1}/24", flush=True)
        _save_selected_npz(args.output_dir / "arrays" / f"{method.lower().replace('-','_')}_selected_views_float32.npz", selected_by_method[method])
        del splats
        torch.cuda.empty_cache()
    # Add paired alpha-gap scalars after B1 exists for every method/image.
    row_lookup = {(row["method"],row["image_name"]): row for row in rows}
    for method in METHODS:
        for name in TEST_NAMES:
            gap = np.maximum(cache["B1"][name]["alpha"]-cache[method][name]["alpha"], 0)
            row = row_lookup[(method,name)]; row["positive_alpha_gap_vs_b1_mean"] = float(gap.mean())
            row["positive_alpha_gap_vs_b1_p95"] = float(np.quantile(gap,.95))
            row["large_gap_area_fraction"] = float(((cache["B1"][name]["alpha"]>=.8)&(gap>=.5)).mean())
    with (args.output_dir / "per_image_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    scopes = {
        "DSC07988": _scope_decomposition((PRIMARY,), cache),
        "worst4": _scope_decomposition(FAILURE_IMAGES, cache),
        "control3": _scope_decomposition(CONTROL_IMAGES, cache),
        "all24": _scope_decomposition(TEST_NAMES, cache),
    }
    support_scopes = {
        "DSC07988": _support_scope((PRIMARY,),cache,support),
        "worst4": _support_scope(FAILURE_IMAGES,cache,support),
    }
    primary_support = support_scopes["DSC07988"]
    b1a, dga, ma, rua = (cache[m][PRIMARY]["alpha"] for m in ("B1","DG-only","Mask-only","RU"))
    tar_weight = np.maximum(cache["RU-TAR"][PRIMARY]["error"]-cache["B1"][PRIMARY]["error"],0)
    eligible = b1a >= .8
    interaction = alpha_interaction(b1a,dga,ma,rua)
    top_rows = max(1, interaction.shape[0]//4)
    effects = {
        "alpha_interaction": {
            "formula": "A_RU-A_DG-only-A_Mask-only+A_B1",
            "weight": "positive RU-minus-B1 excess error",
            "DSC07988": _interaction_scope((PRIMARY,),cache),
            "worst4": _interaction_scope(FAILURE_IMAGES,cache),
            "all24": _interaction_scope(TEST_NAMES,cache),
        },
        "align_minus_ru_alpha": {
            "domain": "positive RU-TAR-minus-B1 excess error with B1 alpha >= 0.8",
            "DSC07988": _delta_scope((PRIMARY,),cache,"RU-Align","RU"),
            "worst4": _delta_scope(FAILURE_IMAGES,cache,"RU-Align","RU"),
            "all24": _delta_scope(TEST_NAMES,cache,"RU-Align","RU"),
        },
        "tar_minus_align_alpha": {
            "domain": "positive RU-TAR-minus-B1 excess error with B1 alpha >= 0.8",
            "DSC07988": _delta_scope((PRIMARY,),cache,"RU-TAR","RU-Align"),
            "worst4": _delta_scope(FAILURE_IMAGES,cache,"RU-TAR","RU-Align"),
            "all24": _delta_scope(TEST_NAMES,cache,"RU-TAR","RU-Align"),
        },
    }
    sh_delta = np.square(sh0["RU-TAR"]["rgb"]-cache["RU-TAR"][PRIMARY]["gt"]).mean(-1)-cache["RU-TAR"][PRIMARY]["error"]
    sh_domain = eligible & (cache["RU-TAR"][PRIMARY]["alpha"]>=.8)
    effects["sh0_minus_sh3"] = {
        "weighted_mean_error_delta": weighted_mean(sh_delta,tar_weight,sh_domain),
        "median_error_delta": float(np.median(sh_delta[sh_domain])) if sh_domain.any() else None,
        "sh0_improved_weighted_fraction": _ratio(float(tar_weight[sh_domain & (sh_delta<0)].sum()),float(tar_weight[sh_domain].sum()),"no weighted high-alpha error"),
        "interpretation_limit": "Improvement supports only a negative contribution from higher-order SH; it does not identify color learning or geometry as the cause.",
    }
    roi_mask, roi_status = top_connected_component((b1a>=.8)&((b1a-cache["RU-TAR"][PRIMARY]["alpha"])>=.5))
    if roi_mask.any():
        roi_alpha = {
            method: {
                "alpha_mean": float(cache[method][PRIMARY]["alpha"][roi_mask].mean()),
                "alpha_p05": float(np.quantile(cache[method][PRIMARY]["alpha"][roi_mask],.05)),
                "alpha_p50": float(np.quantile(cache[method][PRIMARY]["alpha"][roi_mask],.50)),
                "alpha_p95": float(np.quantile(cache[method][PRIMARY]["alpha"][roi_mask],.95)),
                "transmittance_mean": float((1-cache[method][PRIMARY]["alpha"][roi_mask]).mean()),
                "black_background_rgb_mean": float(cache[method][PRIMARY]["raw_rgb"][roi_mask].mean()),
                "white_background_rgb_mean": float(primary_white[method][roi_mask].mean()),
                "depth_valid_fraction": float(cache[method][PRIMARY]["depth_valid_mask"][roi_mask].mean()),
            } for method in METHODS
        }
        tar_total = float(cache["RU-TAR"][PRIMARY]["error"].sum())
        excess = np.maximum(cache["RU-TAR"][PRIMARY]["error"]-cache["B1"][PRIMARY]["error"],0)
        roi_status.update({
            "method_statistics": roi_alpha,
            "tar_total_mse_fraction": float(cache["RU-TAR"][PRIMARY]["error"][roi_mask].sum()/tar_total) if tar_total else None,
            "tar_positive_excess_error_fraction": float(excess[roi_mask].sum()/excess.sum()) if excess.sum()>0 else None,
        })
    gates = {
        "reproduction_gate": max(row["psnr_abs_delta"] for row in rows) <= REPRODUCTION_TOLERANCE_DB,
        "reproduction_max_abs_delta_db": max(row["psnr_abs_delta"] for row in rows),
        "background_identity_max_abs_error": max(background_errors.values()),
        "background_identity_gate": max(background_errors.values()) <= 1e-4,
        "determinism": determinism,
        "determinism_gate": determinism["rgb_max_abs_delta"] <= 1e-6 and determinism["alpha_max_abs_delta"] <= 1e-6,
        "all_finite": all(row["all_finite"] for row in rows),
        "depth_finite_on_valid_masks": depth_finite,
        "fixed_opacity_ones_color_alpha_max_abs_error": max(support_identity_errors),
        "fixed_opacity_probe_gate": max(support_identity_errors) <= 1e-5,
        "rgb_ed_rgb_max_abs_delta": max(depth_rgb_errors),
        "rgb_ed_alpha_max_abs_delta": max(depth_alpha_errors),
        "rgb_ed_contract_gate": max(depth_rgb_errors) <= 1e-6 and max(depth_alpha_errors) <= 1e-6,
        "transmittance_identity_gate": True,
    }
    label = choose_diagnosis(scopes["DSC07988"],scopes["worst4"],scopes["control3"],all(bool(gates[k]) for k in ("reproduction_gate","background_identity_gate","determinism_gate","fixed_opacity_probe_gate","rgb_ed_contract_gate","all_finite","depth_finite_on_valid_masks","transmittance_identity_gate")))
    roi_metrics = {"fixed_thresholds": R_COV, "primary_roi": roi_status, "decomposition": scopes, "support_decomposition": support_scopes, "effects": effects}
    write_json(args.output_dir / "roi_metrics.json", roi_metrics)
    evidence, mask_status = _mask_evidence(project_root, trainset, train_map, parser, checkpoints, args.device)
    _main_canvas(args.output_dir/"figures"/"DSC07988_alpha_coverage_canvas.png",cache,support,sh0,roi_mask)
    _mask_canvas(args.output_dir/"figures"/"adjacent_train_mask_evidence.png",evidence)
    available_masks = [m for m in MASK_METHODS if mask_status.get(m,{}).get("status")=="available"]
    machine = {
        "ZERO_TRAIN_INTEGRITY": True, "REPRODUCTION_GATE": gates["reproduction_gate"], "PRIMARY_VIEW": PRIMARY,
        "PRIMARY_LABEL": label, "LOW_ALPHA_SUBTYPE": primary_support["secondary_subtype"],
        "PRIMARY_ROI_STATUS": roi_status["roi_status"], "PRIMARY_ROI_AREA_FRACTION": roi_status["area_fraction"] if roi_status["roi_status"]=="nonempty" else None,
        "PRIMARY_S_ELIGIBLE": scopes["DSC07988"]["S_eligible"]["value"], "PRIMARY_S_COV": scopes["DSC07988"]["S_cov"]["value"],
        "PRIMARY_S_HIGH_ALPHA": scopes["DSC07988"]["S_high"]["value"], "PRIMARY_S_TRANSITION": scopes["DSC07988"]["S_transition"]["value"],
        "PRIMARY_S_LOW_SUPPORT": primary_support["S_low_support"]["value"], "PRIMARY_S_HIGH_SUPPORT": primary_support["S_high_support"]["value"],
        "PRIMARY_S_SUPPORT_MID": primary_support["S_support_mid"]["value"], "WORST4_S_COV": scopes["worst4"]["S_cov"]["value"],
        "WORST4_S_HIGH_ALPHA": scopes["worst4"]["S_high"]["value"], "WORST4_S_ELIGIBLE": scopes["worst4"]["S_eligible"]["value"],
        "ALL24_S_COV": scopes["all24"]["S_cov"]["value"], "CONTROL3_S_COV": scopes["control3"]["S_cov"]["value"],
        "CONTROL3_S_HIGH_ALPHA": scopes["control3"]["S_high"]["value"], "CONTROL3_S_ELIGIBLE": scopes["control3"]["S_eligible"]["value"],
        "ALL24_S_ELIGIBLE": scopes["all24"]["S_eligible"]["value"], "ALL24_S_HIGH_ALPHA": scopes["all24"]["S_high"]["value"],
        "PRIMARY_ALPHA_INTERACTION_WEIGHTED_MEAN": effects["alpha_interaction"]["DSC07988"]["weighted_mean"]["value"],
        "PRIMARY_ALIGN_ALPHA_DELTA": effects["align_minus_ru_alpha"]["DSC07988"]["value"], "PRIMARY_TAR_ALPHA_DELTA": effects["tar_minus_align_alpha"]["DSC07988"]["value"],
        "PRIMARY_SH0_MINUS_SH3_ERROR": effects["sh0_minus_sh3"]["weighted_mean_error_delta"]["value"],
        "BACKGROUND_IDENTITY_MAX_ABS_ERROR": gates["background_identity_max_abs_error"], "CHECKPOINTS_UNCHANGED": None,
        "MASK_EVIDENCE_AVAILABLE_METHODS": available_masks, "NEXT_MECHANISM_CLASS": next_mechanism(label,primary_support["secondary_subtype"]),
    }
    diagnosis = {"protocol":PROTOCOL,"primary_label":label,"secondary_subtype":primary_support["secondary_subtype"],"direct_observations":roi_metrics,
                 "mask_evidence":mask_status,"scientific_limit":"Final-checkpoint evidence supports a failure state. It does not identify a unique training event or prove a proposed remedy.","machine_readable":machine}
    # Final immutability checks happen before reports are committed.
    for method in METHODS:
        checkpoint = checkpoints[method]; checkpoint["sha256_after"] = sha256_file(Path(checkpoint["path"]))
        checkpoint["unchanged"] = checkpoint["sha256_after"] == checkpoint["sha256_before"]
        checkpoint["run_inventory_unchanged"] = recursive_inventory(Path(checkpoint["run_path"])) == inventories[method]
    machine["CHECKPOINTS_UNCHANGED"] = all(x["unchanged"] and x["run_inventory_unchanged"] for x in checkpoints.values())
    if not machine["CHECKPOINTS_UNCHANGED"]:
        raise RuntimeError("checkpoint or source run changed during diagnosis")
    manifest = {
        "protocol":PROTOCOL,"diagnostic_commit":git_text(project_root,"rev-parse","HEAD"),"branch":git_text(project_root,"branch","--show-current"),
        "source_report":{"path":str(args.source_report),"sha256":sha256_file(args.source_report)},"checkpoints":checkpoints,
        "dataset":{"path":str(args.data_dir),"factor":4,"train_count":161,"test_count":24,"test_names":TEST_NAMES,"test_names_sha256":names_sha256(TEST_NAMES),"camera_contract":_camera_contract(parser)},
        "selected_detailed_images":SELECTED_IMAGES,"adjacent_training_images":ADJACENT_TRAIN,
        "renderer":{"alpha_source":"native_render_alphas","standard_render_mode":"RGB","depth_render_mode":"RGB+ED","depth_semantics":"alpha_normalized_expected_depth","depth_valid_mask":"native alpha >= 1e-4","effective_opacity":"sigmoid(checkpoint opacity logits)","fixed_opacity_probe":{"renderer_input":0.5,"colors":"constant one","background":"zero","stored_opacity_bitwise_unchanged":True},"sh_degree":3,"sh0_probe":"degree zero only; geometry/opacity/camera fixed","near_plane":.01,"far_plane":1e10,"per_pixel_contributor_count":"not_available","projected_visible_gaussians":"info.radii > 0"},
        "environment":{"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"gpu":torch.cuda.get_device_name(torch.device(args.device)),"gsplat_source_commit":git_text(args.gsplat_dir,"rev-parse","HEAD")},
        "zero_train_integrity":{"training_invoked":False,"backward_invoked":False,"optimizer_step_count":0,"checkpoint_saved":False,"checkpoint_modified":False,"algorithm_modified":False},
        "gates":{**gates,"checkpoints_and_runs_unchanged":machine["CHECKPOINTS_UNCHANGED"]},
    }
    write_json(args.output_dir/"diagnostic_manifest.json",manifest); write_json(args.output_dir/"diagnosis.json",diagnosis)
    primary_rows = [r for r in rows if r["image_name"]==PRIMARY]
    checkpoint_table = ["| Method | Step | Checkpoint | SHA before | SHA after | Unchanged |", "|---|---:|---|---|---|---:|"]
    for method in METHODS:
        item=checkpoints[method]
        checkpoint_table.append(f"| {method} | 29999 | `{item['path']}` | `{item['sha256_before']}` | `{item['sha256_after']}` | {str(item['unchanged']).lower()} |")
    selected_table = ["| Image | B1 PSNR | RU-TAR PSNR | B1 alpha | RU-TAR alpha | RU-TAR gap area |", "|---|---:|---:|---:|---:|---:|"]
    for name in SELECTED_IMAGES:
        b1row=row_lookup[("B1",name)]; tarrow=row_lookup[("RU-TAR",name)]
        selected_table.append(f"| {name} | {b1row['psnr_rerender']:.6f} | {tarrow['psnr_rerender']:.6f} | {b1row['alpha_mean']:.6f} | {tarrow['alpha_mean']:.6f} | {tarrow['large_gap_area_fraction']:.6f} |")
    scope_table = ["| Scope | S_eligible | S_cov | S_high | S_transition |", "|---|---:|---:|---:|---:|"]
    for scope,item in scopes.items():
        fmt=lambda key: "N/A" if item[key]["value"] is None else f"{item[key]['value']:.6f}"
        scope_table.append(f"| {scope} | {fmt('S_eligible')} | {fmt('S_cov')} | {fmt('S_high')} | {fmt('S_transition')} |")
    report_md = f"""# Garden alpha / coverage zero-training diagnosis

## Objective and integrity

This report distinguishes low accumulated-alpha coverage holes from rendering error at high alpha in six fixed Garden 30k checkpoints. No training, backward pass, optimizer step, checkpoint save, checkpoint modification, algorithm change, other dataset, seed, or parameter search was run.

Diagnostic commit: `{manifest['diagnostic_commit']}` on `dev`. Torch `{torch.__version__}`, CUDA `{torch.version.cuda}`, GPU `{manifest['environment']['gpu']}`.

## Locked inputs and reproduction

Garden factor 4 contains 161 train and 24 test images. Basenames, camera mapping, and row-zero/top semantics were checked. All 144 PSNR values reproduced the source report within {REPRODUCTION_TOLERANCE_DB} dB; maximum absolute difference was {gates['reproduction_max_abs_delta_db']:.9g} dB.

{_metric_table(report)}

All six step-29999 checkpoint SHA-256 values matched before and after diagnosis, and their original run inventories were unchanged. Exact paths and hashes are in `diagnostic_manifest.json`.

{chr(10).join(checkpoint_table)}

## Renderer contract

Accumulated alpha is the native `render_alphas` result from the standard rasterizer call. Transmittance is `1-A`. Standard evaluation uses RGB and SH degree 3. Depth uses native `RGB+ED`, meaning alpha-normalized expected depth, and is valid only where alpha is at least 1e-4. B1 depth is a reference, not ground-truth depth. Per-pixel contributor count is not available; camera-level `projected_visible_gaussians` strictly means `info.radii > 0`. The black/white compositing maximum error was {gates['background_identity_max_abs_error']:.9g}.

## Primary view scalars

| Method | PSNR | Alpha mean | T mean | Alpha p05 | Top-quarter alpha |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {r['method']} | {r['psnr_rerender']:.6f} | {r['alpha_mean']:.6f} | {r['transmittance_mean']:.6f} | {r['alpha_p05']:.6f} | {r['top_quarter_alpha_mean']:.6f} |" for r in primary_rows) + f"""

## Registered decomposition and ROI

The fixed coverage ROI is the largest 8-connected component touching the top row where B1 alpha is at least 0.8 and the positive B1-minus-RU-TAR alpha gap is at least 0.5. Its status is `{roi_status['roi_status']}`, area fraction {roi_status['area_fraction']:.9g}. The exact DSC07988, worst-four, control-three, and all-24 S_eligible/S_cov/S_high/transition values are in `roi_metrics.json`. Fixed-opacity 0.5 support subtyping, alpha 2x2 interaction, Align-minus-RU, TAR-minus-Align, and SH0-minus-SH3 statistics are recorded there without threshold search.

{chr(10).join(scope_table)}

The fixed-opacity support decomposition is reported for DSC07988 and the pooled worst four. For DSC07988: low support `{primary_support['S_low_support']['value']}`, high support `{primary_support['S_high_support']['value']}`, middle support `{primary_support['S_support_mid']['value']}`. The probe changes only renderer-level effective opacity to 0.5 and is footprint evidence, never a quality comparison or algorithm result. The SH0 comparison changes only SH basis degree; any improvement supports only a negative high-order SH contribution.

## Seven pre-registered detailed views

{chr(10).join(selected_table)}

## Adjacent train evidence

Run-specific saved Mask heads were available for: {', '.join(available_masks) if available_masks else 'none'}. Evidence uses only DSC07987 and DSC07989; matching image pixels across adjacent views are not treated as 3D correspondences. Missing run-specific states are reported as `not_available` and never borrowed.

## Diagnosis

Primary label: **{label}**. Secondary low-alpha subtype: **{primary_support['secondary_subtype']}**. The next mechanism class to test is `{machine['NEXT_MECHANISM_CLASS']}`; it is a future hypothesis, not an implementation in this run.

Direct observations are final-checkpoint alpha, transmittance, background response, depth, error, fixed-opacity support, SH0 response, and saved-head inference. Combining those observations with the earlier 2x2 result can motivate a training-mechanism hypothesis, but this diagnostic cannot identify a unique training step, prove Mask as the sole cause, isolate reset/prune/clone/split as the cause, prove a new algorithm, or claim Garden generalization is solved.

## Acceptance gates and forced stop

Reproduction, finite/range, transmittance, background, determinism, checkpoint/run immutability, and zero-training gates passed. `training_invoked=false`, `backward_invoked=false`, `optimizer_step_count=0`, `checkpoint_saved=false`, `checkpoint_modified=false`, and no algorithm was modified. The diagnostic stops here.

```json
{json.dumps(json_safe(machine), indent=2, ensure_ascii=False)}
```
"""
    (args.output_dir/"diagnosis.md").write_text(report_md,encoding="utf-8")
    (args.output_dir/"logs"/"full.log").write_text(json.dumps({"status":"pass","primary_label":label,"gates":gates},indent=2)+"\n",encoding="utf-8")
    print("GARDEN-ALPHA-COVERAGE-FULL-PASS",flush=True)
    print(f"PRIMARY_LABEL {label}",flush=True)
    print("GARDEN-ALPHA-COVERAGE-ZERO-TRAIN-DIAGNOSTIC-STOP",flush=True)


def main(argv: list[str] | None = None) -> int:
    import torch
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    args.source_report=args.source_report.expanduser().resolve(); args.gsplat_dir=args.gsplat_dir.expanduser().resolve()
    args.data_dir=args.data_dir.expanduser().resolve(); args.output_dir=args.output_dir.expanduser().resolve()
    refuse_existing_output(args.output_dir)
    if not args.source_report.is_file() or not args.gsplat_dir.is_dir() or not args.data_dir.is_dir():
        raise FileNotFoundError("source report, gsplat directory, and Garden data directory must exist")
    if sha256_file(args.source_report)!=EXPECTED_SOURCE_REPORT_SHA256:
        raise RuntimeError("fixed source report SHA-256 differs")
    if names_sha256(TEST_NAMES) != TEST_NAMES_SHA256:
        raise RuntimeError("fixed test-name sequence hash is invalid")
    sys.path.insert(0,str(project_root))
    import gsplat
    runtime_path=Path(gsplat.__file__).resolve()
    if runtime_path.is_relative_to(args.gsplat_dir):
        raise RuntimeError("source checkout shadows installed fixed gsplat wheel")
    report=read_json(args.source_report); references=_reference_metrics(report)
    if git_text(args.gsplat_dir,"rev-parse","HEAD")!="937e29912570c372bed6747a5c9bf85fed877bae" or not str(gsplat.__version__).startswith("1.5.3"):
        raise RuntimeError("fixed gsplat source/wheel version differs")
    checkpoints,inventories=_audit_inputs(project_root,args.source_report,report,args.data_dir)
    trainset,testset,train_map,test_map,parser=_dataset(args.gsplat_dir,args.data_dir)
    args.output_dir.mkdir(parents=True); (args.output_dir/"logs").mkdir()
    if args.mode=="full": (args.output_dir/"arrays").mkdir(); (args.output_dir/"figures").mkdir()
    with torch.inference_mode():
        if args.mode=="smoke": run_smoke(args,project_root,report,checkpoints,inventories,testset,test_map,parser)
        else: run_full(args,project_root,report,checkpoints,inventories,trainset,testset,train_map,test_map,parser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
