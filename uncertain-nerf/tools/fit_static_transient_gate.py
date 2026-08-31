#!/usr/bin/env python3
"""Audit, smoke-test, and run the frozen PURI-GS-ST Phase 4A gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import platform
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.static_transient_gate import (
    aggregate_summary,
    confusion_counts,
    decide_gate,
    ensure_output_absent,
    load_gate_config,
    metrics_from_counts,
    psnr,
    sha256_file,
    tree_sha256,
    write_json,
)
from puri_gs.transient_layer import (
    FrozenStaticGaussians,
    TransientGaussianLayer,
    compose_premultiplied,
    projection_error_percentile,
    run_cpu_math_contract,
    static_transient_loss,
)


EXPECTED_GSPLAT_COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"


@dataclass
class FrameInput:
    index: int
    image_name: str
    kind: str
    target_rgb: torch.Tensor
    K: torch.Tensor
    camtoworld: torch.Tensor
    mask_path: Path
    parser_local_index: int
    parser_global_index: int
    camera_id: int


@dataclass
class InputAudit:
    contract: dict[str, Any]
    frames: list[FrameInput]
    commit: str
    branch: str
    checkpoint_sha256: str
    derived_tree_sha256: str


def _command_output(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _git(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    return _command_output(["git", *args], cwd)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_parser_classes(gsplat_dir: Path):
    examples_dir = gsplat_dir / "examples"
    if not (examples_dir / "datasets" / "colmap.py").is_file():
        raise RuntimeError(f"pinned gsplat COLMAP parser is missing: {examples_dir}")
    sys.path.insert(0, str(examples_dir))
    try:
        from datasets.colmap import Dataset, Parser
    except ImportError as error:
        raise RuntimeError("cannot import the pinned gsplat Parser/Dataset") from error
    return Parser, Dataset


def _load_rgb_u8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    if not set(np.unique(values).tolist()).issubset({0, 255}):
        raise RuntimeError(f"GT mask is not binary: {path}")
    return values > 0


def _audit_rasterizer(gsplat_dir: Path) -> dict[str, Any]:
    if _git("rev-parse", "HEAD", cwd=gsplat_dir) != EXPECTED_GSPLAT_COMMIT:
        raise RuntimeError("gsplat checkout is not the pinned v1.5.3 commit")
    import gsplat
    from gsplat.rendering import rasterization

    version = getattr(gsplat, "__version__", None)
    if not str(version).startswith("1.5.3"):
        raise RuntimeError(f"runtime gsplat is not 1.5.3: {version}")
    parameters = list(inspect.signature(rasterization).parameters)
    expected = [
        "means", "quats", "scales", "opacities", "colors", "viewmats", "Ks", "width", "height"
    ]
    if parameters[: len(expected)] != expected:
        raise RuntimeError("runtime rasterization API does not match gsplat v1.5.3")
    rendering_source = (gsplat_dir / "gsplat" / "rendering.py").read_text(encoding="utf-8")
    torch_source = (gsplat_dir / "gsplat" / "cuda" / "_torch_impl.py").read_text(encoding="utf-8")
    if "quaternions of the Gaussians (wxyz convension)" not in rendering_source:
        raise RuntimeError("cannot verify gsplat quaternion order")
    premultiplied_evidence = "render_colors = render_colors + backgrounds[..., None, None, :] * ("
    if premultiplied_evidence not in torch_source or "1.0 - render_alphas" not in torch_source:
        raise RuntimeError("cannot verify rasterizer alpha/background semantics")
    return {
        "runtime_version": str(version),
        "runtime_module_path": str(Path(gsplat.__file__).resolve()),
        "source_checkout": str(gsplat_dir),
        "source_commit": EXPECTED_GSPLAT_COMMIT,
        "rasterization_signature": str(inspect.signature(rasterization)),
        "return_shapes": {
            "render_colors": "[C,H,W,D]",
            "render_alphas": "[C,H,W,1]",
            "metadata": "dict",
        },
        "premultiplied_rgb_semantics": (
            "With black backgrounds, render_colors is sum_i(weight_i * color_i); "
            "the background term is background * (1-render_alphas), so RGB is already premultiplied."
        ),
        "quaternion_order": "wxyz",
        "view_matrix_contract": "rasterization receives inverse(camtoworld)",
    }


def _checkpoint_contract(checkpoint_path: Path, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    expected = config["input_contract"]
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != expected["expected_checkpoint_sha256"]:
        raise RuntimeError("Room B1 checkpoint SHA differs from the Phase 3 fixed asset")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or sorted(checkpoint) != ["splats", "step"]:
        raise RuntimeError("B1 checkpoint must contain only step and splats")
    if int(checkpoint["step"]) != expected["expected_checkpoint_step"]:
        raise RuntimeError("B1 checkpoint step is not 9999")
    splats = checkpoint["splats"]
    required = set(FrozenStaticGaussians.REQUIRED_KEYS)
    if not isinstance(splats, dict) or set(splats) != required:
        raise RuntimeError("B1 splat keys differ from the frozen checkpoint contract")
    tensor_inventory: dict[str, Any] = {}
    for name in sorted(splats):
        value = splats[name]
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"B1 tensor is absent or non-finite: {name}")
        tensor_inventory[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    del checkpoint
    return checkpoint_sha, {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha,
        "step": expected["expected_checkpoint_step"],
        "keys": ["splats", "step"],
        "splat_tensors": tensor_inventory,
        "read_only": True,
    }


def _audit_manifest_and_parser(
    data_dir: Path,
    gsplat_dir: Path,
    derived_dir: Path,
    config: dict[str, Any],
) -> tuple[list[FrameInput], dict[str, Any], str]:
    expected = config["input_contract"]
    manifest_path = derived_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    derived_sha, file_count = tree_sha256(derived_dir)
    if manifest_sha != expected["expected_derived_manifest_sha256"]:
        raise RuntimeError("parser-exact v2 manifest SHA differs from Phase 3")
    if derived_sha != expected["expected_derived_tree_sha256"] or file_count != 33:
        raise RuntimeError("parser-exact v2 tree differs from the Phase 3 fixed asset")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("scene") != "room"
        or manifest.get("data_factor") != 4
        or manifest.get("test_every") != 8
        or not isinstance(entries, list)
        or len(entries) != 16
        or [entry.get("kind") for entry in entries] != ["clean"] * 8 + ["transient"] * 8
    ):
        raise RuntimeError("parser-exact v2 manifest structure is invalid")

    Parser, Dataset = _load_parser_classes(gsplat_dir)
    parser = Parser(data_dir=str(data_dir), factor=4, normalize=True, test_every=8)
    dataset = Dataset(parser, split="train", val_every=0)
    local_by_name: dict[str, int] = {}
    for local_index, global_value in enumerate(dataset.indices):
        global_index = int(global_value)
        name = parser.image_names[global_index]
        if name in local_by_name:
            raise RuntimeError(f"duplicate parser training image: {name}")
        local_by_name[name] = local_index

    selected_names = [entry["image_name"] for entry in entries]
    if len(set(selected_names)) != 16 or not set(selected_names).issubset(local_by_name):
        raise RuntimeError("synthetic manifest image names do not map one-to-one to parser training views")

    base_by_name: dict[str, np.ndarray] = {}
    mapping_rows: list[dict[str, Any]] = []
    frames: list[FrameInput] = []
    for entry in entries:
        name = entry["image_name"]
        local_index = local_by_name[name]
        global_index = int(dataset.indices[local_index])
        item = dataset[local_index]
        base = item["image"].clamp(0, 255).to(torch.uint8).cpu().numpy().copy()
        base_by_name[name] = base
        target_path = derived_dir / entry["derived_image"]
        mask_path = derived_dir / entry["ground_truth_mask"]
        target = _load_rgb_u8(target_path)
        mask = _load_mask(mask_path)
        if target.shape != base.shape or mask.shape != base.shape[:2]:
            raise RuntimeError(f"parser-exact shape mismatch for {name}")
        if entry["kind"] == "clean":
            if bool(mask.any()) or not np.array_equal(target, base):
                raise RuntimeError(f"clean parser-exact frame differs from parser target: {name}")
        else:
            destination = entry.get("destination")
            if not isinstance(destination, dict):
                raise RuntimeError(f"patched frame lacks destination metadata: {name}")
            x, y = int(destination["x"]), int(destination["y"])
            patch_width, patch_height = int(destination["width"]), int(destination["height"])
            expected_mask = np.zeros_like(mask)
            expected_mask[y : y + patch_height, x : x + patch_width] = True
            if not np.array_equal(mask, expected_mask):
                raise RuntimeError(f"GT rectangle does not match manifest: {name}")
            source_name = entry.get("source_image_name")
            if source_name not in selected_names:
                raise RuntimeError(f"patch source is not one of the fixed selected frames: {name}")
        camera_id = int(parser.camera_ids[global_index])
        K = item["K"].float().cpu()
        camtoworld = item["camtoworld"].float().cpu()
        rotation = camtoworld[:3, :3]
        inverse_error = float((camtoworld @ torch.linalg.inv(camtoworld) - torch.eye(4)).abs().max().item())
        orthonormal_error = float((rotation.T @ rotation - torch.eye(3)).abs().max().item())
        determinant = float(torch.linalg.det(rotation).item())
        if inverse_error > 1e-5 or orthonormal_error > 1e-4 or abs(determinant - 1.0) > 1e-4:
            raise RuntimeError(f"camera-to-world contract failed for {name}")
        target_tensor = torch.from_numpy(target.copy()).float() / 255.0
        layer = TransientGaussianLayer(target_tensor, K, camtoworld)
        projection_p95 = projection_error_percentile(layer, K, camtoworld)
        if projection_p95 > config["evaluation"]["projection_p95_max_pixels"]:
            raise RuntimeError(f"camera-plane projection failed before CUDA for {name}")
        del layer
        mapping_rows.append(
            {
                "image_name": name,
                "kind": entry["kind"],
                "manifest_index": int(entry["index"]),
                "dataset_local_index": local_index,
                "parser_global_index": global_index,
                "camera_id": camera_id,
                "height": int(target.shape[0]),
                "width": int(target.shape[1]),
                "K": K.tolist(),
                "camtoworld": camtoworld.tolist(),
                "inverse_max_error": inverse_error,
                "rotation_orthonormal_max_error": orthonormal_error,
                "rotation_determinant": determinant,
                "grid_projection_p95_pixels": projection_p95,
            }
        )
        frames.append(
            FrameInput(
                index=int(entry["index"]),
                image_name=name,
                kind=entry["kind"],
                target_rgb=target_tensor,
                K=K,
                camtoworld=camtoworld,
                mask_path=mask_path,
                parser_local_index=local_index,
                parser_global_index=global_index,
                camera_id=camera_id,
            )
        )

    for entry in entries[8:]:
        name = entry["image_name"]
        base = base_by_name[name]
        target = (frames[int(entry["index"])].target_rgb.numpy() * 255.0).round().astype(np.uint8)
        destination = entry["destination"]
        x, y = int(destination["x"]), int(destination["y"])
        patch_width, patch_height = int(destination["width"]), int(destination["height"])
        source = base_by_name[entry["source_image_name"]]
        source_x = (int(entry["index"]) * 29) % (source.shape[1] - patch_width + 1)
        source_y = (int(entry["index"]) * 31) % (source.shape[0] - patch_height + 1)
        expected_target = base.copy()
        expected_target[y : y + patch_height, x : x + patch_width] = source[
            source_y : source_y + patch_height, source_x : source_x + patch_width
        ]
        if not np.array_equal(target, expected_target):
            raise RuntimeError(f"patched pixels do not match the fixed v2 generator: {name}")

    return frames, {
        "derived_dir": str(derived_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "tree_sha256": derived_sha,
        "file_count": file_count,
        "target_contract": manifest["target_contract"],
        "frame_mapping": mapping_rows,
        "status": "PASS",
    }, derived_sha


def audit_inputs(args: argparse.Namespace, config: dict[str, Any]) -> InputAudit:
    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    if branch != "dev":
        raise RuntimeError(f"Phase 4A must remain on dev, found {branch}")
    data_dir = args.data_dir.expanduser().resolve()
    gsplat_dir = args.gsplat_dir.expanduser().resolve()
    derived_dir = args.derived_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    required_paths = [data_dir, gsplat_dir, derived_dir, checkpoint_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Phase 4A fixed inputs are missing: {missing}")
    rasterizer = _audit_rasterizer(gsplat_dir)
    checkpoint_sha, checkpoint_contract = _checkpoint_contract(checkpoint_path, config)
    frames, derived_contract, derived_sha = _audit_manifest_and_parser(
        data_dir, gsplat_dir, derived_dir, config
    )
    cpu_contract = run_cpu_math_contract()
    if cpu_contract["status"] != "PASS":
        raise RuntimeError("CPU math contract failed")
    contract = {
        "status": "PASS",
        "branch": branch,
        "commit": commit,
        "data_dir": str(data_dir),
        "gsplat": rasterizer,
        "checkpoint": checkpoint_contract,
        "derived": derived_contract,
        "camera_convention": {
            "parser_field": "parser.camtoworlds[index] / Dataset[item]['camtoworld']",
            "shape": [4, 4],
            "meaning": "camera-to-world homogeneous transform [R_wc|t_wc; 0 0 0 1]",
            "rasterizer_viewmat": "inverse(camtoworld)",
        },
        "cpu_math_contract": cpu_contract,
    }
    return InputAudit(contract, frames, commit, branch, checkpoint_sha, derived_sha)


def _tensor_state_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_static_module(checkpoint_path: Path, device: torch.device) -> tuple[FrozenStaticGaussians, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    splats = checkpoint["splats"]
    state_digest = _tensor_state_digest(splats)
    module = FrozenStaticGaussians(splats).to(device)
    del checkpoint, splats
    for parameter in module.parameters():
        if parameter.requires_grad:
            raise RuntimeError("static Gaussian parameter was not frozen")
    return module, state_digest


def _module_state_digest(module: FrozenStaticGaussians) -> str:
    return _tensor_state_digest({name: value for name, value in module.gaussians.items()})


def _fused_ssim():
    try:
        from fused_ssim import fused_ssim
    except ImportError as error:
        raise RuntimeError("the pinned gsplat environment is missing fused_ssim") from error
    return fused_ssim


def _make_optimizer(layer: TransientGaussianLayer, config: dict[str, Any]) -> torch.optim.Adam:
    values = config["transient"]
    return torch.optim.Adam(
        [
            {"params": [layer.opacity_logits], "lr": values["opacity_lr"], "name": "opacity_logits"},
            {"params": [layer.color_logits], "lr": values["color_lr"], "name": "color_logits"},
        ],
        betas=(values["adam_beta1"], values["adam_beta2"]),
        eps=values["adam_epsilon"],
    )


def _directional_cuda_test(device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    fused_ssim = _fused_ssim()
    height = width = 64
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    static = torch.stack((0.15 + 0.55 * xx, 0.2 + 0.5 * yy, 0.25 + 0.25 * (xx + yy)), dim=-1)
    K = torch.tensor([[64.0, 0.0, 31.5], [0.0, 64.0, 31.5], [0.0, 0.0, 1.0]], device=device)
    camtoworld = torch.eye(4, device=device)

    def run(patched: bool) -> dict[str, float]:
        target = static.clone()
        patch_mask = torch.zeros(height, width, dtype=torch.bool, device=device)
        if patched:
            patch_mask[20:44, 20:44] = True
            target[patch_mask] = torch.tensor([0.95, 0.05, 0.75], device=device)
        layer = TransientGaussianLayer(target, K, camtoworld).to(device)
        optimizer = _make_optimizer(layer, config)
        with torch.no_grad():
            transient_pre, alpha, _ = layer.render(K, camtoworld)
            initial_composite = compose_premultiplied(static, transient_pre, alpha)
            initial_alpha = float(alpha.mean().item())
            initial_patch_error = float((initial_composite[patch_mask] - target[patch_mask]).abs().mean().item()) if patched else 0.0
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            transient_pre, alpha, _ = layer.render(K, camtoworld)
            composite = compose_premultiplied(static, transient_pre, alpha)
            loss, _ = static_transient_loss(
                composite,
                target,
                alpha,
                fused_ssim,
                ssim_lambda=config["transient"]["ssim_lambda"],
                alpha_weight=config["transient"]["alpha_weight"],
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            transient_pre, alpha, _ = layer.render(K, camtoworld)
            composite = compose_premultiplied(static, transient_pre, alpha)
            result = {
                "initial_mean_alpha": initial_alpha,
                "final_mean_alpha": float(alpha.mean().item()),
                "inside_alpha": float(alpha[..., 0][patch_mask].mean().item()) if patched else 0.0,
                "outside_alpha": float(alpha[..., 0][~patch_mask].mean().item()) if patched else 0.0,
                "initial_patch_error": initial_patch_error,
                "final_patch_error": float((composite[patch_mask] - target[patch_mask]).abs().mean().item()) if patched else 0.0,
                "composite_static_mae": float((composite - static).abs().mean().item()),
                "finite": float(torch.isfinite(composite).all().item()),
            }
        del optimizer, layer
        return result

    clean = run(False)
    patched = run(True)
    return {
        "clean": clean,
        "patched": patched,
        "clean_null_pass": clean["final_mean_alpha"] < clean["initial_mean_alpha"] and clean["finite"] == 1.0,
        "patch_on_pass": (
            patched["inside_alpha"] > patched["outside_alpha"]
            and patched["final_patch_error"] < patched["initial_patch_error"]
            and patched["finite"] == 1.0
        ),
    }


def run_cuda_contract_suite(
    static_module: FrozenStaticGaussians,
    frame: FrameInput,
    static_rgb: torch.Tensor,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 4A CUDA contract requires an available CUDA device")
    target = frame.target_rgb.to(device)
    K = frame.K.to(device)
    camtoworld = frame.camtoworld.to(device)
    layer = TransientGaussianLayer(target, K, camtoworld).to(device)
    projection_p95 = projection_error_percentile(layer, K, camtoworld)

    from gsplat.rendering import rasterization

    with torch.inference_mode():
        direct_static, _, _ = rasterization(
            means=static_module.gaussians["means"],
            quats=static_module.gaussians["quats"],
            scales=torch.exp(static_module.gaussians["scales"]),
            opacities=torch.sigmoid(static_module.gaussians["opacities"]),
            colors=torch.cat((static_module.gaussians["sh0"], static_module.gaussians["shN"]), dim=1),
            viewmats=torch.linalg.inv(camtoworld[None]),
            Ks=K[None],
            width=target.shape[1],
            height=target.shape[0],
            near_plane=0.01,
            far_plane=1e10,
            packed=False,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=True,
            rasterize_mode="classic",
            distributed=False,
            camera_model="pinhole",
            sh_degree=3,
            with_ut=False,
            with_eval3d=False,
        )
        static_baseline_difference = float(
            (direct_static[0].clamp(0.0, 1.0) - static_rgb).abs().max().item()
        )

    coverage = TransientGaussianLayer(target, K, camtoworld).to(device)
    with torch.no_grad():
        coverage.color_logits.fill_(torch.logit(torch.tensor(1.0 - 1e-4)).item())
        coverage.opacity_logits.fill_(0.0)
        _, coverage_alpha, _ = coverage.render(K, camtoworld)
    border = config["evaluation"]["coverage_inner_border_pixels"]
    interior = coverage_alpha[border:-border, border:-border, 0]
    finite_ratio = float(torch.isfinite(interior).float().mean().item())
    hole_ratio = float((interior < config["evaluation"]["coverage_alpha_hole_threshold"]).float().mean().item())

    fused_ssim = _fused_ssim()
    transient_pre, alpha, _ = layer.render(K, camtoworld)
    composite = compose_premultiplied(static_rgb, transient_pre, alpha)
    loss, loss_parts = static_transient_loss(
        composite,
        target,
        alpha,
        fused_ssim,
        ssim_lambda=config["transient"]["ssim_lambda"],
        alpha_weight=config["transient"]["alpha_weight"],
    )
    loss.backward()
    color_grad = layer.color_logits.grad
    opacity_grad = layer.opacity_logits.grad
    color_grad_pass = color_grad is not None and bool(torch.isfinite(color_grad).all()) and float(color_grad.abs().sum().item()) > 0.0
    opacity_grad_pass = opacity_grad is not None and bool(torch.isfinite(opacity_grad).all()) and float(opacity_grad.abs().sum().item()) > 0.0
    static_gradient_leak = any(
        parameter.grad is not None and float(parameter.grad.abs().sum().item()) > 0.0
        for parameter in static_module.parameters()
    )
    geometry_frozen = all(
        not tensor.requires_grad for tensor in (layer.means, layer.quaternions, layer.log_scales)
    )
    cpu_contract = run_cpu_math_contract()
    directional = _directional_cuda_test(device, config)
    result = {
        "status": "PASS",
        "projection_p95_pixels": projection_p95,
        "static_baseline_render_max_abs_difference": static_baseline_difference,
        "coverage_finite_ratio": finite_ratio,
        "coverage_alpha_hole_ratio_below_0_01": hole_ratio,
        "loss": float(loss.item()),
        "loss_parts": {name: float(value.item()) for name, value in loss_parts.items()},
        "transient_color_gradient_finite_nonzero": color_grad_pass,
        "transient_opacity_gradient_finite_nonzero": opacity_grad_pass,
        "static_gradient_leak": static_gradient_leak,
        "transient_geometry_frozen": geometry_frozen,
        "cpu_compositing_contract": cpu_contract,
        "directional_cuda_tests": directional,
    }
    result["PREMULTIPLIED_COMPOSITING_PASS"] = cpu_contract["status"] == "PASS"
    result["GRID_PROJECTION_PASS"] = projection_p95 <= config["evaluation"]["projection_p95_max_pixels"]
    result["GRID_COVERAGE_PASS"] = finite_ratio == 1.0 and hole_ratio <= config["evaluation"]["coverage_alpha_hole_ratio_max"]
    result["STATIC_GRADIENT_LEAK"] = static_gradient_leak
    result["TRANSIENT_GEOMETRY_FROZEN"] = geometry_frozen
    result["TRANSIENT_GRADIENT_PASS"] = color_grad_pass and opacity_grad_pass
    result["STATIC_BASELINE_RENDER_MATCH"] = static_baseline_difference <= 1e-6
    if not (
        result["PREMULTIPLIED_COMPOSITING_PASS"]
        and result["GRID_PROJECTION_PASS"]
        and result["GRID_COVERAGE_PASS"]
        and not static_gradient_leak
        and geometry_frozen
        and result["TRANSIENT_GRADIENT_PASS"]
        and result["STATIC_BASELINE_RENDER_MATCH"]
        and directional["clean_null_pass"]
        and directional["patch_on_pass"]
    ):
        result["status"] = "FAIL"
    del coverage, layer
    torch.cuda.empty_cache()
    return result


def _actual_frame_smoke(
    static_module: FrozenStaticGaussians,
    frame: FrameInput,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    fused_ssim = _fused_ssim()
    target = frame.target_rgb.to(device)
    K = frame.K.to(device)
    camtoworld = frame.camtoworld.to(device)
    static = static_module.render(K, camtoworld, target.shape[1], target.shape[0])
    layer = TransientGaussianLayer(target, K, camtoworld).to(device)
    optimizer = _make_optimizer(layer, config)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        transient_pre, alpha, _ = layer.render(K, camtoworld)
        composite = compose_premultiplied(static, transient_pre, alpha)
        loss, _ = static_transient_loss(
            composite,
            target,
            alpha,
            fused_ssim,
            ssim_lambda=config["transient"]["ssim_lambda"],
            alpha_weight=config["transient"]["alpha_weight"],
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("actual-frame smoke loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return {"status": "PASS", "image_name": frame.image_name, "steps": 2, "losses": losses}


def _benchmark_static(
    module: FrozenStaticGaussians, frame: FrameInput, device: torch.device
) -> dict[str, Any]:
    K = frame.K.to(device)
    camtoworld = frame.camtoworld.to(device)
    height, width = frame.target_rgb.shape[:2]
    for _ in range(10):
        module.render(K, camtoworld, width, height)
    torch.cuda.synchronize(device)
    batch_fps = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            module.render(K, camtoworld, width, height)
        end.record()
        torch.cuda.synchronize(device)
        elapsed_seconds = start.elapsed_time(end) / 1000.0
        batch_fps.append(20.0 / elapsed_seconds)
    return {"fps": float(np.median(batch_fps)), "batch_fps": batch_fps, "warmup_renders": 10, "timed_renders": 100}


def _safe_frame_name(index: int, image_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(image_name).name)
    return f"{index:02d}_{cleaned}"


def _save_rgb(path: Path, value: torch.Tensor) -> None:
    array = (value.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _save_gray(path: Path, value: torch.Tensor) -> None:
    array = (value.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def _save_error(path: Path, value: torch.Tensor) -> None:
    error = value.detach().cpu().numpy()
    scale = max(float(np.quantile(error, 0.99)), 1e-8)
    normalized = np.clip(error / scale, 0.0, 1.0)
    heat = np.stack((normalized, 0.25 * (1.0 - normalized), 1.0 - normalized), axis=-1)
    Image.fromarray((heat * 255).round().astype(np.uint8), mode="RGB").save(path)


def _save_overlay(path: Path, predicted: np.ndarray, target: np.ndarray) -> None:
    overlay = np.zeros(target.shape + (3,), dtype=np.uint8)
    overlay[predicted & target] = (0, 220, 0)
    overlay[predicted & ~target] = (255, 0, 255)
    overlay[~predicted & target] = (255, 0, 0)
    Image.fromarray(overlay, mode="RGB").save(path)


def _frame_metrics(
    frame: FrameInput,
    static: torch.Tensor,
    target: torch.Tensor,
    transient_pre: torch.Tensor,
    alpha: torch.Tensor,
    composite: torch.Tensor,
    gt_mask: np.ndarray,
    fit_seconds: float,
    peak_vram_bytes: int,
    gaussian_count: int,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    alpha_2d = alpha[..., 0].detach().cpu().numpy()
    predicted = alpha_2d >= 0.5
    counts = confusion_counts(predicted, gt_mask)
    metrics = metrics_from_counts(counts)
    static_np = static.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    composite_np = composite.detach().cpu().numpy()
    inside = gt_mask
    outside = ~inside
    static_patch_mae = float(np.mean(np.abs(static_np[inside] - target_np[inside]))) if bool(inside.any()) else 0.0
    composite_patch_mae = float(np.mean(np.abs(composite_np[inside] - target_np[inside]))) if bool(inside.any()) else 0.0
    relative_reduction = (static_patch_mae - composite_patch_mae) / (static_patch_mae + 1e-12) if bool(inside.any()) else 0.0
    background_change_l1 = float(np.mean(np.abs(composite_np[outside] - static_np[outside]).sum(axis=-1)))
    return {
        "index": frame.index,
        "image_name": frame.image_name,
        "kind": frame.kind,
        **metrics,
        "mean_alpha_inside_gt": float(alpha_2d[inside].mean()) if bool(inside.any()) else 0.0,
        "mean_alpha_outside_gt": float(alpha_2d[outside].mean()),
        "mean_transient_alpha": float(alpha_2d.mean()),
        "p95_transient_alpha": float(np.quantile(alpha_2d, 0.95)),
        "mean_abs_composite_static": float(np.mean(np.abs(composite_np - static_np))),
        "psnr_composite_target": psnr(composite_np, target_np),
        "psnr_static_target": psnr(static_np, target_np),
        "static_patch_mae": static_patch_mae,
        "composite_patch_mae": composite_patch_mae,
        "relative_patch_mae_reduction": relative_reduction,
        "background_change_l1": background_change_l1,
        "transient_gaussian_count": gaussian_count,
        "fit_time_seconds": fit_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "transient_checkpoint_bytes": checkpoint_bytes,
        "transient_premultiplied_rgb_mean": float(transient_pre.mean().item()),
    }


def _fit_one_frame(
    frame: FrameInput,
    static_rgb: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    fused_ssim = _fused_ssim()
    target = frame.target_rgb.to(device)
    K = frame.K.to(device)
    camtoworld = frame.camtoworld.to(device)
    layer = TransientGaussianLayer(target, K, camtoworld).to(device)
    optimizer = _make_optimizer(layer, config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for step in range(config["transient"]["fit_steps_per_frame"]):
        optimizer.zero_grad(set_to_none=True)
        transient_pre, alpha, _ = layer.render(K, camtoworld)
        composite = compose_premultiplied(static_rgb, transient_pre, alpha)
        loss, _ = static_transient_loss(
            composite,
            target,
            alpha,
            fused_ssim,
            ssim_lambda=config["transient"]["ssim_lambda"],
            alpha_weight=config["transient"]["alpha_weight"],
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite Phase 4A loss for {frame.image_name} at step {step}")
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    with torch.no_grad():
        transient_pre, alpha, _ = layer.render(K, camtoworld)
        composite = compose_premultiplied(static_rgb, transient_pre, alpha)
        final_loss, _ = static_transient_loss(
            composite,
            target,
            alpha,
            fused_ssim,
            ssim_lambda=config["transient"]["ssim_lambda"],
            alpha_weight=config["transient"]["alpha_weight"],
        )
    if not bool(torch.isfinite(composite).all()) or float(composite.min()) < -1e-6 or float(composite.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"composite range contract failed for {frame.image_name}")

    safe_name = _safe_frame_name(frame.index, frame.image_name)
    parameter_path = output_dir / "transient_params" / f"{safe_name}.pt"
    torch.save(layer.serializable_state(frame.image_name), parameter_path)
    visual_dir = output_dir / "visualizations" / safe_name
    visual_dir.mkdir(parents=True, exist_ok=False)
    gt_mask = _load_mask(frame.mask_path)
    predicted = alpha[..., 0].detach().cpu().numpy() >= config["evaluation"]["alpha_threshold"]
    _save_rgb(visual_dir / "static_rgb.png", static_rgb)
    _save_rgb(visual_dir / "target_rgb.png", target)
    _save_rgb(visual_dir / "transient_premultiplied_rgb.png", transient_pre)
    _save_gray(visual_dir / "transient_alpha.png", alpha[..., 0])
    _save_rgb(visual_dir / "composite_rgb.png", composite)
    _save_gray(visual_dir / "gt_mask.png", torch.from_numpy(gt_mask.astype(np.float32)))
    _save_gray(visual_dir / "predicted_mask.png", torch.from_numpy(predicted.astype(np.float32)))
    _save_overlay(visual_dir / "tp_fp_fn_overlay.png", predicted, gt_mask)
    _save_error(visual_dir / "static_error.png", (static_rgb - target).abs().mean(dim=-1))
    _save_error(visual_dir / "composite_error.png", (composite - target).abs().mean(dim=-1))
    metrics = _frame_metrics(
        frame,
        static_rgb,
        target,
        transient_pre,
        alpha,
        composite,
        gt_mask,
        fit_seconds,
        int(peak_vram),
        int(layer.means.shape[0]),
        parameter_path.stat().st_size,
    )
    metrics["final_loss"] = float(final_loss.item())
    del optimizer, layer, target, K, camtoworld, transient_pre, alpha, composite
    torch.cuda.empty_cache()
    return metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _environment(audit: InputAudit, device: torch.device) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "packages": {name: _package_version(name) for name in ("gsplat", "numpy", "Pillow", "fused-ssim", "pycolmap")},
        "branch": audit.branch,
        "commit": audit.commit,
    }


def _default_output(commit: str) -> Path:
    return PROJECT_ROOT / "analysis" / f"static_transient_gate_{commit[:7]}"


def run_smoke(args: argparse.Namespace, config: dict[str, Any]) -> int:
    audit = audit_inputs(args, config)
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    static, state_before = _load_static_module(checkpoint_path, device)
    frame = audit.frames[0]
    K = frame.K.to(device)
    camtoworld = frame.camtoworld.to(device)
    static_rgb = static.render(K, camtoworld, frame.target_rgb.shape[1], frame.target_rgb.shape[0])
    suite = run_cuda_contract_suite(static, frame, static_rgb, device, config)
    patched_frame = audit.frames[8]
    actual_smoke = _actual_frame_smoke(static, patched_frame, device, config)
    state_after = _module_state_digest(static)
    checkpoint_after = sha256_file(checkpoint_path)
    result = {
        "status": "PASS" if suite["status"] == "PASS" and actual_smoke["status"] == "PASS" and state_after == state_before and checkpoint_after == audit.checkpoint_sha256 else "FAIL",
        "cuda_contract": suite,
        "actual_patched_frame_smoke": actual_smoke,
        "static_state_unchanged": state_after == state_before,
        "checkpoint_unchanged": checkpoint_after == audit.checkpoint_sha256,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


def run_formal(args: argparse.Namespace, config: dict[str, Any]) -> int:
    audit = audit_inputs(args, config)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else _default_output(audit.commit)
    expected_output = _default_output(audit.commit).resolve()
    if output_dir != expected_output:
        raise RuntimeError(f"Phase 4A output must be the unique commit-scoped path: {expected_output}")
    ensure_output_absent(output_dir)
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    static, static_state_before = _load_static_module(checkpoint_path, device)
    benchmark_before = _benchmark_static(static, audit.frames[0], device)
    first = audit.frames[0]
    first_static = static.render(
        first.K.to(device), first.camtoworld.to(device), first.target_rgb.shape[1], first.target_rgb.shape[0]
    )
    cuda_contract = run_cuda_contract_suite(static, first, first_static, device, config)
    if cuda_contract["status"] != "PASS":
        print(json.dumps(cuda_contract, indent=2))
        raise RuntimeError("Phase 4A CUDA hard contract failed; formal fitting was not started")

    output_dir.mkdir(parents=True)
    for name in ("static_cache", "transient_params", "metrics", "visualizations"):
        (output_dir / name).mkdir()
    write_json(output_dir / "input_contract.json", audit.contract)
    write_json(output_dir / "unit_test_summary.json", {"cpu": audit.contract["cpu_math_contract"], "cuda": cuda_contract})
    write_json(output_dir / "config.json", config)
    run_command = shlex.join([sys.executable, *sys.argv])
    (output_dir / "actual_run_command.txt").write_text(run_command + "\n", encoding="utf-8")
    write_json(output_dir / "environment.json", _environment(audit, device))

    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for frame in audit.frames:
        if frame.index == 0:
            static_rgb = first_static
        else:
            static_rgb = static.render(
                frame.K.to(device),
                frame.camtoworld.to(device),
                frame.target_rgb.shape[1],
                frame.target_rgb.shape[0],
            )
        if static_rgb.requires_grad:
            raise RuntimeError("static render cache was not detached")
        safe_name = _safe_frame_name(frame.index, frame.image_name)
        torch.save(static_rgb.detach().cpu(), output_dir / "static_cache" / f"{safe_name}.pt")
        row = _fit_one_frame(frame, static_rgb, output_dir, device, config)
        rows.append(row)
        print(
            f"PHASE4A_FRAME {frame.index + 1}/16 {frame.image_name} "
            f"fit_seconds={row['fit_time_seconds']:.3f} mean_alpha={row['mean_transient_alpha']:.6f}",
            flush=True,
        )
        if frame.index == 0:
            del first_static
    total_wall_time = time.perf_counter() - total_start

    static_state_after = _module_state_digest(static)
    checkpoint_after = sha256_file(checkpoint_path)
    reference_after = static.render(
        first.K.to(device), first.camtoworld.to(device), first.target_rgb.shape[1], first.target_rgb.shape[0]
    )
    reference_before = torch.load(
        output_dir / "static_cache" / f"{_safe_frame_name(first.index, first.image_name)}.pt",
        map_location=device,
        weights_only=True,
    )
    static_render_difference = float((reference_after - reference_before).abs().max().item())
    benchmark_after = _benchmark_static(static, first, device)
    fps_relative_difference = abs(benchmark_after["fps"] - benchmark_before["fps"]) / benchmark_before["fps"]
    engineering = {
        "transient_gaussian_counts": [int(row["transient_gaussian_count"]) for row in rows],
        "mean_fit_time_per_frame_seconds": float(np.mean([row["fit_time_seconds"] for row in rows])),
        "total_fit_time_seconds": float(sum(row["fit_time_seconds"] for row in rows)),
        "total_gate_wall_time_seconds": total_wall_time,
        "peak_vram_bytes": int(max(row["peak_vram_bytes"] for row in rows)),
        "static_fps_before": benchmark_before["fps"],
        "static_fps_after": benchmark_after["fps"],
        "static_fps_relative_difference": fps_relative_difference,
        "static_fps_benchmark_before": benchmark_before,
        "static_fps_benchmark_after": benchmark_after,
        "static_render_max_abs_difference": static_render_difference,
        "transient_checkpoint_total_bytes": int(sum(row["transient_checkpoint_bytes"] for row in rows)),
        "transient_checkpoint_mean_bytes": float(np.mean([row["transient_checkpoint_bytes"] for row in rows])),
        "static_inference_loads_transient_parameters": False,
    }
    summary = aggregate_summary(rows, engineering)
    hard_gates = {
        "PASS_EXECUTION": len(rows) == 16,
        "STATIC_CHECKPOINT_UNCHANGED": checkpoint_after == audit.checkpoint_sha256 and static_state_after == static_state_before,
        "STATIC_GRADIENT_LEAK": cuda_contract["STATIC_GRADIENT_LEAK"],
        "TRANSIENT_GEOMETRY_FROZEN": cuda_contract["TRANSIENT_GEOMETRY_FROZEN"],
        "PREMULTIPLIED_COMPOSITING_PASS": cuda_contract["PREMULTIPLIED_COMPOSITING_PASS"],
        "GRID_PROJECTION_PASS": cuda_contract["GRID_PROJECTION_PASS"],
        "GRID_COVERAGE_PASS": cuda_contract["GRID_COVERAGE_PASS"],
    }
    gate = decide_gate(summary, hard_gates, config)
    metrics_dir = output_dir / "metrics"
    _write_csv(metrics_dir / "per_frame.csv", rows)
    write_json(metrics_dir / "per_frame.json", rows)
    write_json(metrics_dir / "summary.json", summary)
    write_json(metrics_dir / "gate.json", gate)
    derived_after, derived_file_count_after = tree_sha256(args.derived_dir.expanduser().resolve())
    provenance = {
        "branch": audit.branch,
        "commit": audit.commit,
        "commit_short": audit.commit[:7],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before": audit.checkpoint_sha256,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_unchanged": checkpoint_after == audit.checkpoint_sha256,
        "static_state_sha256_before": static_state_before,
        "static_state_sha256_after": static_state_after,
        "static_state_unchanged": static_state_after == static_state_before,
        "derived_tree_sha256_before": audit.derived_tree_sha256,
        "derived_tree_sha256_after": derived_after,
        "derived_file_count_after": derived_file_count_after,
        "derived_tree_unchanged": derived_after == audit.derived_tree_sha256,
        "optimizer_scope": "one independent transient layer per frame",
        "static_optimizer_created": False,
        "default_strategy_created": False,
        "static_checkpoint_written": False,
        "transient_densification": False,
        "transient_pruning": False,
        "gt_mask_training_uses": 0,
        "run_command": run_command,
        "decision": gate["decision"],
    }
    write_json(output_dir / "provenance.json", provenance)
    print(json.dumps({"status": "PASS_EXECUTION", "decision": gate["decision"], "output_dir": str(output_dir)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "puri_gs_static_transient_gate.yaml")
    parser.add_argument("--gsplat-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_gate_config(config_path)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if args.dry_run:
        branch = _git("branch", "--show-current")
        commit = _git("rev-parse", "HEAD")
        planned = {
            "status": "DRY_RUN_PASS" if branch == "dev" else "DRY_RUN_BRANCH_FAIL",
            "branch": branch,
            "commit": commit,
            "config": str(config_path),
            "planned_output": str(_default_output(commit)),
            "formal_fit_started": False,
            "inputs_opened": False,
        }
        print(json.dumps(planned, indent=2))
        return 0 if branch == "dev" else 2
    if args.preflight:
        audit = audit_inputs(args, config)
        print(json.dumps(audit.contract, indent=2))
        return 0
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("smoke and formal Phase 4A execution require CUDA")
    if args.smoke:
        return run_smoke(args, config)
    return run_formal(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
