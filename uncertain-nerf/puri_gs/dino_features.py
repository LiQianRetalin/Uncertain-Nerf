"""Frozen, offline-only DINOv2 feature extraction for PURI-GS-RU."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


MODEL_NAME = "dinov2_vits14_reg"
FEATURE_DIM = 384
PATCH_SIZE = 14
COARSE_GRID = 16
FINE_GRID = 36
COARSE_INPUT = COARSE_GRID * PATCH_SIZE
FINE_INPUT = FINE_GRID * PATCH_SIZE
FEATURE_EXTRACTOR_VERSION = "puri-gs-ru-dinov2-v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest without loading the whole weight file in RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_dir: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot read DINOv2 repository commit: {result.stderr.strip()}")
    return result.stdout.strip()


def _unwrap_state_dict(value: Any) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError("DINOv2 checkpoint must be a state-dict mapping")
    for key in ("model", "state_dict", "teacher"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            value = nested
            break
    state = {
        str(key).removeprefix("module.").removeprefix("backbone."): tensor
        for key, tensor in value.items()
        if isinstance(tensor, Tensor)
    }
    if not state:
        raise ValueError("DINOv2 checkpoint contains no tensor state")
    return state


def load_frozen_dinov2(
    repo_dir: str | Path,
    weight_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the fixed register-token ViT-S/14 from local source and local weights.

    ``pretrained=False`` is intentional: this function must never initiate a network
    download.  The caller has to supply both assets explicitly.
    """

    repo = Path(repo_dir).expanduser().resolve()
    weights = Path(weight_path).expanduser().resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv2 hubconf.py is missing: {repo / 'hubconf.py'}")
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv2 weight file is missing: {weights}")

    model = torch.hub.load(
        str(repo), MODEL_NAME, source="local", pretrained=False
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(_unwrap_state_dict(checkpoint), strict=True)
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_count != 0:
        raise RuntimeError("DINOv2 must have zero trainable parameters")
    metadata = {
        "model_name": MODEL_NAME,
        "feature_dim": FEATURE_DIM,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "weight_path": str(weights),
        "weight_sha256": sha256_file(weights),
        "repository_path": str(repo),
        "repository_commit": git_commit(repo),
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
    }
    return model, metadata


def preprocess_rgb(images: Tensor, input_size: int) -> Tensor:
    """Apply the fixed resize and ImageNet normalization used by DINOv2.

    Accepted layouts are ``[B,H,W,3]`` and ``[B,3,H,W]`` with RGB values in
    ``[0,1]``.  Aspect ratio is deliberately mapped to the fixed square grid so
    cached ground truth and rendered images follow exactly the same transform.
    """

    if images.ndim != 4:
        raise ValueError(f"expected a 4D RGB batch, got {tuple(images.shape)}")
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    elif images.shape[1] != 3:
        raise ValueError(f"expected RGB channels, got {tuple(images.shape)}")
    if not torch.is_floating_point(images):
        images = images.float()
    if not torch.isfinite(images).all():
        raise ValueError("DINOv2 input contains non-finite values")
    if images.numel() and (images.min() < 0 or images.max() > 1):
        raise ValueError("DINOv2 input values must be in [0, 1]")
    resized = F.interpolate(
        images,
        size=(input_size, input_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = resized.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (resized - mean) / std


def extract_patch_grid(model: nn.Module, images: Tensor, grid_size: int) -> Tensor:
    """Extract normalized patch tokens as ``[B,384,grid,grid]``."""

    expected_input = grid_size * PATCH_SIZE
    prepared = preprocess_rgb(images.detach(), expected_input)
    with torch.no_grad():
        output = model.forward_features(prepared)
    if not isinstance(output, Mapping) or "x_norm_patchtokens" not in output:
        raise RuntimeError("DINOv2 forward_features did not return x_norm_patchtokens")
    tokens = output["x_norm_patchtokens"]
    expected_shape = (prepared.shape[0], grid_size * grid_size, FEATURE_DIM)
    if tuple(tokens.shape) != expected_shape:
        raise RuntimeError(
            f"unexpected DINOv2 patch shape {tuple(tokens.shape)}, expected {expected_shape}"
        )
    grid = tokens.transpose(1, 2).reshape(
        prepared.shape[0], FEATURE_DIM, grid_size, grid_size
    )
    if not torch.isfinite(grid).all():
        raise RuntimeError("DINOv2 produced non-finite features")
    return grid.detach()


@dataclass(frozen=True)
class CachedFeatureRecord:
    image_name: str
    file: str
    original_height: int
    original_width: int
    coarse_shape: tuple[int, int, int]
    fine_shape: tuple[int, int, int]


class FeatureCache:
    """Validated one-file-per-image feature cache with one-time lazy CPU loads."""

    def __init__(
        self,
        directory: str | Path,
        *,
        expected_weight_sha256: str,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"feature cache manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_name") != MODEL_NAME:
            raise ValueError("feature cache uses the wrong DINOv2 model")
        if manifest.get("model_weight_sha256") != expected_weight_sha256:
            raise ValueError("feature cache DINOv2 weight SHA-256 does not match runtime")
        if manifest.get("feature_extractor_version") != FEATURE_EXTRACTOR_VERSION:
            raise ValueError("feature cache extractor version does not match runtime")
        records = manifest.get("images")
        if not isinstance(records, list) or not records:
            raise ValueError("feature cache manifest has no images")
        self.scene = str(manifest.get("scene"))
        self.records: dict[str, dict[str, Any]] = {}
        for record in records:
            image_name = str(record["image_name"])
            if image_name in self.records:
                raise ValueError(f"duplicate feature-cache image mapping: {image_name}")
            self.records[image_name] = record
        self._loaded: dict[str, dict[str, Tensor]] = {}
        self.load_seconds = 0.0

    def load(self, image_name: str, grid_size: int) -> Tensor:
        if grid_size not in (COARSE_GRID, FINE_GRID):
            raise ValueError(f"unsupported feature grid: {grid_size}")
        record = self.records.get(image_name)
        if record is None:
            raise KeyError(f"training image is missing from feature cache: {image_name}")
        if image_name not in self._loaded:
            path = self.directory / str(record["file"])
            started = perf_counter()
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.load_seconds += perf_counter() - started
            if not isinstance(payload, dict):
                raise ValueError(f"invalid feature payload: {path}")
            self._loaded[image_name] = payload
        key = "coarse" if grid_size == COARSE_GRID else "fine"
        feature = self._loaded[image_name].get(key)
        expected = (FEATURE_DIM, grid_size, grid_size)
        if not isinstance(feature, Tensor) or tuple(feature.shape) != expected:
            raise ValueError(
                f"cached {key} feature for {image_name} has wrong shape; expected {expected}"
            )
        if feature.dtype != torch.float32 or not torch.isfinite(feature).all():
            raise ValueError(f"cached {key} feature for {image_name} is invalid")
        return feature

