"""Mip-aware multi-resolution hash-grid encoders for V7."""

import math
from itertools import product

import torch
from torch import nn

from v5.encoding import DirectionEncoder


def _level_attenuation(footprint, resolutions, dtype):
    """Low-pass each level using the sample's world-space cone footprint."""
    if footprint is None:
        return None
    footprint = footprint.reshape(-1, 1).clamp_min(0.0)
    resolutions = torch.as_tensor(resolutions, device=footprint.device, dtype=footprint.dtype)
    # A level is smoothly suppressed once its voxel width is smaller than the footprint.
    attenuation = torch.exp(-0.5 * (footprint * resolutions[None, :]).square())
    return attenuation.to(dtype)


class MipHashGridEncoder(nn.Module):
    _PRIMES = (1, 2654435761, 805459861)

    def __init__(self, aabb, n_levels=16, min_resolution=16, max_resolution=2048,
                 features_per_level=2, log2_hashmap_size=19):
        super().__init__()
        aabb = torch.as_tensor(aabb, dtype=torch.float32)
        if aabb.shape != (2, 3) or torch.any(aabb[1] <= aabb[0]):
            raise ValueError("aabb must have shape [2, 3] with max > min")
        self.register_buffer("aabb", aabb)
        self.n_levels = int(n_levels)
        self.features_per_level = int(features_per_level)
        self.max_hash_size = 1 << int(log2_hashmap_size)
        growth = math.exp((math.log(max_resolution) - math.log(min_resolution)) /
                          max(self.n_levels - 1, 1))
        self.resolutions = [int(math.floor(min_resolution * growth**level))
                            for level in range(self.n_levels)]
        self.tables = nn.ModuleList()
        self.table_sizes = []
        for resolution in self.resolutions:
            size = min(self.max_hash_size, resolution**3)
            table = nn.Embedding(size, self.features_per_level)
            nn.init.uniform_(table.weight, -1.0e-4, 1.0e-4)
            self.tables.append(table)
            self.table_sizes.append(size)
        self.output_dim = self.n_levels * self.features_per_level
        self.register_buffer("corner_offsets", torch.tensor(
            list(product((0, 1), repeat=3)), dtype=torch.long))

    def normalize(self, points):
        return (points - self.aabb[0]) / (self.aabb[1] - self.aabb[0])

    def _indices(self, corners, resolution, table_size):
        if resolution**3 <= table_size:
            return corners[..., 0] + resolution * (
                corners[..., 1] + resolution * corners[..., 2])
        hashed = (corners[..., 0] * self._PRIMES[0]
                  ^ corners[..., 1] * self._PRIMES[1]
                  ^ corners[..., 2] * self._PRIMES[2])
        return torch.remainder(hashed, table_size)

    def forward(self, points, footprint=None):
        original_shape = points.shape[:-1]
        x = self.normalize(points.reshape(-1, 3)).clamp(0.0, 1.0 - 1.0e-6)
        if footprint is not None:
            # Convert an isotropic world-space radius to conservative normalized units.
            footprint = footprint.reshape(-1) / (self.aabb[1] - self.aabb[0]).min()
        attenuation = _level_attenuation(footprint, self.resolutions, x.dtype)
        features = []
        offsets = self.corner_offsets.to(x.device)
        for level, (resolution, table, table_size) in enumerate(zip(
                self.resolutions, self.tables, self.table_sizes)):
            pos = x * (resolution - 1)
            base = torch.floor(pos).long()
            frac = pos - base.float()
            corners = (base[:, None, :] + offsets[None]).clamp(0, resolution - 1)
            corner_features = table(self._indices(corners, resolution, table_size))
            weights = torch.where(offsets[None].bool(), frac[:, None],
                                  1.0 - frac[:, None]).prod(dim=-1)
            level_features = (corner_features * weights[..., None]).sum(dim=1)
            if attenuation is not None:
                level_features = level_features * attenuation[:, level, None]
            features.append(level_features)
        return torch.cat(features, dim=-1).reshape(*original_shape, self.output_dim)


class TinyCudaNNMipHashGridEncoder(nn.Module):
    def __init__(self, aabb, n_levels=16, min_resolution=16, max_resolution=2048,
                 features_per_level=2, log2_hashmap_size=19):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("The tcnn hash backend requires a CUDA device")
        try:
            import tinycudann as tcnn
        except ImportError as exc:
            raise RuntimeError("backend=tcnn requires tiny-cuda-nn; see V7_GUIDE.md") from exc
        aabb = torch.as_tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.n_levels = int(n_levels)
        self.features_per_level = int(features_per_level)
        self.output_dim = self.n_levels * self.features_per_level
        growth = math.exp((math.log(max_resolution) - math.log(min_resolution)) /
                          max(self.n_levels - 1, 1))
        self.resolutions = [int(math.floor(min_resolution * growth**level))
                            for level in range(self.n_levels)]
        self.encoder = tcnn.Encoding(n_input_dims=3, encoding_config={
            "otype": "HashGrid", "n_levels": self.n_levels,
            "n_features_per_level": self.features_per_level,
            "log2_hashmap_size": int(log2_hashmap_size),
            "base_resolution": int(min_resolution), "per_level_scale": float(growth),
            "interpolation": "Smoothstep",
        })

    def normalize(self, points):
        return (points - self.aabb[0]) / (self.aabb[1] - self.aabb[0])

    def forward(self, points, footprint=None):
        shape = points.shape[:-1]
        normalized = self.normalize(points.reshape(-1, 3)).clamp(0.0, 1.0)
        encoded = self.encoder(normalized).reshape(-1, self.n_levels, self.features_per_level)
        if footprint is not None:
            normalized_footprint = footprint.reshape(-1) / (self.aabb[1] - self.aabb[0]).min()
            attenuation = _level_attenuation(normalized_footprint, self.resolutions, encoded.dtype)
            encoded = encoded * attenuation[..., None]
        return encoded.reshape(*shape, self.output_dim).to(points.dtype)


def make_hash_encoder(backend="torch", **kwargs):
    backend = str(backend).lower()
    if backend == "torch":
        return MipHashGridEncoder(**kwargs)
    if backend == "tcnn":
        return TinyCudaNNMipHashGridEncoder(**kwargs)
    raise ValueError(f"Unknown hash backend {backend!r}; expected 'torch' or 'tcnn'")


__all__ = ["DirectionEncoder", "MipHashGridEncoder", "TinyCudaNNMipHashGridEncoder",
           "make_hash_encoder"]
