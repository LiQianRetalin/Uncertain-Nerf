import math
from itertools import product

import torch
from torch import nn


class HashGridEncoder(nn.Module):
    """Pure PyTorch multi-resolution hash-grid encoder.

    The implementation intentionally favors portability and testability over a
    custom CUDA kernel. Coordinates are normalized with the persisted scene
    AABB before trilinear interpolation.
    """

    _PRIMES = (1, 2654435761, 805459861)

    def __init__(
        self,
        aabb,
        n_levels=16,
        min_resolution=16,
        max_resolution=2048,
        features_per_level=2,
        log2_hashmap_size=19,
    ):
        super().__init__()
        aabb = torch.as_tensor(aabb, dtype=torch.float32)
        if aabb.shape != (2, 3) or torch.any(aabb[1] <= aabb[0]):
            raise ValueError("aabb must have shape [2, 3] with max > min")
        self.register_buffer("aabb", aabb)
        self.n_levels = int(n_levels)
        self.features_per_level = int(features_per_level)
        self.max_hash_size = 1 << int(log2_hashmap_size)
        growth = math.exp(
            (math.log(max_resolution) - math.log(min_resolution))
            / max(self.n_levels - 1, 1)
        )
        self.resolutions = [
            int(math.floor(min_resolution * growth**level))
            for level in range(self.n_levels)
        ]
        self.tables = nn.ModuleList()
        self.table_sizes = []
        for resolution in self.resolutions:
            size = min(self.max_hash_size, resolution**3)
            table = nn.Embedding(size, self.features_per_level)
            nn.init.uniform_(table.weight, -1.0e-4, 1.0e-4)
            self.tables.append(table)
            self.table_sizes.append(size)
        self.output_dim = self.n_levels * self.features_per_level
        self.register_buffer(
            "corner_offsets",
            torch.tensor(list(product((0, 1), repeat=3)), dtype=torch.long),
        )

    def normalize(self, points):
        return (points - self.aabb[0]) / (self.aabb[1] - self.aabb[0])

    def _indices(self, corners, resolution, table_size):
        if resolution**3 <= table_size:
            return (
                corners[..., 0]
                + resolution
                * (corners[..., 1] + resolution * corners[..., 2])
            )
        hashed = (
            corners[..., 0] * self._PRIMES[0]
            ^ corners[..., 1] * self._PRIMES[1]
            ^ corners[..., 2] * self._PRIMES[2]
        )
        return torch.remainder(hashed, table_size)

    def forward(self, points):
        original_shape = points.shape[:-1]
        x = self.normalize(points.reshape(-1, 3))
        x = x.clamp(0.0, 1.0 - 1.0e-6)
        features = []
        offsets = self.corner_offsets.to(x.device)
        for resolution, table, table_size in zip(
            self.resolutions, self.tables, self.table_sizes
        ):
            pos = x * (resolution - 1)
            base = torch.floor(pos).long()
            frac = pos - base.float()
            corners = base[:, None, :] + offsets[None, :, :]
            corners = corners.clamp(0, resolution - 1)
            indices = self._indices(corners, resolution, table_size)
            corner_features = table(indices)
            weights = torch.where(
                offsets[None, :, :].bool(),
                frac[:, None, :],
                1.0 - frac[:, None, :],
            ).prod(dim=-1)
            features.append((corner_features * weights[..., None]).sum(dim=1))
        encoded = torch.cat(features, dim=-1)
        return encoded.reshape(*original_shape, self.output_dim)


class DirectionEncoder(nn.Module):
    def __init__(self, n_frequencies=4):
        super().__init__()
        self.n_frequencies = int(n_frequencies)
        self.output_dim = 3 * (1 + 2 * self.n_frequencies)
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(self.n_frequencies, dtype=torch.float32)
        )

    def forward(self, directions):
        directions = torch.nn.functional.normalize(directions, dim=-1)
        pieces = [directions]
        scaled = directions[..., None, :] * self.frequencies[:, None]
        pieces.extend([torch.sin(scaled).flatten(-2), torch.cos(scaled).flatten(-2)])
        return torch.cat(pieces, dim=-1)
