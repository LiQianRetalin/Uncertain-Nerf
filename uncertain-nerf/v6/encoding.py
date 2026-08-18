"""Hash-grid encoders used by V6.

The pure PyTorch implementation keeps CPU tests and non-CUDA machines usable.
On Ubuntu/CUDA, ``backend=tcnn`` selects tiny-cuda-nn's fused hash encoder.
"""

import math

import torch
from torch import nn

from v5.encoding import DirectionEncoder, HashGridEncoder as TorchHashGridEncoder


class TinyCudaNNHashGridEncoder(nn.Module):
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
        if not torch.cuda.is_available():
            raise RuntimeError("The tcnn hash backend requires a CUDA device")
        try:
            import tinycudann as tcnn
        except ImportError as exc:
            raise RuntimeError(
                "backend=tcnn requires tiny-cuda-nn. Follow V6_GUIDE.md section 4."
            ) from exc
        aabb = torch.as_tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.output_dim = int(n_levels) * int(features_per_level)
        per_level_scale = math.exp(
            (math.log(max_resolution) - math.log(min_resolution))
            / max(int(n_levels) - 1, 1)
        )
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": int(n_levels),
                "n_features_per_level": int(features_per_level),
                "log2_hashmap_size": int(log2_hashmap_size),
                "base_resolution": int(min_resolution),
                "per_level_scale": float(per_level_scale),
                "interpolation": "Smoothstep",
            },
        )

    def normalize(self, points):
        return (points - self.aabb[0]) / (self.aabb[1] - self.aabb[0])

    def forward(self, points):
        shape = points.shape[:-1]
        normalized = self.normalize(points.reshape(-1, 3)).clamp(0.0, 1.0)
        encoded = self.encoder(normalized)
        return encoded.reshape(*shape, self.output_dim).to(points.dtype)


def make_hash_encoder(backend="torch", **kwargs):
    backend = str(backend).lower()
    if backend == "torch":
        return TorchHashGridEncoder(**kwargs)
    if backend == "tcnn":
        return TinyCudaNNHashGridEncoder(**kwargs)
    raise ValueError(f"Unknown hash backend {backend!r}; expected 'torch' or 'tcnn'")


__all__ = [
    "DirectionEncoder",
    "TorchHashGridEncoder",
    "TinyCudaNNHashGridEncoder",
    "make_hash_encoder",
]
