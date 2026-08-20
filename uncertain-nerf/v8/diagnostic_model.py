"""Matched reconstruction fields for the V8 baseline/V7.5 gate."""

import torch
from torch import nn
from torch.nn import functional as F

from v7.encoding import DirectionEncoder, make_hash_encoder


class DiagnosticRadianceField(nn.Module):
    """V7 reconstruction trunk with an optional, gradient-isolated UQ head.

    ``mode=baseline`` has no learnable uncertainty parameters.  A constant
    sentinel logit is appended only because the already-tested V7 renderer has
    a five-channel internal interface; it is never used by reconstruction.

    ``mode=v7_5`` adds a side head whose input is detached.  RGB and density are
    therefore numerically independent of both the head output and its loss.
    Reconstruction modules are initialized before the optional head so matched
    seeds produce identical reconstruction parameters in both modes.
    """

    VALID_MODES = ("baseline", "v7_5")

    def __init__(self, aabb, mode="baseline", hidden_dim=64, hidden_layers=2,
                 hash_levels=16, hash_min_resolution=16,
                 hash_max_resolution=2048, hash_features=2,
                 hash_log2_size=19, direction_frequencies=4,
                 density_bias=-1.0, hash_backend="torch", mip_enabled=True):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        self.mode = mode
        self.appearance_dim = 0
        self.density_bias = float(density_bias)
        self.mip_enabled = bool(mip_enabled)
        self.position_encoder = make_hash_encoder(
            backend=hash_backend, aabb=aabb, n_levels=hash_levels,
            min_resolution=hash_min_resolution,
            max_resolution=hash_max_resolution,
            features_per_level=hash_features,
            log2_hashmap_size=hash_log2_size)
        self.direction_encoder = DirectionEncoder(direction_frequencies)
        layers = []
        in_dim = self.position_encoder.output_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=False)])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        color_in = hidden_dim + self.direction_encoder.output_dim
        self.color_head = nn.Sequential(
            nn.Linear(color_in, hidden_dim), nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 3))
        nn.init.constant_(self.density_head.bias, 0.0)

        if mode == "v7_5":
            # Building the optional side head must not advance the global RNG:
            # matched baseline/V7.5 runs must draw exactly the same training
            # rays after identical reconstruction parameters are initialized.
            with torch.random.fork_rng(devices=[]):
                self.uncertainty_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(inplace=False),
                    nn.Linear(hidden_dim // 2, 1))
                nn.init.constant_(self.uncertainty_head[-1].bias, -2.0)
        else:
            self.uncertainty_head = None

    @property
    def aabb(self):
        return self.position_encoder.aabb

    def reconstruction_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("uncertainty_head."):
                yield parameter

    def uncertainty_parameters(self):
        if self.uncertainty_head is not None:
            yield from self.uncertainty_head.parameters()

    def reconstruction_named_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("uncertainty_head."):
                yield name, parameter

    def forward(self, points, directions, appearance=None, footprint=None):
        del appearance
        encoded = self.position_encoder(
            points, footprint if self.mip_enabled else None)
        hidden = self.trunk(encoded)
        direction_features = self.direction_encoder(directions)
        rgb_raw = self.color_head(torch.cat(
            [self.feature_head(hidden), direction_features], dim=-1))
        sigma_raw = self.density_head(hidden)
        if self.uncertainty_head is None:
            # Renderer compatibility only: sigmoid(-30) is effectively zero.
            uncertainty_logit = sigma_raw.detach().new_full(sigma_raw.shape, -30.0)
        else:
            # Mathematical V7.5 boundary: d L_UQ / d theta_recon = 0.
            uncertainty_logit = self.uncertainty_head(hidden.detach())
        return torch.cat([rgb_raw, sigma_raw, uncertainty_logit], dim=-1)

    def activate_density(self, sigma_raw):
        return F.softplus(sigma_raw + self.density_bias)

    def density(self, points, footprint=None):
        encoded = self.position_encoder(
            points, footprint if self.mip_enabled else None)
        return self.activate_density(self.density_head(self.trunk(encoded)))

    def uncertainty(self, points, footprint=None):
        if self.uncertainty_head is None:
            return points.new_zeros(points.shape[:-1])
        encoded = self.position_encoder(
            points, footprint if self.mip_enabled else None)
        hidden = self.trunk(encoded).detach()
        return torch.sigmoid(self.uncertainty_head(hidden)).squeeze(-1)


__all__ = ["DiagnosticRadianceField"]
