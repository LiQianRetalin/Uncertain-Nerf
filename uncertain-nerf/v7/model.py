import torch
from torch import nn
from torch.nn import functional as F

from .encoding import DirectionEncoder, make_hash_encoder


class RadianceFieldV7(nn.Module):
    """Shared radiance field with a gradient-isolated uncertainty head."""

    def __init__(self, aabb, hidden_dim=64, hidden_layers=2, hash_levels=16,
                 hash_min_resolution=16, hash_max_resolution=2048, hash_features=2,
                 hash_log2_size=19, direction_frequencies=4, appearance_dim=0,
                 density_bias=-1.0, hash_backend="torch", mip_enabled=True):
        super().__init__()
        self.position_encoder = make_hash_encoder(
            backend=hash_backend, aabb=aabb, n_levels=hash_levels,
            min_resolution=hash_min_resolution, max_resolution=hash_max_resolution,
            features_per_level=hash_features, log2_hashmap_size=hash_log2_size)
        self.direction_encoder = DirectionEncoder(direction_frequencies)
        self.appearance_dim = int(appearance_dim)
        self.density_bias = float(density_bias)
        self.mip_enabled = bool(mip_enabled)
        layers = []
        in_dim = self.position_encoder.output_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=False)])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=False),
            nn.Linear(hidden_dim // 2, 1))
        color_in = hidden_dim + self.direction_encoder.output_dim + self.appearance_dim
        self.color_head = nn.Sequential(nn.Linear(color_in, hidden_dim), nn.ReLU(inplace=False),
                                        nn.Linear(hidden_dim, 3))
        nn.init.constant_(self.density_head.bias, 0.0)
        nn.init.constant_(self.uncertainty_head[-1].bias, -2.0)

    @property
    def aabb(self):
        return self.position_encoder.aabb

    def forward(self, points, directions, appearance=None, footprint=None):
        encoded = self.position_encoder(points, footprint if self.mip_enabled else None)
        hidden = self.trunk(encoded)
        direction_features = self.direction_encoder(directions)
        if appearance is None:
            appearance = hidden.new_zeros(*hidden.shape[:-1], self.appearance_dim)
        rgb_raw = self.color_head(torch.cat(
            [self.feature_head(hidden), direction_features, appearance], dim=-1))
        sigma_raw = self.density_head(hidden)
        # This detach is the hard V7 gradient boundary: uncertainty losses cannot
        # update HashGrid, shared trunk, density, feature or RGB parameters.
        uncertainty_logit = self.uncertainty_head(hidden.detach())
        return torch.cat([rgb_raw, sigma_raw, uncertainty_logit], dim=-1)

    def activate_density(self, sigma_raw):
        return F.softplus(sigma_raw + self.density_bias)

    def density(self, points, footprint=None):
        encoded = self.position_encoder(points, footprint if self.mip_enabled else None)
        return self.activate_density(self.density_head(self.trunk(encoded)))

    def uncertainty(self, points, footprint=None):
        encoded = self.position_encoder(points, footprint if self.mip_enabled else None)
        hidden = self.trunk(encoded).detach()
        return torch.sigmoid(self.uncertainty_head(hidden))


__all__ = ["RadianceFieldV7"]
