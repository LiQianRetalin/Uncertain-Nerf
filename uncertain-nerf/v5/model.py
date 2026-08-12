import torch
from torch import nn
from torch.nn import functional as F

from .encoding import DirectionEncoder, HashGridEncoder


class RadianceFieldV5(nn.Module):
    """Hash-encoded radiance field with density and uncertainty heads."""

    def __init__(
        self,
        aabb,
        hidden_dim=64,
        hidden_layers=2,
        dropout=0.1,
        hash_levels=16,
        hash_min_resolution=16,
        hash_max_resolution=2048,
        hash_features=2,
        hash_log2_size=19,
        direction_frequencies=4,
    ):
        super().__init__()
        self.position_encoder = HashGridEncoder(
            aabb=aabb,
            n_levels=hash_levels,
            min_resolution=hash_min_resolution,
            max_resolution=hash_max_resolution,
            features_per_level=hash_features,
            log2_hashmap_size=hash_log2_size,
        )
        self.direction_encoder = DirectionEncoder(direction_frequencies)
        layers = []
        in_dim = self.position_encoder.output_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=False)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.density_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim + self.direction_encoder.output_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 3),
        )

    @property
    def aabb(self):
        return self.position_encoder.aabb

    def forward(self, points, directions):
        encoded = self.position_encoder(points)
        hidden = self.trunk(encoded)
        sigma_raw = self.density_head(hidden)
        uncertainty_logit = self.uncertainty_head(hidden)
        direction_features = self.direction_encoder(directions)
        rgb_raw = self.color_head(
            torch.cat([self.feature_head(hidden), direction_features], dim=-1)
        )
        return torch.cat([rgb_raw, sigma_raw, uncertainty_logit], dim=-1)

    def density_uncertainty(self, points):
        hidden = self.trunk(self.position_encoder(points))
        sigma = F.relu(self.density_head(hidden))
        uncertainty = torch.sigmoid(self.uncertainty_head(hidden))
        return sigma, uncertainty
