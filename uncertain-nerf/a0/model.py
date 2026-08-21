"""Classic NeRF MLPs used by the A0 recovery experiment."""

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoder(nn.Module):
    """The sinusoidal encoder from the original NeRF implementation."""

    def __init__(self, input_dim=3, frequencies=10, include_input=True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.frequencies = int(frequencies)
        self.include_input = bool(include_input)
        bands = 2.0 ** torch.arange(self.frequencies, dtype=torch.float32)
        self.register_buffer("frequency_bands", bands, persistent=False)
        multiplier = (1 if self.include_input else 0) + 2 * self.frequencies
        self.output_dim = self.input_dim * multiplier

    def forward(self, values):
        pieces = [values] if self.include_input else []
        angles = values[..., None, :] * self.frequency_bands[:, None]
        # Original ordering: sin/cos are adjacent within every frequency.
        periodic = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-2)
        pieces.append(periodic.flatten(-3))
        return torch.cat(pieces, dim=-1)


class NeRFMLP(nn.Module):
    """Eight-layer NeRF field with the original view-dependent color head."""

    def __init__(self, point_dim, view_dim, depth=8, width=256,
                 skips=(4,), output_channels=5):
        super().__init__()
        self.depth = int(depth)
        self.width = int(width)
        self.point_dim = int(point_dim)
        self.view_dim = int(view_dim)
        self.skips = tuple(int(value) for value in skips)

        point_layers = []
        for layer in range(self.depth):
            input_dim = self.point_dim if layer == 0 else self.width
            if layer > 0 and (layer - 1) in self.skips:
                input_dim += self.point_dim
            point_layers.append(nn.Linear(input_dim, self.width))
        self.point_layers = nn.ModuleList(point_layers)
        self.density_head = nn.Linear(self.width, 1)
        self.feature_head = nn.Linear(self.width, self.width)
        self.view_layer = nn.Linear(self.width + self.view_dim, self.width // 2)
        self.color_head = nn.Linear(self.width // 2, 3)
        self.extra_head = (nn.Linear(self.width, int(output_channels) - 4)
                           if int(output_channels) > 4 else None)

    def forward(self, encoded_points, encoded_views):
        hidden = encoded_points
        for index, layer in enumerate(self.point_layers):
            hidden = F.relu(layer(hidden), inplace=False)
            if index in self.skips and index + 1 < len(self.point_layers):
                hidden = torch.cat([encoded_points, hidden], dim=-1)
        density = self.density_head(hidden)
        feature = self.feature_head(hidden)
        view_hidden = F.relu(
            self.view_layer(torch.cat([feature, encoded_views], dim=-1)),
            inplace=False)
        color = self.color_head(view_hidden)
        outputs = [color, density]
        if self.extra_head is not None:
            outputs.append(self.extra_head(hidden))
        return torch.cat(outputs, dim=-1)

    def density_parameters(self):
        yield from self.point_layers.parameters()
        yield from self.density_head.parameters()

    def color_parameters(self):
        yield from self.feature_head.parameters()
        yield from self.view_layer.parameters()
        yield from self.color_head.parameters()


class OriginalNeRF(nn.Module):
    """Matched coarse/fine classic NeRF with no uncertainty parameters."""

    def __init__(self, depth=8, width=256, fine_depth=8, fine_width=256,
                 point_frequencies=10, view_frequencies=4,
                 use_viewdirs=True, output_channels=5):
        super().__init__()
        if not use_viewdirs:
            raise ValueError("A0 requires the original NeRF view-direction branch")
        self.point_encoder = PositionalEncoder(3, point_frequencies)
        self.view_encoder = PositionalEncoder(3, view_frequencies)
        self.coarse = NeRFMLP(
            self.point_encoder.output_dim, self.view_encoder.output_dim,
            depth=depth, width=width, output_channels=output_channels)
        self.fine = NeRFMLP(
            self.point_encoder.output_dim, self.view_encoder.output_dim,
            depth=fine_depth, width=fine_width,
            output_channels=output_channels)

    def query(self, field, points, viewdirs, chunk=65536):
        shape = points.shape[:-1]
        directions = viewdirs[:, None, :].expand_as(points)
        flat_points = points.reshape(-1, 3)
        flat_directions = directions.reshape(-1, 3)
        outputs = []
        for start in range(0, len(flat_points), int(chunk)):
            end = min(start + int(chunk), len(flat_points))
            outputs.append(field(
                self.point_encoder(flat_points[start:end]),
                self.view_encoder(flat_directions[start:end])))
        return torch.cat(outputs, dim=0).reshape(*shape, -1)

    def density_parameters(self):
        yield from self.coarse.density_parameters()
        yield from self.fine.density_parameters()

    def color_parameters(self):
        yield from self.coarse.color_parameters()
        yield from self.fine.color_parameters()

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def global_gradient_norm(parameters):
    squares = [parameter.grad.detach().float().square().sum()
               for parameter in parameters if parameter.grad is not None]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


__all__ = [
    "NeRFMLP", "OriginalNeRF", "PositionalEncoder", "global_gradient_norm",
]
