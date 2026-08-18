import torch
from torch import nn


class OccupancyGrid(nn.Module):
    """EMA density grid used to compact network queries during ray marching."""

    def __init__(self, aabb, resolution=128, threshold=0.01, decay=0.95,
                 warmup_steps=256):
        super().__init__()
        self.resolution = int(resolution)
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.register_buffer("aabb", torch.as_tensor(aabb, dtype=torch.float32))
        self.register_buffer("values", torch.zeros(
            self.resolution, self.resolution, self.resolution))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.active = False

    def _indices(self, points):
        normalized = (points - self.aabb[0]) / (self.aabb[1] - self.aabb[0])
        valid = ((normalized >= 0.0) & (normalized <= 1.0)).all(dim=-1)
        indices = (normalized * self.resolution).long().clamp(0, self.resolution - 1)
        return indices, valid

    def occupied(self, points):
        if not self.active:
            return torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
        shape = points.shape[:-1]
        indices, valid = self._indices(points.reshape(-1, 3))
        values = self.values[indices[:, 0], indices[:, 1], indices[:, 2]]
        return (valid & (values > self.threshold)).reshape(shape)

    @torch.no_grad()
    def update(self, field, step, n_samples=262144, chunk=65536):
        device = self.aabb.device
        points = self.aabb[0] + torch.rand(int(n_samples), 3, device=device) * (
            self.aabb[1] - self.aabb[0])
        densities = []
        for start in range(0, len(points), int(chunk)):
            densities.append(field.density(points[start:start + int(chunk)]).reshape(-1))
        density = torch.cat(densities)
        voxel_diagonal = torch.linalg.vector_norm(
            (self.aabb[1] - self.aabb[0]) / self.resolution)
        occupancy = 1.0 - torch.exp(-density * voxel_diagonal)
        indices, _ = self._indices(points)
        flat = indices[:, 0] * self.resolution**2 + indices[:, 1] * self.resolution + indices[:, 2]
        self.values.mul_(self.decay)
        flat_values = self.values.view(-1)
        if hasattr(flat_values, "scatter_reduce_"):
            flat_values.scatter_reduce_(0, flat, occupancy, reduce="amax", include_self=True)
        else:  # pragma: no cover - only for old PyTorch builds
            for index, value in zip(flat.tolist(), occupancy.tolist()):
                flat_values[index] = max(float(flat_values[index]), value)
        self.updates.add_(1)
        self.active = int(step) >= self.warmup_steps

    def get_extra_state(self):
        return {"active": bool(self.active)}

    def set_extra_state(self, state):
        self.active = bool(state.get("active", False))


__all__ = ["OccupancyGrid"]
