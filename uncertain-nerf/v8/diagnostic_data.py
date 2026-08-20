"""UQ-independent sampling shared exactly by baseline and V7.5."""

import math

import torch

from v5.data import pixels_to_rays
from v6.data import SceneData, load_scene
from v7.data import _edge_maps, pixel_ray_radii


class FixedDiagnosticRaySampler:
    """Mix COLMAP-prior, Sobel-edge and uniform rays without UQ feedback."""

    def __init__(self, scene, batch_size=1024, prior_ratio=0.25,
                 edge_ratio=0.25, probability_floor=0.05):
        self.scene = scene
        self.batch_size = int(batch_size)
        self.prior_ratio = float(prior_ratio)
        self.edge_ratio = float(edge_ratio)
        self.probability_floor = float(probability_floor)
        if not 0 <= self.prior_ratio <= 1 or not 0 <= self.edge_ratio <= 1:
            raise ValueError("prior_ratio and edge_ratio must be in [0, 1]")
        if self.prior_ratio + self.edge_ratio > 1:
            raise ValueError("prior_ratio + edge_ratio must be at most 1")
        self.edge_maps = _edge_maps(scene.images).detach()

    def _uniform_pixels(self, count):
        device = self.scene.images.device
        train = torch.as_tensor(self.scene.train_indices, device=device)
        height, width = self.scene.images.shape[1:3]
        image_ids = train[torch.randint(len(train), (count,), device=device)]
        return (image_ids, torch.randint(height, (count,), device=device),
                torch.randint(width, (count,), device=device))

    def _edge_pixels(self, count):
        device = self.scene.images.device
        train = torch.as_tensor(self.scene.train_indices, device=device)
        image_ids = train[torch.randint(len(train), (count,), device=device)]
        height, width = self.edge_maps.shape[1:]
        ys = torch.empty(count, dtype=torch.long, device=device)
        xs = torch.empty(count, dtype=torch.long, device=device)
        for image_id in image_ids.unique():
            chosen = torch.where(image_ids == image_id)[0]
            probabilities = (self.edge_maps[image_id].reshape(-1).clamp_min(0.0)
                             + self.probability_floor)
            pixels = torch.multinomial(probabilities, len(chosen), replacement=True)
            ys[chosen], xs[chosen] = pixels // width, pixels % width
        return image_ids, ys, xs

    def sample(self):
        scene, device = self.scene, self.scene.images.device
        n_prior = min(int(math.ceil(self.batch_size * self.prior_ratio)),
                      len(scene.prior_depth))
        n_edge = int(round(self.batch_size * self.edge_ratio))
        n_uniform = self.batch_size - n_prior - n_edge
        prior_indices = torch.randint(
            len(scene.prior_depth), (n_prior,), device=device)
        groups = [(scene.prior_image[prior_indices], scene.prior_y[prior_indices],
                   scene.prior_x[prior_indices]), self._edge_pixels(n_edge),
                  self._uniform_pixels(n_uniform)]
        image_ids = torch.cat([group[0] for group in groups])
        ys = torch.cat([group[1] for group in groups])
        xs = torch.cat([group[2] for group in groups])
        prior_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        prior_depth = torch.zeros(self.batch_size, device=device)
        prior_mask[:n_prior] = True
        prior_depth[:n_prior] = scene.prior_depth[prior_indices]
        order = torch.randperm(self.batch_size, device=device)
        image_ids, ys, xs = image_ids[order], ys[order], xs[order]
        prior_mask, prior_depth = prior_mask[order], prior_depth[order]
        rays_o, rays_d = pixels_to_rays(
            scene.poses, scene.intrinsics, image_ids, ys, xs,
            radial_distortion=scene.radial_distortion)
        return {"rays_o": rays_o, "rays_d": rays_d,
                "ray_radii": pixel_ray_radii(scene.intrinsics, image_ids),
                "target": scene.images[image_ids, ys, xs, :3],
                "image_ids": image_ids, "ys": ys, "xs": xs,
                "prior_mask": prior_mask, "prior_depth": prior_depth}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        del state


__all__ = ["FixedDiagnosticRaySampler", "SceneData", "load_scene"]
