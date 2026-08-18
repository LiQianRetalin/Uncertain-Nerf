"""LLFF loading plus edge/uncertainty-adaptive ray sampling for V7."""

import math

import torch
from torch.nn import functional as F

from v6.data import SceneData, load_scene
from v5.data import pixels_to_rays


def pixel_ray_radii(intrinsics, image_ids):
    matrix = intrinsics[image_ids]
    return 0.5 * torch.sqrt(matrix[:, 0, 0].reciprocal().square()
                            + matrix[:, 1, 1].reciprocal().square())


def _edge_maps(images):
    gray = (images[..., :3] * images.new_tensor([0.299, 0.587, 0.114])).sum(dim=-1)
    gray = gray[:, None]
    kx = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edges = torch.sqrt(gx.square() + gy.square()).squeeze(1)
    return edges / edges.flatten(1).mean(dim=-1)[:, None, None].clamp_min(1.0e-6)


class AdaptiveRaySampler:
    def __init__(self, scene, batch_size=1024, prior_ratio=0.3, edge_ratio=0.25,
                 uncertainty_ratio=0.25, uncertainty_ema=0.1,
                 probability_floor=0.05):
        self.scene = scene
        self.batch_size = int(batch_size)
        self.prior_ratio = float(prior_ratio)
        self.edge_ratio = float(edge_ratio)
        self.uncertainty_ratio = float(uncertainty_ratio)
        self.uncertainty_ema = float(uncertainty_ema)
        self.probability_floor = float(probability_floor)
        if self.prior_ratio + self.edge_ratio + self.uncertainty_ratio > 1.0 + 1.0e-8:
            raise ValueError("prior/edge/uncertainty sampling ratios must sum to at most 1")
        self.edge_maps = _edge_maps(scene.images).detach()
        self.uncertainty_maps = torch.ones_like(self.edge_maps)

    def _weighted_pixels(self, count, maps):
        device = self.scene.images.device
        train = torch.as_tensor(self.scene.train_indices, device=device)
        image_ids = train[torch.randint(len(train), (count,), device=device)]
        height, width = maps.shape[1:]
        ys = torch.empty(count, dtype=torch.long, device=device)
        xs = torch.empty(count, dtype=torch.long, device=device)
        for image_id in image_ids.unique():
            chosen = torch.where(image_ids == image_id)[0]
            probabilities = maps[image_id].reshape(-1).clamp_min(0.0) + self.probability_floor
            pixels = torch.multinomial(probabilities, len(chosen), replacement=True)
            ys[chosen], xs[chosen] = pixels // width, pixels % width
        return image_ids, ys, xs

    def _uniform_pixels(self, count):
        device = self.scene.images.device
        train = torch.as_tensor(self.scene.train_indices, device=device)
        height, width = self.scene.images.shape[1:3]
        image_ids = train[torch.randint(len(train), (count,), device=device)]
        return (image_ids, torch.randint(height, (count,), device=device),
                torch.randint(width, (count,), device=device))

    def sample(self):
        scene, device = self.scene, self.scene.images.device
        n_prior = min(int(math.ceil(self.batch_size * self.prior_ratio)),
                      len(scene.prior_depth))
        n_edge = int(round(self.batch_size * self.edge_ratio))
        n_uncertainty = int(round(self.batch_size * self.uncertainty_ratio))
        n_uniform = self.batch_size - n_prior - n_edge - n_uncertainty
        if n_uniform < 0:
            n_edge = max(0, n_edge + n_uniform)
            n_uniform = 0
        prior_indices = torch.randint(len(scene.prior_depth), (n_prior,), device=device)
        groups = [(scene.prior_image[prior_indices], scene.prior_y[prior_indices],
                   scene.prior_x[prior_indices])]
        groups.append(self._weighted_pixels(n_edge, self.edge_maps))
        groups.append(self._weighted_pixels(n_uncertainty, self.uncertainty_maps))
        groups.append(self._uniform_pixels(n_uniform))
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
        rays_o, rays_d = pixels_to_rays(scene.poses, scene.intrinsics, image_ids, ys, xs,
                                        radial_distortion=scene.radial_distortion)
        return {"rays_o": rays_o, "rays_d": rays_d,
                "ray_radii": pixel_ray_radii(scene.intrinsics, image_ids),
                "target": scene.images[image_ids, ys, xs, :3], "image_ids": image_ids,
                "ys": ys, "xs": xs, "prior_mask": prior_mask,
                "prior_depth": prior_depth}

    @torch.no_grad()
    def update_uncertainty(self, image_ids, ys, xs, uncertainty):
        old = self.uncertainty_maps[image_ids, ys, xs]
        values = (1.0 - self.uncertainty_ema) * old + self.uncertainty_ema * uncertainty.detach()
        self.uncertainty_maps[image_ids, ys, xs] = values.clamp(0.0, 1.0)

    def state_dict(self):
        return {"uncertainty_maps": self.uncertainty_maps.detach().cpu()}

    def load_state_dict(self, state):
        values = state.get("uncertainty_maps")
        if values is not None and tuple(values.shape) == tuple(self.uncertainty_maps.shape):
            self.uncertainty_maps.copy_(values.to(self.uncertainty_maps.device))


__all__ = ["AdaptiveRaySampler", "SceneData", "load_scene", "pixel_ray_radii",
           "pixels_to_rays"]
