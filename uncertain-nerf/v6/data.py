"""V6 scene loading reuses the tested COLMAP/LLFF camera pipeline from V5."""

import numpy as np
import torch

from v5.data import RaySampler, SceneData, _build_source_views, pixels_to_rays
from v5.data import load_scene as _load_scene_v5


def load_scene(
    datadir,
    factor=4,
    holdout=8,
    n_source_views=4,
    device="cpu",
    aabb_scale=1.25,
    val_every=8,
):
    scene = _load_scene_v5(
        datadir,
        factor=factor,
        holdout=holdout,
        n_source_views=n_source_views,
        device=device,
    )
    scale = float(aabb_scale)
    if scale < 1.0:
        raise ValueError("aabb_scale must be at least 1.0")
    center = 0.5 * (scene.aabb[0] + scene.aabb[1])
    half_extent = 0.5 * (scene.aabb[1] - scene.aabb[0]) * scale
    scene.aabb = center[None] + scene.aabb.new_tensor([[-1.0], [1.0]]) * half_extent[None]
    original_train = np.asarray(scene.train_indices, dtype=np.int64)
    if int(val_every) > 0 and len(original_train) >= 4:
        val = original_train[:: int(val_every)]
        if len(val) == len(original_train):
            val = val[:1]
        val_set = set(map(int, val))
        train = np.asarray([i for i in original_train if int(i) not in val_set], dtype=np.int64)
        scene.train_indices = train
        scene.val_indices = np.asarray(val, dtype=np.int64)
        scene.source_views = _build_source_views(
            scene.poses, scene.train_indices, n_source_views
        ).to(scene.images.device)
        keep_prior = torch.zeros_like(scene.prior_image, dtype=torch.bool)
        for image_index in scene.train_indices:
            keep_prior |= scene.prior_image == int(image_index)
        scene.prior_image = scene.prior_image[keep_prior]
        scene.prior_y = scene.prior_y[keep_prior]
        scene.prior_x = scene.prior_x[keep_prior]
        scene.prior_depth = scene.prior_depth[keep_prior]
        if scene.prior_depth.numel() == 0:
            raise ValueError("No COLMAP priors remain after the V6 validation split")
    else:
        # Tiny synthetic tests cannot afford a third split; real paper runs must use val_every>0.
        scene.val_indices = scene.test_indices.copy()
    train_ids = torch.as_tensor(scene.train_indices, device=scene.images.device)
    train_images = scene.images[train_ids, ..., :3]
    scene.image_mean = scene.images[..., :3].mean(dim=(1, 2))
    scene.image_std = scene.images[..., :3].std(dim=(1, 2)).clamp_min(0.05)
    scene.training_color_mean = train_images.mean(dim=(0, 1, 2))
    return scene


__all__ = ["RaySampler", "SceneData", "load_scene", "pixels_to_rays"]
