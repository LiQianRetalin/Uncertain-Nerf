"""LLFF loader and uniform ray sampler for the original A0 baseline."""

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import torch

from load_llff import load_llff_data
from .rendering import pixels_to_rays


@dataclass
class A0Scene:
    images: torch.Tensor
    poses: torch.Tensor
    render_poses: torch.Tensor
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    height: int
    width: int
    focal: float
    bounds: torch.Tensor
    fingerprint: str


def split_indices(image_count, holdout=8, val_every=8):
    all_indices = np.arange(int(image_count), dtype=np.int64)
    test = all_indices[::int(holdout)] if int(holdout) > 0 else np.array([], dtype=np.int64)
    test_set = set(map(int, test))
    original_train = np.asarray(
        [index for index in all_indices if int(index) not in test_set],
        dtype=np.int64)
    if int(val_every) > 0 and len(original_train) >= 4:
        val = original_train[::int(val_every)]
    else:
        val = np.array([], dtype=np.int64)
    val_set = set(map(int, val))
    train = np.asarray(
        [index for index in original_train if int(index) not in val_set],
        dtype=np.int64)
    if not len(train):
        raise ValueError("A0 split produced no training images")
    return train, val, test


def _fingerprint(datadir, factor, train, val, test):
    digest = hashlib.sha256()
    poses_path = os.path.join(datadir, "poses_bounds.npy")
    with open(poses_path, "rb") as handle:
        digest.update(handle.read())
    image_dir = os.path.join(datadir, f"images_{int(factor)}")
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(datadir, "images")
    for name in sorted(os.listdir(image_dir)):
        if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png"):
            digest.update(name.encode("utf-8"))
    digest.update(str(int(factor)).encode())
    for values in (train, val, test):
        digest.update(np.asarray(values, dtype=np.int64).tobytes())
    return digest.hexdigest()


def load_a0_scene(datadir, factor=2, holdout=8, val_every=8, device="cpu"):
    images, poses, bounds, render_poses, _ = load_llff_data(
        datadir, factor=int(factor), recenter=True, bd_factor=0.75,
        spherify=False)
    height, width, focal = poses[0, :3, -1]
    train, val, test = split_indices(len(images), holdout, val_every)
    fingerprint = _fingerprint(datadir, factor, train, val, test)
    return A0Scene(
        images=torch.from_numpy(images[..., :3]).to(device=device, dtype=torch.float32),
        poses=torch.from_numpy(poses[:, :3, :4]).to(device=device, dtype=torch.float32),
        render_poses=torch.from_numpy(render_poses[:, :3, :4]).to(
            device=device, dtype=torch.float32),
        train_indices=train, val_indices=val, test_indices=test,
        height=int(height), width=int(width), focal=float(focal),
        bounds=torch.from_numpy(bounds).to(device=device, dtype=torch.float32),
        fingerprint=fingerprint)


class UniformRaySampler:
    """Uniform pixels over all A0 training images; no priors or UQ feedback."""

    def __init__(self, scene, batch_size=1024):
        self.scene = scene
        self.batch_size = int(batch_size)

    def sample(self):
        scene = self.scene
        device = scene.images.device
        train = torch.as_tensor(scene.train_indices, device=device)
        image_ids = train[torch.randint(
            len(train), (self.batch_size,), device=device)]
        ys = torch.randint(
            scene.height, (self.batch_size,), device=device)
        xs = torch.randint(
            scene.width, (self.batch_size,), device=device)
        rays_o, rays_d = pixels_to_rays(
            scene.poses, image_ids, ys, xs,
            scene.height, scene.width, scene.focal)
        return {
            "rays_o": rays_o, "rays_d": rays_d,
            "target": scene.images[image_ids, ys, xs],
            "image_ids": image_ids, "ys": ys, "xs": xs,
        }


__all__ = [
    "A0Scene", "UniformRaySampler", "load_a0_scene", "split_indices",
]
