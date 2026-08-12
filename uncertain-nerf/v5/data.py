import hashlib
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from load_llff import load_llff_data
from .camera import undistort_simple_radial
from .colmap import read_model


@dataclass
class SceneData:
    images: torch.Tensor
    poses: torch.Tensor
    intrinsics: torch.Tensor
    radial_distortion: torch.Tensor
    render_poses: torch.Tensor
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    near: float
    far: float
    aabb: torch.Tensor
    source_views: torch.Tensor
    prior_image: torch.Tensor
    prior_y: torch.Tensor
    prior_x: torch.Tensor
    prior_depth: torch.Tensor
    fingerprint: str
    hwf: tuple


def _image_names(datadir):
    image_dir = os.path.join(datadir, "images")
    names = sorted(
        name for name in os.listdir(image_dir)
        if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png")
    )
    if not names:
        raise ValueError(f"No images found in {image_dir}")
    return names


def _fingerprint(paths, image_names):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    for name in image_names:
        digest.update(name.encode("utf-8"))
    return digest.hexdigest()


def _build_source_views(poses, train_indices, n_views):
    centers = poses[:, :3, 3].cpu()
    forwards = -poses[:, :3, 2].cpu()
    forwards = torch.nn.functional.normalize(forwards, dim=-1)
    source = torch.full((poses.shape[0], n_views), -1, dtype=torch.long)
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long)
    baseline_scale = torch.pdist(centers[train_tensor]).median().clamp_min(1.0e-6) if len(train_indices) > 1 else torch.tensor(1.0)
    for ref in range(poses.shape[0]):
        candidates = train_tensor[train_tensor != ref]
        if not len(candidates):
            continue
        distance = torch.linalg.norm(centers[candidates] - centers[ref], dim=-1) / baseline_scale
        angle = 1.0 - (forwards[candidates] * forwards[ref]).sum(dim=-1).clamp(-1.0, 1.0)
        score = distance + 0.25 * angle
        valid = distance > 1.0e-4
        candidates = candidates[valid]
        score = score[valid]
        order = torch.argsort(score)[:n_views]
        source[ref, : len(order)] = candidates[order]
    return source


def _transform_points(points, transform):
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=-1)
    return (transform @ homogeneous.T).T[:, :3]


def load_scene(datadir, factor=4, holdout=8, n_source_views=8, device="cpu"):
    loaded = load_llff_data(
        datadir, factor=factor, recenter=True, bd_factor=0.75,
        spherify=False, return_meta=True,
    )
    images, poses, bounds, render_poses, _, meta = loaded
    names = _image_names(datadir)
    if len(names) != len(images):
        raise ValueError("LLFF image count does not match poses_bounds.npy")
    model_dir = os.path.join(datadir, "sparse", "0")
    cameras, colmap_images, points = read_model(model_dir)
    by_name = {os.path.basename(image.name): image for image in colmap_images.values()}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"COLMAP image names do not match images/: {missing[:5]}")
    height, width = images.shape[1:3]
    intrinsics = []
    radial_distortion = []
    for name in names:
        image = by_name[name]
        camera = cameras[image.camera_id]
        matrix = camera.intrinsic_matrix().copy()
        radial_distortion.append(camera.radial_distortion())
        matrix[0, :] /= factor
        matrix[1, :] /= factor
        if abs(camera.width / factor - width) > 1.5 or abs(camera.height / factor - height) > 1.5:
            raise ValueError(f"COLMAP resolution for {name} does not match LLFF factor {factor}")
        intrinsics.append(matrix)
    intrinsics = np.stack(intrinsics).astype(np.float32)
    radial_distortion = np.asarray(radial_distortion, dtype=np.float32)
    colmap_centers = np.stack([by_name[name].center() for name in names])
    transformed_centers = _transform_points(colmap_centers, meta["world_transform"])
    center_error = np.linalg.norm(transformed_centers - poses[:, :3, 3], axis=-1)
    center_scale = max(float(np.linalg.norm(np.ptp(poses[:, :3, 3], axis=0))), 1.0)
    if float(center_error.max()) > 1.0e-3 * center_scale:
        raise ValueError(
            "poses_bounds.npy and sparse/0 do not describe the same COLMAP reconstruction "
            f"(maximum camera-center error {center_error.max():.6g})"
        )
    name_to_index = {name: i for i, name in enumerate(names)}
    prior = {}
    accepted_points = []
    for name in names:
        image = by_name[name]
        rotation = image.rotation()
        for xy, point_id in zip(image.xys, image.point3D_ids):
            if point_id < 0 or point_id not in points:
                continue
            point = points[int(point_id)]
            if point.error > 2.0 or len(point.image_ids) < 3:
                continue
            depth = float((rotation @ point.xyz + image.tvec)[2]) * meta["scene_scale"]
            if depth <= 0:
                continue
            x = int(round(xy[0] / factor))
            y = int(round(xy[1] / factor))
            if not (0 <= x < width and 0 <= y < height):
                continue
            key = (name_to_index[name], y, x)
            if key not in prior or depth < prior[key]:
                prior[key] = depth
            accepted_points.append(point.xyz)
    if not prior:
        raise ValueError("No valid COLMAP sparse depth priors after filtering")
    transformed = _transform_points(np.unique(np.asarray(accepted_points), axis=0), meta["world_transform"])
    low, high = np.percentile(transformed, [1.0, 99.0], axis=0)
    extent = np.maximum(high - low, 1.0e-3)
    aabb = np.stack([low - 0.1 * extent, high + 0.1 * extent]).astype(np.float32)
    poses_t = torch.from_numpy(poses[:, :3, :4].astype(np.float32))
    test = np.arange(len(images))[::holdout] if holdout > 0 else np.array([], dtype=np.int64)
    train = np.asarray([i for i in range(len(images)) if i not in set(test)], dtype=np.int64)
    val = test.copy()
    source_views = _build_source_views(poses_t, train, n_source_views)
    prior_items = sorted((img, y, x, depth) for (img, y, x), depth in prior.items() if img in set(train))
    if not prior_items:
        raise ValueError("No COLMAP priors belong to the LLFF training split")
    prior_array = np.asarray(prior_items, dtype=np.float64)
    model_files = [
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name in ("cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt")
    ]
    fingerprint = _fingerprint([os.path.join(datadir, "poses_bounds.npy")] + sorted(model_files), names)
    return SceneData(
        images=torch.from_numpy(images.astype(np.float32)).to(device),
        poses=poses_t.to(device),
        intrinsics=torch.from_numpy(intrinsics).to(device),
        radial_distortion=torch.from_numpy(radial_distortion).to(device),
        render_poses=torch.from_numpy(render_poses[:, :3, :4].astype(np.float32)).to(device),
        train_indices=train,
        val_indices=val,
        test_indices=test,
        near=float(bounds.min() * 0.9),
        far=float(bounds.max()),
        aabb=torch.from_numpy(aabb).to(device),
        source_views=source_views.to(device),
        prior_image=torch.from_numpy(prior_array[:, 0].astype(np.int64)).to(device),
        prior_y=torch.from_numpy(prior_array[:, 1].astype(np.int64)).to(device),
        prior_x=torch.from_numpy(prior_array[:, 2].astype(np.int64)).to(device),
        prior_depth=torch.from_numpy(prior_array[:, 3].astype(np.float32)).to(device),
        fingerprint=fingerprint,
        hwf=(height, width, float(np.median(intrinsics[:, 0, 0]))),
    )


def pixels_to_rays(poses, intrinsics, image_ids, ys, xs, radial_distortion=None):
    pose = poses[image_ids]
    matrix = intrinsics[image_ids]
    distorted_xy = torch.stack(
        [
            (xs.to(dtype=matrix.dtype) - matrix[:, 0, 2]) / matrix[:, 0, 0],
            (ys.to(dtype=matrix.dtype) - matrix[:, 1, 2]) / matrix[:, 1, 1],
        ],
        dim=-1,
    )
    if radial_distortion is None:
        radial_k = distorted_xy.new_zeros(())
    else:
        radial_distortion = torch.as_tensor(
            radial_distortion, dtype=distorted_xy.dtype, device=distorted_xy.device
        )
        radial_k = (
            radial_distortion
            if radial_distortion.ndim == 0
            else radial_distortion[image_ids]
        )
    normalized_xy = undistort_simple_radial(distorted_xy, radial_k)
    directions = torch.stack(
        [
            normalized_xy[:, 0],
            -normalized_xy[:, 1],
            -torch.ones_like(normalized_xy[:, 0]),
        ],
        dim=-1,
    )
    rays_d = torch.einsum("bij,bj->bi", pose[:, :3, :3], directions)
    rays_o = pose[:, :3, 3]
    return rays_o, rays_d


class RaySampler:
    def __init__(self, scene, batch_size=1024, prior_ratio=0.3):
        self.scene = scene
        self.batch_size = int(batch_size)
        self.prior_ratio = float(prior_ratio)

    def sample(self):
        scene = self.scene
        device = scene.images.device
        n_prior = min(int(math.ceil(self.batch_size * self.prior_ratio)), len(scene.prior_depth))
        n_random = self.batch_size - n_prior
        prior_indices = torch.randint(len(scene.prior_depth), (n_prior,), device=device)
        prior_images = scene.prior_image[prior_indices]
        prior_y = scene.prior_y[prior_indices]
        prior_x = scene.prior_x[prior_indices]
        train = torch.as_tensor(scene.train_indices, device=device)
        random_images = train[torch.randint(len(train), (n_random,), device=device)]
        height, width = scene.images.shape[1:3]
        random_y = torch.randint(height, (n_random,), device=device)
        random_x = torch.randint(width, (n_random,), device=device)
        image_ids = torch.cat([prior_images, random_images])
        ys = torch.cat([prior_y, random_y])
        xs = torch.cat([prior_x, random_x])
        order = torch.randperm(self.batch_size, device=device)
        image_ids, ys, xs = image_ids[order], ys[order], xs[order]
        prior_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        prior_depth = torch.zeros(self.batch_size, dtype=torch.float32, device=device)
        prior_mask[:n_prior] = True
        prior_depth[:n_prior] = scene.prior_depth[prior_indices]
        prior_mask, prior_depth = prior_mask[order], prior_depth[order]
        rays_o, rays_d = pixels_to_rays(
            scene.poses,
            scene.intrinsics,
            image_ids,
            ys,
            xs,
            radial_distortion=scene.radial_distortion,
        )
        target = scene.images[image_ids, ys, xs, :3]
        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target": target,
            "image_ids": image_ids,
            "ys": ys,
            "xs": xs,
            "prior_mask": prior_mask,
            "prior_depth": prior_depth,
        }
