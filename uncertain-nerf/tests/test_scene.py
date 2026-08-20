from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from v5.data import RaySampler, load_scene


def create_scene(root: Path, camera_model="PINHOLE", radial_k=-0.08):
    image_dir = root / "images"
    model_dir = root / "sparse" / "0"
    image_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    centers = [-0.3, -0.1, 0.1, 0.3]
    poses_bounds = []
    for index, center in enumerate(centers, start=1):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[..., 0] = index * 40
        image[..., 1] = np.arange(16, dtype=np.uint8)[None, :] * 8
        imageio.imwrite(image_dir / f"frame{index:03d}.png", image)
        # LLFF/NeRF camera space looks down -Z.  The COLMAP fixture below uses
        # +Z depth, so poses_bounds must store the corresponding backward axis;
        # otherwise every generated ray points away from the sparse scene AABB.
        rotation = np.diag([1.0, 1.0, -1.0])
        pose = np.concatenate(
            [rotation, np.array([[center], [0.0], [0.0]]), np.array([[16.0], [16.0], [12.0]])],
            axis=1,
        )
        poses_bounds.append(np.concatenate([pose.reshape(-1), [1.0, 4.0]]))
    np.save(root / "poses_bounds.npy", np.stack(poses_bounds))
    if camera_model == "PINHOLE":
        camera_line = "1 PINHOLE 16 16 12 12 8 8"
    elif camera_model == "SIMPLE_RADIAL":
        camera_line = f"1 SIMPLE_RADIAL 16 16 12 8 8 {radial_k}"
    else:
        raise ValueError(f"Unsupported test camera model {camera_model}")
    (model_dir / "cameras.txt").write_text(
        f"# camera\n{camera_line}\n", encoding="utf-8"
    )
    points = [(-0.2, -0.2, 2.0), (0.2, -0.2, 2.2), (-0.2, 0.2, 2.4), (0.2, 0.2, 2.6)]
    image_lines = ["# images"]
    for image_id, center in enumerate(centers, start=1):
        image_lines.append(
            f"{image_id} 1 0 0 0 {-center} 0 0 1 frame{image_id:03d}.png"
        )
        observations = []
        for point_id, (x, y, z) in enumerate(points, start=1):
            normalized_x = (x - center) / z
            normalized_y = y / z
            scale = 1.0
            if camera_model == "SIMPLE_RADIAL":
                scale += radial_k * (normalized_x ** 2 + normalized_y ** 2)
            u = 12 * normalized_x * scale + 8
            v = 12 * normalized_y * scale + 8
            observations.extend([str(u), str(v), str(point_id)])
        image_lines.append(" ".join(observations))
    (model_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    point_lines = ["# points"]
    for point_id, (x, y, z) in enumerate(points, start=1):
        track = " ".join(f"{image_id} {point_id - 1}" for image_id in range(1, 5))
        point_lines.append(f"{point_id} {x} {y} {z} 255 255 255 0.1 {track}")
    (model_dir / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")


def test_scene_loading_and_prior_sampling(tmp_path):
    create_scene(tmp_path)
    scene = load_scene(str(tmp_path), factor=2, holdout=2, n_source_views=2)
    assert scene.images.shape == (4, 8, 8, 3)
    assert len(scene.prior_depth) > 0
    assert scene.aabb.shape == (2, 3)
    assert torch.all(scene.aabb[1] > scene.aabb[0])
    batch = RaySampler(scene, batch_size=10, prior_ratio=0.3).sample()
    assert batch["rays_o"].shape == (10, 3)
    assert int(batch["prior_mask"].sum()) == 3
    assert torch.all(batch["prior_depth"][batch["prior_mask"]] > 0)


def test_scene_loading_with_simple_radial_camera(tmp_path):
    create_scene(tmp_path, camera_model="SIMPLE_RADIAL", radial_k=-0.08)
    scene = load_scene(str(tmp_path), factor=2, holdout=2, n_source_views=2)
    assert torch.allclose(scene.radial_distortion, torch.full((4,), -0.08))
    assert torch.allclose(scene.intrinsics[:, 0, 0], torch.full((4,), 6.0))
    assert torch.allclose(scene.intrinsics[:, 0, 2], torch.full((4,), 4.0))
    batch = RaySampler(scene, batch_size=10, prior_ratio=0.3).sample()
    assert torch.isfinite(batch["rays_d"]).all()
