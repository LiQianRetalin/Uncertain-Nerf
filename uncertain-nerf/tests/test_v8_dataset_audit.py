import struct

import numpy as np

from v8_robot.dataset_audit import audit_fern_dataset


def _write_png_header(path, width=2016, height=1512):
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
    )


def _make_fern(root, image_count=2):
    (root / "images").mkdir(parents=True)
    (root / "images_2").mkdir()
    (root / "images_4").mkdir()
    (root / "sparse" / "0").mkdir(parents=True)
    for index in range(image_count):
        stem = f"image{index:03d}"
        (root / "images" / f"{stem}.jpg").write_bytes(b"jpeg-placeholder")
        _write_png_header(root / "images_2" / f"{stem}.png")
        _write_png_header(
            root / "images_4" / f"{stem}.png", width=1008, height=756
        )
    for filename, count in (
        ("cameras.bin", 1),
        ("images.bin", image_count),
        ("points3D.bin", 5),
    ):
        (root / "sparse" / "0" / filename).write_bytes(struct.pack("<Q", count))
    poses = np.ones((image_count, 17), dtype=np.float64)
    poses[:, -2] = 0.5
    poses[:, -1] = 5.0
    np.save(root / "poses_bounds.npy", poses)


def test_complete_fern_dataset_passes(tmp_path):
    _make_fern(tmp_path)
    result = audit_fern_dataset(tmp_path, expected_images=2)
    assert result["decision"] == "PASS"
    assert result["counts"]["colmap_images"] == 2
    assert result["images_2_dimensions"] == [[2016, 1512]]
    assert result["images_4_dimensions"] == [[1008, 756]]
    assert result["read_only"] is True


def test_mismatched_frames_fail(tmp_path):
    _make_fern(tmp_path)
    (tmp_path / "images_2" / "image001.png").rename(
        tmp_path / "images_2" / "wrong-frame.png"
    )
    result = audit_fern_dataset(tmp_path, expected_images=2)
    assert result["decision"] == "FAIL"
    assert "images and images_2 do not contain the same frame stems" in result["failures"]


def test_wrong_factor_four_dimensions_fail(tmp_path):
    _make_fern(tmp_path)
    _write_png_header(
        tmp_path / "images_4" / "image001.png", width=2016, height=1512
    )
    result = audit_fern_dataset(tmp_path, expected_images=2)
    assert result["decision"] == "FAIL"
    assert "images_4 contains inconsistent image dimensions" in result["failures"]


def test_factor_four_sorted_alias_names_are_supported(tmp_path):
    _make_fern(tmp_path)
    for index, image in enumerate(sorted((tmp_path / "images_4").iterdir())):
        image.rename(tmp_path / "images_4" / f"image{index:03d}.png")
    result = audit_fern_dataset(tmp_path, expected_images=2)
    assert result["decision"] == "PASS"
