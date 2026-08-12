import struct

import numpy as np

from v5.colmap import read_model


def write_text_model(path):
    path.mkdir()
    (path / "cameras.txt").write_text("# camera\n1 PINHOLE 16 12 10 11 8 6\n", encoding="utf-8")
    (path / "images.txt").write_text(
        "# images\n1 1 0 0 0 0 0 0 1 frame.png\n8 6 7\n", encoding="utf-8"
    )
    (path / "points3D.txt").write_text(
        "# points\n7 0 0 2 255 255 255 0.1 1 0\n", encoding="utf-8"
    )


def write_binary_model(path, model_id=1, camera_params=(10, 11, 8, 6)):
    path.mkdir()
    with (path / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, model_id, 16, 12))
        handle.write(struct.pack("<" + "d" * len(camera_params), *camera_params))
    with (path / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<i7di", 1, 1, 0, 0, 0, 0, 0, 0, 1))
        handle.write(b"frame.png\x00")
        handle.write(struct.pack("<Qddq", 1, 8, 6, 7))
    with (path / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<QdddBBBd", 7, 0, 0, 2, 255, 255, 255, 0.1))
        handle.write(struct.pack("<Qii", 1, 1, 0))


def assert_model(path):
    cameras, images, points = read_model(str(path))
    assert cameras[1].model == "PINHOLE"
    assert np.allclose(cameras[1].intrinsic_matrix(), [[10, 0, 8], [0, 11, 6], [0, 0, 1]])
    assert images[1].name == "frame.png"
    assert images[1].point3D_ids.tolist() == [7]
    assert np.allclose(points[7].xyz, [0, 0, 2])


def test_read_text_model(tmp_path):
    path = tmp_path / "text"
    write_text_model(path)
    assert_model(path)


def test_read_binary_model(tmp_path):
    path = tmp_path / "binary"
    write_binary_model(path)
    assert_model(path)


def test_binary_is_preferred(tmp_path):
    path = tmp_path / "both"
    write_binary_model(path)
    (path / "cameras.txt").write_text("invalid", encoding="utf-8")
    (path / "images.txt").write_text("invalid", encoding="utf-8")
    (path / "points3D.txt").write_text("invalid", encoding="utf-8")
    assert_model(path)


def test_read_simple_radial_binary_model(tmp_path):
    path = tmp_path / "simple_radial"
    write_binary_model(path, model_id=2, camera_params=(10, 8, 6, -0.075))
    cameras, _, _ = read_model(str(path))
    camera = cameras[1]
    assert camera.model == "SIMPLE_RADIAL"
    assert np.allclose(camera.intrinsic_matrix(), [[10, 0, 8], [0, 10, 6], [0, 0, 1]])
    assert np.isclose(camera.radial_distortion(), -0.075)
