import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from tests.test_ontogo_patio_high import _write_prepared_fixture
from tools.export_ontogo_colmap_common import OUTPUT_SCHEMA, export, validate_common_input


def test_export_preserves_frozen_poses_pixels_split_and_points(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "common"
    _write_prepared_fixture(source)
    manifest = export(source, output)

    assert manifest["schema"] == OUTPUT_SCHEMA
    assert manifest["sfm_reestimated"] is False
    assert manifest["frame_count"] == 267
    assert manifest["train_count"] == 221
    assert manifest["test_count"] == 45
    assert manifest["initial_point_count"] == 10_000
    assert manifest["max_rotation_roundtrip_abs_error"] < 1e-12
    assert len(list((output / "images").glob("*.png"))) == 267
    with Image.open(next((output / "images").glob("*.png"))) as image:
        assert image.size == (8, 6)

    with (output / "sparse" / "0" / "cameras.bin").open("rb") as stream:
        count, camera_id, model_id, width, height = struct.unpack("<QiiQQ", stream.read(32))
        params = struct.unpack("<4d", stream.read(32))
    assert (count, camera_id, model_id, width, height) == (1, 1, 1, 8, 6)
    assert params == (6.0, 6.0, 4.0, 3.0)

    with (output / "sparse" / "0" / "images.bin").open("rb") as stream:
        assert struct.unpack("<Q", stream.read(8))[0] == 267
    with (output / "sparse" / "0" / "points3D.bin").open("rb") as stream:
        assert struct.unpack("<Q", stream.read(8))[0] == 10_000
    assert json.loads((output / "common_input_protocol.json").read_text())["conversion_only"]
    validation = validate_common_input(source, output)
    assert validation["decision"] == "PATIO_HIGH_COMMON_COLMAP_VALID"
    assert validation["projection_matrices_verified"] is True
