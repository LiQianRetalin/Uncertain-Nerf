import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tools.export_android_colmap_common import export


def test_android_export_cli_can_run_directly():
    result = subprocess.run(
        [sys.executable, "tools/export_android_colmap_common.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _write_source(root: Path) -> None:
    images = root / "images_4"
    sparse = root / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    names = ["0clean000.JPG", "1extra000.JPG", "2clutter000.JPG"]
    for index, name in enumerate(names):
        Image.new("RGB", (8, 6), (20 + index, 30, 40)).save(images / name)
    with (sparse / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQ8d", 1, 1, 4, 32, 24, 24.0, 24.0, 16.0, 12.0, 0.0, 0.0, 0.0, 0.0))
    with (sparse / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(names)))
        for image_id, name in enumerate(names, 1):
            stream.write(struct.pack("<i7di", image_id, 1.0, 0.0, 0.0, 0.0, float(image_id), 0.0, 0.0, 1))
            stream.write(name.encode() + b"\0")
            stream.write(struct.pack("<Q", 0))
    with (sparse / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<Q3d3BdQ", 7, 1.0, 2.0, 3.0, 10, 20, 30, 0.1, 0))


def test_android_export_is_lossless_pinhole_format_conversion(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)
    manifest = export(source, output)
    assert manifest["conversion_only"] is True
    assert manifest["sfm_reestimated"] is False
    assert manifest["frame_count"] == 3
    assert manifest["train_count"] == 1
    assert manifest["test_count"] == 1
    assert manifest["excluded_clean_count"] == 1
    assert len(list((output / "images").glob("*.png"))) == 3
    with (output / "sparse" / "0" / "cameras.bin").open("rb") as stream:
        count, camera_id, model_id, width, height = struct.unpack("<QiiQQ", stream.read(32))
    assert (count, camera_id, model_id, width, height) == (1, 1, 1, 8, 6)
