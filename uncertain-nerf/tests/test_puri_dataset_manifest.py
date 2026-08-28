import json
import struct

from puri_gs.dataset_manifest import render_manifest_markdown, scan_dataset_root


def _png(path, width=16, height=12):
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
    )


def _colmap_scene(root, names):
    (root / "images").mkdir(parents=True)
    (root / "images_4").mkdir()
    (root / "sparse" / "0").mkdir(parents=True)
    for name in names:
        _png(root / "images" / name)
        _png(root / "images_4" / name, 4, 3)
    for filename, count in (
        ("cameras.bin", 1),
        ("images.bin", len(names)),
        ("points3D.bin", 4),
    ):
        (root / "sparse" / "0" / filename).write_bytes(struct.pack("<Q", count))


def test_manifest_marks_mip_partial_and_detects_keyword_dynamic_split(tmp_path):
    data_root = tmp_path / "nerf-data"
    (data_root / "mipnerf360").mkdir(parents=True)
    android = data_root / "nerf_robustnerf" / "robustnerf" / "android"
    _colmap_scene(android, ["2clutter001.png", "1extra001.png", "0clean001.png"])
    fern = data_root / "nerf_example_data" / "nerf_llff_data" / "fern"
    _colmap_scene(fern, ["image000.png", "image001.png"])

    manifest = scan_dataset_root(data_root)
    by_name = {item["name"]: item for item in manifest["datasets"]}
    assert by_name["mipnerf360"]["status"] == "PARTIAL"
    assert "garden" in by_name["mipnerf360"]["missing_items"]
    robust = by_name["nerf_robustnerf"]
    assert robust["recommended_first_dynamic_scene"] == "android"
    assert robust["scenes"][0]["split_keyword_counts"]["clutter"] == 1
    assert robust["scenes"][0]["split_keyword_counts"]["extra"] == 1
    assert manifest["read_only"] is True
    rendered = render_manifest_markdown(manifest)
    assert "PARTIAL" in rendered
    json.dumps(manifest)
