import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from puri_gs.ontogo import COLORS_FILE, POINTS_FILE, PROTOCOL_FILE, sha256_file
from puri_gs.ontogo_corner import (
    DATASET_FORMAT,
    PROTOCOL_SCHEMA,
    OnTheGoCornerParser,
    validate_prepared_corner,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_corner_fixture(root: Path) -> None:
    image_dir = root / "images_4"
    image_dir.mkdir(parents=True)
    train = list(range(12, 113))
    test = [*range(12), *range(113, 121)]
    train_set, test_set = set(train), set(test)
    frames = []
    records = []
    for index in range(122):
        matrix = np.eye(4)
        matrix[0, 3] = index / 100.0
        source_name = f"IMG_{index:04d}.JPG"
        frames.append(
            {
                "file_path": f"./images/{source_name}",
                "transform_matrix": matrix.tolist(),
            }
        )
        split = (
            "clutter"
            if index in train_set
            else "extra"
            if index in test_set
            else "excluded"
        )
        name = f"{split}_{index:04d}_IMG_{index:04d}.png"
        path = image_dir / name
        Image.new("RGB", (8, 6), (index % 255, 20, 30)).save(path)
        records.append(
            {
                "frame_index": index,
                "source_name": source_name,
                "source_sha256": "fixture",
                "source_width": 32,
                "source_height": 24,
                "file": name,
                "width": 8,
                "height": 6,
                "sha256": sha256_file(path),
            }
        )

    transforms = {
        "fl_x": 24.0,
        "fl_y": 24.0,
        "cx": 16.0,
        "cy": 12.0,
        "w": 32.0,
        "h": 24.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "is_fisheye": False,
        "frames": frames,
    }
    split = {"clutter": train, "extra": test}
    transforms_path = root / "transforms.json"
    split_path = root / "split.json"
    transforms_path.write_text(json.dumps(transforms), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    points = np.linspace(-1, 1, 30_000, dtype=np.float32).reshape(10_000, 3)
    colors = np.tile(np.array([[10, 20, 30]], dtype=np.uint8), (10_000, 1))
    np.save(root / POINTS_FILE, points, allow_pickle=False)
    np.save(root / COLORS_FILE, colors, allow_pickle=False)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "dataset_format": DATASET_FORMAT,
        "scene": "corner",
        "data_factor": 4,
        "frame_count": 122,
        "train_count": 101,
        "test_count": 20,
        "train_indices": train,
        "test_indices": test,
        "unassigned_frame_indices": [121],
        "transforms_sha256": sha256_file(transforms_path),
        "split_sha256": sha256_file(split_path),
        "initial_point_count": len(points),
        "initial_points_sha256": sha256_file(root / POINTS_FILE),
        "initial_colors_sha256": sha256_file(root / COLORS_FILE),
        "images": records,
    }
    (root / PROTOCOL_FILE).write_text(json.dumps(protocol), encoding="utf-8")


def test_corner_contract_and_parser_keep_official_split(tmp_path):
    _write_corner_fixture(tmp_path)
    contract = validate_prepared_corner(tmp_path, verify_image_hashes=True)
    assert contract["train_indices"] == list(range(12, 113))
    assert contract["test_indices"] == [*range(12), *range(113, 121)]
    assert contract["image_names"][0].startswith("extra_0000_")
    assert contract["image_names"][12].startswith("clutter_0012_")
    assert contract["image_names"][121].startswith("excluded_0121_")

    parser = OnTheGoCornerParser(str(tmp_path), factor=4, normalize=False)
    assert parser.camtoworlds.shape == (122, 4, 4)
    assert np.allclose(parser.camtoworlds[0], np.diag([1.0, -1.0, -1.0, 1.0]))
    assert parser.Ks_dict[1][0, 0] == pytest.approx(6.0)
    assert parser.imsize_dict[1] == (8, 6)
    assert parser.points.shape == (10_000, 3)


def test_corner_contract_rejects_split_or_point_tampering(tmp_path):
    _write_corner_fixture(tmp_path)
    split_path = tmp_path / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["extra"][0] = 12
    split_path.write_text(json.dumps(split), encoding="utf-8")
    protocol_path = tmp_path / PROTOCOL_FILE
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["split_sha256"] = sha256_file(split_path)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        validate_prepared_corner(tmp_path)

    _write_corner_fixture(tmp_path / "clean")
    points_path = tmp_path / "clean" / POINTS_FILE
    with points_path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="point initialization SHA-256 mismatch"):
        validate_prepared_corner(tmp_path / "clean")


def test_p03_patches_are_adapter_timing_and_accounting_only():
    gsplat = (ROOT / "patches" / "gsplat_v1.5.3_p03_corner.patch").read_text(
        encoding="utf-8"
    )
    assert "OnTheGoCornerParser" in gsplat
    assert "ontogo-corner" in gsplat
    for forbidden in ("loss.backward", "mask_", "strategy.step"):
        assert forbidden not in gsplat

    robust = (ROOT / "patches" / "p03_robustsplat_timing_config.patch").read_text(
        encoding="utf-8"
    )
    spotless = (ROOT / "patches" / "p03_spotless_timing_memory.patch").read_text(
        encoding="utf-8"
    )
    internal_smoke = (ROOT / "patches" / "p03_gsplat_smoke_finiteness.patch").read_text(
        encoding="utf-8"
    )
    for patch in (robust, spotless, internal_smoke):
        assert "P03_SMOKE_FINITE_LOSS_GRADIENT=PASS" in patch
        assert "torch.isfinite(loss)" in patch
    for patch in (robust, spotless):
        assert "P03_TIMING_ONLY" in patch
        assert "torch.cuda.synchronize()" in patch
        assert '"per_image"' in patch
    assert "p03_resolved_config.json" in robust
    assert "p03_resolved_config.json" in spotless

    validator = (ROOT / "tools" / "p03_validate_inputs.py").read_text(
        encoding="utf-8"
    )
    assert '"lower_bound": 0.5' in validator
    assert '"upper_bound": 0.9' in validator
    assert "native/common pixels differ" in validator

    pipeline = (ROOT / "scripts" / "p03_server_pipeline.sh").read_text(
        encoding="utf-8"
    )
    assert "gpu_cap_seconds=43200" in pipeline
    assert "P03_WORK_ROOT已存在；拒绝覆盖或隐式续跑" in pipeline
    assert "P03_SMOKE_AUDIT" in pipeline
    assert pipeline.index("timing_P03-corner-ru") < pipeline.index(
        "quality_render_P03-corner-ru"
    )
    assert "P03_COMPLETE_STOP" in pipeline
    assert "--loss_type robust --semantics --no-cluster --lower_bound 0.5 --upper_bound 0.9" in pipeline
    assert "--ubp" not in pipeline
    assert '--gsplat-dir "$gsplat_src"' in pipeline
