import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from puri_gs.ontogo import (
    COLORS_FILE,
    DATASET_FORMAT,
    POINTS_FILE,
    PROTOCOL_FILE,
    PROTOCOL_SCHEMA,
    OnTheGoPatioHighParser,
    sha256_file,
    validate_prepared_patio_high,
)
from tools.prepare_ontogo_patio_high import _camera_contract, _opencv_camtoworlds


ROOT = Path(__file__).resolve().parents[1]


def _write_prepared_fixture(root: Path) -> None:
    image_dir = root / "images_4"
    image_dir.mkdir(parents=True)
    frames = []
    train = list(range(45, 266))
    test = list(range(45))
    train_set, test_set = set(train), set(test)
    records = []
    for index in range(267):
        matrix = np.eye(4)
        matrix[0, 3] = index / 100.0
        source_name = f"IMG_{index:04d}.JPG"
        frames.append(
            {
                "file_path": f"./images/{source_name}",
                "transform_matrix": matrix.tolist(),
            }
        )
        if index in train_set:
            prefix = "clutter"
        elif index in test_set:
            prefix = "extra"
        else:
            prefix = "excluded"
        filename = f"{prefix}_{index:04d}_IMG_{index:04d}.png"
        path = image_dir / filename
        Image.new("RGB", (8, 6), (index % 255, 20, 30)).save(path)
        records.append(
            {
                "frame_index": index,
                "source_name": source_name,
                "file": filename,
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

    values = np.linspace(-1.0, 1.0, 30_000, dtype=np.float32).reshape(10_000, 3)
    colors = np.tile(np.array([[10, 20, 30]], dtype=np.uint8), (10_000, 1))
    np.save(root / POINTS_FILE, values, allow_pickle=False)
    np.save(root / COLORS_FILE, colors, allow_pickle=False)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "dataset_format": DATASET_FORMAT,
        "scene": "patio_high",
        "data_factor": 4,
        "frame_count": 267,
        "train_count": 221,
        "test_count": 45,
        "train_indices": train,
        "test_indices": test,
        "unassigned_frame_indices": [266],
        "transforms_sha256": sha256_file(transforms_path),
        "split_sha256": sha256_file(split_path),
        "initial_point_count": len(values),
        "initial_points_sha256": sha256_file(root / POINTS_FILE),
        "initial_colors_sha256": sha256_file(root / COLORS_FILE),
        "images": records,
    }
    (root / PROTOCOL_FILE).write_text(json.dumps(protocol), encoding="utf-8")


def test_prepared_contract_preserves_official_split_and_parser_convention(tmp_path):
    _write_prepared_fixture(tmp_path)
    contract = validate_prepared_patio_high(tmp_path, verify_image_hashes=True)
    assert contract["train_indices"] == list(range(45, 266))
    assert contract["test_indices"] == list(range(45))
    assert contract["image_names"][0].startswith("extra_0000_")
    assert contract["image_names"][45].startswith("clutter_0045_")
    assert contract["image_names"][266].startswith("excluded_0266_")

    parser = OnTheGoPatioHighParser(
        str(tmp_path), factor=4, normalize=False, calibration_index=1
    )
    assert parser.camtoworlds.shape == (267, 4, 4)
    assert np.allclose(parser.camtoworlds[0], np.diag([1.0, -1.0, -1.0, 1.0]))
    assert parser.Ks_dict[1][0, 0] == pytest.approx(6.0)
    assert parser.imsize_dict[1] == (8, 6)
    assert parser.points.shape == (10_000, 3)
    assert parser.points_rgb.shape == (10_000, 3)


def test_contract_accepts_only_the_frozen_factor_and_detects_bad_split(tmp_path):
    _write_prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="factor 4"):
        validate_prepared_patio_high(tmp_path, factor=2)
    split_path = tmp_path / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["extra"][0] = 45
    split_path.write_text(json.dumps(split), encoding="utf-8")
    protocol_path = tmp_path / PROTOCOL_FILE
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["split_sha256"] = sha256_file(split_path)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        validate_prepared_patio_high(tmp_path)


def test_tampered_sparse_initialization_is_rejected(tmp_path):
    _write_prepared_fixture(tmp_path)
    points_path = tmp_path / POINTS_FILE
    with points_path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="point initialization SHA-256 mismatch"):
        validate_prepared_patio_high(tmp_path)


def test_camera_contract_scales_intrinsics_and_flips_opengl_axes():
    transforms = {
        "fl_x": 40.0,
        "fl_y": 44.0,
        "cx": 20.0,
        "cy": 16.0,
        "k1": 0.1,
        "k2": 0.2,
        "p1": 0.01,
        "p2": 0.02,
    }
    k, distortion = _camera_contract(transforms)
    assert np.allclose(k, [[10.0, 0.0, 5.0], [0.0, 11.0, 4.0], [0.0, 0.0, 1.0]])
    assert np.allclose(distortion, [0.1, 0.2, 0.01, 0.02])
    frames = [{"transform_matrix": np.eye(4).tolist()}]
    converted = _opencv_camtoworlds(frames)
    assert np.allclose(converted[0], np.diag([1.0, -1.0, -1.0, 1.0]))


def test_ontogo_patch_is_small_and_keeps_the_ru_algorithm_untouched():
    patch = (ROOT / "patches" / "gsplat_v1.5.3_puri_gs_ontogo.patch").read_text(
        encoding="utf-8"
    )
    assert "OnTheGoPatioHighParser" in patch
    assert "dataset_format" in patch
    assert "loss.backward" not in patch
    assert "mask_" not in patch
    assert "strategy" not in patch
    prepare = (ROOT / "scripts" / "prepare_puri_gs_ontogo.sh").read_text(
        encoding="utf-8"
    )
    assert "prepare_puri_gs_efficiency_audit.sh" in prepare
    assert "937e29912570c372bed6747a5c9bf85fed877bae" in prepare
    launcher = (ROOT / "run_puri_gs.py").read_text(encoding="utf-8")
    assert 'choices=("colmap", "ontogo-patio-high")' in launcher
    assert "Patio-High requires --train-keyword clutter --test-keyword extra" in launcher
    assert "dataset_protocol.json" in launcher
    audit = (ROOT / "tools" / "audit_ontogo_patio_high.py").read_text(
        encoding="utf-8"
    )
    assert "PATIO_HIGH_PROTOCOL_PASS" in audit
    assert "verify_image_hashes" in audit
    preflight = (ROOT / "scripts" / "preflight_puri_gs_ontogo.sh").read_text(
        encoding="utf-8"
    )
    assert "PATIO_HIGH_SERVER_PREFLIGHT=PASS" in preflight
    assert "puri_gs_ontogo_patio_high_v1.tar" in preflight
    assert "4b4d6a6a1e03328b1ff377e9994ab8dfc2c2b5d9d45abb2c8595d52be791abd9" in preflight
    assert "tests/test_ontogo_patio_high.py" in preflight
