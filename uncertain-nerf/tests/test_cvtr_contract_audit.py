import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from puri_gs.cvtr import ViewEvidence
from tools import analyze_cvtr_failure_attribution as attribution
from tools.analyze_cvtr_failure_attribution import (
    audit_derived_and_mapping,
    checkpoint_audit,
    compare_reproduced_metrics,
    ensure_outputs_absent,
)


def test_checkpoint_sha_and_state_are_read_only(tmp_path: Path):
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "step": 9999,
            "splats": {
                "means": torch.zeros(2, 3),
                "opacities": torch.ones(2),
            },
        },
        checkpoint_path,
    )
    before = checkpoint_path.read_bytes()
    audit, digest = checkpoint_audit(checkpoint_path)
    assert audit["status"] == "PASS"
    assert audit["step"] == 9999
    assert audit["keys"] == ["splats", "step"]
    assert checkpoint_path.read_bytes() == before
    assert digest == attribution.sha256_file(checkpoint_path)


def test_checkpoint_audit_rejects_unexpected_training_state(tmp_path: Path):
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save(
        {"step": 9999, "splats": {"means": torch.zeros(1, 3)}, "optimizer": {}},
        checkpoint_path,
    )
    audit, _ = checkpoint_audit(checkpoint_path)
    assert audit["status"] == "CHECKPOINT_CONTRACT_FAIL"


def test_existing_output_directory_is_never_overwritten(tmp_path: Path):
    output = tmp_path / "analysis"
    output.mkdir()
    try:
        ensure_outputs_absent([output])
    except attribution.ContractFailure as error:
        assert error.status == "OUTPUT_OVERWRITE_ATTEMPT"
    else:
        raise AssertionError("existing output must be rejected")


def test_absent_output_paths_pass_the_nonoverwrite_guard(tmp_path: Path):
    ensure_outputs_absent([tmp_path / "new", tmp_path / "new.tar.gz"])


def test_final_metric_reproduction_uses_absolute_tolerance():
    existing = {
        "precision": 0.5398659696,
        "recall": 0.6598021027,
        "f1": 0.5938387628,
        "iou": 0.4223119989,
        "clean_false_positive_rate": 0.0232244751,
    }
    reproduced = {key: value + 5e-6 for key, value in existing.items()}
    matches, differences = compare_reproduced_metrics(reproduced, existing)
    assert matches
    assert max(differences.values()) <= 1e-5
    reproduced["precision"] += 2e-5
    assert not compare_reproduced_metrics(reproduced, existing)[0]


def _derived_fixture(tmp_path: Path):
    names = [f"frame_{index:02d}.jpg" for index in range(16)]

    class FakeParser:
        def __init__(self, *, data_dir, factor, normalize, test_every):
            assert factor == 4 and normalize and test_every == 8
            self.image_names = names
            self.camera_ids = [0] * len(names)

    class FakeDataset:
        def __init__(self, parser, *, split, val_every):
            assert isinstance(parser, FakeParser)
            assert split == "train" and val_every == 0
            self.indices = np.arange(len(names))

        def __len__(self):
            return len(names)

        def __getitem__(self, index):
            return {
                "image": torch.zeros(4, 5, 3),
                "K": torch.eye(3),
                "camtoworld": torch.eye(4),
            }

    root = tmp_path / "derived"
    (root / "images").mkdir(parents=True)
    (root / "ground_truth_masks").mkdir()
    entries = []
    records = {}
    metadata = []
    for index, name in enumerate(names):
        gt = np.zeros((4, 5), dtype=np.uint8)
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        kind = "clean" if index < 8 else "transient"
        if kind == "transient":
            gt[1, 1] = 255
            rgb[1, 1] = np.array([255, 0, 0], dtype=np.uint8)
        image_rel = f"images/{name}.png"
        mask_rel = f"ground_truth_masks/{name}.png"
        Image.fromarray(rgb).save(root / image_rel)
        Image.fromarray(gt, mode="L").save(root / mask_rel)
        entries.append(
            {
                "index": index,
                "image_name": name,
                "kind": kind,
                "derived_image": image_rel,
                "ground_truth_mask": mask_rel,
                "target_area_ratio": float((gt == 255).mean()),
            }
        )
        records[name] = ViewEvidence(
            image_name=name,
            residual=torch.zeros(4, 5),
            depth=torch.ones(4, 5),
            alpha=torch.ones(4, 5),
            K=torch.eye(3),
            camtoworld=torch.eye(4),
        )
        metadata.append({"image_name": name})
    manifest = {"entries": entries}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest, records, metadata, FakeParser, FakeDataset


def test_derived_data_and_image_camera_contract_pass_on_aligned_fixture(
    tmp_path: Path, monkeypatch
):
    root, manifest, records, metadata, Parser, Dataset = _derived_fixture(tmp_path)
    monkeypatch.setattr(
        attribution, "_load_colmap_classes", lambda gsplat_dir: (Parser, Dataset)
    )
    derived, mapping, rows, frame_data = audit_derived_and_mapping(
        data_dir=tmp_path / "room",
        gsplat_dir=tmp_path / "gsplat",
        derived_dir=root,
        derived_manifest=manifest,
        cache_records=records,
        cache_metadata=metadata,
    )
    assert derived["status"] == "PASS"
    assert mapping["status"] == "PASS"
    assert len(rows) == len(frame_data) == 16
    assert derived["clean_count"] == derived["transient_count"] == 8
    assert derived["file_count"] == 33


def test_derived_data_contract_detects_patch_outside_misalignment(
    tmp_path: Path, monkeypatch
):
    root, manifest, records, metadata, Parser, Dataset = _derived_fixture(tmp_path)
    patched = root / manifest["entries"][8]["derived_image"]
    with Image.open(patched) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    array[0, 0] = np.array([255, 255, 255], dtype=np.uint8)
    Image.fromarray(array).save(patched)
    monkeypatch.setattr(
        attribution, "_load_colmap_classes", lambda gsplat_dir: (Parser, Dataset)
    )
    derived, _, _, _ = audit_derived_and_mapping(
        data_dir=tmp_path / "room",
        gsplat_dir=tmp_path / "gsplat",
        derived_dir=root,
        derived_manifest=manifest,
        cache_records=records,
        cache_metadata=metadata,
    )
    assert derived["status"] == "DERIVED_DATA_ALIGNMENT_FAIL"


def test_analysis_source_contains_inference_mode_and_no_training_calls():
    source = Path(attribution.__file__).read_text(encoding="utf-8")
    assert "with torch.inference_mode():" in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "torch.optim" not in source
    assert "run_puri_gs" not in source
    assert "continuation" not in source.lower().replace("continuation_run_count", "")


def test_cuda_inference_smoke_is_present_but_does_not_write_checkpoint():
    source = Path(attribution.__file__).read_text(encoding="utf-8")
    assert "def single_gaussian_depth_contract" in source
    assert "from gsplat.rendering import rasterization" in source
    assert "torch.save(" not in source
    assert "checkpoint_write_count\": 0" in source
