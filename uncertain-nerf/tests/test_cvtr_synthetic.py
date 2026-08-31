import argparse
from pathlib import Path

import pytest
import torch

from puri_gs.cvtr import binary_mask_metrics
from tools import validate_cvtr_synthetic
from tools.build_cvtr_masks import require_empty_output
from tools.validate_cvtr_synthetic import uniformly_select


def test_uniform_synthetic_frame_selection_is_deterministic():
    names = [f"frame_{index:03d}.png" for index in range(100)]
    first = uniformly_select(names, 16)
    assert first == uniformly_select(names, 16)
    assert len(first) == len(set(first)) == 16
    assert first[0] == names[0]
    assert first[-1] == names[-1]


def test_synthetic_targets_come_from_the_pinned_training_dataset(
    monkeypatch, tmp_path
):
    class FakeParser:
        def __init__(self, *, data_dir, factor, normalize, test_every):
            assert data_dir == str(tmp_path / "room")
            assert factor == 4
            assert normalize is True
            assert test_every == 8
            self.image_names = ["test.jpg", "train_a.jpg", "train_b.jpg"]

    class FakeDataset:
        def __init__(self, parser, *, split, val_every):
            assert isinstance(parser, FakeParser)
            assert split == "train"
            assert val_every == 0
            self.indices = torch.tensor([1, 2])

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            value = 17 + index
            return {"image": torch.full((3, 5, 3), value, dtype=torch.float32)}

    monkeypatch.setattr(
        validate_cvtr_synthetic,
        "_load_colmap_classes",
        lambda gsplat_dir: (FakeParser, FakeDataset),
    )
    images = validate_cvtr_synthetic._load_room_training_images(
        tmp_path / "room", tmp_path / "gsplat"
    )

    assert sorted(images) == ["train_a.jpg", "train_b.jpg"]
    assert images["train_a.jpg"].shape == (3, 5, 3)
    assert images["train_a.jpg"].dtype.name == "uint8"
    assert images["train_a.jpg"][0, 0, 0] == 17
    assert images["train_b.jpg"][0, 0, 0] == 18


def test_synthetic_mask_precision_recall():
    target = torch.zeros(10, 10, dtype=torch.bool)
    target[2:6, 2:6] = True
    predicted = target.clone()
    predicted[2, 2] = False
    predicted[0, 0] = True
    metrics = binary_mask_metrics(predicted, target)
    assert metrics["precision"] == pytest.approx(15 / 16)
    assert metrics["recall"] == pytest.approx(15 / 16)
    assert metrics["f1"] == pytest.approx(15 / 16)
    assert metrics["iou"] == pytest.approx(15 / 17)


def test_output_directory_must_not_be_overwritten(tmp_path: Path):
    output = tmp_path / "new"
    require_empty_output(output)
    (output / "existing.txt").write_text("owned by prior run", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not empty"):
        require_empty_output(output)
