import argparse
from pathlib import Path

import pytest
import torch

from puri_gs.cvtr import binary_mask_metrics
from tools.build_cvtr_masks import require_empty_output
from tools.validate_cvtr_synthetic import uniformly_select


def test_uniform_synthetic_frame_selection_is_deterministic():
    names = [f"frame_{index:03d}.png" for index in range(100)]
    first = uniformly_select(names, 16)
    assert first == uniformly_select(names, 16)
    assert len(first) == len(set(first)) == 16
    assert first[0] == names[0]
    assert first[-1] == names[-1]


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
