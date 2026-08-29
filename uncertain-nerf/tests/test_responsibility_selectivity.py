import numpy as np
import torch
from PIL import Image

from tools.analyze_responsibility_selectivity import (
    _write_csv,
    _write_json,
    average_ranks,
    deterministic_uniform_indices,
    edge_overlap_statistics,
    save_selectivity_figure,
    sobel_gradient,
    spearman_numpy,
)


def test_uniform_frame_selection_and_tie_aware_ranks_are_deterministic():
    assert deterministic_uniform_indices(122, 8) == [0, 17, 35, 52, 69, 86, 104, 121]
    np.testing.assert_allclose(average_ranks(np.array([10, 10, 30])), [1.5, 1.5, 3.0])
    assert np.isclose(
        spearman_numpy(np.array([1, 2, 3]), np.array([10, 20, 30])), 1.0
    )


def test_sobel_and_overlap_statistics_are_finite():
    rgb = torch.zeros((1, 9, 9, 3), dtype=torch.float32)
    rgb[:, :, 5:, :] = 1.0
    gradient = sobel_gradient(rgb)
    assert gradient.shape == (1, 9, 9)
    assert torch.isfinite(gradient).all()
    assert gradient.max() == 1.0

    q = np.ones((9, 9), dtype=np.float32)
    q[:, 4:6] = 0.4
    residual = np.zeros((9, 9), dtype=np.float32)
    residual[:, 4:6] = 0.8
    stats = edge_overlap_statistics(q, gradient[0].numpy(), residual)
    assert 0.0 <= stats["edge_overlap_q_0_8"] <= 1.0
    assert 0.0 <= stats["edge_overlap_q_0_5"] <= 1.0
    assert np.isfinite(list(stats.values())).all()


def test_selectivity_visualization_writes_seven_panel_png(tmp_path):
    rgb = np.linspace(0, 1, 12 * 16 * 3, dtype=np.float32).reshape(12, 16, 3)
    scalar = np.linspace(0, 1, 12 * 16, dtype=np.float32).reshape(12, 16)
    output = tmp_path / "selectivity.png"
    save_selectivity_figure(
        output,
        ground_truth=rgb,
        render=rgb,
        residual=scalar,
        responsibility=1.0 - 0.8 * scalar,
        gradient=scalar,
        valid=np.ones((12, 16), dtype=bool),
        title="synthetic",
    )
    assert output.is_file()
    assert Image.open(output).size[0] == 7 * 320
    rows = [{"image_name": "synthetic.png", "score": 0.5}]
    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "summary.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)
    assert csv_path.is_file() and json_path.is_file()
