import numpy as np
import torch

from puri_gs.cvtr import (
    CVTR_STAGE_NAMES,
    ViewEvidence,
    compute_cvtr_stage_trace,
)
from tools.analyze_cvtr_failure_attribution import (
    auroc_rank,
    confusion_counts,
    count_first_failures,
    distribution_summary,
    first_failure_map,
    location_attribution,
    metrics_from_counts,
    patch_boundary_and_interior,
    run_cpu_self_test,
    save_visualizations,
    sobel_gradient,
    stage_metric_tables,
    stage_transition_statistics,
    top_fraction_mask,
)


def _stage_fixture():
    masks = np.zeros((16, 8, 2, 2), dtype=bool)
    targets = np.zeros((16, 2, 2), dtype=bool)
    kinds = ["clean"] * 8 + ["transient"] * 8
    targets[8:, 0, 0] = True
    masks[8:, :, 0, 0] = True
    masks[:, 0, 1, 1] = True
    masks[:, 1:, 1, 1] = False
    return masks, targets, kinds


def test_micro_confusion_matrix_is_exact():
    predicted = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [1, 0]], dtype=bool)
    assert confusion_counts(predicted, target) == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_metrics_from_counts_are_exact():
    metrics = metrics_from_counts({"tp": 3, "fp": 1, "tn": 5, "fn": 2})
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.6
    assert np.isclose(metrics["f1"], 2 * 0.75 * 0.6 / 1.35)
    assert metrics["iou"] == 0.5


def test_zero_predicted_positive_uses_null_precision():
    metrics = metrics_from_counts({"tp": 0, "fp": 0, "tn": 7, "fn": 2})
    assert metrics["precision"] is None
    assert metrics["f1"] is None
    assert metrics["recall"] == 0.0


def test_clean_fpr_and_macro_metrics_are_reported_without_fake_precision():
    masks, targets, kinds = _stage_fixture()
    metrics, rows, per_frame = stage_metric_tables(masks, targets, kinds)
    s0 = metrics[CVTR_STAGE_NAMES[0]]
    assert s0["clean"]["micro_fpr"] == 0.25
    assert s0["clean"]["macro"]["mean"] == 0.25
    assert s0["patched_micro"]["precision"] == 0.5
    assert rows[0]["clean_micro_fpr"] == 0.25
    assert per_frame[0][f"{CVTR_STAGE_NAMES[0]}_fp"] == 1
    assert per_frame[0][f"{CVTR_STAGE_NAMES[0]}_clean_fpr"] == 0.25


def test_macro_mean_median_min_and_max_are_correct():
    masks, targets, kinds = _stage_fixture()
    masks[8, :, 0, 0] = False
    metrics, _, _ = stage_metric_tables(masks, targets, kinds)
    recall = metrics[CVTR_STAGE_NAMES[-1]]["patched_macro"]["recall"]
    assert recall == {"mean": 0.875, "median": 1.0, "min": 0.0, "max": 1.0}


def test_stage_transition_statistics_identify_largest_changes():
    masks, targets, kinds = _stage_fixture()
    masks[8:, 2:, 0, 0] = False
    metrics, _, _ = stage_metric_tables(masks, targets, kinds)
    transitions = stage_transition_statistics(metrics)
    assert transitions["largest_TP_loss_stage"] == CVTR_STAGE_NAMES[2]
    assert transitions["largest_recall_loss_stage"] == CVTR_STAGE_NAMES[2]
    assert transitions["largest_FP_reduction_stage"] == CVTR_STAGE_NAMES[1]


def test_first_failure_stage_counts_each_final_fn_once():
    stages = np.ones((8, 2, 3), dtype=bool)
    stages[0:, 0, 0] = False
    stages[3:, 0, 1] = False
    stages[7:, 0, 2] = False
    final_fn = np.zeros((2, 3), dtype=bool)
    final_fn[0] = True
    counts = count_first_failures(stages, final_fn)
    assert counts[CVTR_STAGE_NAMES[0]] == 1
    assert counts[CVTR_STAGE_NAMES[3]] == 1
    assert counts[CVTR_STAGE_NAMES[7]] == 1
    assert sum(counts.values()) == 3


def test_first_failure_map_keeps_the_earliest_removal_when_s6_readds():
    stages = np.ones((8, 1, 1), dtype=bool)
    stages[4, 0, 0] = False
    stages[5, 0, 0] = False
    stages[6, 0, 0] = True
    stages[7, 0, 0] = False
    assert first_failure_map(stages)[0, 0] == 4


def test_patch_boundary_and_interior_use_a_three_pixel_band():
    mask = np.zeros((13, 13), dtype=bool)
    mask[1:12, 1:12] = True
    boundary, interior = patch_boundary_and_interior(mask, boundary_width=3)
    assert np.array_equal(boundary | interior, mask)
    assert not np.any(boundary & interior)
    assert interior[6, 6]
    assert boundary[1, 1]


def test_sobel_rgb_gradient_is_zero_for_constant_and_positive_at_step():
    constant = np.ones((7, 7, 3), dtype=np.float32)
    assert np.allclose(sobel_gradient(constant), 0.0)
    step = constant.copy()
    step[:, 4:] = 0.0
    gradient = sobel_gradient(step)
    assert gradient[:, 3:5].max() > 0


def test_sobel_depth_gradient_respects_valid_mask():
    depth = np.tile(np.arange(7, dtype=np.float32), (7, 1))
    valid = np.ones((7, 7), dtype=bool)
    valid[:, 0] = False
    gradient = sobel_gradient(depth, valid)
    assert np.all(gradient[:, 0] == 0)
    assert gradient[:, 3].mean() > 0


def test_auroc_rank_perfect_reversed_and_tied():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert auroc_rank(np.array([0.0, 0.1, 0.8, 1.0]), labels) == 1.0
    assert auroc_rank(np.array([1.0, 0.8, 0.1, 0.0]), labels) == 0.0
    assert auroc_rank(np.ones(4), labels) == 0.5


def test_auroc_returns_null_without_both_classes():
    assert auroc_rank(np.arange(4), np.ones(4, dtype=bool)) is None


def test_top_ten_percent_mask_uses_only_valid_values():
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    valid = np.ones_like(values, dtype=bool)
    valid[9, 9] = False
    selected = top_fraction_mask(values, valid, 0.10)
    assert not selected[9, 9]
    assert 9 <= selected.sum() <= 10


def test_distribution_summary_reports_threshold_fraction():
    summary = distribution_summary(np.array([0.0, 1.0, 2.0, 3.0]), threshold=1.5)
    assert summary["mean"] == 1.5
    assert summary["p50"] == 1.5
    assert summary["fraction_above_threshold"] == 0.5


def test_cpu_artificial_dry_run_has_zero_training_calls():
    result = run_cpu_self_test()
    assert result["status"] == "PASS"
    assert result["optimizer_step_count"] == 0
    assert result["backward_call_count"] == 0


def test_artificial_visualization_writes_main_and_focus_pngs(tmp_path):
    residual = torch.zeros(9, 9)
    residual[3:6, 3:6] = 1.0
    source = ViewEvidence(
        image_name="source.jpg",
        residual=residual,
        depth=torch.full((9, 9), 2.0),
        alpha=torch.ones(9, 9),
        K=torch.tensor([[6.0, 0.0, 4.0], [0.0, 6.0, 4.0], [0.0, 0.0, 1.0]]),
        camtoworld=torch.eye(4),
    )
    neighbors = [
        ViewEvidence(
            image_name=f"neighbor_{index}.jpg",
            residual=torch.zeros(9, 9),
            depth=torch.full((9, 9), 2.0),
            alpha=torch.ones(9, 9),
            K=source.K,
            camtoworld=torch.eye(4),
        )
        for index in range(3)
    ]
    trace = compute_cvtr_stage_trace(source, neighbors, 0.5)
    target = np.zeros((1, 9, 9), dtype=bool)
    target[0, 3:6, 3:6] = True
    rgb = np.zeros((1, 9, 9, 3), dtype=np.uint8)
    depth = np.full((1, 9, 9), 2.0, dtype=np.float32)
    _, _, locations = location_attribution(
        traces=[trace],
        targets=target,
        parser_rgbs=rgb,
        depths=depth,
        threshold=0.5,
    )
    scales = save_visualizations(
        output_dir=tmp_path,
        names=["source.jpg"],
        traces=[trace],
        targets=target,
        parser_rgbs=rgb,
        rendered_rgbs=np.zeros((1, 9, 9, 3), dtype=np.float32),
        depths=depth,
        location_arrays=locations,
    )
    assert (tmp_path / "visualizations" / "source.jpg.png").is_file()
    assert (tmp_path / "visualizations" / "focus_source.jpg.png").is_file()
    assert scales["rgb_gradient_vmax"] > 0
    assert scales["depth_gradient_vmax"] > 0
