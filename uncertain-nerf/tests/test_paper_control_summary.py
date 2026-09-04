import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from puri_gs.delayed_absgrad import topology_event_summary
from puri_gs.paper_controls import (
    MASK_FIELDS, MASK_SAMPLE_STEPS, PaperControlRecorder, schedule_from_config,
)
from tools.audit_puri_gs_paper_controls import TEST_NAMES, audit_run, config_audit
from tools.summarize_puri_gs_garden_paper_controls import (
    make_summary, paired_comparison, quality_gates, render_markdown, representative_canvas,
)
from test_garden_causal_summary import _make_run, _write_json
from test_paper_controls import config


def control_fixture(root, name="ru_align"):
    metrics = {"psnr": 27.8, "ssim": 0.88, "lpips": 0.06}
    _make_run(root, config_name=f"puri_gs_{name}_full30k.yaml", quadrant="Y11", metrics=metrics)
    cfg = config(name)
    split = {"protocol": "every-nth-test", "train": [f"train_{i}.JPG" for i in range(161)], "test": TEST_NAMES}
    for path in (root, root / "independent_eval"):
        _write_json(path / "dataset_split.json", split)
        _write_json(path / "config.yaml", cfg)
    (root / "independent_eval/run_command.txt").write_text("python simple_trainer.py --ckpt ckpt.pt")
    with (root / "independent_eval/per_image_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_name", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(dict(image_name=n, **metrics) for n in TEST_NAMES)
    train = json.loads((root / "train_metrics.json").read_text())
    train["step"] = 29999
    _write_json(root / "train_metrics.json", train)
    schedule = schedule_from_config(cfg)
    saved = {**asdict(schedule), **{k: cfg[k] for k in (
        "bootstrap_switch_step", "mask_begin_step", "mask_threshold", "mask_erode_kernel")},
        "topology_events": topology_event_summary(delayed_topology=True, mask_pause_after_reset=300, delayed_schedule=schedule)}
    _write_json(root / "aux/training_schedule.json", saved)
    _write_json(root / "aux/ru_training_validation.json", {"gradient_isolation_pass": True, "dino_trainable_parameter_count": 0})
    _write_json(root / "DINO_time.json", {"mask_update_count": 29400, "mask_pause_count": 600, "render_feature_calls": 30000})
    splats = {name: torch.zeros((1000, *shape)) for name, shape in {
        "means": (3,), "scales": (3,), "quats": (4,), "opacities": (), "sh0": (1, 3), "shN": (15, 3)}.items()}
    torch.save({"step": 29999, "splats": splats}, root / "ckpts/ckpt_29999_rank0.pt")
    recorder = PaperControlRecorder(root, cfg, 30000)
    for step in schedule.refine_steps(30000):
        recorder.record_topology(step=step, gaussian_count_before=1000, gaussian_count_after=1000,
            split_count=0, clone_count=0, prune_count=0, reset_event=int(schedule.should_reset(step)))
    for step in MASK_SAMPLE_STEPS:
        row = dict.fromkeys(MASK_FIELDS, 0.5)
        row.update(step=step, mask_supervision_scale="coarse" if step < 10000 else "fine", head_paused=0)
        recorder.mask_writer.writerow(row)
    # Construct a complete small artifact fixture, independent of the renderer.
    recorder.mask_steps = list(MASK_SAMPLE_STEPS)
    recorder.iterations, recorder.full_renders, recorder.coarse_renders = 30000, 30000, 10000
    recorder.finish()
    _write_json(root / "resolved_config_diff.json", config_audit()["resolved_config_diff"])
    return root


def test_complete_control_artifact_audit_rejects_corruption(tmp_path):
    root = control_fixture(tmp_path / "align")
    assert audit_run(root, evaluation=True)["topology"]["event_count"] == 100
    csv_path = root / "independent_eval/per_image_metrics.csv"
    original = csv_path.read_text()
    csv_path.write_text(original.replace(TEST_NAMES[0], "wrong.JPG"))
    with pytest.raises(ValueError, match="order"):
        audit_run(root, evaluation=True)
    csv_path.write_text(original)
    ckpt = root / "ckpts/ckpt_29999_rank0.pt"
    payload = torch.load(ckpt, weights_only=True)
    payload["splats"]["means"][0, 0] = float("nan")
    torch.save(payload, ckpt)
    with pytest.raises(ValueError, match="checkpoint Gaussian"):
        audit_run(root)


def test_standard_checkpoint_cannot_contain_mask_head(tmp_path):
    root = control_fixture(tmp_path / "tar", "ru_tar")
    assert audit_run(root, evaluation=True)["topology"]["event_count"] == 120
    ckpt = root / "ckpts/ckpt_29999_rank0.pt"
    payload = torch.load(ckpt, weights_only=True)
    payload["splats"]["mask_head"] = torch.zeros(1)
    torch.save(payload, ckpt)
    with pytest.raises(ValueError, match="standard Gaussian"):
        audit_run(root)


def sample_run(metrics, count):
    return {"metrics": metrics, "training": {"gaussian_count": count},
            "per_image": [dict(image_name=n, **metrics) for n in TEST_NAMES],
            "split": {"test": TEST_NAMES}}


def test_effects_bootstrap_directions_gates_and_capacity():
    b1 = sample_run(dict(psnr=27.716728, ssim=0.873957, lpips=0.068151), 3798503)
    ru = sample_run(dict(psnr=26.65, ssim=0.8573, lpips=0.0905), 2274197)
    align = sample_run(dict(psnr=27.6, ssim=0.87, lpips=0.075), 2500000)
    tar = sample_run(dict(psnr=27.8, ssim=0.88, lpips=0.065), 3000000)
    tar["topology"] = {"phases": {"late": dict(split_count=400000, clone_count=200000, prune_count=100000)}}
    paired = paired_comparison(align, tar)
    for metric in ("psnr", "ssim", "lpips"):
        row = paired["paired"][metric]
        assert row["favorable_count"] == 24
        assert row["ci95"] == pytest.approx([row["mean_difference"]] * 2)
    gates = quality_gates(b1, tar)
    assert gates["recovery_label"] == "GARDEN_CLEAN_TOLERANCE_RECOVERED"
    assert not gates["count_reference_pass"]
    assert all(gates["b1_metric_checks"].values())
    representative = {"selection_rule": "fixed", "images": [{"image_name": TEST_NAMES[4]}]}
    runs = {"B1": b1, "RU": ru, "RU-Align": align, "RU-TAR": tar}
    result = make_summary(runs, representative)
    assert result["capacity"]["per_million_gaussians"]["psnr"] == pytest.approx(0.4)
    assert result["next_stage_inputs"]["LATE_ADDED_GAUSSIANS"] == 500000
    tar["training"]["gaussian_count"] = 2400000
    result = make_summary(runs, representative)
    assert result["capacity"]["status"] == "undefined_nonpositive_added_count"
    assert result["capacity"]["per_million_gaussians"]["psnr"] is None
    tar["per_image"][4]["psnr"] = 20.0
    assert not quality_gates(b1, tar)["clean_checks"]["worst_view"]


def test_lightweight_canvas_uses_same_ground_truth_and_refuses_overwrite(tmp_path):
    from PIL import Image
    runs = {}
    selected = {"images": [dict(image_name=TEST_NAMES[4], test_index_zero_based=4)] * 6}
    for name in ("B1", "RU", "RU-Align", "RU-TAR"):
        root = tmp_path / name
        folder = root / "independent_eval/renders"
        folder.mkdir(parents=True)
        image = Image.new("RGB", (40, 15), (12, 24, 36))
        image.save(folder / "test_step29999_0004.png")
        runs[name] = {"path": str(root)}
    target = tmp_path / "representatives.png"
    representative_canvas(runs, selected, target)
    with Image.open(target) as image:
        assert image.width == 1536
    with pytest.raises(FileExistsError):
        representative_canvas(runs, selected, target)
