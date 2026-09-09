"""Historical reuse must preserve provenance and reject changed evidence."""
import csv
import json
from pathlib import Path
import shlex

import numpy as np
import pytest
import torch

from puri_gs.ru_part_v3 import write_json
from puri_gs.static_tracks import sha256_file
from puri_gs.v3_parent_reference import (
    AUDITED_COMMIT, REEVAL_COMMIT, load_reference, register_reference, runtime_config,
)
from tools.ru_part_v3_screen import validate_checkpoint


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def evaluation_files(directory, identity, psnr=26.65):
    put(directory / "test_metrics.json", {"psnr": psnr, "ssim": .87, "lpips": .07, "num_GS": 2})
    with (directory / "per_image_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_name", "psnr", "ssim", "lpips"])
        writer.writerows((name, psnr, .87, .07) for name in identity["test_basenames"])
    put(directory / "ru_validation.json", {
        "standard_checkpoint_load_pass": True, "evaluation_imported_dino": False,
        "evaluation_loaded_mask_head": False, "evaluation_rasterization_count_ratio": 1,
        "evaluation_warmup_render_count": 10, "raw_latency_sample_count": 24,
        "evaluation_image_save_disabled": False,
    })
    put(directory / "efficiency_metrics.json", {"render_fps": 140., "gaussian_count": 2,
                                               "warmup_render_count": 10, "raw_latency_sample_count": 24})
    for name in ("DSC07988_alpha.npy", "DSC07988_abs_error.npy"):
        np.save(directory / name, np.ones((3, 4)))


@pytest.fixture
def historical(tmp_path):
    project = tmp_path
    root, source, data = (tmp_path / name for name in ("screen", "historical", "garden"))
    evaluation = root / "eval-existing-parent"
    for path in (root, source / "ckpts", data / "images_4_png", data / "sparse/0", evaluation):
        path.mkdir(parents=True, exist_ok=True)
    split = {"train": [f"train{i}.JPG" for i in range(161)],
             "test": ["DSC07988.JPG"] + [f"test{i}.JPG" for i in range(23)]}
    hashes = {}
    for name in split["train"]:
        path = data / "images_4_png" / Path(name).with_suffix(".png")
        path.write_bytes(name.encode())
        hashes[name] = sha256_file(path)
    sfm = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        path = data / "sparse/0" / name
        path.write_bytes(name.encode())
        sfm[name] = sha256_file(path)
    identity = {"train_basenames": split["train"], "test_basenames": split["test"],
                "train_image_sha256": hashes, "sfm_sha256": sfm}
    cfg = """ckpt: null
init_type: sfm
max_steps: 30000
puri_gs_ru_enabled: true
strategy: !!python/object:puri_gs.delayed_absgrad.DelayedAbsGradStrategy
  absgrad: true
  grow_grad2d: 0.0006
  schedule: !!python/object:puri_gs.delayed_absgrad.DelayedAbsGradSchedule
    densify_stop_step: 20000
"""
    (source / "cfg.yml").write_text(cfg)
    for stage in ("smoke-parent", "smoke-v3"):
        put(root / f"{stage}.status.json", {"status": "SMOKE_COMPLETE", "exit_code": 0})
        put(root / stage / "v3_input_manifest.json", identity)
        put(root / stage / "v3_camera_sequence.json", [i % 161 for i in range(600)])
    (root / "smoke-parent/cfg.yml").write_text(cfg + "ru_v3_mode: parent\nru_v3_stop_step: 599\n")
    config = {"profile": "ru", "seed": 42}
    put(source / "config.yaml", config)
    put(project / "configs/puri_gs_ru_part_v3_parent_garden30k.yaml", {**config, "v3_screening": "parent"})
    runtime = {"repository_commit": AUDITED_COMMIT, "platform": "linux", "python": "3.10",
               "torch": {"gpu": "L20", "version": "2.4", "cuda_runtime": "12.1"},
               "packages": {"numpy": "1.26"}, "gsplat": {"commit": "937e"}}
    put(source / "environment.json", runtime)
    put(evaluation / "environment.json", {**runtime, "repository_commit": REEVAL_COMMIT})
    for path in (source, evaluation):
        put(path / "dataset_split.json", split)
    put(source / "train_metrics.json", {"step": 29999, "gaussian_count": 2, "training_time_seconds": 1300.})
    shapes = {"means": (2, 3), "scales": (2, 3), "quats": (2, 4), "opacities": (2,), "sh0": (2, 1, 3), "shN": (2, 15, 3)}
    checkpoint = source / "ckpts/ckpt_29999_rank0.pt"
    torch.save({"step": 29999, "splats": {key: torch.ones(shape) for key, shape in shapes.items()}}, checkpoint)
    (source / "run_command.txt").write_text(shlex.join(["CUDA_VISIBLE_DEVICES=1", "python", "simple_trainer.py",
                                                       "--data_dir", str(data)]))
    (evaluation / "run_command.txt").write_text(shlex.join([
        "CUDA_VISIBLE_DEVICES=0", "python", "simple_trainer.py", "--data_dir", str(data), "--ckpt", str(checkpoint),
        "--eval_warmup_renders", "10", "--data_factor", "4", "--test_every", "8", "--sh_degree", "3"]))
    evaluation.with_suffix(".exitcode").write_text("0\n")
    evaluation_files(evaluation, identity)
    put(source / "independent_eval/test_metrics.json", {"psnr": 26.65, "ssim": .87, "lpips": .07})
    env = {"gpu": 0, "data": str(data), "commit": "registration-commit", "trainer_sha256": "unchanged-trainer"}
    def register():
        return register_reference(root, project, source, evaluation, env, validate_checkpoint)
    return root, source, evaluation, env, identity, register


def test_reuse_is_a_reference_not_a_fabricated_train_state(historical, monkeypatch):
    root, source, evaluation, env, identity, register = historical
    reference = register()
    assert reference["psnr_reproduction_abs_difference"] == 0
    assert reference["historical_training_gpu"] == "1"
    assert reference["training_time_comparability"] == "NOT_ASSESSABLE"
    assert reference["historical_input_hashes"] == reference["historical_camera_sequence"] == "NOT_RECORDED"
    assert not (root / "parent.status.json").exists()
    assert not (root / "parent").exists()
    assert not (source / "v3_input_manifest.json").exists()
    assert register() == reference  # idempotent read-back; no re-evaluation/training
    from tools import ru_part_v3_screen as screen
    monkeypatch.setattr(screen, "output_root", lambda: root)
    assert screen.require_parent_ready()["checkpoint"] == reference["checkpoint"]
    command = screen.command_for("v3", {**env, "python": "python", "gsplat": "gsplat", "dino_repo": "dino",
                                         "dino_weight": "weights", "feature_cache": "features", "track_cache": "tracks"})
    assert "--checkpoint" not in command and "--replay-checkpoint" not in command
    assert command[command.index("--v3-stop-step") + 1] == "29999"


@pytest.mark.parametrize("change", ["config", "strategy", "steps", "split", "metric", "checkpoint", "image", "gpu", "dino", "exit"])
def test_registration_rejects_incompatible_or_broken_evidence(historical, change):
    root, source, evaluation, env, identity, register = historical
    if change == "config":
        put(source / "config.yaml", {"profile": "ru_part", "seed": 42})
    elif change == "strategy":
        path = source / "cfg.yml"
        path.write_text(path.read_text().replace("DelayedAbsGradStrategy", "LineageDelayedAbsGradStrategy"))
    elif change == "steps":
        path = source / "cfg.yml"
        path.write_text(path.read_text().replace("30000", "10000"))
    elif change == "split":
        value = json.loads((source / "dataset_split.json").read_text())
        value["train"].reverse()
        put(source / "dataset_split.json", value)
    elif change == "metric":
        evaluation_files(evaluation, identity, psnr=26.66)
    elif change == "checkpoint":
        path = evaluation / "run_command.txt"
        path.write_text(path.read_text().replace("ckpt_29999_rank0.pt", "another.pt"))
    elif change == "image":
        (Path(env["data"]) / "images_4_png/train0.png").write_bytes(b"changed")
    elif change == "gpu":
        path = evaluation / "run_command.txt"
        path.write_text(path.read_text().replace("DEVICES=0", "DEVICES=6"))
    elif change == "dino":
        path = evaluation / "ru_validation.json"
        put(path, {**json.loads(path.read_text()), "evaluation_imported_dino": True})
    else:
        evaluation.with_suffix(".exitcode").write_text("1")
    with pytest.raises(ValueError):
        register()
    assert not (root / "parent_reference.json").exists()


def test_registered_artifact_mutation_is_rejected(historical):
    root, source, evaluation, env, identity, register = historical
    register()
    (source / "ckpts/ckpt_29999_rank0.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_reference(root, verify=True)


def test_cfg_reading_never_constructs_python_objects(tmp_path):
    path = tmp_path / "cfg.yml"
    path.write_text("strategy: !!python/object/apply:os.system ['exit 1']\n")
    with pytest.raises((ValueError, AttributeError)):
        runtime_config(path)


def test_report_uses_new_inference_but_leaves_historical_training_time_unknown(historical):
    root, source, evaluation, env, identity, register = historical
    reference = register()
    put(root / "v3.status.json", {"status": "TRAIN_COMPLETE", "checkpoint_sha256": "v3-checkpoint"})
    put(root / "eval-v3.status.json", {"status": "EVAL_COMPLETE",
                                     "evaluation_fingerprint": reference["evaluation_fingerprint"]})
    put(root / "v3/v3_input_manifest.json", identity)
    put(root / "v3/v3_camera_sequence.json", [i % 161 for i in range(30000)])
    put(root / "v3/train_metrics.json", {"gaussian_count": 2, "training_time_seconds": 1301.})
    # Even an identical software environment must not legitimize old training cost.
    put(root / "v3/environment.json", json.loads((source / "environment.json").read_text()))
    evaluation_files(root / "eval-v3", identity, psnr=28.)
    roi = root / "fixed_roi.npy"
    np.save(roi, np.ones((3, 4), dtype=np.bool_))
    from puri_gs.v3_report import build_report
    result = build_report(root, roi_path=roi)
    assert result["invalid"] == [] and result["missing"] == []
    assert result["metrics"]["parent"]["psnr"] == 26.65
    assert result["delta_vs_parent"]["psnr"] == pytest.approx(1.35)
    assert result["resource_gates"]["fps_ratio"] is True
    assert result["resource_gates"]["training_time_ratio"] is None
    assert "training_time" not in result["ratios"]
    assert result["status"] == "COMPARISON_INCOMPLETE"
    # A device/protocol difference invalidates only the FPS comparison.
    changed = {**reference["evaluation_fingerprint"], "gpu_index": "6"}
    put(root / "eval-v3.status.json", {"status": "EVAL_COMPLETE", "evaluation_fingerprint": changed})
    assert build_report(root, roi_path=roi)["resource_gates"]["fps_ratio"] is None
