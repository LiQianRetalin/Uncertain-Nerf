import csv
import json
from pathlib import Path

import torch

from tools.audit_android_ru_pairing import _load_run, decide


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_run(root: Path, profile: str, commit: str = "a" * 40) -> Path:
    root.mkdir(parents=True)
    common = {
        "profile": profile,
        "gsplat_version": "1.5.3",
        "seed": 42,
        "total_steps": 30000,
        "training": {"data_factor": 4, "test_every": 8},
    }
    if profile == "b1":
        common.update(
            {
                "strategy": {
                    "type": "default",
                    "absgrad": True,
                    "grow_grad2d": 0.0006,
                },
                "sh_degree": 3,
                "ssim_lambda": 0.2,
            }
        )
    else:
        common.update(
            {
                "absgrad": True,
                "grow_grad2d": 0.0006,
                "sh_degree": 3,
                "ssim_lambda": 0.2,
            }
        )
    _write_json(root / "config.yaml", common)
    train = [f"clutter_{index:03d}.JPG" for index in range(122)]
    test = [f"extra_{index:03d}.JPG" for index in range(19)]
    split = {
        "protocol": "filename-keyword",
        "train_keyword": "clutter",
        "test_keyword": "extra",
        "train": train,
        "test": test,
    }
    _write_json(root / "dataset_split.json", split)
    environment = {
        "repository_commit": commit,
        "torch": {
            "gpu": "NVIDIA L20",
            "version": "2.4.0+cu121",
            "cuda_runtime": "12.1",
        },
        "gsplat": {"version": "1.5.3", "commit": "9" * 40},
        "packages": {"torchmetrics": "1.4", "fused-ssim": "0.1"},
    }
    _write_json(root / "environment.json", environment)
    _write_json(root / "independent_eval" / "environment.json", environment)
    _write_json(root / "independent_eval" / "dataset_split.json", split)
    _write_json(
        root / "independent_eval" / "ru_validation.json",
        {
            "standard_checkpoint_load_pass": True,
            "evaluation_imported_dino": False,
            "evaluation_loaded_mask_head": False,
            "evaluation_rasterization_count_ratio": 1.0,
        },
    )
    metrics_path = root / "independent_eval" / "per_image_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["image_name", "psnr", "ssim", "lpips"]
        )
        writer.writeheader()
        for name in test:
            writer.writerow({"image_name": name, "psnr": 30, "ssim": 0.9, "lpips": 0.1})
    (root / "ckpts").mkdir(parents=True)
    torch.save(
        {"step": 29999, "splats": {"means": torch.zeros(4, 3)}},
        root / "ckpts" / "ckpt_29999_rank0.pt",
    )
    (root / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (root / "run_command.txt").write_text(
        "simple_trainer.py --train_keyword clutter --test_keyword extra\n",
        encoding="utf-8",
    )
    return root


def test_strict_android_pairing_accepts_matched_runs(tmp_path: Path):
    b1 = _load_run(_make_run(tmp_path / "b1", "b1"), "b1")
    ru = _load_run(_make_run(tmp_path / "ru", "ru"), "ru")
    report = decide(b1, ru)
    assert report["decision"] == "ANDROID_PAIRING_VALID"
    assert report["mismatches"] == []
    assert report["b1"]["checkpoint"]["gaussian_count"] == 4


def test_strict_android_pairing_rejects_different_training_commits(tmp_path: Path):
    b1 = _load_run(_make_run(tmp_path / "b1", "b1", "a" * 40), "b1")
    ru = _load_run(_make_run(tmp_path / "ru", "ru", "b" * 40), "ru")
    report = decide(b1, ru)
    assert report["decision"] == "ANDROID_PAIRING_INVALID"
    assert "same_training_commit" in report["mismatches"]


def test_strict_android_pairing_rejects_metric_order_drift(tmp_path: Path):
    b1_path = _make_run(tmp_path / "b1", "b1")
    ru_path = _make_run(tmp_path / "ru", "ru")
    metrics = ru_path / "independent_eval" / "per_image_metrics.csv"
    lines = metrics.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    metrics.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = decide(_load_run(b1_path, "b1"), _load_run(ru_path, "ru"))
    assert report["decision"] == "ANDROID_PAIRING_INVALID"
    assert "metric_rows_follow_test_order_ru" in report["mismatches"]
