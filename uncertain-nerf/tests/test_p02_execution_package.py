from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def test_p02_run_plan_is_exact_and_bounded() -> None:
    with (ROOT / "reports" / "run_plan_P02.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [row["run_id"] for row in rows] == [
        "P02-android-robustsplat",
        "P02-android-sls-mlp",
        "P02-patio_high-robustsplat",
        "P02-patio_high-sls-mlp",
    ]
    assert {row["seed"] for row in rows} == {"42"}
    assert {row["loader_data_factor"] for row in rows} == {"1"}
    assert {row["max_attempts"] for row in rows} == {"2"}


def test_p02_server_pipeline_preserves_scope_and_protocol() -> None:
    script = (ROOT / "scripts" / "p02_server_pipeline.sh").read_text(
        encoding="utf-8"
    )
    assert 'run_ids=(' in script
    assert "smoke_updates=$((smoke_updates + 100))" in script
    assert 'if [[ "$smoke_updates" -gt 800 ]]' in script
    assert '--iterations "$steps" --seed 42 --resolution 1 --eval' in script
    assert "--loss_type robust --semantics --no-cluster" in script
    assert "--ubp" not in script
    assert "--data_factor 1" in script
    assert 'mkdir -p "$view/sparse/0"' in script
    assert 'ln -s "$dataset/sparse/0/$name" "$view/sparse/0/$name"' in script
    assert 'robust_dataset="$work_root/dataset_views/robustsplat/' in script
    assert "P02_TIMING_WARMUP=10" in script
    assert "P02_TIMING_REPEATS=3" in script
    assert "Corner" not in script
    assert "Garden" not in script


def test_p02_minimal_patches_are_pinned_to_expected_files() -> None:
    robust = (ROOT / "patches" / "p02_robustsplat_seed_float_timing.patch").read_text(
        encoding="utf-8"
    )
    spotless = (ROOT / "patches" / "p02_spotless_seed_float_timing.patch").read_text(
        encoding="utf-8"
    )
    assert "diff --git a/train.py b/train.py" in robust
    assert "diff --git a/render.py b/render.py" in robust
    assert "diff --git a/utils/general_utils.py b/utils/general_utils.py" in robust
    assert "scene/dataset_readers.py" not in robust
    assert spotless.count("diff --git") == 1
    assert "diff --git a/examples/spotless_trainer.py" in spotless
    assert "P02_FLOAT_PREDICTIONS" in robust and "P02_FLOAT_PREDICTIONS" in spotless


def test_p02_sls_notebook_hash_matches_pinned_spotless_commit() -> None:
    extractor = (ROOT / "tools" / "p02_extract_sls_features.py").read_text(
        encoding="utf-8"
    )
    assert (
        'NOTEBOOK_SHA256 = "a19857e7c659a82341fee76dfb61d595f82e50fca310e5bbab6c8783bdc546e9"'
        in extractor
    )
    assert 'MODEL_ID = "sd2-community/stable-diffusion-2-1"' in extractor
    assert (
        'MODEL_REVISION = "bb2154823665391b4fb29b0b9cf82a198964ee05"'
        in extractor
    )
    assert "MODEL_DOWNLOAD_ATTEMPTS = 30" in extractor
    assert '"unet/diffusion_pytorch_model.bin"' in extractor
    assert '"text_encoder/pytorch_model.bin"' in extractor
    assert '"vae/diffusion_pytorch_model.bin"' in extractor
    assert "_prefetch_model_files()" in extractor
    assert "revision_count != 3" in extractor


def test_blocked_report_does_not_claim_results() -> None:
    status = json.loads((ROOT / "reports" / "p02" / "status.json").read_text("utf-8"))
    with (ROOT / "reports" / "p02" / "summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        summary = list(csv.DictReader(stream))
    assert status["status"] == "BLOCKED_SERVER_SSH_AUTHENTICATION"
    assert status["run_count_started"] == 0
    assert all(row["status"] == "BLOCKED_NOT_RUN" for row in summary)
    assert all(not row["psnr"] and not row["ssim"] and not row["lpips"] for row in summary)


def test_external_source_cache_is_ignored() -> None:
    patterns = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/P02-external/" in patterns
