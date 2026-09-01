import argparse
from pathlib import Path

from puri_gs.config import load_experiment_config
from run_puri_gs import PROJECT_ROOT, _build_command


def test_ru_checkpoint_evaluation_passes_no_training_network_assets(tmp_path):
    config = load_experiment_config(PROJECT_ROOT / "configs" / "puri_gs_ru_full30k.yaml")
    args = argparse.Namespace(
        data_factor=None,
        train_keyword=None,
        test_keyword=None,
        checkpoint=tmp_path / "ckpt_29999_rank0.pt",
        resume_checkpoint=None,
        cvtr_mask_dir=None,
        dino_repo_dir=None,
        dino_weight_path=None,
        feature_cache_dir=None,
        max_steps=None,
    )
    command = _build_command(
        args,
        config,
        tmp_path / "gsplat",
        tmp_path / "data",
        tmp_path / "eval",
    )
    joined = " ".join(map(str, command))
    assert "--ckpt" in command
    assert "puri_gs_ru_enabled" not in joined
    assert "dino" not in joined.casefold()
    assert "feature_cache" not in joined
    assert "mask_head" not in joined


def test_trainer_patch_does_not_import_dino_on_module_import():
    patch = (PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_ru.patch").read_text()
    assert "from puri_gs.dino_features import" not in patch
    assert "from puri_gs.ru_training import PURIGSRUTraining" in patch

