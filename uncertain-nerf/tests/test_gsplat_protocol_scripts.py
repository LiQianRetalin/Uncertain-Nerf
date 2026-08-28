from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_training_script_matches_paper_comparable_split_and_scale():
    script = _script("train_gsplat_robot_baseline.sh")
    assert "--data_factor 4" in script
    assert "--test_every 8" in script
    assert "--val_every 0" in script
    assert "--eval_split test" in script
    assert "--eval_steps -1" in script
    assert "images_4" in script
    assert "--data_factor 2" not in script


def test_evaluation_script_uses_the_same_held_out_test_protocol():
    script = _script("evaluate_gsplat_robot_baseline.sh")
    assert "--data_factor 4" in script
    assert "--test_every 8" in script
    assert "--val_every 0" in script
    assert "--eval_split test" in script
    assert "--data_factor 2" not in script
