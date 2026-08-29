import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.analyze_android_pairwise import (
    _draw_pairwise_plot,
    _write_csv,
    _write_json,
    build_file_manifest,
    build_summaries,
    paired_bootstrap,
    split_eval_canvas,
    validate_metric_consistency,
)


def _write_canvas(path: Path, value: int) -> None:
    gt = np.full((8, 10, 3), value, dtype=np.uint8)
    render = np.full((8, 10, 3), value + 1, dtype=np.uint8)
    Image.fromarray(np.concatenate([gt, render], axis=1)).save(path)


def test_manifest_checks_identical_splits_render_names_and_embedded_gt(tmp_path):
    split = {"train": ["clutter.png"], "test": ["extra.png"]}
    for profile in ("b0", "b1", "a1"):
        train = tmp_path / profile
        evaluate = tmp_path / f"{profile}_eval"
        (evaluate / "renders").mkdir(parents=True)
        (evaluate / "stats").mkdir()
        train.mkdir()
        (train / "dataset_split.json").write_text(json.dumps(split))
        (evaluate / "dataset_split.json").write_text(json.dumps(split))
        _write_canvas(evaluate / "renders" / "test_step9999_0000.png", 50)
        (evaluate / "stats" / "test_step9999.json").write_text(
            json.dumps({"psnr": 20.0, "ssim": 0.8, "lpips": 0.2})
        )

    manifest, rows = build_file_manifest(tmp_path, expected_count=1)
    assert manifest["embedded_ground_truth_identical"]
    assert manifest["ground_truth_files"] == ["extra.png"]
    assert rows == [
        {
            "image_name": "extra.png",
            "eval_file": "test_step9999_0000.png",
        }
    ]
    gt, render = split_eval_canvas(
        tmp_path / "b0_eval" / "renders" / "test_step9999_0000.png"
    )
    assert gt.shape == render.shape == (8, 10, 3)
    metric_rows = [
        {
            "B0_psnr": 20.0,
            "B0_ssim": 0.8,
            "B0_lpips": 0.2,
            "B1_psnr": 20.0,
            "B1_ssim": 0.8,
            "B1_lpips": 0.2,
            "A1_psnr": 20.0,
            "A1_ssim": 0.8,
            "A1_lpips": 0.2,
        }
    ]
    consistency = validate_metric_consistency(tmp_path, metric_rows)
    assert consistency["a1"]["psnr"]["absolute_difference"] == 0.0


def test_bootstrap_and_pairwise_stable_decision_are_deterministic():
    rows = []
    for index in range(19):
        row = {"image_name": f"image_{index:02d}.png"}
        for baseline in ("B0", "B1"):
            row[f"A1_minus_{baseline}_psnr"] = 0.1 + index * 0.001
            row[f"A1_minus_{baseline}_ssim"] = 0.002
            row[f"A1_minus_{baseline}_lpips"] = -0.003
        rows.append(row)
    pairwise, bootstrap = build_summaries(rows)
    assert pairwise["decision"] == "PAIRWISE_STABLE"
    assert bootstrap["decision"] == "PAIRWISE_STABLE"
    first = paired_bootstrap([1.0, 2.0, 3.0])
    second = paired_bootstrap([1.0, 2.0, 3.0])
    assert first == second


def test_pairwise_plot_writes_png_without_matplotlib(tmp_path):
    rows = []
    for index in range(3):
        row = {"image_name": f"frame_{index}.png"}
        for baseline in ("B0", "B1"):
            for metric in ("psnr", "ssim", "lpips"):
                row[f"A1_minus_{baseline}_{metric}"] = (index - 1) * 0.1
        rows.append(row)
    output = tmp_path / "plot.png"
    _draw_pairwise_plot(rows, "psnr", output)
    assert output.is_file()
    assert Image.open(output).size == (1400, 560)
    csv_path = tmp_path / "rows.csv"
    json_path = tmp_path / "rows.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)
    assert csv_path.read_text(encoding="utf-8").startswith("image_name")
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["image_name"] == "frame_0.png"
