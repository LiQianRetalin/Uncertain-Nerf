import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from puri_gs.static_transient_gate import (
    EXPECTED_CONFIG,
    aggregate_summary,
    confusion_counts,
    decide_gate,
    ensure_output_absent,
    load_gate_config,
    metrics_from_counts,
)
from puri_gs.transient_layer import run_cpu_math_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "puri_gs_static_transient_gate.yaml"


def _row(kind: str):
    counts = {"tp": 80, "fp": 10, "tn": 910, "fn": 0} if kind == "transient" else {"tp": 0, "fp": 0, "tn": 1000, "fn": 0}
    return {
        "kind": kind,
        **metrics_from_counts(counts),
        "mean_alpha_inside_gt": 0.9 if kind == "transient" else 0.0,
        "mean_alpha_outside_gt": 0.001,
        "relative_patch_mae_reduction": 0.7 if kind == "transient" else 0.0,
        "background_change_l1": 0.001,
        "mean_transient_alpha": 0.005,
        "p95_transient_alpha": 0.01,
        "mean_abs_composite_static": 0.001,
        "psnr_composite_target": 35.0,
        "psnr_static_target": 30.0,
    }


def _engineering():
    return {
        "static_render_max_abs_difference": 0.0,
        "static_fps_relative_difference": 0.005,
    }


def _hard_gates():
    return {
        "PASS_EXECUTION": True,
        "STATIC_CHECKPOINT_UNCHANGED": True,
        "STATIC_GRADIENT_LEAK": False,
        "TRANSIENT_GEOMETRY_FROZEN": True,
        "PREMULTIPLIED_COMPOSITING_PASS": True,
        "GRID_PROJECTION_PASS": True,
        "GRID_COVERAGE_PASS": True,
    }


def test_frozen_config_loads_and_rejects_mutation(tmp_path: Path):
    assert load_gate_config(CONFIG_PATH) == EXPECTED_CONFIG
    changed = copy.deepcopy(EXPECTED_CONFIG)
    changed["transient"]["alpha_weight"] = 0.01
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen protocol"):
        load_gate_config(path)


def test_metrics_and_full_pass_decision():
    counts = confusion_counts([True, True, False, False], [True, False, True, False])
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    metrics = metrics_from_counts(counts)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    rows = [_row("clean") for _ in range(8)] + [_row("transient") for _ in range(8)]
    summary = aggregate_summary(rows, _engineering())
    gate = decide_gate(summary, _hard_gates(), EXPECTED_CONFIG)
    assert gate["decision"] == "ST_GATE_PASS"
    assert gate["hard_gate_pass"]


def test_clean_activation_forces_fail():
    rows = [_row("clean") for _ in range(8)] + [_row("transient") for _ in range(8)]
    for row in rows[:8]:
        row["mean_transient_alpha"] = 0.2
        row["fp"] = 100
        row["tn"] = 900
        row.update(metrics_from_counts({"tp": 0, "fp": 100, "tn": 900, "fn": 0}))
    summary = aggregate_summary(rows, _engineering())
    gate = decide_gate(summary, _hard_gates(), EXPECTED_CONFIG)
    assert gate["decision"] == "ST_GATE_FAIL"
    assert not gate["clean_gate_pass"]


def test_hard_failure_stops_numerical_gate_evaluation():
    rows = [_row("clean") for _ in range(8)] + [_row("transient") for _ in range(8)]
    summary = aggregate_summary(rows, _engineering())
    hard = _hard_gates()
    hard["GRID_COVERAGE_PASS"] = False
    gate = decide_gate(summary, hard, EXPECTED_CONFIG)
    assert gate["decision"] == "ST_GATE_FAIL"
    assert gate["numerical_gates_evaluated"] is False
    assert gate["selectivity_checks"] == {}


def test_output_overwrite_guard_and_cpu_math(tmp_path: Path):
    absent = tmp_path / "new"
    ensure_output_absent(absent)
    absent.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        ensure_output_absent(absent)
    assert run_cpu_math_contract()["status"] == "PASS"


def test_cli_dry_run_does_not_open_server_inputs():
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "fit_static_transient_gate.py"),
        "--gsplat-dir", "/missing/gsplat",
        "--data-dir", "/missing/room",
        "--derived-dir", "/missing/derived",
        "--checkpoint", "/missing/checkpoint.pt",
        "--dry-run",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRY_RUN_PASS"
    assert payload["formal_fit_started"] is False
    assert payload["inputs_opened"] is False
