#!/usr/bin/env python3
"""Create the Phase 4A decision reports and compact evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.static_transient_gate import ensure_output_absent, write_json


MARKDOWN_REPORT = PROJECT_ROOT / "reports" / "PHASE_4A_STATIC_TRANSIENT_GATE.md"
JSON_REPORT = PROJECT_ROOT / "reports" / "phase_4a_static_transient_gate.json"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return str(value)


def _report_payload(run_dir: Path) -> dict[str, Any]:
    provenance = _read_json(run_dir / "provenance.json")
    input_contract = _read_json(run_dir / "input_contract.json")
    summary = _read_json(run_dir / "metrics" / "summary.json")
    gate = _read_json(run_dir / "metrics" / "gate.json")
    per_frame = _read_json(run_dir / "metrics" / "per_frame.json")
    unit_tests = _read_json(run_dir / "unit_test_summary.json")
    return {
        "schema_version": 1,
        "phase": "PURI-GS-ST Phase 4A",
        "decision": gate["decision"],
        "run_dir": str(run_dir),
        "provenance": provenance,
        "input_contract_status": input_contract["status"],
        "hard_gates": gate["hard_gates"],
        "gate_groups": {
            "hard_gate_pass": gate["hard_gate_pass"],
            "selectivity_gate_pass": gate["selectivity_gate_pass"],
            "clean_gate_pass": gate["clean_gate_pass"],
            "reconstruction_gate_pass": gate["reconstruction_gate_pass"],
            "static_inference_gate_pass": gate["static_inference_gate_pass"],
        },
        "checks": {
            "selectivity": gate["selectivity_checks"],
            "clean": gate["clean_checks"],
            "reconstruction": gate["reconstruction_checks"],
            "static_inference": gate["static_inference_checks"],
        },
        "summary": summary,
        "per_frame": per_frame,
        "unit_test_summary": unit_tests,
    }


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    patched = summary["patched_micro"]
    engineering = summary["engineering"]
    hard = payload["hard_gates"]
    rows = [
        "# PURI-GS Phase 4A：显式静态—瞬态 Gaussian 分解可行性门",
        "",
        f"结论：`{payload['decision']}`",
        "",
        "本报告只回答冻结 B1 静态地图时，每视图容量受限瞬态 Gaussian 层的合成门可行性；不启动联合训练或 Phase 4B。",
        "",
        "## 执行硬门",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
    ]
    rows.extend(f"| {name} | `{value}` |" for name, value in hard.items())
    rows.extend(
        [
            "",
            "## Patched 八帧 micro 指标",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| Precision | {_number(patched['precision'])} |",
            f"| Recall | {_number(patched['recall'])} |",
            f"| F1 | {_number(patched['f1'])} |",
            f"| IoU | {_number(patched['iou'])} |",
            f"| F1 >= 0.60 帧数 | {summary['patched_frames_f1_at_least_0_60']}/8 |",
            f"| predicted mask ratio（帧均值） | {_number(summary['patched_mean_predicted_mask_ratio'])} |",
            f"| GT mask ratio（帧均值） | {_number(summary['patched_mean_ground_truth_mask_ratio'])} |",
            f"| mean alpha inside GT | {_number(summary['patched_mean_alpha_inside_gt'])} |",
            f"| mean alpha outside GT | {_number(summary['patched_mean_alpha_outside_gt'])} |",
            "",
            "## Clean 八帧关闭指标",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| Clean FPR | {_number(summary['clean_fpr'])} |",
            f"| predicted mask ratio | {_number(summary['clean_mean_predicted_mask_ratio'])} |",
            f"| mean transient alpha | {_number(summary['clean_mean_transient_alpha'])} |",
            f"| max per-frame mean alpha | {_number(summary['clean_max_frame_mean_transient_alpha'])} |",
            f"| mean p95 transient alpha | {_number(summary['clean_mean_p95_transient_alpha'])} |",
            f"| mean abs(composite-static) | {_number(summary['clean_mean_abs_composite_static'])} |",
            f"| PSNR(composite,target) | {_number(summary['clean_mean_psnr_composite_target'])} |",
            f"| PSNR(static,target) | {_number(summary['clean_mean_psnr_static_target'])} |",
            "",
            "## 重建与静态保护",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| patch MAE 相对降低 | {_number(summary['patched_mean_relative_patch_mae_reduction'])} |",
            f"| patched 背景变化 L1/像素 | {_number(summary['patched_mean_background_change_l1'])} |",
            f"| 静态渲染最大差异 | {_number(engineering['static_render_max_abs_difference'])} |",
            f"| 静态 FPS before | {_number(engineering['static_fps_before'])} |",
            f"| 静态 FPS after | {_number(engineering['static_fps_after'])} |",
            f"| 静态 FPS 相对差异 | {_number(engineering['static_fps_relative_difference'])} |",
            "",
            "## 工程事实",
            "",
            f"- 平均每帧拟合时间：{_number(engineering['mean_fit_time_per_frame_seconds'])} 秒。",
            "",
            f"- 总拟合时间：{_number(engineering['total_fit_time_seconds'])} 秒。",
            "",
            f"- 峰值显存：{engineering['peak_vram_bytes']} bytes。",
            "",
            f"- 瞬态参数总大小：{engineering['transient_checkpoint_total_bytes']} bytes。",
            "",
            "- 静态推理未加载瞬态参数，未保存新的静态 checkpoint。",
            "",
            "阶段在该决策处停止；未解冻静态层，未做 densification/pruning，未调节固定超参数。",
            "",
        ]
    )
    return "\n".join(rows)


def _archive(run_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path, str]:
    commit_short = payload["provenance"]["commit_short"]
    archive_path = PROJECT_ROOT / "analysis" / f"puri_gs_phase4a_static_transient_{commit_short}.tar.gz"
    checksum_path = PROJECT_ROOT / "analysis" / f"puri_gs_phase4a_static_transient_{commit_short}.sha256"
    ensure_output_absent(archive_path)
    ensure_output_absent(checksum_path)
    required = [
        MARKDOWN_REPORT,
        JSON_REPORT,
        run_dir / "provenance.json",
        run_dir / "input_contract.json",
        run_dir / "config.json",
        run_dir / "unit_test_summary.json",
        run_dir / "actual_run_command.txt",
        run_dir / "environment.json",
        run_dir / "metrics" / "per_frame.csv",
        run_dir / "metrics" / "per_frame.json",
        run_dir / "metrics" / "summary.json",
        run_dir / "metrics" / "gate.json",
    ]
    visualizations = sorted((run_dir / "visualizations").glob("*/transient_alpha.png"))
    visualizations += sorted((run_dir / "visualizations").glob("*/tp_fp_fn_overlay.png"))
    visualizations += sorted((run_dir / "visualizations").glob("*/composite_error.png"))
    required.extend(visualizations)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"evidence package inputs are missing: {missing}")
    prefix = f"puri_gs_phase4a_static_transient_{commit_short}"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in required:
            if path in (MARKDOWN_REPORT, JSON_REPORT):
                relative = Path("reports") / path.name
            else:
                relative = Path("run") / path.relative_to(run_dir)
            archive.add(path, arcname=(Path(prefix) / relative).as_posix(), recursive=False)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    return archive_path, checksum_path, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    if branch != "dev":
        raise RuntimeError("Phase 4A reporting must remain on dev")
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else (
        PROJECT_ROOT / "analysis" / f"static_transient_gate_{commit[:7]}"
    ).resolve()
    if run_dir.name != f"static_transient_gate_{commit[:7]}":
        raise RuntimeError("run directory does not match the current commit")
    ensure_output_absent(MARKDOWN_REPORT)
    ensure_output_absent(JSON_REPORT)
    payload = _report_payload(run_dir)
    if payload["provenance"]["commit"] != commit:
        raise RuntimeError("run provenance does not match the current commit")
    MARKDOWN_REPORT.write_text(_markdown(payload), encoding="utf-8")
    write_json(JSON_REPORT, payload)
    archive_path, checksum_path, digest = _archive(run_dir, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "markdown_report": str(MARKDOWN_REPORT),
                "json_report": str(JSON_REPORT),
                "evidence_archive": str(archive_path),
                "evidence_sha256_file": str(checksum_path),
                "evidence_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
