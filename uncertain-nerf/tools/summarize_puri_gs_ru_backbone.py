#!/usr/bin/env python3
"""Produce the final RU backbone state from the five frozen prerequisite reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required decision report is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    status = value.get("decision", value.get("status")) if isinstance(value, dict) else None
    if not isinstance(value, dict) or not isinstance(status, str):
        raise ValueError(f"invalid decision report: {path}")
    return {**value, "resolved_status": status}


def decide(decisions: dict[str, str]) -> str:
    prerequisites = {
        "provenance": "RU_PROVENANCE_COMPLETE",
        "android_pairing": "ANDROID_PAIRING_VALID",
        "android_efficiency": "ANDROID_EFFICIENCY_PASS",
        "phase_r": "RU_RECONSTRUCTION_PASS",
        "room_efficiency": "EFFICIENCY_AUDIT_PASS",
    }
    failed = [name for name, expected in prerequisites.items() if decisions.get(name) != expected]
    if failed:
        raise RuntimeError("final backbone decision is blocked by: " + ", ".join(failed))
    if decisions.get("garden") not in {"GARDEN_CLEAN_PASS", "GARDEN_CLEAN_FAIL"}:
        raise RuntimeError("final backbone decision is blocked by: garden")
    if decisions.get("ontogo") not in {"ONTOGO_DYNAMIC_PASS", "ONTOGO_DYNAMIC_FAIL"}:
        raise RuntimeError("final backbone decision is blocked by: ontogo")
    garden_pass = decisions.get("garden") == "GARDEN_CLEAN_PASS"
    ontogo_pass = decisions.get("ontogo") == "ONTOGO_DYNAMIC_PASS"
    if garden_pass and ontogo_pass:
        return "RU_BACKBONE_ACCEPTED"
    if garden_pass:
        return "RU_CLEAN_COMPACT_ONLY"
    if ontogo_pass:
        return "RU_DYNAMIC_WITH_CLEAN_TRADEOFF"
    return "RU_SCENE_SPECIFIC"


def render_markdown(report: dict[str, Any]) -> str:
    rows = [f"| {name} | `{value}` |" for name, value in report["inputs"].items()]
    return "\n".join(
        [
            "# PURI-GS-RU 主干判定",
            "",
            f"最终状态：`{report['decision']}`",
            "",
            "| 输入 | 状态 |",
            "| --- | --- |",
            *rows,
            "",
            "该状态由冻结门槛机械生成；不触发调参、uncertainty、多随机种子或新场景实验。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--android-pairing", type=Path, required=True)
    parser.add_argument("--android-efficiency", type=Path, required=True)
    parser.add_argument("--phase-r", type=Path, required=True)
    parser.add_argument("--room-efficiency", type=Path, required=True)
    parser.add_argument("--garden", type=Path, required=True)
    parser.add_argument("--ontogo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "provenance": _read(args.provenance.resolve())["resolved_status"],
        "android_pairing": _read(args.android_pairing.resolve())["resolved_status"],
        "android_efficiency": _read(args.android_efficiency.resolve())["resolved_status"],
        "phase_r": _read(args.phase_r.resolve())["resolved_status"],
        "room_efficiency": _read(args.room_efficiency.resolve())["resolved_status"],
        "garden": _read(args.garden.resolve())["resolved_status"],
        "ontogo": _read(args.ontogo.resolve())["resolved_status"],
    }
    report = {
        "protocol": "puri-gs-ru-backbone-decision-1",
        "decision": decide(inputs),
        "inputs": inputs,
    }
    output_dir = args.output_dir.expanduser().resolve()
    json_path = output_dir / "ru_backbone_decision.json"
    markdown_path = output_dir / "RU_BACKBONE_DECISION.md"
    if json_path.exists() or markdown_path.exists():
        raise RuntimeError("refusing to overwrite an existing backbone decision")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
