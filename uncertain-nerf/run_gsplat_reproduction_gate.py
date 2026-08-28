#!/usr/bin/env python3
"""Check a 30k Mip-NeRF 360 Garden run against the pinned gsplat reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v8_robot.gate import load_json
from v8_robot.reproduction_gate import evaluate_reproduction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True, help="Official trainer test stats JSON")
    parser.add_argument(
        "--limits",
        default="configs/gsplat153_garden_reproduction_limits.json",
    )
    parser.add_argument("--output", help="Optional decision JSON")
    args = parser.parse_args()

    result = evaluate_reproduction(load_json(args.stats), load_json(args.limits))
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
