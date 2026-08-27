#!/usr/bin/env python3
"""CLI for the read-only Fern input audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v8_robot.dataset_audit import audit_fern_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the LLFF Fern directory")
    parser.add_argument("--expected-images", type=int, default=20)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    result = audit_fern_dataset(args.data, args.expected_images)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
