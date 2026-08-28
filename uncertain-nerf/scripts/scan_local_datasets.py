#!/usr/bin/env python3
"""Generate the read-only PURI-GS local dataset JSON and Markdown manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from puri_gs.dataset_manifest import DEFAULT_DATA_ROOT, scan_dataset_root, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--json-output", type=Path, default=Path("reports/local_dataset_manifest.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/local_dataset_manifest.md")
    )
    args = parser.parse_args()
    manifest = scan_dataset_root(args.root)
    write_manifest(manifest, args.json_output, args.markdown_output)
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
