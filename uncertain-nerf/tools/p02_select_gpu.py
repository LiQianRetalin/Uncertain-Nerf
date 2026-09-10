#!/usr/bin/env python3
"""Select an idle physical GPU using the frozen P02 priority."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path


PRIORITY = [6, 7, 0, 1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--max-memory-mib", type=int, default=2048)
    return parser.parse_args()


def _query(command: list[str]) -> list[list[str]]:
    output = subprocess.check_output(command, text=True, encoding="utf-8")
    return [
        [value.strip() for value in row]
        for row in csv.reader(io.StringIO(output))
        if row
    ]


def main() -> int:
    args = parse_args()
    rows = _query(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_rows = _query(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    busy_uuids = {row[0] for row in process_rows if row and row[0] != "[N/A]"}
    inventory = []
    by_index = {}
    for row in rows:
        entry = {
            "physical_index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "memory_used_mib": int(row[3]),
            "memory_total_mib": int(row[4]),
            "utilization_percent": int(row[5]),
            "has_compute_process": row[1] in busy_uuids,
        }
        entry["idle"] = (
            not entry["has_compute_process"]
            and entry["memory_used_mib"] <= args.max_memory_mib
            and entry["utilization_percent"] <= args.max_utilization
        )
        inventory.append(entry)
        by_index[entry["physical_index"]] = entry
    selected = next(
        (by_index[index] for index in PRIORITY if index in by_index and by_index[index]["idle"]),
        None,
    )
    result = {
        "schema": "puri-gs-p02-gpu-selection-v1",
        "priority": PRIORITY,
        "inventory": inventory,
        "selected": selected,
        "process_rows": process_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if selected is None:
        print("P02_GPU_SELECTION=NO_IDLE_GPU")
        return 2
    print(selected["physical_index"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
