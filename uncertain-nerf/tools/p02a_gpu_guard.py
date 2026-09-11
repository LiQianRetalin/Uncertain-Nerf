#!/usr/bin/env python3
"""Assert that the fixed P02-A physical GPU is idle immediately before a timed run."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def query(command: list[str]) -> list[list[str]]:
    text = subprocess.check_output(command, text=True, encoding="utf-8")
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(text)) if row]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--append-jsonl", type=Path, required=True)
    args = parser.parse_args()
    inventory = query(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    processes = query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
    row = next(item for item in inventory if int(item[0]) == args.physical_gpu)
    uuid = row[1]
    competing = [item for item in processes if item[0] == uuid]
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "label": args.label,
        "physical_gpu": args.physical_gpu, "uuid": uuid, "name": row[2],
        "memory_used_mib": int(row[3]), "memory_total_mib": int(row[4]),
        "utilization_percent": int(row[5]), "compute_processes": competing,
        "idle_pass": not competing and int(row[3]) <= 2048 and int(row[5]) <= 10,
    }
    args.append_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.append_jsonl.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    if not result["idle_pass"]:
        raise RuntimeError(f"selected GPU is no longer idle before {args.label}: {result}")
    print(f"P02A_GPU_GUARD_PASS={args.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
