"""One-time GPU selection for a serial, same-device Garden screening."""
from __future__ import annotations

import csv
import subprocess

GPU_PRIORITY = (6, 7, 0, 1, 2, 3, 4, 5)


def gpu_inventory():
    def query(fields):
        return subprocess.check_output(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"], text=True,
        )
    rows = list(csv.reader(query("--query-gpu=index,uuid,name,memory.used,utilization.gpu").splitlines()))
    processes = list(csv.reader(query("--query-compute-apps=gpu_uuid,pid").splitlines()))
    busy_uuids = {row[0].strip() for row in processes if len(row) == 2}
    result = []
    for raw in rows:
        index, uuid, name, memory, utilization = (value.strip() for value in raw)
        result.append({"index": int(index), "uuid": uuid, "name": name,
                       "memory_used_mib": int(memory), "utilization_percent": int(utilization),
                       "compute_process_present": uuid in busy_uuids})
    return result


def gpu_idle(row):
    # Allow small driver/display allocations, but never share with a compute job.
    return (not row["compute_process_present"] and row["memory_used_mib"] <= 512
            and row["utilization_percent"] <= 5)


def select_gpu(rows, requested="auto", *, pinned=None):
    """Do not silently change the device after any measured stage has started."""
    requested = str(requested)
    if requested != "auto" and (not requested.isdigit() or int(requested) < 0):
        raise ValueError("--gpu must be auto or a non-negative physical GPU index")
    if pinned is not None:
        if requested != "auto" and int(requested) != pinned:
            raise ValueError(f"Screening measurements already use GPU {pinned}; device changes are disabled")
        order = (pinned,)
    else:
        order = GPU_PRIORITY if requested == "auto" else (int(requested),)
    indexed = {row["index"]: row for row in rows}
    for index in order:
        if index in indexed and gpu_idle(indexed[index]):
            return indexed[index]
    suffix = f"; GPU {pinned} is pinned to preserve comparability" if pinned is not None else ""
    raise RuntimeError(f"No idle GPU in priority {list(order)}{suffix}; no task was started")
