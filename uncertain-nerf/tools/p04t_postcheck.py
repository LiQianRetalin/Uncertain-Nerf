"""Independent read-only audit of P04-T budget, states and event evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def assert_true(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    plan = read_csv(work / "run_plan.csv")[0]
    status = json.loads((work / "status.json").read_text())
    assert_true(status["stage"] == "COMPLETE" and status["completed_step"] == 19999,
                "P04T 20k controlled stop not complete")
    assert_true(int(plan["planned_schedule_steps"]) == 30000, "P04T original schedule shortened")
    updates = read_csv(work / "training_updates_ledger.csv")
    assert_true(len(updates) == 20000 and [int(row["step"]) for row in updates] == list(range(20000)),
                "P04T updates are not exactly one normal 20k prefix")
    assert_true(set(row["attempt"] for row in updates) == {"1"}, "P04T unexpected training attempt")
    calls = read_csv(work / "call_budget_ledger.csv")
    reservations = [row for row in calls if int(row["charged"]) == 1]
    outcomes = [row for row in calls if int(row["charged"]) == 0]
    assert_true(len(reservations) == len(outcomes), "P04T call reservation/outcome mismatch")
    charged = len(reservations)
    assert_true(charged == int(status["charged_extra_calls"]) and charged <= 300,
                "P04T extra-call budget mismatch")
    kinds = Counter(row["kind"] for row in reservations)
    assert_true(kinds["A_exact_probe_forward"] == 200,
                "P04T true-window A probe count not 200")
    assert_true(kinds["D_clone_render_forward"] == 8
                and kinds["D_clone_photo_backward"] == 8
                and kinds["D_and_R_exact_probe_forward"] == 8,
                "P04T D clone call count not 8 each")
    assert_true(charged == 17 + 200 + 24, "P04T preflight or training call inventory changed")
    assert_true(float(status["gpu_wall_upper_bound_seconds"]) <= 7200,
                "P04T GPU time upper bound exceeded")
    states = []
    required = {"completed_step", "next_step", "splats", "gaussian_optimizers",
                "schedulers", "strategy_state", "ru_state", "shadow_rows", "selected_ids",
                "sampler", "rng", "cfg_sha256", "feature_manifest_sha256", "sh_degree_next"}
    for step in (10999, 18999, 19999):
        path = work / "states" / f"complete_after_step{step}.pt"
        manifest = json.loads(path.with_suffix(".json").read_text())
        assert_true(path.is_file() and sha(path) == manifest["sha256"],
                    f"P04T snapshot {step} SHA mismatch")
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert_true(required <= data.keys() and data["completed_step"] == step
                    and data["next_step"] == step + 1,
                    f"P04T snapshot {step} key/step mismatch")
        assert_true(set(data["splats"]) == {"means", "scales", "quats", "opacities", "sh0", "shN"},
                    f"P04T snapshot {step} Gaussian keys changed")
        assert_true(set(data["gaussian_optimizers"]) == set(data["splats"]),
                    f"P04T snapshot {step} Gaussian optimizer keys incomplete")
        assert_true(data["cfg_sha256"] == sha(work / "outputs" / plan["run_id"] / "cfg.yml"),
                    f"P04T snapshot {step} config changed")
        assert_true(data["feature_manifest_sha256"] == "459cf0694d82056f4b4b7064a9f125222a4b13f53244189cdc582b12a22dc782",
                    f"P04T snapshot {step} feature identity changed")
        expected_phase = min((step + 1) // 1000, 3)
        assert_true(data["sh_degree_next"] == expected_phase,
                    f"P04T snapshot {step} SH phase mismatch")
        assert_true(data["sampler"]["next_step"] == step + 1,
                    f"P04T snapshot {step} camera sampler position mismatch")
        states.append({"completed_step": step, "path_server": str(path),
                       "sha256": manifest["sha256"], "bytes": path.stat().st_size,
                       "gaussians": len(data["splats"]["means"]),
                       "sh_degree_next": data["sh_degree_next"],
                       "sampler_next_step": data["sampler"]["next_step"],
                       "keys_complete": True, "readback": "PASS"})
    with (work / "state_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(states[0]))
        writer.writeheader(); writer.writerows(states)
    windows = {}
    for name, start, end in (("W1", 10901, 11000), ("W2", 18901, 19000)):
        alignment = json.loads((work / f"{name}_alignment.json").read_text())
        assert_true(alignment["count_exact"] and alignment["grad_pass"]
                    and alignment["event_step"] == end,
                    f"P04T {name} original strategy mismatch")
        path = work / f"{name}_per_step_probe.npz"
        manifest = json.loads((work / f"{name}_data_manifest.json").read_text())
        assert_true(sha(path) == manifest["sha256"], f"P04T {name} data changed")
        data = np.load(path, allow_pickle=False)
        assert_true(np.array_equal(data["steps"], np.arange(start, end + 1)),
                    f"P04T {name} step sequence invalid")
        assert_true(np.array_equal(data["d_steps"], np.array([start, start + 33, start + 66, end])),
                    f"P04T {name} D moments invalid")
        assert_true(np.array_equal(data["original_count"], data["z"].sum(axis=0)),
                    f"P04T {name} original projection count differs")
        windows[name] = {"step_start": start, "event_step": end,
                         "A_probe_steps": len(data["steps"]), "D_moments": data["d_steps"].tolist(),
                         "probe_ids": len(data["ids"]), "data_sha256": manifest["sha256"],
                         "alignment": alignment}
    output = {"status": "P04T_POSTCHECK_PASS", "normal_updates": len(updates),
              "training_attempts": 1, "charged_extra_calls": charged,
              "call_kinds": dict(kinds), "gpu_wall_upper_bound_seconds": status["gpu_wall_upper_bound_seconds"],
              "snapshots": states, "windows": windows}
    (work / "postcheck.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(output["status"])


if __name__ == "__main__":
    main()
