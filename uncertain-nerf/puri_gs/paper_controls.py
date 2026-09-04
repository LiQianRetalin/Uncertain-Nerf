"""Fixed RU paper-control schedules and observation-only, bounded diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from puri_gs.config import parse_refine_windows, validate_experiment_config
from puri_gs.delayed_absgrad import DelayedAbsGradSchedule, topology_event_summary

MASK_SAMPLE_STEPS = (500, 2000, 5000, 9999, 10000, 15000, 18000, 19999, 20000, 23999, 29999)
NOT_AVAILABLE = "not_available"
TOPOLOGY_FIELDS = (
    "step", "mask_supervision_scale", "head_paused", "reset_event",
    "gaussian_count_before", "split_count", "clone_count", "prune_count",
    "gaussian_count_after", "eligible_count", "grad2d_mean", "grad2d_p50",
    "grad2d_p90", "grad2d_p99", "opacity_mean", "opacity_p10", "opacity_p50",
)
MASK_FIELDS = (
    "step", "mask_supervision_scale", "head_paused", "raw_mask_mean",
    "raw_mask_p10", "raw_mask_p50", "raw_mask_p90", "hard_static_retention",
    "eroded_static_retention", "rgb_residual_mean",
)


def write_json_new(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def schedule_from_config(config: dict) -> DelayedAbsGradSchedule:
    return DelayedAbsGradSchedule(
        **{key: config[key] for key in (
            "densify_start_step", "densify_stop_step", "densify_every",
            "opacity_reset_start_step", "opacity_reset_every", "mask_pause_after_reset",
        )},
        refine_windows=parse_refine_windows(config.get("refine_windows"), config["total_steps"]),
    )


def scale_for_step(step: int, switch_step: int, begin_step: int = 500) -> str:
    if step < begin_step:
        return "not_applicable"
    return "coarse" if step < switch_step else "fine"


def resolved_schedule(config: dict, run_steps: int | None = None) -> dict:
    schedule = schedule_from_config(config)
    steps = config["total_steps"] if run_steps is None else run_steps
    events = topology_event_summary(
        delayed_topology=True, total_steps=steps,
        mask_pause_after_reset=config["mask_pause_after_reset"], delayed_schedule=schedule,
    )
    fine = sum(step >= config["bootstrap_switch_step"] for step in events["refine_steps"])
    return {
        "scale_switch_step": config["bootstrap_switch_step"],
        "mask_begin_step": config["mask_begin_step"],
        "run_steps": steps, "declared_total_steps": config["total_steps"],
        "refine_steps": events["refine_steps"],
        "refine_event_count": events["refine_event_count"],
        "reset_steps": events["reset_steps"],
        "head_pause_ranges": events["mask_pause_segments"],
        "stats_window_inclusive": events["statistics_window_inclusive"],
        "stats_semantics": {
            "start_step": 0,
            "stop_exclusive": min(steps, schedule.statistics_stop_step),
            "includes_current_step_before_refine": True,
            "normalization": "grad2d / count.clamp_min(1)",
            "clear": "after each refine: grad2d/count and enabled radii; reset adds no clear",
            "topology_order": "clone, split (one parent to two children), prune, clear, opacity reset",
        },
        "post_backward_order": events["post_backward_order"],
        "fine_refine_event_count": fine,
        "scale_refine_overlap": fine / len(events["refine_steps"]) if events["refine_steps"] else None,
    }


def resolved_config_diff(before: dict, after: dict) -> dict:
    """Keep identity metadata visible and audit every algorithm/config field."""
    validate_experiment_config(before)
    validate_experiment_config(after)
    changes = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)
    }
    metadata = {key: changes.pop(key) for key in ("paper_control",) if key in changes}
    return {"identity_metadata": metadata, "algorithm_changes": changes}


def check_control_diffs(ru: dict, align: dict, tar: dict) -> dict:
    align_diff, late_diff = resolved_config_diff(ru, align), resolved_config_diff(align, tar)
    if set(align_diff["algorithm_changes"]) != {"bootstrap_switch_step"}:
        raise ValueError("RU-Align must change only scale switch")
    if set(late_diff["algorithm_changes"]) != {"refine_windows"}:
        raise ValueError("RU-TAR must change only the late topology window factor")
    return {"RU_Align_minus_RU": align_diff, "RU_TAR_minus_RU_Align": late_diff}


def _finite_row(row: dict, fields: tuple) -> None:
    for field in fields:
        value = row[field]
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite {field} at step {row['step']}")


class PaperControlRecorder:
    """No tensor logging on ordinary steps; optional topology quantiles omitted."""

    def __init__(self, directory: Path, config: dict, run_steps: int):
        self.directory, self.config = directory, config
        self.schedule = resolved_schedule(config, run_steps)
        self.topology_steps: list[int] = []
        self.mask_steps: list[int] = []
        self.previous_count: int | None = None
        self.full_renders = self.coarse_renders = self.iterations = 0
        self.current_step: int | None = None
        self.render_sizes: list[tuple[int, int]] = []
        self.streams = []
        for name, fields in (("topology_events.csv", TOPOLOGY_FIELDS), ("mask_aggregates.csv", MASK_FIELDS)):
            stream = (directory / name).open("x", newline="", encoding="utf-8")
            self.streams.append(stream)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            setattr(self, "topology_writer" if name.startswith("topology") else "mask_writer", writer)
        write_json_new(directory / "resolved_schedule.json", self.schedule)

    def _paused(self, step: int) -> bool:
        return any(r["pause_start_step"] <= step <= r["pause_stop_step"] for r in self.schedule["head_pause_ranges"])

    def record_topology(self, *, step: int, **counts: int) -> None:
        row = dict.fromkeys(TOPOLOGY_FIELDS, NOT_AVAILABLE)
        row.update(counts, step=step,
                   mask_supervision_scale=scale_for_step(step, self.config["bootstrap_switch_step"]),
                   head_paused=int(self._paused(step)))
        if row["reset_event"] != int(step in self.schedule["reset_steps"]):
            raise ValueError("actual reset event differs from the schedule")
        expected = self.schedule["refine_steps"]
        if len(self.topology_steps) >= len(expected) or step != expected[len(self.topology_steps)]:
            raise ValueError(f"unexpected or duplicate refine event: {step}")
        _check_counts(row, self.previous_count)
        self.previous_count = row["gaussian_count_after"]
        self.topology_writer.writerow(row)
        self.streams[0].flush()
        self.topology_steps.append(step)

    def begin_iteration(self, step: int, width: int, height: int) -> None:
        if step != self.iterations or self.current_step is not None:
            raise ValueError("non-sequential paper-control iteration")
        self.current_step, self.full_size = step, (width, height)
        self.render_sizes = []

    def note_rasterization(self, width: int, height: int) -> None:
        if self.current_step is None:
            raise ValueError("rasterization outside the audited training iteration")
        self.render_sizes.append((width, height))

    def record_iteration(self, step: int, state: Any) -> None:
        if step != self.current_step:
            raise ValueError("iteration/rasterization audit mismatch")
        coarse = step < self.config["bootstrap_switch_step"]
        expected_sizes = [self.full_size] + ([(224, 224)] if coarse else [])
        if self.render_sizes != expected_sizes:
            raise ValueError(f"unexpected rasterization calls at {step}: {self.render_sizes}")
        self.full_renders += 1
        self.coarse_renders += int(coarse)
        self.iterations += 1
        self.current_step = None
        if step not in MASK_SAMPLE_STEPS:
            return
        import torch
        # Eleven scheduled host transfers in a 30k run; never per-Gaussian arrays.
        raw = state.probability.detach().float().cpu().flatten()
        quantiles = torch.quantile(raw, torch.tensor([0.1, 0.5, 0.9])).tolist()
        row = {
            "step": step, "mask_supervision_scale": scale_for_step(step, self.config["bootstrap_switch_step"]),
            "head_paused": int(state.paused), "raw_mask_mean": raw.mean().item(),
            **dict(zip(("raw_mask_p10", "raw_mask_p50", "raw_mask_p90"), quantiles)),
            "hard_static_retention": (raw > 0.25).float().mean().item(),
            "eroded_static_retention": state.safe_mask.detach().float().mean().item(),
            "rgb_residual_mean": state.residual.detach().float().mean().item(),
        }
        _finite_row(row, MASK_FIELDS[3:])
        if row["head_paused"] != int(self._paused(step)):
            raise ValueError("mask head pause mismatch")
        self.mask_writer.writerow(row)
        self.streams[1].flush()
        self.mask_steps.append(step)

    def finish(self) -> None:
        steps = self.schedule["run_steps"]
        if self.iterations != steps or self.topology_steps != self.schedule["refine_steps"]:
            raise ValueError("incomplete training/refine evidence")
        if self.mask_steps != [s for s in MASK_SAMPLE_STEPS if s < steps]:
            raise ValueError("incomplete mask aggregates")
        write_json_new(self.directory / "paper_control_validation.json", {
            "training_steps": self.iterations,
            "full_rasterization_count": self.full_renders,
            "full_rasterizations_per_step": self.full_renders / steps,
            "coarse_rasterization_count": self.coarse_renders,
            "refine_events_observed": len(self.topology_steps),
            "mask_samples_observed": len(self.mask_steps),
            "count_identity": "after = before + split + clone - prune (split: one parent to two children)",
            "optional_topology_statistics": NOT_AVAILABLE,
            "optional_statistics_reason": "grow/prune return counts only; no extra Gaussian reductions or device synchronization for logging",
            "statistics_affect_training_decisions": False,
        })
        for stream in self.streams:
            stream.close()


def _check_counts(row: dict, previous: int | None) -> None:
    fields = TOPOLOGY_FIELDS[4:9]
    _finite_row(row, fields)
    counts = {key: int(row[key]) for key in fields}
    if any(float(row[key]) != value or value < 0 for key, value in counts.items()):
        raise ValueError("invalid topology count")
    before, after = counts["gaussian_count_before"], counts["gaussian_count_after"]
    if after != before + counts["split_count"] + counts["clone_count"] - counts["prune_count"]:
        raise ValueError("topology count identity failed")
    if previous is not None and before != previous:
        raise ValueError("unlogged Gaussian count change between refine events")


def validate_event_files(directory: Path, config: dict, run_steps: int = 30000) -> dict:
    expected = resolved_schedule(config, run_steps)
    actual = json.loads((directory / "resolved_schedule.json").read_text())
    if actual != expected:
        raise ValueError("resolved schedule differs from the fixed config")
    with (directory / "topology_events.csv").open(newline="") as stream:
        events = list(csv.DictReader(stream))
    if [int(row["step"]) for row in events] != expected["refine_steps"]:
        raise ValueError("refine event set/count differs")
    previous = None
    for row in events:
        step = int(row["step"])
        _check_counts(row, previous)
        previous = int(row["gaussian_count_after"])
        paused = any(r["pause_start_step"] <= step <= r["pause_stop_step"] for r in expected["head_pause_ranges"])
        if row["mask_supervision_scale"] != scale_for_step(step, config["bootstrap_switch_step"]):
            raise ValueError("topology scale differs")
        if int(row["head_paused"]) != int(paused) or int(row["reset_event"]) != int(step in expected["reset_steps"]):
            raise ValueError("topology reset/pause differs")
        for field in TOPOLOGY_FIELDS[9:]:
            if row[field] != NOT_AVAILABLE:
                _finite_row(row, (field,))
    with (directory / "mask_aggregates.csv").open(newline="") as stream:
        masks = list(csv.DictReader(stream))
    if [int(row["step"]) for row in masks] != [s for s in MASK_SAMPLE_STEPS if s < run_steps]:
        raise ValueError("mask sample set differs")
    for row in masks:
        _finite_row(row, MASK_FIELDS[3:])
        step = int(row["step"])
        paused = any(r["pause_start_step"] <= step <= r["pause_stop_step"] for r in expected["head_pause_ranges"])
        if row["mask_supervision_scale"] != scale_for_step(step, config["bootstrap_switch_step"]) or int(row["head_paused"]) != int(paused):
            raise ValueError("mask scale/pause differs")
        if any(not 0 <= float(row[f]) <= 1 for f in MASK_FIELDS[3:-1]) or float(row["rgb_residual_mean"]) < 0:
            raise ValueError("mask aggregates out of range")
    phases = {}
    for name, start, stop in (("early", 10000, 20000), ("late", 20000, 24000)):
        subset = [row for row in events if start <= int(row["step"]) < stop]
        phases[name] = {"events": len(subset), **{
            field: sum(int(row[field]) for row in subset)
            for field in ("split_count", "clone_count", "prune_count")
        }}
    return {"pass": True, "event_count": len(events), "final_topology_count": previous,
            "phases": phases, "mask_sample_count": len(masks)}
