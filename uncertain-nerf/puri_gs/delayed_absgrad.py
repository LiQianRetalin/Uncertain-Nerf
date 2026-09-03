"""Delayed AbsGrad schedule and the single gsplat DefaultStrategy subclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:  # Keep schedule-only CPU tests usable without a gsplat installation.
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import reset_opa
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    DefaultStrategy = object  # type: ignore[assignment,misc]
    reset_opa = None


@dataclass(frozen=True)
class DelayedAbsGradSchedule:
    densify_start_step: int = 10_000
    densify_stop_step: int = 20_000
    densify_every: int = 100
    opacity_reset_start_step: int = 15_000
    opacity_reset_every: int = 3_000
    mask_pause_after_reset: int = 300

    def __post_init__(self) -> None:
        if not 0 <= self.densify_start_step < self.densify_stop_step:
            raise ValueError("invalid densification interval")
        if self.densify_every <= 0 or self.opacity_reset_every <= 0:
            raise ValueError("densification/reset intervals must be positive")
        if not self.densify_start_step <= self.opacity_reset_start_step < self.densify_stop_step:
            raise ValueError("opacity reset must start inside the densification interval")
        if self.mask_pause_after_reset < 0:
            raise ValueError("mask pause must be non-negative")

    def densification_allowed(self, step: int) -> bool:
        return self.densify_start_step <= step < self.densify_stop_step

    def should_refine(self, step: int) -> bool:
        return self.densification_allowed(step) and step % self.densify_every == 0

    def should_reset(self, step: int) -> bool:
        return (
            self.opacity_reset_start_step <= step < self.densify_stop_step
            and (step - self.opacity_reset_start_step) % self.opacity_reset_every == 0
        )

    def mask_update_paused(self, step: int) -> bool:
        return mask_update_paused_after_resets(
            step,
            self.reset_steps(self.densify_stop_step),
            self.mask_pause_after_reset,
        )

    def refine_steps(self, total_steps: int) -> tuple[int, ...]:
        stop = min(total_steps, self.densify_stop_step)
        return tuple(
            step
            for step in range(self.densify_start_step, stop)
            if self.should_refine(step)
        )

    def reset_steps(self, total_steps: int) -> tuple[int, ...]:
        stop = min(total_steps, self.densify_stop_step)
        return tuple(
            step
            for step in range(self.opacity_reset_start_step, stop)
            if self.should_reset(step)
        )


def mask_update_paused_after_resets(
    step: int, reset_steps: tuple[int, ...], pause_after_reset: int
) -> bool:
    """Pause only after reset events that actually occur in the chosen topology."""

    if pause_after_reset <= 0:
        return False
    return any(reset < step <= reset + pause_after_reset for reset in reset_steps)


def topology_event_summary(
    *,
    delayed_topology: bool,
    total_steps: int = 30_000,
    mask_pause_after_reset: int = 0,
    delayed_schedule: DelayedAbsGradSchedule | None = None,
) -> dict[str, Any]:
    """Resolve the pinned B1/RU topology into an auditable event contract."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if mask_pause_after_reset < 0:
        raise ValueError("mask_pause_after_reset must be non-negative")

    if delayed_topology:
        schedule = delayed_schedule or DelayedAbsGradSchedule()
        refine_steps = schedule.refine_steps(total_steps)
        reset_steps = schedule.reset_steps(total_steps)
        statistics_window = [0, min(total_steps, schedule.densify_stop_step) - 1]
        name = "ru_delayed_default_strategy"
        post_backward_order = "before_gaussian_optimizer"
        reset_behavior = "explicit_delayed_schedule"
    else:
        # Pinned gsplat 1.5.3 uses:
        #   if step % self.reset_every == 0 & step > 0:
        # Python parses this as a chained comparison through (0 & step), so the
        # final 0 > 0 term is false. T=0 must preserve that actual B1 behavior.
        refine_stop = min(total_steps, 15_000)
        refine_steps = tuple(range(600, refine_stop, 100))
        reset_steps = ()
        statistics_window = [0, refine_stop - 1]
        name = "b1_gsplat_default_strategy"
        post_backward_order = "after_gaussian_optimizer"
        reset_behavior = "pinned_gsplat_1.5.3_expression_produces_no_events"

    pause_segments = [
        {
            "reset_step": step,
            "pause_start_step": step + 1,
            "pause_stop_step": step + mask_pause_after_reset,
        }
        for step in reset_steps
        if step + 1 < total_steps and mask_pause_after_reset > 0
    ]
    pause_steps = sum(
        min(total_steps - 1, item["pause_stop_step"])
        - item["pause_start_step"]
        + 1
        for item in pause_segments
    )
    return {
        "name": name,
        "delayed_topology": delayed_topology,
        "statistics_window_inclusive": statistics_window,
        "refine_steps": list(refine_steps),
        "split_steps": list(refine_steps),
        "clone_steps": list(refine_steps),
        "prune_steps": list(refine_steps),
        "reset_steps": list(reset_steps),
        "refine_event_count": len(refine_steps),
        "split_event_count": len(refine_steps),
        "clone_event_count": len(refine_steps),
        "prune_event_count": len(refine_steps),
        "reset_event_count": len(reset_steps),
        "reset_behavior": reset_behavior,
        "post_backward_order": post_backward_order,
        "mask_pause_after_reset": mask_pause_after_reset,
        "mask_pause_segments": pause_segments,
        "mask_pause_step_count": pause_steps,
    }


class DelayedAbsGradStrategy(DefaultStrategy):  # type: ignore[misc]
    """Minimal DefaultStrategy override that adds a reset start boundary."""

    def __init__(self, *, schedule: DelayedAbsGradSchedule, **kwargs: Any) -> None:
        if reset_opa is None:
            raise ImportError("gsplat is required to construct DelayedAbsGradStrategy")
        super().__init__(
            refine_start_iter=schedule.densify_start_step - 1,
            refine_stop_iter=schedule.densify_stop_step,
            refine_every=schedule.densify_every,
            reset_every=schedule.opacity_reset_every,
            absgrad=True,
            **kwargs,
        )
        self.schedule = schedule

    def step_post_backward(
        self,
        params,
        optimizers,
        state,
        step: int,
        info,
        packed: bool = False,
    ) -> None:
        if step >= self.schedule.densify_stop_step:
            return

        # Statistics are accumulated from step zero even though topology is frozen.
        self._update_state(params, state, info, packed=packed)

        if self.schedule.should_refine(step):
            n_duplicate, n_split = self._grow_gs(params, optimizers, state, step)
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_duplicate} GSs duplicated, {n_split} GSs split, "
                    f"{n_prune} GSs pruned. Now having {len(params['means'])} GSs."
                )
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if self.schedule.should_reset(step):
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )
            if self.verbose:
                print(f"Step {step}: opacity reset applied.")


def delayed_strategy_from_default(
    strategy: Any, schedule: DelayedAbsGradSchedule
) -> DelayedAbsGradStrategy:
    """Copy the official DefaultStrategy thresholds into the delayed subclass."""

    if DefaultStrategy is object or not isinstance(strategy, DefaultStrategy):
        raise TypeError("PURI-GS-RU requires gsplat DefaultStrategy")
    fields = (
        "prune_opa",
        "grow_grad2d",
        "grow_scale3d",
        "grow_scale2d",
        "prune_scale3d",
        "prune_scale2d",
        "refine_scale2d_stop_iter",
        "pause_refine_after_reset",
        "revised_opacity",
        "verbose",
        "key_for_gradient",
    )
    kwargs = {field: getattr(strategy, field) for field in fields}
    return DelayedAbsGradStrategy(schedule=schedule, **kwargs)
