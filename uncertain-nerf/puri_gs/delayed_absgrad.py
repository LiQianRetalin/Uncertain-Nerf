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
        if step <= self.opacity_reset_start_step or self.mask_pause_after_reset == 0:
            return False
        reset_index = (
            step - self.opacity_reset_start_step - 1
        ) // self.opacity_reset_every
        most_recent = self.opacity_reset_start_step + reset_index * self.opacity_reset_every
        steps_after_reset = step - most_recent
        return self.should_reset(most_recent) and (
            1 <= steps_after_reset <= self.mask_pause_after_reset
        )


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
