"""Training-only PURI-GS-RU integration kept outside the standard eval path."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from torch import Tensor

from puri_gs.delayed_absgrad import (
    DelayedAbsGradSchedule,
    mask_update_paused_after_resets,
    topology_event_summary,
)
from puri_gs.config import parse_refine_windows
from puri_gs.dino_features import (
    COARSE_GRID,
    FEATURE_DIM,
    FINE_GRID,
    FeatureCache,
    extract_patch_grid,
    load_frozen_dinov2,
)
from puri_gs.semantic_mask import (
    ResidualHistogram,
    ResidualHistogramConfig,
    StaticResponsibilityHead,
    cosine_static_target,
    hard_static_mask,
    residual_interval_mask,
    semantic_mask_loss,
)


class RUIterationState:
    def __init__(
        self,
        *,
        gt_feature: Tensor,
        predicted_grid: Tensor,
        safe_mask: Tensor,
        probability: Tensor,
        paused: bool,
    ) -> None:
        self.gt_feature = gt_feature
        self.predicted_grid = predicted_grid
        self.safe_mask = safe_mask
        self.probability = probability
        self.paused = paused
        self.mask_loss: Tensor | None = None
        self.similarity: Tensor | None = None
        self.residual: Tensor | None = None
        self.components: dict[str, Tensor] | None = None


def prepare_render_for_dino(render: Tensor) -> Tensor:
    """Detach and project raw Gaussian RGB into DINOv2's valid image range."""

    return render.detach().clamp(0.0, 1.0)


class PURIGSRUTraining:
    """Own DINO, the mask head, cache, histogram, diagnostics, and aux state."""

    def __init__(
        self,
        *,
        cfg: Any,
        parser: Any,
        trainset: Any,
        device: str | torch.device,
        delayed_topology_enabled: bool = True,
    ) -> None:
        self.cfg = cfg
        self.parser = parser
        self.trainset = trainset
        self.device = torch.device(device)
        self.result_dir = Path(cfg.result_dir)
        self.aux_dir = self.result_dir / "aux"
        self.aux_dir.mkdir(parents=True, exist_ok=True)

        self.dino, self.dino_metadata = load_frozen_dinov2(
            cfg.dino_repo_dir, cfg.dino_weight_path, device=self.device
        )
        self.cache = FeatureCache(
            cfg.feature_cache_dir,
            expected_weight_sha256=self.dino_metadata["weight_sha256"],
        )
        expected_names = {
            self.parser.image_names[int(index)] for index in self.trainset.indices
        }
        missing = sorted(expected_names.difference(self.cache.records))
        if missing:
            raise RuntimeError(
                f"feature cache misses {len(missing)} training images; first={missing[0]}"
            )

        self.head = StaticResponsibilityHead(
            feature_dim=cfg.dino_feature_dim, hidden_dim=cfg.mask_hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.head.parameters(),
            lr=cfg.mask_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.histogram = ResidualHistogram(
            ResidualHistogramConfig(
                bins=cfg.residual_hist_bins,
                momentum=cfg.residual_hist_momentum,
                lower_quantile=cfg.residual_lower_quantile,
                upper_quantile=cfg.residual_upper_quantile,
            )
        )
        self.delayed_topology_enabled = delayed_topology_enabled
        self.schedule = (
            DelayedAbsGradSchedule(
                densify_start_step=cfg.densify_start_step,
                densify_stop_step=cfg.densify_stop_step,
                densify_every=cfg.densify_every,
                opacity_reset_start_step=cfg.opacity_reset_start_step,
                opacity_reset_every=cfg.opacity_reset_every,
                mask_pause_after_reset=cfg.mask_pause_after_reset,
                refine_windows=parse_refine_windows(getattr(cfg, "refine_windows", None)),
            )
            if delayed_topology_enabled
            else None
        )
        self.topology_events = topology_event_summary(
            delayed_topology=delayed_topology_enabled,
            total_steps=cfg.max_steps,
            mask_pause_after_reset=cfg.mask_pause_after_reset,
            delayed_schedule=self.schedule,
        )
        self.reset_steps = tuple(self.topology_events["reset_steps"])
        self.dino_render_seconds = 0.0
        self.dino_render_calls = 0
        self.mask_update_count = 0
        self.mask_pause_count = 0
        self._last_state: RUIterationState | None = None
        self._diagnostic_files = self._open_diagnostics()
        self.paper_recorder = None
        if getattr(cfg, "puri_gs_paper_control", None):
            from puri_gs.paper_controls import PaperControlRecorder, schedule_from_config

            config = json.loads((self.result_dir / "config.yaml").read_text())
            if (config.get("paper_control") != cfg.puri_gs_paper_control
                    or config["bootstrap_switch_step"] != cfg.bootstrap_switch_step
                    or schedule_from_config(config) != self.schedule):
                raise ValueError("runtime paper-control schedule differs from launch config")
            self.paper_recorder = PaperControlRecorder(self.result_dir, config, cfg.max_steps)

        environment = {
            **self.dino_metadata,
            "mask_head_parameter_count": sum(
                parameter.numel() for parameter in self.head.parameters()
            ),
            "feature_cache_dir": str(self.cache.directory),
            "feature_cache_scene": self.cache.scene,
        }
        (self.aux_dir / "dino_environment.json").write_text(
            json.dumps(environment, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "PURI-GS-RU DINO: "
            f"parameters={environment['parameter_count']}, "
            f"trainable={environment['trainable_parameter_count']}, "
            f"weight_sha256={environment['weight_sha256']}, "
            f"repository_commit={environment['repository_commit']}"
        )

    def _open_diagnostics(self) -> dict[str, tuple[Any, csv.writer]]:
        specifications = {
            "mask_mean_curve.csv": ["step", "mask_mean"],
            "static_ratio_curve.csv": ["step", "static_ratio"],
            "Gaussian_count_curve.csv": ["step", "gaussian_count"],
            "loss_curve.csv": [
                "step",
                "photo_loss",
                "mask_loss",
                "cosine_loss",
                "residual_loss",
                "static_prior",
                "weight_regularizer",
            ],
        }
        opened: dict[str, tuple[Any, csv.writer]] = {}
        for name, header in specifications.items():
            path = self.result_dir / name
            stream = path.open("w", newline="", encoding="utf-8")
            writer = csv.writer(stream)
            writer.writerow(header)
            opened[name] = (stream, writer)
        return opened

    def grid_for_step(self, step: int) -> int:
        return COARSE_GRID if step < self.cfg.bootstrap_switch_step else FINE_GRID

    def _image_name(self, image_id: Tensor) -> str:
        item = int(image_id.reshape(-1)[0].detach().cpu().item())
        parser_index = int(self.trainset.indices[item])
        return str(self.parser.image_names[parser_index])

    def prepare_photo_mask(
        self,
        *,
        step: int,
        image_id: Tensor,
        pixels: Tensor,
    ) -> RUIterationState:
        """Predict the detached static mask needed by the Gaussian photo loss."""

        grid_size = self.grid_for_step(step)
        image_name = self._image_name(image_id)
        gt_feature = (
            self.cache.load(image_name, grid_size)
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )

        self.optimizer.zero_grad(set_to_none=True)
        self.head.train()
        predicted_grid = self.head(gt_feature)
        probability = F.interpolate(
            predicted_grid,
            size=tuple(pixels.shape[1:3]),
            mode="bilinear",
            align_corners=False,
        )
        safe_mask = hard_static_mask(
            probability.detach(),
            threshold=self.cfg.mask_threshold,
            kernel_size=self.cfg.mask_erode_kernel,
        )
        state = RUIterationState(
            gt_feature=gt_feature,
            predicted_grid=predicted_grid,
            safe_mask=safe_mask,
            probability=probability.detach(),
            paused=mask_update_paused_after_resets(
                step, self.reset_steps, self.cfg.mask_pause_after_reset
            ),
        )
        self._last_state = state
        return state

    def complete_mask_supervision(
        self,
        *,
        step: int,
        state: RUIterationState,
        pixels: Tensor,
        full_render: Tensor,
        render_for_mask: Tensor,
    ) -> None:
        """Compute the independent mask loss after the Gaussian backward pass."""

        grid_size = self.grid_for_step(step)
        started = perf_counter()
        render_feature = extract_patch_grid(
            self.dino, prepare_render_for_dino(render_for_mask), grid_size
        )
        self.dino_render_seconds += perf_counter() - started
        self.dino_render_calls += 1
        similarity_grid = cosine_static_target(state.gt_feature, render_feature)

        residual = torch.abs(full_render.detach() - pixels.detach()).mean(
            dim=-1, keepdim=False
        ).unsqueeze(1)
        lower_threshold, upper_threshold = self.histogram.update(residual)
        lower_mask = residual_interval_mask(residual, lower_threshold)
        upper_mask = residual_interval_mask(residual, upper_threshold)
        mask_loss, components = semantic_mask_loss(
            state.predicted_grid,
            similarity_grid,
            lower_mask,
            upper_mask,
            image_size=tuple(pixels.shape[1:3]),
            step=step,
            head=self.head,
            cosine_weight=self.cfg.mask_cos_weight,
            residual_weight=self.cfg.mask_residual_weight,
            static_prior_weight=self.cfg.mask_static_prior_weight,
            static_prior_decay=self.cfg.mask_static_prior_decay,
        )
        state.mask_loss = mask_loss
        state.probability = components["mask_probability"].detach()
        state.similarity = components["feature_similarity"].detach()
        state.residual = residual.detach()
        state.components = components

    def update_mask_head(self, state: RUIterationState) -> None:
        if state.mask_loss is None:
            raise RuntimeError("mask supervision must be completed before its update")
        if state.paused:
            self.mask_pause_count += 1
            self.optimizer.zero_grad(set_to_none=True)
            return
        state.mask_loss.backward()
        gradients = [
            parameter.grad for parameter in self.head.parameters() if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("mask-head gradients are missing or non-finite")
        if any(parameter.grad is not None for parameter in self.dino.parameters()):
            raise RuntimeError("DINOv2 unexpectedly received gradients")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.mask_update_count += 1

    def assert_photo_gradient_isolation(self) -> None:
        head_gradients = [parameter.grad for parameter in self.head.parameters()]
        if any(
            gradient is not None and torch.count_nonzero(gradient).item() != 0
            for gradient in head_gradients
        ):
            raise RuntimeError("Gaussian photo loss unexpectedly updated the mask head")
        if any(parameter.grad is not None for parameter in self.dino.parameters()):
            raise RuntimeError("Gaussian photo loss unexpectedly updated DINOv2")

    def record(
        self,
        *,
        step: int,
        state: RUIterationState,
        photo_loss: Tensor,
        gaussian_count: int,
    ) -> None:
        if state.mask_loss is None or state.components is None:
            raise RuntimeError("cannot record incomplete mask supervision")
        self._diagnostic_files["mask_mean_curve.csv"][1].writerow(
            [step, float(state.probability.mean().item())]
        )
        self._diagnostic_files["static_ratio_curve.csv"][1].writerow(
            [step, float(state.safe_mask.mean().item())]
        )
        self._diagnostic_files["Gaussian_count_curve.csv"][1].writerow(
            [step, gaussian_count]
        )
        self._diagnostic_files["loss_curve.csv"][1].writerow(
            [
                step,
                float(photo_loss.detach().item()),
                float(state.mask_loss.detach().item()),
                float(state.components["cosine_loss"].detach().item()),
                float(state.components["residual_loss"].detach().item()),
                float(state.components["static_prior"].detach().item()),
                float(state.components["weight_regularizer"].detach().item()),
            ]
        )
        if step % 100 == 0:
            for stream, _ in self._diagnostic_files.values():
                stream.flush()
        if self.paper_recorder is not None:
            self.paper_recorder.record_iteration(step, state)

    def save_auxiliary(self, step: int, *, training_seconds: float) -> None:
        torch.save(
            self.head.state_dict(), self.aux_dir / f"mask_head_step{step}.pt"
        )
        torch.save(
            self.optimizer.state_dict(),
            self.aux_dir / f"mask_optimizer_step{step}.pt",
        )
        torch.save(
            self.histogram.state_dict(),
            self.aux_dir / f"residual_hist_step{step}.pt",
        )
        schedule = {
            "semantic_mask_enabled": True,
            "delayed_topology_enabled": self.delayed_topology_enabled,
            "bootstrap_switch_step": self.cfg.bootstrap_switch_step,
            "mask_begin_step": self.cfg.mask_begin_step,
            "mask_threshold": self.cfg.mask_threshold,
            "mask_erode_kernel": self.cfg.mask_erode_kernel,
            "mask_pause_after_reset": self.cfg.mask_pause_after_reset,
            "topology_events": self.topology_events,
        }
        if self.schedule is not None:
            fields = asdict(self.schedule)
            if not self.schedule.refine_windows:
                fields.pop("refine_windows")
            schedule.update(fields)
        (self.aux_dir / "training_schedule.json").write_text(
            json.dumps(schedule, indent=2) + "\n", encoding="utf-8"
        )
        timing = {
            "render_feature_calls": self.dino_render_calls,
            "render_feature_seconds": self.dino_render_seconds,
            "render_feature_mean_seconds": (
                self.dino_render_seconds / self.dino_render_calls
                if self.dino_render_calls
                else None
            ),
            "cache_load_seconds": self.cache.load_seconds,
            "training_seconds": training_seconds,
            "render_feature_fraction_of_training": (
                self.dino_render_seconds / training_seconds
                if training_seconds > 0
                else None
            ),
            "mask_update_count": self.mask_update_count,
            "mask_pause_count": self.mask_pause_count,
        }
        (self.result_dir / "DINO_time.json").write_text(
            json.dumps(timing, indent=2) + "\n", encoding="utf-8"
        )
        validation = {
            "gradient_isolation_pass": True,
            "dino_trainable_parameter_count": self.dino_metadata[
                "trainable_parameter_count"
            ],
            "mask_updates_completed": self.mask_update_count,
            "runtime_photo_to_mask_checks_completed": self.dino_render_calls,
        }
        (self.aux_dir / "ru_training_validation.json").write_text(
            json.dumps(validation, indent=2) + "\n", encoding="utf-8"
        )
        for stream, _ in self._diagnostic_files.values():
            stream.flush()
        if self.paper_recorder is not None:
            self.paper_recorder.finish()

    def save_visuals(self, full_render: Tensor) -> None:
        state = self._last_state
        if state is None or state.residual is None or state.similarity is None:
            raise RuntimeError("cannot save RU visuals before the first iteration")
        render = full_render[0].detach().clamp(0, 1).mul(255).byte().cpu().numpy()
        probability = state.probability[0, 0].clamp(0, 1).mul(255).byte().cpu().numpy()
        hard = state.safe_mask[0, 0].clamp(0, 1).mul(255).byte().cpu().numpy()
        residual = state.residual[0, 0].clamp(0, 1).mul(255).byte().cpu().numpy()
        similarity = state.similarity[0, 0].clamp(0, 1).mul(255).byte().cpu().numpy()
        imageio.imwrite(self.result_dir / "rgb_render.png", render)
        imageio.imwrite(self.result_dir / "mask_probability.png", probability)
        imageio.imwrite(self.result_dir / "hard_static_mask.png", hard)
        imageio.imwrite(self.result_dir / "residual.png", residual)
        imageio.imwrite(self.result_dir / "feature_similarity.png", similarity)


def scale_intrinsics(Ks: Tensor, *, width: int, height: int, target: int = 224) -> Tensor:
    """Scale a full camera to the fixed square coarse DINO rasterization."""

    scaled = Ks.clone()
    scaled[:, 0, :] *= float(target) / float(width)
    scaled[:, 1, :] *= float(target) / float(height)
    return scaled
