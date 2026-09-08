"""Training-only RU-PART controller and topology strategy."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from torch import Tensor

from puri_gs.delayed_absgrad import DelayedAbsGradSchedule, DelayedAbsGradStrategy
from puri_gs.prospective_topology import (
    BIRTH_INITIAL_OPACITY,
    N_MAX,
    RHO0,
    append_birth_rows,
    compute_default_grow_masks,
    initialize_birth_geometry,
    initialize_footprint_scores,
    pair_births_with_clones,
    project_gaussians,
    rgb_to_sh0,
    sparse_footprints_from_projection,
    validate_parameter_state_shapes,
)
from puri_gs.static_tracks import evidence_upsample, load_static_track_cache, scaled_grid_intrinsics, CameraRecord

try:
    from gsplat.strategy.ops import duplicate, split
except ImportError:  # CPU-only unit tests can import the controller.
    duplicate = split = None


EVENT_FIELDS = (
    "step", "intervention_mode", "diagnostic_only", "gaussians_before",
    "fixed_support_mean", "standard_clone_proposed",
    "standard_split_proposed", "post_grow_prune_candidate_count",
    "track_candidates_visible", "track_candidates_geometry_valid",
    "candidate_B_positive", "candidate_alpha_mass_positive", "candidate_B_gt_H0",
    "birth_pareto_dominates_clone", "counterfactual_birth_selected",
    "coverage_birth_accepted", "regular_clone_replaced",
    "regular_clone_executed", "split_executed", "prune_executed",
    "budget_rejected_clone", "budget_rejected_birth", "budget_rejected_split",
    "accepted_track_id_unique", "birth_survived_100", "birth_survived_500",
    "mean_B_accepted", "mean_H0_accepted", "mean_prospective_alpha_mass",
    "mean_fixed_support_gain", "gaussians_after", "fixed_support_probe_ms",
    "topology_total_ms",
)
REPRESENTATIVE_STEPS = (10_000, 13_300, 16_600, 19_900)
ACCEPTED_FIELDS = (
    "step", "track_id", "replaced_clone_id", "B_birth", "H0_birth",
    "B_clone", "H0_clone", "prospective_alpha_mass",
)


class RUPARTController:
    def __init__(self, *, cache_path: str | Path, parser: Any, trainset: Any, cfg: Any, device: str | torch.device) -> None:
        self.payload = load_static_track_cache(cache_path)
        self.device = torch.device(device)
        self.cfg = cfg
        self.intervention_mode = str(cfg.ru_part_mode)
        if self.intervention_mode not in {"noop", "current"}:
            raise ValueError("RUPARTController is valid only for noop/current modes")
        self.parser = parser
        self.trainset = trainset
        runtime_names = [parser.image_names[int(i)] for i in trainset.indices]
        if runtime_names != list(self.payload["train_basenames"]):
            raise ValueError("static-track train-view mapping differs from runtime")
        self.global_to_train = {int(global_id): local for local, global_id in enumerate(trainset.indices)}
        self.evidence = self.payload["track_evidence_binary"].float().to(self.device)
        if self.evidence.shape[0] != len(runtime_names):
            raise ValueError("static-track evidence view count mismatch")
        self.track_ids = self.payload["track_ids"].long()
        self.track_xyz = self.payload["track_world_xyz"].float()
        self.track_views = self.payload["track_view_ids"].long()
        self.track_rgb = self.payload["track_rgb_median"].float()
        self.born_track_ids: set[int] = set()
        self._geometry: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._event: dict[str, Any] | None = None
        self.full_rgb_rasterizations = 0
        self.fixed_support_probe_count = 0
        self.gaussian_backward_count = 0
        self.test_images_opened_during_training = 0
        lookup = torch.full((len(parser.image_names),), -1, dtype=torch.long, device=self.device)
        for global_id, local_id in self.global_to_train.items():
            lookup[global_id] = local_id
        self.global_to_train_lookup = lookup
        self.sum_gaussians_over_train_steps = 0
        self.peak_gaussian_count = 0
        self.result_dir = Path(cfg.result_dir)
        self.aux_dir = self.result_dir / "ru_part_aux"
        self.aux_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.result_dir / "ru_part_topology_events.csv"
        self._event_stream = self.event_path.open("w", newline="", encoding="utf-8")
        self._event_writer = csv.DictWriter(self._event_stream, fieldnames=EVENT_FIELDS)
        self._event_writer.writeheader()
        self.accepted_path = self.result_dir / "ru_part_accepted_candidates.csv"
        self._accepted_stream = self.accepted_path.open("w", newline="", encoding="utf-8")
        self._accepted_writer = csv.DictWriter(self._accepted_stream, fieldnames=ACCEPTED_FIELDS)
        self._accepted_writer.writeheader()
        self._accepted: dict[int, dict[str, int]] = {}
        self._representatives: list[dict[str, Any]] = []
        self._precompute_geometry()

    def _camera(self, train_view_id: int) -> CameraRecord:
        global_id = int(self.trainset.indices[train_view_id])
        camera_id = self.parser.camera_ids[global_id]
        width, height = self.parser.imsize_dict[camera_id]
        return CameraRecord(
            train_view_id,
            self.parser.image_names[global_id],
            np.asarray(self.parser.camtoworlds[global_id]),
            np.asarray(self.parser.Ks_dict[camera_id]),
            int(width), int(height),
        )

    def _precompute_geometry(self) -> None:
        for row, track_id_tensor in enumerate(self.track_ids):
            view_ids = [int(v) for v in self.track_views[row].tolist() if int(v) >= 0]
            cameras = [self._camera(v) for v in view_ids]
            try:
                scales, quat = initialize_birth_geometry(
                    self.track_xyz[row].numpy(),
                    [camera.camtoworld for camera in cameras],
                    [scaled_grid_intrinsics(camera) for camera in cameras],
                )
            except ValueError:
                continue
            self._geometry[int(track_id_tensor)] = (scales, quat)

    def global_image_id_from_train(self, image_id: int | Tensor) -> int:
        """Convert Dataset.image_id (split-local) to the parser/cache ID."""
        item = int(image_id.item()) if isinstance(image_id, Tensor) else int(image_id)
        if not 0 <= item < len(self.trainset.indices):
            raise ValueError("RU-PART training image index is out of range")
        return int(self.trainset.indices[item])

    def static_evidence(self, global_image_id: int | Tensor, *, target_size: tuple[int, int] | None = None, device: torch.device | None = None) -> Tensor:
        if isinstance(global_image_id, Tensor):
            global_image_id = int(global_image_id.item())
        if global_image_id not in self.global_to_train:
            raise ValueError("RU-PART attempted to use a non-training view")
        value = self.evidence[self.global_to_train[global_image_id]]
        if target_size is not None:
            value = evidence_upsample(value, target_size)[0, 0]
        return value.to(device or self.device)

    def rescue_loss(self, render: Tensor, target: Tensor, alpha: Tensor, safe_mask: Tensor, global_image_id: int | Tensor) -> Tensor:
        if getattr(self, "intervention_mode", "current") == "noop":
            raise RuntimeError("noop mode must not attach rescue loss to the training graph")
        if render.shape != target.shape or render.ndim != 4 or render.shape[0] != 1:
            raise ValueError("RU-PART supports batch=1 RGB training only")
        height, width = render.shape[1:3]
        evidence = self.static_evidence(global_image_id, target_size=(height, width), device=render.device)
        alpha_bhw = alpha[..., 0] if alpha.ndim == 4 else alpha
        hard = safe_mask[:, 0]
        weight = ((1.0 - hard) * evidence[None] * (1.0 - alpha_bhw)).detach()
        return (weight[..., None] * torch.abs(render - target)).mean()

    def note_rgb_rasterization(self) -> None:
        self.full_rgb_rasterizations += 1

    def note_gaussian_backward(self) -> None:
        self.gaussian_backward_count += 1

    def note_gaussian_count(self, count: int) -> None:
        self.sum_gaussians_over_train_steps += int(count)
        self.peak_gaussian_count = max(self.peak_gaussian_count, int(count))

    def prepare_event(
        self,
        *,
        step: int,
        global_image_id: int,
        standard_alpha: Tensor,
        fixed_support: Tensor,
        standard_rgb: Tensor,
        target_rgb: Tensor,
        camtoworld: Tensor,
        K: Tensor,
        fixed_support_probe_ms: float,
    ) -> None:
        if not (10_000 <= step < 20_000 and step % 100 == 0):
            raise ValueError("fixed-support probes are allowed only at the 100 RU topology events")
        if standard_alpha.shape != fixed_support.shape:
            raise ValueError("standard alpha and fixed support must have identical shapes")
        self.fixed_support_probe_count += 1
        alpha = F.adaptive_avg_pool2d(standard_alpha.permute(0, 3, 1, 2), (36, 36))[0, 0].detach()
        support = F.adaptive_avg_pool2d(fixed_support.permute(0, 3, 1, 2), (36, 36))[0, 0].detach()
        train_view_id = self.global_to_train[int(global_image_id)]
        camera = self._camera(train_view_id)
        K_grid = torch.as_tensor(scaled_grid_intrinsics(camera), device=self.device, dtype=K.dtype)
        self._event = {
            "step": step, "global_image_id": int(global_image_id), "train_view_id": train_view_id,
            "alpha": alpha, "support": support, "evidence": self.static_evidence(int(global_image_id), device=self.device),
            "viewmat": torch.linalg.inv(camtoworld[0]).detach(), "K_grid": K_grid,
            "probe_ms": float(fixed_support_probe_ms),
        }
        if step in REPRESENTATIVE_STEPS:
            self._event["standard_rgb"] = F.adaptive_avg_pool2d(
                standard_rgb.permute(0, 3, 1, 2), (36, 36)
            )[0].permute(1, 2, 0).detach()
            self._event["target_rgb"] = F.adaptive_avg_pool2d(
                target_rgb.permute(0, 3, 1, 2), (36, 36)
            )[0].permute(1, 2, 0).detach()

    def take_event(self, step: int) -> dict[str, Any]:
        if self._event is None or self._event["step"] != step:
            raise RuntimeError("RU-PART topology event has no matching fixed-support probe")
        event, self._event = self._event, None
        return event

    @torch.no_grad()
    def record_noop_event(
        self,
        params: Mapping[str, Tensor],
        state: Mapping[str, Any],
        strategy: Any,
        step: int,
    ) -> None:
        """Score the current intervention without changing parameters or topology."""

        if self.intervention_mode != "noop":
            raise RuntimeError("record_noop_event is exclusive to noop mode")
        started = perf_counter()
        event = self.take_event(step)
        before = len(params["means"])
        proposals = compute_default_grow_masks(params, state, strategy, step)
        clone_indices = torch.where(proposals.clone)[0]
        split_indices = torch.where(proposals.split)[0]
        track_rows = self.visible_track_rows(event["train_view_id"])
        decision_support = event["support"].float().cpu()
        decision_alpha = event["alpha"].float().cpu()
        decision_evidence = event["evidence"].float().cpu()

        birth_footprints = []
        if len(track_rows):
            ids_cpu = self.track_ids[track_rows]
            means = self.track_xyz[track_rows].to(params["means"].device)
            scales = torch.as_tensor(
                np.stack([self._geometry[int(value)][0] for value in ids_cpu.tolist()]),
                device=means.device,
            )
            quats = torch.as_tensor(
                np.stack([self._geometry[int(value)][1] for value in ids_cpu.tolist()]),
                device=means.device,
            )
            radii, means2d, _, conics = project_gaussians(
                means, quats, scales.exp(), event["viewmat"], event["K_grid"]
            )
            birth_footprints = sparse_footprints_from_projection(
                ids_cpu, means2d.cpu(), conics.cpu(), radii.cpu()
            )
            initialize_footprint_scores(
                birth_footprints,
                decision_support,
                decision_alpha,
                decision_evidence,
                initial_opacity=BIRTH_INITIAL_OPACITY,
            )

        clone_footprints = []
        if len(clone_indices):
            radii, means2d, _, conics = project_gaussians(
                params["means"][clone_indices],
                params["quats"][clone_indices],
                params["scales"][clone_indices].exp(),
                event["viewmat"],
                event["K_grid"],
            )
            clone_footprints = sparse_footprints_from_projection(
                clone_indices.cpu(), means2d.cpu(), conics.cpu(), radii.cpu()
            )
            initialize_footprint_scores(
                clone_footprints,
                decision_support,
                decision_alpha,
                decision_evidence,
                initial_opacity=BIRTH_INITIAL_OPACITY,
            )

        clone_ids = clone_indices.detach().cpu().tolist()
        clone_grads = proposals.average_grad2d[clone_indices].detach().cpu().tolist()
        replacements, updated_support = pair_births_with_clones(
            birth_footprints,
            clone_footprints,
            decision_support,
            decision_evidence,
            limit=len(clone_indices),
            clone_tiebreak=dict(zip(clone_ids, clone_grads)),
        )
        accepted_b = [item.benefit for item in replacements]
        accepted_h = [item.harm for item in replacements]
        accepted_a = [item.alpha_mass for item in replacements]
        self.record_event({
            "step": step,
            "intervention_mode": "noop",
            "diagnostic_only": 1,
            "gaussians_before": before,
            "fixed_support_mean": float(decision_support.mean()),
            "standard_clone_proposed": len(clone_indices),
            "standard_split_proposed": len(split_indices),
            "track_candidates_visible": int(
                (self.track_views == event["train_view_id"]).any(1).sum()
            ),
            "track_candidates_geometry_valid": len(track_rows),
            "candidate_B_positive": sum(item.initial_benefit > 0 for item in birth_footprints),
            "candidate_alpha_mass_positive": sum(item.alpha_mass > 0 for item in birth_footprints),
            "candidate_B_gt_H0": sum(
                item.initial_benefit > item.initial_harm for item in birth_footprints
            ),
            "birth_pareto_dominates_clone": len(replacements),
            "counterfactual_birth_selected": len(replacements),
            "coverage_birth_accepted": 0,
            "regular_clone_replaced": 0,
            "mean_B_accepted": float(np.mean(accepted_b)) if accepted_b else 0.0,
            "mean_H0_accepted": float(np.mean(accepted_h)) if accepted_h else 0.0,
            "mean_prospective_alpha_mass": float(np.mean(accepted_a)) if accepted_a else 0.0,
            "mean_fixed_support_gain": float((updated_support - decision_support).mean()),
            "gaussians_after": before,
            "fixed_support_probe_ms": event["probe_ms"],
            "topology_total_ms": (perf_counter() - started) * 1000.0,
        })

    def visible_track_rows(self, train_view_id: int) -> Tensor:
        member = (self.track_views == int(train_view_id)).any(dim=1)
        unspent = torch.tensor([int(v) not in self.born_track_ids for v in self.track_ids.tolist()])
        geometry = torch.tensor([int(v) in self._geometry for v in self.track_ids.tolist()])
        return torch.where(member & unspent & geometry)[0]

    def birth_rows(self, selected_track_ids: list[int], params: Mapping[str, Tensor]) -> dict[str, Tensor]:
        index = {int(track_id): row for row, track_id in enumerate(self.track_ids.tolist())}
        device, dtype = params["means"].device, params["means"].dtype
        rows = [index[value] for value in selected_track_ids]
        scales = torch.as_tensor(np.stack([self._geometry[value][0] for value in selected_track_ids]), device=device, dtype=dtype)
        quats = torch.as_tensor(np.stack([self._geometry[value][1] for value in selected_track_ids]), device=device, dtype=dtype)
        rgb = self.track_rgb[rows].to(device=device, dtype=dtype)
        result = {
            "means": self.track_xyz[rows].to(device=device, dtype=dtype),
            "scales": scales,
            "quats": quats,
            "opacities": torch.full(
                (len(rows),),
                torch.logit(torch.tensor(BIRTH_INITIAL_OPACITY)).item(),
                device=device,
                dtype=dtype,
            ),
            "sh0": rgb_to_sh0(rgb)[:, None, :],
            "shN": torch.zeros((len(rows), *params["shN"].shape[1:]), device=device, dtype=dtype),
        }
        if set(result) != set(params):
            raise ValueError("RU-PART supports the standard Gaussian-only SH parameter schema")
        return result

    def record_event(self, row: Mapping[str, Any]) -> None:
        unknown = set(row).difference(EVENT_FIELDS)
        if unknown:
            raise ValueError(f"unknown RU-PART event fields: {sorted(unknown)}")
        self._event_writer.writerow({field: row.get(field, 0) for field in EVENT_FIELDS})
        self._event_stream.flush()

    def record_acceptances(self, step: int, replacements: list[Any]) -> None:
        for item in replacements:
            if not (
                item.benefit > 0
                and item.benefit > item.harm
                and item.benefit > item.clone_benefit
                and item.harm <= item.clone_harm
                and item.alpha_mass > 0
            ):
                raise RuntimeError("accepted birth violates the preregistered marginal/Pareto gate")
            self._accepted_writer.writerow({
                "step": step, "track_id": item.birth_id,
                "replaced_clone_id": item.clone_id,
                "B_birth": item.benefit, "H0_birth": item.harm,
                "B_clone": item.clone_benefit, "H0_clone": item.clone_harm,
                "prospective_alpha_mass": item.alpha_mass,
            })
        self._accepted_stream.flush()

    def save_representative(
        self,
        event: Mapping[str, Any],
        selected_footprints: list[Any],
    ) -> None:
        """Save only the four step-preregistered 36x36 decision panels."""

        step = int(event["step"])
        if step not in REPRESENTATIVE_STEPS:
            return
        support = event["support"].detach().float().cpu().clamp(0, 1)
        evidence = event["evidence"].detach().float().cpu().clamp(0, 1)
        alpha = event["alpha"].detach().float().cpu().clamp(0, 1)
        accepted = torch.zeros_like(support)
        for footprint in selected_footprints:
            tile = (0.1 * footprint.gaussian.detach().float().cpu()).clamp(0, 1)
            region = accepted[footprint.y0:footprint.y1, footprint.x0:footprint.x1]
            region.copy_(torch.maximum(region, tile))
        arrays = (
            ("GT", event["target_rgb"].detach().float().cpu().clamp(0, 1)),
            ("RGB", event["standard_rgb"].detach().float().cpu().clamp(0, 1)),
            ("A", alpha),
            ("F", support),
            ("s", evidence),
            ("q_starve", evidence * (1.0 - support)),
            ("q_harm", 1.0 - evidence),
            ("accepted_birth", accepted),
        )
        scale, header = 5, 18
        panels = []
        for label, value in arrays:
            if value.ndim == 2:
                value = value[..., None].expand(-1, -1, 3)
            pixels = value.mul(255).round().byte().numpy()
            panel = Image.fromarray(pixels, mode="RGB").resize(
                (36 * scale, 36 * scale), resample=Image.NEAREST
            )
            framed = Image.new("RGB", (36 * scale, 36 * scale + header), "white")
            framed.paste(panel, (0, header))
            ImageDraw.Draw(framed).text((3, 2), label, fill="black")
            panels.append(framed)
        canvas = Image.new("RGB", (4 * 36 * scale, 2 * (36 * scale + header)), "white")
        for index, panel in enumerate(panels):
            canvas.paste(panel, ((index % 4) * panel.width, (index // 4) * panel.height))
        output = self.aux_dir / f"representative_step{step}.png"
        imageio.imwrite(output, np.asarray(canvas))
        global_id = int(event["global_image_id"])
        self._representatives.append({
            "step": step,
            "global_image_id": global_id,
            "image_name": self.parser.image_names[global_id],
            "accepted_track_ids": [int(item.candidate_id) for item in selected_footprints],
            "path": str(output),
        })

    def lifecycle_counts(self, step: int, state: Mapping[str, Any]) -> tuple[int, int]:
        live = set()
        track_state = state.get("part_track_id")
        if isinstance(track_state, Tensor):
            live = {int(value) for value in track_state.unique().tolist() if int(value) >= 0}
        survived_100 = survived_500 = 0
        for track_id, item in self._accepted.items():
            if step == item["birth_step"] + 100:
                item["survived_100"] = int(track_id in live)
                survived_100 += item["survived_100"]
            if step == item["birth_step"] + 500:
                item["survived_500"] = int(track_id in live)
                survived_500 += item["survived_500"]
        return survived_100, survived_500

    def finish(self, *, training_seconds: float, peak_memory_gib: float | None = None) -> None:
        if peak_memory_gib is None and torch.cuda.is_available():
            peak_memory_gib = torch.cuda.max_memory_allocated(self.device) / 1024**3
        manifest_path = Path(self.cfg.ru_part_track_cache).with_name("static_track_manifest.json")
        track_audit = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        summary = {
            "full_rgb_rasterizations_per_train_step": self.full_rgb_rasterizations / max(int(self.cfg.max_steps), 1),
            "fixed_support_probe_count": self.fixed_support_probe_count,
            "gaussian_backward_per_step": self.gaussian_backward_count / max(int(self.cfg.max_steps), 1),
            "test_images_opened_during_track_build": track_audit.get("test_file_opened_count"),
            "test_images_opened_during_training": self.test_images_opened_during_training,
            "checkpoint_schema_unchanged": True,
            "evaluation_imported_dino": None,
            "evaluation_loaded_track_cache": None,
            "evaluation_loaded_mask_head": None,
            "CUDA_modified": False,
            "gsplat_version": "1.5.3",
            "training_time_seconds": training_seconds,
            "peak_memory_gib": peak_memory_gib,
            "static_track_cache_bytes": Path(self.cfg.ru_part_track_cache).stat().st_size,
            "static_track_build_time_seconds": track_audit.get("build_time_seconds"),
            "sum_gaussians_over_train_steps": self.sum_gaussians_over_train_steps,
            "peak_gaussian_count": self.peak_gaussian_count,
            "born_track_count": len(self.born_track_ids),
            "birth_survival": self._accepted,
        }
        (self.result_dir / "ru_part_global_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (self.aux_dir / "representative_visuals.json").write_text(
            json.dumps({
                "selection_rule": "current training view at fixed topology steps 10000,13300,16600,19900",
                "decision_grid": [36, 36],
                "items": self._representatives,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save({"born_track_ids": sorted(self.born_track_ids)}, self.aux_dir / "born_track_ids.pt")
        self._event_stream.flush()
        self._accepted_stream.flush()

    def replay_state_dict(self) -> dict[str, Any]:
        return {
            "intervention_mode": self.intervention_mode,
            "born_track_ids": sorted(self.born_track_ids),
            "accepted": self._accepted,
            "full_rgb_rasterizations": self.full_rgb_rasterizations,
            "fixed_support_probe_count": self.fixed_support_probe_count,
            "gaussian_backward_count": self.gaussian_backward_count,
            "sum_gaussians_over_train_steps": self.sum_gaussians_over_train_steps,
            "peak_gaussian_count": self.peak_gaussian_count,
        }

    def load_replay_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["intervention_mode"] != self.intervention_mode:
            raise ValueError("replay intervention mode differs from runtime")
        self.born_track_ids = {int(value) for value in state["born_track_ids"]}
        self._accepted = {
            int(key): dict(value) for key, value in state["accepted"].items()
        }
        for field in (
            "full_rgb_rasterizations",
            "fixed_support_probe_count",
            "gaussian_backward_count",
            "sum_gaussians_over_train_steps",
            "peak_gaussian_count",
        ):
            setattr(self, field, int(state[field]))


LINEAGE_KEYS = ("uid", "lineage_id", "parent_uid", "track_id", "birth_step")


class LineageDelayedAbsGradStrategy(DelayedAbsGradStrategy):
    """Default delayed topology with stable identity metadata and prune reasons."""

    def __init__(
        self,
        *,
        schedule: DelayedAbsGradSchedule,
        result_dir: str | Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(schedule=schedule, **kwargs)
        self._next_uid = 0
        self._prune_stream = Path(result_dir, "ru_part_lineage_prune.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._prune_writer = csv.DictWriter(
            self._prune_stream,
            fieldnames=(
                "step", "uid", "lineage_id", "parent_uid", "track_id",
                "birth_step", "opacity", "opacity_threshold", "max_scale",
                "scale_threshold", "radius", "radius_threshold",
                "reason_opacity", "reason_scale3d", "reason_scale2d",
            ),
        )
        self._prune_writer.writeheader()

    def ensure_lineage(self, state: dict[str, Any], count: int, device: torch.device) -> None:
        present = [key in state for key in LINEAGE_KEYS]
        if any(present) and not all(present):
            raise RuntimeError("lineage state is partially initialized")
        if all(present):
            if any(len(state[key]) != count for key in LINEAGE_KEYS):
                raise RuntimeError("lineage state length differs from Gaussian count")
            self._next_uid = max(self._next_uid, int(state["uid"].max().item()) + 1)
            return
        uid = torch.arange(count, dtype=torch.int64, device=device)
        state["uid"] = uid
        state["lineage_id"] = uid.clone()
        state["parent_uid"] = torch.full_like(uid, -1)
        state["track_id"] = torch.full_like(uid, -1)
        state["birth_step"] = torch.full_like(uid, -1)
        self._next_uid = count

    def _allocate_uids(self, count: int, device: torch.device) -> Tensor:
        result = torch.arange(
            self._next_uid, self._next_uid + count, dtype=torch.int64, device=device
        )
        self._next_uid += count
        return result

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step: int) -> tuple[int, int]:
        self.ensure_lineage(state, len(params["means"]), params["means"].device)
        proposals = compute_default_grow_masks(params, state, self, step)
        clone_indices = torch.where(proposals.clone)[0]
        split_indices = torch.where(proposals.split)[0]
        n_clone, n_split = len(clone_indices), len(split_indices)
        if n_clone:
            parent_uids = state["uid"][clone_indices].clone()
            duplicate(params=params, optimizers=optimizers, state=state, mask=proposals.clone)
            state["uid"][-n_clone:] = self._allocate_uids(n_clone, state["uid"].device)
            state["parent_uid"][-n_clone:] = parent_uids
        if n_split:
            split_mask = torch.cat((
                proposals.split,
                torch.zeros(n_clone, dtype=torch.bool, device=proposals.split.device),
            ))
            parent_uids = state["uid"][split_mask].clone()
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=split_mask,
                revised_opacity=self.revised_opacity,
            )
            state["uid"][-2 * n_split:] = self._allocate_uids(
                2 * n_split, state["uid"].device
            )
            state["parent_uid"][-2 * n_split:] = parent_uids.repeat(2)
        return n_clone, n_split

    @torch.no_grad()
    def _prune_gs(self, params, optimizers, state, step: int) -> int:
        self.ensure_lineage(state, len(params["means"]), params["means"].device)
        opacity = torch.sigmoid(params["opacities"].flatten())
        max_scale = torch.exp(params["scales"]).max(dim=-1).values
        radius = state.get("radii")
        reason_opacity = opacity < self.prune_opa
        reason_scale3d = torch.zeros_like(reason_opacity)
        reason_scale2d = torch.zeros_like(reason_opacity)
        if step > self.reset_every:
            reason_scale3d = max_scale > self.prune_scale3d * state["scene_scale"]
            if step < self.refine_scale2d_stop_iter:
                reason_scale2d = radius > self.prune_scale2d
        prune_mask = reason_opacity | reason_scale3d | reason_scale2d
        for index in torch.where(prune_mask)[0].tolist():
            self._prune_writer.writerow({
                "step": step,
                "uid": int(state["uid"][index]),
                "lineage_id": int(state["lineage_id"][index]),
                "parent_uid": int(state["parent_uid"][index]),
                "track_id": int(state["track_id"][index]),
                "birth_step": int(state["birth_step"][index]),
                "opacity": float(opacity[index]),
                "opacity_threshold": float(self.prune_opa),
                "max_scale": float(max_scale[index]),
                "scale_threshold": float(self.prune_scale3d * state["scene_scale"]),
                "radius": float(radius[index]) if isinstance(radius, Tensor) else -1.0,
                "radius_threshold": float(self.prune_scale2d),
                "reason_opacity": int(reason_opacity[index]),
                "reason_scale3d": int(reason_scale3d[index]),
                "reason_scale2d": int(reason_scale2d[index]),
            })
        if torch.any(prune_mask):
            self._prune_stream.flush()
        actual = super()._prune_gs(params, optimizers, state, step)
        if actual != int(prune_mask.sum()):
            raise RuntimeError("recorded prune mask differs from actual prune count")
        return actual

    def assign_birth_lineage(
        self, state: dict[str, Any], selected_track_ids: list[int], step: int
    ) -> None:
        count = len(selected_track_ids)
        if not count:
            return
        device = state["uid"].device
        new_uids = self._allocate_uids(count, device)
        state["uid"][-count:] = new_uids
        state["lineage_id"][-count:] = new_uids
        state["parent_uid"][-count:] = -1
        state["track_id"][-count:] = torch.as_tensor(selected_track_ids, device=device)
        state["birth_step"][-count:] = int(step)


class RUPARTStrategy(LineageDelayedAbsGradStrategy):
    """Delayed strategy replacing eligible clone slots with prospective births."""

    def __init__(self, *, schedule: DelayedAbsGradSchedule, **kwargs: Any) -> None:
        super().__init__(schedule=schedule, **kwargs)
        self.part_controller: RUPARTController | None = None

    def step_post_backward(self, params, optimizers, state, step: int, info, packed: bool = False) -> None:
        if self.part_controller is None:
            raise RuntimeError("RU-PART strategy has no controller")
        if packed:
            raise ValueError("RU-PART supports packed=False only")
        if step >= self.schedule.statistics_stop_step:
            return
        self._update_state(params, state, info, packed=False)
        if self.schedule.should_refine(step):
            self._part_refine(params, optimizers, state, step)
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()
        if self.schedule.should_reset(step):
            from gsplat.strategy.ops import reset_opa
            reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * 2.0)

    @torch.no_grad()
    def _part_refine(self, params, optimizers, state, step: int) -> None:
        if duplicate is None or split is None:
            raise ImportError("gsplat is required for RU-PART topology")
        started = perf_counter()
        controller = self.part_controller
        assert controller is not None
        event = controller.take_event(step)
        before = len(params["means"])
        self.ensure_lineage(state, before, params["means"].device)
        if "part_track_id" not in state:
            state["part_track_id"] = torch.full(
                (before,), -1, dtype=torch.long, device=params["means"].device
            )
            state["part_birth_step"] = torch.full_like(state["part_track_id"], -1)
        survived_100, survived_500 = controller.lifecycle_counts(step, state)
        proposals = compute_default_grow_masks(params, state, self, step)
        clone_indices = torch.where(proposals.clone)[0]
        split_indices = torch.where(proposals.split)[0]
        track_rows = controller.visible_track_rows(event["train_view_id"])
        previously_born = set(controller.born_track_ids)
        decision_support = event["support"].float().cpu()
        decision_alpha = event["alpha"].float().cpu()
        decision_evidence = event["evidence"].float().cpu()

        birth_footprints = []
        if len(track_rows):
            ids_cpu = controller.track_ids[track_rows]
            ids = ids_cpu.to(params["means"].device)
            means = controller.track_xyz[track_rows].to(params["means"].device)
            scales = torch.as_tensor(np.stack([controller._geometry[int(v)][0] for v in ids_cpu.tolist()]), device=means.device)
            quats = torch.as_tensor(np.stack([controller._geometry[int(v)][1] for v in ids_cpu.tolist()]), device=means.device)
            radii, means2d, _, conics = project_gaussians(means, quats, scales.exp(), event["viewmat"], event["K_grid"])
            birth_footprints = sparse_footprints_from_projection(
                ids.cpu(), means2d.cpu(), conics.cpu(), radii.cpu()
            )
            initialize_footprint_scores(
                birth_footprints, decision_support, decision_alpha, decision_evidence,
                initial_opacity=0.1,
            )

        clone_footprints = []
        if len(clone_indices):
            radii, means2d, _, conics = project_gaussians(
                params["means"][clone_indices], params["quats"][clone_indices], params["scales"][clone_indices].exp(),
                event["viewmat"], event["K_grid"],
            )
            clone_footprints = sparse_footprints_from_projection(
                clone_indices.cpu(), means2d.cpu(), conics.cpu(), radii.cpu()
            )
            initialize_footprint_scores(
                clone_footprints, decision_support, decision_alpha, decision_evidence,
                initial_opacity=0.1,
            )

        clone_id_values = clone_indices.detach().cpu().tolist()
        clone_grad_values = proposals.average_grad2d[clone_indices].detach().cpu().tolist()
        replacements, updated_support = pair_births_with_clones(
            birth_footprints,
            clone_footprints,
            decision_support,
            decision_evidence,
            limit=len(clone_indices),
            clone_tiebreak=dict(zip(clone_id_values, clone_grad_values)),
        )
        replaced = {item.clone_id for item in replacements}
        selected = [item.birth_id for item in replacements]
        clone_mask = proposals.clone.clone()
        if replaced:
            clone_mask[torch.as_tensor(sorted(replaced), device=clone_mask.device)] = False

        # Conservative pre-prune cap.  Reject low-AbsGrad clones, then low-utility births,
        # then low-AbsGrad split tokens, exactly in the preregistered order.
        budget_rejected_clone = budget_rejected_birth = budget_rejected_split = 0
        predicted = before + int(clone_mask.sum()) + len(selected) + int(proposals.split.sum())
        excess = max(0, predicted - N_MAX)
        if excess:
            candidates = torch.where(clone_mask)[0]
            order = candidates[torch.argsort(proposals.average_grad2d[candidates], stable=True)]
            drop = order[:excess]
            clone_mask[drop] = False
            budget_rejected_clone = len(drop)
            excess -= len(drop)
        if excess:
            replacements.sort(key=lambda item: (item.benefit - item.harm, item.birth_id))
            drop_count = min(excess, len(replacements))
            dropped = {item.birth_id for item in replacements[:drop_count]}
            replacements = [item for item in replacements if item.birth_id not in dropped]
            selected = [item.birth_id for item in replacements]
            budget_rejected_birth = drop_count
            excess -= drop_count
            replaced = {item.clone_id for item in replacements}
        split_mask = proposals.split.clone()
        if excess:
            candidates = torch.where(split_mask)[0]
            order = candidates[torch.argsort(proposals.average_grad2d[candidates], stable=True)]
            drop = order[:excess]
            split_mask[drop] = False
            budget_rejected_split = len(drop)
            excess -= len(drop)
        if excess:
            raise RuntimeError("RU-PART cannot satisfy the global Gaussian cap")

        regular_clone_count = int(clone_mask.sum())
        if regular_clone_count:
            clone_parent_uids = state["uid"][clone_mask].clone()
            duplicate(params=params, optimizers=optimizers, state=state, mask=clone_mask)
            state["uid"][-regular_clone_count:] = self._allocate_uids(
                regular_clone_count, state["uid"].device
            )
            state["parent_uid"][-regular_clone_count:] = clone_parent_uids
        extended_split = torch.cat((split_mask, torch.zeros(regular_clone_count, dtype=torch.bool, device=split_mask.device)))
        split_count = int(split_mask.sum())
        if split_count:
            split_parent_uids = state["uid"][extended_split].clone()
            split(params=params, optimizers=optimizers, state=state, mask=extended_split, revised_opacity=self.revised_opacity)
            state["uid"][-2 * split_count:] = self._allocate_uids(
                2 * split_count, state["uid"].device
            )
            state["parent_uid"][-2 * split_count:] = split_parent_uids.repeat(2)
        if selected:
            rows = controller.birth_rows(selected, params)
            append_birth_rows(params, optimizers, state, rows)
            self.assign_birth_lineage(state, selected, step)
            state["part_track_id"][-len(selected):] = torch.as_tensor(selected, device=params["means"].device)
            state["part_birth_step"][-len(selected):] = step
            controller.born_track_ids.update(selected)
            for track_id in selected:
                controller._accepted[track_id] = {"birth_step": step, "survived_100": -1, "survived_500": -1}
        predicted_actual = before + regular_clone_count + len(selected) + split_count
        if len(params["means"]) != predicted_actual:
            raise RuntimeError("N_pred_grow differs from actual pre-prune Gaussian count")
        prune_count = self._prune_gs(params, optimizers, state, step)
        validate_parameter_state_shapes(params, optimizers, state)
        if len(params["means"]) > N_MAX:
            raise RuntimeError("RU-PART Gaussian hard cap exceeded")

        accepted_b = [item.benefit for item in replacements]
        accepted_h = [item.harm for item in replacements]
        accepted_a = [item.alpha_mass for item in replacements]
        controller.record_acceptances(step, replacements)
        selected_set = set(selected)
        controller.save_representative(
            event,
            [item for item in birth_footprints if item.candidate_id in selected_set],
        )
        controller.record_event({
            "step": step, "intervention_mode": "current", "diagnostic_only": 0,
            "gaussians_before": before, "fixed_support_mean": float(decision_support.mean()),
            "standard_clone_proposed": len(clone_indices), "standard_split_proposed": len(split_indices),
            "post_grow_prune_candidate_count": prune_count, "track_candidates_visible": int((controller.track_views == event["train_view_id"]).any(1).sum()),
            "track_candidates_geometry_valid": len(track_rows), "candidate_B_positive": sum(item.initial_benefit > 0 for item in birth_footprints),
            "candidate_alpha_mass_positive": sum(item.alpha_mass > 0 for item in birth_footprints),
            "candidate_B_gt_H0": sum(item.initial_benefit > item.initial_harm for item in birth_footprints),
            "birth_pareto_dominates_clone": len(replacements),
            "counterfactual_birth_selected": len(replacements),
            "coverage_birth_accepted": len(selected),
            "regular_clone_replaced": len(replaced), "regular_clone_executed": regular_clone_count,
            "split_executed": split_count, "prune_executed": prune_count,
            "budget_rejected_clone": budget_rejected_clone, "budget_rejected_birth": budget_rejected_birth,
            "budget_rejected_split": budget_rejected_split,
            "accepted_track_id_unique": int(
                len(selected) == len(set(selected)) and not previously_born.intersection(selected)
            ),
            "birth_survived_100": survived_100, "birth_survived_500": survived_500,
            "mean_B_accepted": float(np.mean(accepted_b)) if accepted_b else 0.0,
            "mean_H0_accepted": float(np.mean(accepted_h)) if accepted_h else 0.0,
            "mean_prospective_alpha_mass": float(np.mean(accepted_a)) if accepted_a else 0.0,
            "mean_fixed_support_gain": float((updated_support - decision_support).mean()),
            "gaussians_after": len(params["means"]), "fixed_support_probe_ms": event["probe_ms"],
            "topology_total_ms": (perf_counter() - started) * 1000.0,
        })


def ru_part_strategy_from_default(
    strategy: Any,
    schedule: DelayedAbsGradSchedule,
    *,
    intervention_mode: str = "current",
    result_dir: str | Path = ".",
) -> LineageDelayedAbsGradStrategy:
    fields = (
        "prune_opa", "grow_grad2d", "grow_scale3d", "grow_scale2d", "prune_scale3d",
        "prune_scale2d", "refine_scale2d_stop_iter", "pause_refine_after_reset",
        "revised_opacity", "verbose", "key_for_gradient",
    )
    strategy_class = (
        RUPARTStrategy if intervention_mode == "current" else LineageDelayedAbsGradStrategy
    )
    return strategy_class(
        schedule=schedule,
        result_dir=result_dir,
        **{field: getattr(strategy, field) for field in fields},
    )
