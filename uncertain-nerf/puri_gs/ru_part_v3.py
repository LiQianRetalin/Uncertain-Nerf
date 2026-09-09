"""The single V3 L1 candidate, with bounded screening diagnostics.

No PART controller, rasterizer, matcher, or topology implementation lives here.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from puri_gs.static_tracks import (
    TrackBuildConfig, build_config_dict, evidence_upsample, hash_basenames,
    load_static_track_cache, sha256_file,
)

PROTOCOL = {
    "BRANCH": "ru-part", "TRAINING_FROM_SCRATCH": True,
    "RESUMED_FAILED_SNAPSHOT": False,
    "HISTORICAL_REPLAY_STATUS": "REPLAY_NOT_EQUIVALENT",
    "HISTORICAL_REPLAY_RECLASSIFIED": False,
    "NEW_PROTOCOL": "V3_FROM_SCRATCH_SCREENING",
    "EXACT_200_STEP_REPLAY_REQUIRED": False,
    "NEW_ALGORITHM": "RU-PART-V3", "DEDICATED_BIRTH_ENABLED": False,
    "PART_CAP_ENABLED": False, "EXTRA_TOPOLOGY_SCORING_ENABLED": False,
    "MULTI_SEED_RUN": False, "CROSS_SCENE_RUN": False,
    "NEXT_ALGORITHM_STARTED": False,
}
DISABLED_COUNTS = {name: 0 for name in (
    "birth", "probe", "scoring", "cap", "lineage", "extra_training_rasterization", "per_step_matching",
)}
IMPLEMENTATION_REVISION = "v3-single-q-pinned-transfer-v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def static_rescue_l1(render, target, final_mask, support, *, step):
    """Unweighted mean over ALL B,H,W,RGB elements; no alpha dependency.

    The caller adds (1 - Parent.ssim_lambda) times this scalar to Parent loss.
    M is the exact already-thresholded, already-eroded Parent [B,1,H,W] mask.
    C is the existing bilinear support [B,1,H,W], without further processing.
    """
    loss, _ = _static_rescue_terms(render, target, final_mask, support, step=step)
    return loss


def _static_rescue_terms(render, target, final_mask, support, *, step):
    """Return the loss and its SAME detached Q for activation diagnostics."""
    if render.shape != target.shape or render.ndim != 4 or render.shape[-1] != 3:
        raise ValueError("RGB must have matching [B,H,W,3] shapes")
    expected = (render.shape[0], 1, render.shape[1], render.shape[2])
    if final_mask.shape != expected or support.shape != expected:
        raise ValueError("M and C must be [B,1,H,W] matching RGB")
    if step < 500:
        return render.sum() * 0.0, None
    q = (1.0 - final_mask.detach()) * support.detach()
    return (q.permute(0, 2, 3, 1) * (render - target).abs()).mean(), q


def runtime_identity(parser, trainset, valset):
    names = [parser.image_names[int(i)] for i in trainset.indices]
    tests = [parser.image_names[int(i)] for i in valset.indices]
    if len(names) != 161 or len(tests) != 24 or set(names) & set(tests):
        raise ValueError("INVALID_RUN: Garden must have 161 train / 24 test views")
    cameras = []
    for local, index in enumerate(trainset.indices):
        index = int(index)
        camera_id = parser.camera_ids[index]
        width, height = parser.imsize_dict[camera_id]
        cameras.append({
            "view_id": local, "basename": names[local],
            "camtoworld": np.asarray(parser.camtoworlds[index]).round(12).tolist(),
            "K": np.asarray(parser.Ks_dict[camera_id]).round(12).tolist(),
            "width": width, "height": height,
        })
    return {
        "train_basenames": names, "test_basenames": tests,
        "dataset_split_sha256": hash_basenames(["train:" + n for n in names] + ["test:" + n for n in tests]),
        "camera_manifest_sha256": hash_basenames([
            json.dumps(row, sort_keys=True, separators=(",", ":")) for row in cameras
        ]),
        "cameras": cameras,
    }


class StaticSupport:
    def __init__(self, path, *, identity, cfg, device):
        started = time.perf_counter()
        self.path = Path(path)
        payload = load_static_track_cache(path)
        for key in ("train_basenames", "dataset_split_sha256"):
            if payload[key] != identity[key]:
                raise ValueError(f"INVALID_RUN: static cache {key} differs from runtime")
        # v2 builds normalization with float64 points; Parent parses float32 points.
        # Recompute v2's exact canonical hash with its own unmodified loader, then
        # check the runtime geometry separately, without changing the matcher.
        from tools.build_puri_gs_static_tracks import load_camera_records
        gsplat_dir = Path(sys.modules["datasets.colmap"].__file__).resolve().parents[2]
        all_cameras, _ = load_camera_records(Path(cfg.data_dir), gsplat_dir, 4)
        originals = [c for i, c in enumerate(all_cameras) if i % 8 != 0]
        first = identity["cameras"][0]
        sx = first["width"] / originals[0].width
        sy = first["height"] / originals[0].height
        canonical = []
        max_pose_error = max_K_error = 0.0
        for local, (camera, runtime) in enumerate(zip(originals, identity["cameras"])):
            K = camera.K.copy()
            K[0] *= sx
            K[1] *= sy
            pose_error = float(np.max(np.abs(camera.camtoworld - np.asarray(runtime["camtoworld"]))))
            K_error = float(np.max(np.abs(K - np.asarray(runtime["K"]))))
            max_pose_error, max_K_error = max(max_pose_error, pose_error), max(max_K_error, K_error)
            if (camera.basename != runtime["basename"] or pose_error > 1e-5 or K_error > 1e-5):
                raise ValueError("INVALID_RUN: runtime geometry differs from v2 builder geometry")
            canonical.append({"view_id": local, "basename": camera.basename,
                              "camtoworld": camera.camtoworld.round(12).tolist(),
                              "K": K.round(12).tolist(), "width": int(round(camera.width * sx)),
                              "height": int(round(camera.height * sy))})
        canonical_hash = hash_basenames([json.dumps(c, sort_keys=True, separators=(",", ":")) for c in canonical])
        if canonical_hash != payload["camera_manifest_sha256"]:
            raise ValueError("INVALID_RUN: v2 canonical camera hash differs")
        if payload["test_basenames_hash_only"] != hash_basenames(identity["test_basenames"]):
            raise ValueError("INVALID_RUN: static cache test split differs")
        if Path(payload["dataset_root_canonical"]).resolve() != Path(cfg.data_dir).resolve():
            raise ValueError("INVALID_RUN: static cache data root differs")
        expected = {**build_config_dict(TrackBuildConfig()), "near": cfg.near_plane,
                    "far": cfg.far_plane, "factor": 4}
        if payload["build_config"] != expected:
            raise ValueError("INVALID_RUN: static cache build configuration differs")
        feature_manifest = Path(cfg.feature_cache_dir, "manifest.json")
        if sha256_file(feature_manifest) != payload["dino_feature_manifest_sha256"]:
            raise ValueError("INVALID_RUN: static cache feature manifest differs")
        self.grid = payload["track_evidence_binary"].float()  # CPU, load once
        if not bool(self.grid.any()):
            raise ValueError("EVIDENCE_INACTIVE")
        # Pin the small CPU grid once; transfer only the current view on the
        # training CUDA stream. The pinned source lives for the entire run.
        if torch.device(device).type == "cuda":
            self.grid = self.grid.pin_memory()
        self.device = device
        self.payload = payload
        manifest = self.path.with_name("static_track_manifest.json")
        original = json.loads(manifest.read_text()) if manifest.is_file() else {}
        self.audit = {
            "cache_file_sha256": sha256_file(path), "payload_sha256": payload["payload_sha256"],
            "v2_camera_sha256": canonical_hash,
            "runtime_pose_max_abs_error": max_pose_error, "runtime_K_max_abs_error": max_K_error,
            "build_config": payload["build_config"], "build_git_commit": payload["build_git_commit"],
            "original_cache_build_seconds": original.get("build_time_seconds"),
            "feature_extraction_seconds": json.loads(feature_manifest.read_text()).get("feature_extraction_seconds"),
            "grid_nonzero_fraction_by_view": (self.grid > 0).float().mean((1, 2)).tolist(),
            "grid_mean_by_view": self.grid.mean((1, 2)).tolist(),
            "upsample": "bilinear / align_corners=False / no second threshold",
            "cpu_grid_pinned": self.grid.is_pinned(),
            "transfer": "current view only / non_blocking / current stream",
            "cache_prepare_seconds": time.perf_counter() - started,
            "historical_source_image_content_hashes": "NOT_RECORDED_BY_V2_SCHEMA",
        }

    def current(self, local_image_id, size):
        # Dataset.image_id is split-local; caller obtains it on CPU before H2D.
        return evidence_upsample(
            self.grid[local_image_id].to(self.device, non_blocking=True), size,
        ).detach()


class ScreeningRun:
    def __init__(self, runner):
        from puri_gs.delayed_absgrad import DelayedAbsGradStrategy
        from puri_gs.config import RU_FIXED_FIELDS
        self.runner = runner
        self.cfg = cfg = runner.cfg
        self.out = Path(cfg.result_dir)
        self.mode = cfg.ru_v3_mode
        self.stop = cfg.ru_v3_stop_step
        if (type(cfg.strategy) is not DelayedAbsGradStrategy or runner.ru_training.ru_part is not None
                or self.mode not in ("parent", "v3") or self.stop not in (599, 29999)):
            raise ValueError("INVALID_RUN: V3 requires the unwrapped standard RU strategy")
        for field in RU_FIXED_FIELDS:
            if field in ("method", "total_steps", "absgrad", "grow_grad2d", "mask_enabled", "seed"):
                continue
            if getattr(cfg, field) != RU_FIXED_FIELDS[field]:
                raise ValueError(f"INVALID_RUN: runtime field {field} differs")
        if (cfg.max_steps != 30000 or cfg.steps_scaler != 1 or cfg.batch_size != 1
                or not cfg.strategy.absgrad or cfg.strategy.grow_grad2d != 0.0006
                or cfg.ckpt is not None or getattr(cfg, "resume_ckpt", None) is not None
                or cfg.pose_opt or cfg.app_opt or cfg.depth_loss or cfg.use_bilateral_grid
                or cfg.opacity_reg or cfg.scale_reg or cfg.init_type != "sfm"
                or cfg.random_bkgd or cfg.patch_size is not None or not cfg.normalize_world_space):
            raise ValueError("INVALID_RUN: unsupported runtime change or resume")
        strategy_defaults = {"prune_opa": .005, "grow_scale3d": .01, "grow_scale2d": .05,
                             "prune_scale3d": .1, "prune_scale2d": .15,
                             "refine_scale2d_stop_iter": 0, "pause_refine_after_reset": 0,
                             "revised_opacity": False, "key_for_gradient": "means2d"}
        for field, expected in strategy_defaults.items():
            if getattr(cfg.strategy, field) != expected:
                raise ValueError(f"INVALID_RUN: gsplat 1.5.3 strategy field {field} differs")
        self.identity = runtime_identity(runner.parser, runner.trainset, runner.valset)
        self.support = StaticSupport(cfg.ru_v3_track_cache, identity=self.identity, cfg=cfg,
                                     device=runner.device) if self.mode == "v3" else None
        self.identity["train_image_sha256"] = {
            runner.parser.image_names[int(i)]: sha256_file(runner.parser.image_paths[int(i)])
            for i in runner.trainset.indices
        }
        self.identity["sfm_sha256"] = {
            n: sha256_file(Path(cfg.data_dir, "sparse/0", n))
            for n in ("cameras.bin", "images.bin", "points3D.bin")
        }
        self.identity["normalization"] = np.asarray(runner.parser.transform).tolist()
        write_json(self.out / "v3_input_manifest.json", self.identity)
        write_json(self.out / "v3_run_manifest.json", {
            **PROTOCOL, "mode": self.mode, "last_step_planned": self.stop,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "strategy_class": type(cfg.strategy).__name__,
            "disabled_call_counts": DISABLED_COUNTS,
            "cache": self.support.audit if self.support else None,
            "data_identity": "v3_input_manifest.json", "seed": 42,
            "resolved_strategy": {**strategy_defaults, "absgrad": cfg.strategy.absgrad,
                                  "grow_grad2d": cfg.strategy.grow_grad2d},
            "runtime_gpu_name": torch.cuda.get_device_name(runner.device),
            "runtime_cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        })
        self.stream = (self.out / "v3_progress.jsonl").open("w", encoding="utf-8")
        self.topology = (self.out / "v3_adc.csv").open("w", newline="", encoding="utf-8")
        self.topo_writer = None
        cfg.strategy.event_recorder = self.record_topology
        self.active = torch.zeros((), device=runner.device, dtype=torch.int64)
        self.loss_sum = torch.zeros((), device=runner.device)
        self.profile_ms = []
        self.sequence = []
        self.rasterizations = 0
        self.backwards = 0
        self.gradient_checks = []
        self.ended = False

    def record_topology(self, **row):
        if self.topo_writer is None:
            self.topo_writer = csv.DictWriter(self.topology, fieldnames=list(row))
            self.topo_writer.writeheader()
        self.topo_writer.writerow(row)
        self.topology.flush()

    def begin_step(self, step):
        if self.stop == 599 and 520 <= step < 540:
            torch.cuda.synchronize()
            self.profile_start = time.perf_counter()

    def note_backward(self, step):
        self.backwards += 1
        if step in (500, 599):
            gradients = [p.grad for p in self.runner.splats.values() if p.grad is not None]
            finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
            nonzero = bool(torch.stack([(g != 0).any() for g in gradients]).any())
            if not finite or not nonzero:
                raise FloatingPointError("Gaussian gradients are non-finite or entirely zero")
            self.gradient_checks.append({"step": step, "all_finite": finite, "some_parameter_gradient_nonzero": nonzero})

    def add_loss(self, step, local_id, render, target, mask):
        self.sequence.append(local_id)
        self.c = self.support.current(local_id, render.shape[1:3]) if self.support and step >= 500 else None
        if self.c is None:
            self.q = None
            self.extra = render.new_zeros(())
            return self.extra
        unweighted, self.q = _static_rescue_terms(render, target, mask, self.c, step=step)
        self.extra = (1 - self.cfg.ssim_lambda) * unweighted
        self.active += (self.q.detach().amax() > 0).to(torch.int64)
        self.loss_sum += self.extra.detach()
        return self.extra

    def post_step(self, step, base_loss, total_loss, mask, alpha, elapsed):
        if self.stop == 599 and 520 <= step < 540:
            torch.cuda.synchronize()
            self.profile_ms.append((time.perf_counter() - self.profile_start) * 1000)
        if step % 200 == 0 or step in (499, 500, self.stop):
            row = {
                "step": step, "parent_base_loss": float(base_loss.detach()),
                "added_l1": float(self.extra.detach()), "total_loss": float(total_loss.detach()),
                "mean_C": float(self.c.mean()) if self.c is not None else 0.0,
                "mean_Q": float(self.q.mean()) if self.q is not None else 0.0,
                "mask_acceptance": float(mask.detach().mean()),
                "standard_alpha_mean": float(alpha.detach().mean()),
                "gaussian_count": len(self.runner.splats["means"]),
                "elapsed_seconds": elapsed,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "active_samples": int(self.active), "disabled_call_counts": DISABLED_COUNTS,
            }
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.stream.flush()
            write_json(self.out / "v3_progress.json", row)
        if step == self.stop:
            updates = step + 1
            expected_rasterizations = updates + min(updates, 20000)
            if self.rasterizations != expected_rasterizations or self.backwards != updates:
                raise ValueError("unexpected training rasterization/backward count")
            activation_status = "ACTIVE" if int(self.active) else "NO_Q_SAMPLED"
            fixed_check = None
            if self.mode == "v3" and not int(self.active):
                # A bounded isolated tensor check, never a second Gaussian backward.
                probe = torch.full((1, 2, 2, 3), .5, device=self.runner.device, requires_grad=True)
                probe_loss = .8 * static_rescue_l1(probe, torch.zeros_like(probe),
                    torch.zeros(1, 1, 2, 2, device=probe.device),
                    torch.ones(1, 1, 2, 2, device=probe.device), step=500)
                gradient, = torch.autograd.grad(probe_loss, probe)
                error = float((gradient - .8 / 12).abs().max())
                if error > 2e-7:
                    raise ValueError("fixed-input activation gradient check failed")
                fixed_check = {"type": "FIXED_TENSOR_FUNCTIONAL_CHECK", "gradient_max_abs_error": error}
            write_json(self.out / "v3_training_checks.json", {
                "implementation_revision": IMPLEMENTATION_REVISION,
                "last_step": step, "active_samples": int(self.active),
                "added_loss_sum": float(self.loss_sum), "disabled_call_counts": DISABLED_COUNTS,
                "profile_steps": list(range(520, 540)) if self.profile_ms else [],
                "profile_step_ms": self.profile_ms,
                "profile_median_ms": float(np.median(self.profile_ms)) if self.profile_ms else None,
                "gaussian_backward_per_update": 1,
                "actual_gaussian_backward_count": self.backwards,
                "actual_training_rasterization_count": self.rasterizations,
                "expected_parent_rasterization_count": expected_rasterizations,
                "parameter_gradient_checks": self.gradient_checks,
                "activation_status": activation_status, "fallback_fixed_tensor_check": fixed_check,
                "mask_update_count": self.runner.ru_training.mask_update_count,
                "training_seconds": elapsed,
            })
            write_json(self.out / "v3_camera_sequence.json", self.sequence)
            self.stream.close()
            self.topology.close()


@torch.no_grad()
def export_training_evidence(runner, screening, *, step):
    """Read-only forwards at the SAME Gaussian/head state; never resume optimizers.

    Only two fixed train images are opened. This runs after the timed smoke/full
    training segment and does not affect training's sampling or RNG.
    """
    if screening.support is None:
        return
    from PIL import Image
    from puri_gs.semantic_mask import hard_static_mask
    import torch.nn.functional as F
    out = screening.out / "evidence_check"
    out.mkdir()
    names = screening.identity["train_basenames"]
    selected = ["DSC07987.JPG", "DSC07989.JPG"]
    if not all(name in names for name in selected):
        raise ValueError("INVALID_RUN: prescribed evidence images missing from train manifest")
    rows = []
    for name in selected:
        local = names.index(name)
        data = runner.trainset[local]
        target = data["image"][None].to(runner.device).float() / 255
        height, width = target.shape[1:3]
        render, _, _ = runner.rasterize_splats(
            camtoworlds=data["camtoworld"][None].to(runner.device),
            Ks=data["K"][None].to(runner.device), width=width, height=height,
            sh_degree=min(step // runner.cfg.sh_degree_interval, runner.cfg.sh_degree),
            near_plane=runner.cfg.near_plane, far_plane=runner.cfg.far_plane,
        )
        grid = 16 if step < runner.cfg.bootstrap_switch_step else 36
        feature = runner.ru_training.cache.load(name, grid)[None].to(runner.device)
        probability = runner.ru_training.head(feature)
        probability = F.interpolate(probability, size=(height, width), mode="bilinear", align_corners=False)
        mask = hard_static_mask(probability).detach()
        c = screening.support.current(local, (height, width))
        q = (1 - mask) * c
        residual = (render - target).abs().mean(-1)[0]
        k = max(1, int(np.ceil(residual.numel() * 0.2)))
        # Exactly the top 20%, including deterministic handling of boundary ties.
        hard_indices = torch.argsort(residual.flatten(), descending=True, stable=True)[:k]
        hard_q = q.flatten()[hard_indices]
        row = {
            "image": name, "step": step,
            "C_nonzero_fraction": float((c > 0).float().mean()), "mean_C": float(c.mean()),
            "mask_rejected_fraction": float((1 - mask).mean()),
            "Q_nonzero_fraction": float((q > 0).float().mean()), "mean_Q": float(q.mean()),
            "top20_residual_Q_coverage": float((hard_q > 0).float().mean()),
            "top20_residual_mean_Q": float(hard_q.mean()),
            "source_grid_view_id": local, "source_tensor": "track_evidence_binary",
        }
        rows.append(row)
        tiles = []
        maps = {"GT": target[0], "render": render[0].clamp(0, 1),
                "residual": residual, "M": mask[0, 0], "C": c[0, 0], "Q": q[0, 0]}
        overlay = target[0].clone()
        overlay = overlay * (1 - 0.65 * q[0, 0, ..., None])
        overlay[..., 0] += 0.65 * q[0, 0]
        maps["overlay"] = overlay
        for label, value in maps.items():
            array = (value.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            im = Image.fromarray(array).convert("RGB")
            im.save(out / f"{Path(name).stem}_{label}.png")
            im.thumbnail((420, 300))
            tiles.append(im)
        canvas = Image.new("RGB", (420 * 4, 325 * 2), "white")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        for i, (label, im) in enumerate(zip(maps, tiles)):
            x, y = (i % 4) * 420, (i // 4) * 325
            draw.text((x + 5, y + 3), label, fill="black")
            canvas.paste(im, (x, y + 25))
        canvas.save(out / f"{Path(name).stem}_panel.png")
        torch.save({"fine_grid": screening.support.grid[local],
                    "track_ids": screening.support.payload["track_ids"],
                    "track_view_ids": screening.support.payload["track_view_ids"],
                    "track_patch_xy": screening.support.payload["track_patch_xy"]},
                   out / f"{Path(name).stem}_trace.pt")
    status = "ACTIVE" if any(r["mean_Q"] > 0 for r in rows) else "SUPERVISION_INACTIVE_AT_AUDITED_STATE"
    write_json(out / "evidence_check.json", {
        "status": status, "state": "same run/step Gaussian and saved head",
        "scope": "EARLY_ACTIVATION_ONLY" if step == 599 else "FINAL_TRAINING_STATE",
        "cache": screening.support.audit, "images": rows,
    })
