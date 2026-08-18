import json
import math
import os
import random
import shutil
import time

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from v5.losses import DepthScaleAligner
from v5.robust import anneal, warmup_cosine_lr
from v6.trainer import TrainerV6, _safe_logit, _slice_result, set_seed

from .data import AdaptiveRaySampler, load_scene, pixel_ray_radii, pixels_to_rays
from .losses import termination_distribution_loss, uncertainty_color_nll
from .model import RadianceFieldV7
from .occupancy import OccupancyGrid
from .rendering import distortion_loss, render_rays
from .teacher import VisibilityAwareTeacher, neighborhood_uncertainty_regularization


CHECKPOINT_VERSION = "uncertain-nerf-v7-1"


class _DisabledAppearance(nn.Module):
    def forward(self, image_ids):
        return torch.empty(len(image_ids), 0, device=image_ids.device)


class TrainerV7(TrainerV6):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                                   else "cpu" if args.device == "auto" else args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if args.hash_backend == "tcnn" and self.device.type != "cuda":
            raise ValueError("hash_backend=tcnn requires a CUDA device")
        set_seed(args.seed)
        self.scene = load_scene(args.datadir, factor=args.factor, holdout=args.llffhold,
                                n_source_views=args.teacher_views, device=self.device,
                                aabb_scale=args.aabb_scale, val_every=args.val_every)
        self.field = RadianceFieldV7(
            aabb=self.scene.aabb, hidden_dim=args.netwidth, hidden_layers=args.netdepth,
            hash_levels=args.hash_levels, hash_min_resolution=args.hash_min_resolution,
            hash_max_resolution=args.hash_max_resolution, hash_features=args.hash_features,
            hash_log2_size=args.hash_log2_size, direction_frequencies=args.multires_views,
            appearance_dim=0, density_bias=args.density_bias, hash_backend=args.hash_backend,
            mip_enabled=not args.disable_mip).to(self.device)
        # Static LLFF has one shared appearance; this object only keeps inherited helpers safe.
        self.appearance = _DisabledAppearance().to(self.device)
        self.background_logit = nn.Parameter(
            _safe_logit(self.scene.training_color_mean.detach().clone()))
        self.teacher = VisibilityAwareTeacher(args.teacher_photometric_weight)
        self.occupancy_grid = (OccupancyGrid(
            self.scene.aabb, args.occupancy_resolution, args.occupancy_threshold,
            args.occupancy_decay, args.occupancy_warmup).to(self.device)
            if args.use_occupancy_grid else None)
        self.optimizer = torch.optim.Adam([
            {"params": self.field.parameters(), "lr": args.lrate},
            {"params": [self.background_logit], "lr": args.lrate_background},
        ], betas=(0.9, 0.99), eps=1.0e-8)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(args.amp and self.device.type == "cuda"))
        self.depth_aligner = DepthScaleAligner(
            warmup=args.depth_scale_warmup, interval=args.depth_scale_interval,
            ema=args.depth_scale_ema)
        self.sampler = AdaptiveRaySampler(
            self.scene, args.N_rand, args.prior_ray_ratio, args.edge_ray_ratio,
            args.uncertainty_ray_ratio, args.uncertainty_sampling_ema,
            args.sampling_probability_floor)
        self.experiment_dir = os.path.join(args.basedir, args.expname)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.metrics_path = os.path.join(self.experiment_dir, "train_metrics.jsonl")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_step = 0
        self.training_wall_seconds = self.evaluation_wall_seconds = self._current_elapsed = 0.0
        self.time_to_target_psnr = None
        self._write_run_config()
        checkpoint = self._resolve_checkpoint()
        if checkpoint:
            self.load_checkpoint(checkpoint, load_optimizer=not args.render_only)

    def _mean_appearance(self):
        return torch.empty(1, 0, device=self.device)

    def _render_batch(self, batch, perturb):
        return render_rays(
            self.field, batch["rays_o"], batch["rays_d"], self.scene.near,
            self.scene.far, self._background(len(batch["rays_o"])),
            ray_radii=batch["ray_radii"], n_samples=self.args.N_samples,
            n_importance=self.args.N_importance, perturb=perturb,
            chunk=self.args.netchunk, sampling_space=self.args.sampling_space,
            occupancy_grid=self.occupancy_grid)

    def training_step(self, step):
        args = self.args
        if (self.occupancy_grid is not None and args.occupancy_update_interval > 0
                and step % args.occupancy_update_interval == 0):
            self.occupancy_grid.update(self.field, step, args.occupancy_update_samples,
                                       args.netchunk)
        batch = self.sampler.sample()
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(args.amp and self.device.type == "cuda")
        with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
            rendered = self._render_batch(batch, perturb=True)
            coarse, fine = rendered["coarse"], rendered["fine"]
            fine_mse = F.mse_loss(fine["rgb"], batch["target"])
            coarse_mse = F.mse_loss(coarse["rgb"], batch["target"])
            color_loss = fine_mse + args.coarse_loss_weight * coarse_mse
            color_nll = uncertainty_color_nll(
                fine["rgb"], batch["target"], fine["uncertainty"],
                args.variance_min, args.variance_max)
            distortion = distortion_loss(fine)
            teacher_loss = fine_mse.new_zeros(())
            spatial_reg = fine_mse.new_zeros(())
            teacher_count = 0
            teacher_due = (step >= args.teacher_start_step and args.teacher_interval > 0
                           and step % args.teacher_interval == 0)
            if teacher_due:
                teacher_indices = self._teacher_indices(rendered)
                teacher_count = int(teacher_indices.numel())
                if teacher_count:
                    fine_teacher = _slice_result(fine, teacher_indices)
                    teacher_target, _ = self.teacher(
                        fine_teacher, batch["target"][teacher_indices],
                        batch["image_ids"][teacher_indices], self.scene,
                        anneal(args.tukey_start, args.tukey_end, step,
                               args.robust_anneal_steps),
                        self.field, self._background(1), self.scene.near, self.scene.far,
                        visibility_samples=args.teacher_visibility_samples,
                        visibility_relative_tolerance=args.teacher_visibility_relative_tolerance,
                        visibility_absolute_tolerance=args.teacher_visibility_absolute_tolerance,
                        visibility_min_acc=args.teacher_visibility_min_acc,
                        chunk=args.netchunk, occupancy_grid=self.occupancy_grid)
                    teacher_loss = F.smooth_l1_loss(
                        fine["uncertainty"][teacher_indices], teacher_target)
                    if args.lambda_spatial > 0:
                        spatial_reg = neighborhood_uncertainty_regularization(
                            self.field, fine_teacher["surface_points"], args.spatial_epsilon)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            prior_mask = batch["prior_mask"] & rendered["valid"]
            geometry, geo_stats = termination_distribution_loss(
                fine["weights"].float(), fine["terminal_transmittance"].float(),
                fine["z_vals"].float(), batch["prior_depth"].float(), prior_mask,
                self.depth_aligner.scale, args.depth_distribution_relative_sigma,
                args.depth_distribution_interval_sigma)
        if step >= args.depth_scale_warmup:
            self.depth_aligner.observe(fine["depth"][prior_mask], batch["prior_depth"][prior_mask])
        self.depth_aligner.maybe_update(step)
        total = (color_loss + args.lambda_nll * color_nll + args.lambda_geo * geometry
                 + args.lambda_teacher * teacher_loss + args.lambda_spatial * spatial_reg
                 + args.lambda_distortion * distortion)
        self.scaler.scale(total).backward()
        if args.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.field.parameters(), args.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.sampler.update_uncertainty(batch["image_ids"], batch["ys"], batch["xs"],
                                        fine["uncertainty"])
        base_lr = warmup_cosine_lr(step + 1, args.N_iters, args.lr_warmup,
                                   args.lrate, args.lrate_min)
        self.optimizer.param_groups[0]["lr"] = base_lr
        self.optimizer.param_groups[1]["lr"] = base_lr * (
            args.lrate_background / max(args.lrate, 1.0e-12))
        psnr = -10.0 * torch.log10(fine_mse.clamp_min(1.0e-10))
        return {"loss": float(total.detach()), "color": float(color_loss.detach()),
                "nll": float(color_nll.detach()), "geo": float(geometry.detach()),
                "teacher": float(teacher_loss.detach()), "spatial": float(spatial_reg.detach()),
                "distortion": float(distortion.detach()), "psnr": float(psnr.detach()),
                "lr": base_lr, "scale": self.depth_aligner.scale,
                "teacher_count": teacher_count, "geo_count": geo_stats["count"]}

    def checkpoint_state(self, step):
        return {"version": CHECKPOINT_VERSION, "global_step": int(step),
                "dataset_fingerprint": self.scene.fingerprint,
                "field": self.field.state_dict(), "background_logit": self.background_logit.detach(),
                "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                "depth_aligner": self.depth_aligner.state_dict(),
                "sampler": self.sampler.state_dict(),
                "occupancy_grid": (self.occupancy_grid.state_dict()
                                   if self.occupancy_grid is not None else None),
                "training_wall_seconds": max(self.training_wall_seconds, self._current_elapsed),
                "evaluation_wall_seconds": self.evaluation_wall_seconds,
                "time_to_target_psnr": self.time_to_target_psnr,
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.random.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                "aabb": self.scene.aabb.detach().cpu(),
                "split": {"train": self.scene.train_indices.tolist(),
                          "val": self.scene.val_indices.tolist(),
                          "test": self.scene.test_indices.tolist()}}

    def save_checkpoint(self, step):
        path = os.path.join(self.checkpoint_dir, f"step_{step:06d}.pt")
        torch.save(self.checkpoint_state(step), path)
        shutil.copyfile(path, os.path.join(self.checkpoint_dir, "latest.pt"))
        print(f"Saved V7 checkpoint: {path}")

    def load_checkpoint(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"Expected {CHECKPOINT_VERSION}; V5/V6 checkpoints are incompatible")
        if state.get("dataset_fingerprint") != self.scene.fingerprint:
            raise ValueError("Checkpoint dataset fingerprint does not match the current scene")
        expected = {"train": self.scene.train_indices.tolist(), "val": self.scene.val_indices.tolist(),
                    "test": self.scene.test_indices.tolist()}
        if state.get("split") != expected:
            raise ValueError("Checkpoint split does not match the current V7 config")
        self.field.load_state_dict(state["field"])
        self.background_logit.data.copy_(state["background_logit"].to(self.device))
        self.depth_aligner.load_state_dict(state.get("depth_aligner", {}))
        self.sampler.load_state_dict(state.get("sampler", {}))
        if self.occupancy_grid is not None and state.get("occupancy_grid") is not None:
            self.occupancy_grid.load_state_dict(state["occupancy_grid"])
            self.occupancy_grid.active = int(state["global_step"]) >= self.args.occupancy_warmup
        self.training_wall_seconds = float(state.get("training_wall_seconds", 0.0))
        self.evaluation_wall_seconds = float(state.get("evaluation_wall_seconds", 0.0))
        self._current_elapsed = self.training_wall_seconds
        self.time_to_target_psnr = state.get("time_to_target_psnr")
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state.get("scaler", {}))
            rng = state.get("rng", {})
            if rng:
                random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
                torch.random.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
        self.start_step = int(state["global_step"])
        print(f"Loaded V7 checkpoint {path} at step {self.start_step}")

    def _render_pose(self, pose, intrinsic, radial_distortion=0.0):
        height, width, _ = self.scene.hwf
        ys, xs = torch.meshgrid(torch.arange(height, device=self.device),
                                torch.arange(width, device=self.device), indexing="ij")
        count = height * width
        image_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        rays_o, rays_d = pixels_to_rays(pose[None], intrinsic[None], image_ids,
                                        ys.reshape(-1), xs.reshape(-1), radial_distortion)
        radii = pixel_ray_radii(intrinsic[None], image_ids)
        outputs = {key: [] for key in ("rgb", "depth", "uncertainty", "acc")}
        for start in range(0, count, self.args.chunk):
            end = min(start + self.args.chunk, count)
            result = render_rays(
                self.field, rays_o[start:end], rays_d[start:end], self.scene.near,
                self.scene.far, self._background(end - start), ray_radii=radii[start:end],
                n_samples=self.args.N_samples, n_importance=self.args.N_importance,
                perturb=False, chunk=self.args.netchunk,
                sampling_space=self.args.sampling_space,
                occupancy_grid=self.occupancy_grid)["fine"]
            for key in outputs:
                outputs[key].append(result[key].detach().cpu())
        return {key: torch.cat(value).reshape(
            height, width, -1 if key == "rgb" else 1).squeeze(-1).numpy()
            for key, value in outputs.items()}

    def train(self):
        if self.args.render_only:
            return self.render()
        self.field.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device); torch.cuda.synchronize(self.device)
        progress = trange(self.start_step + 1, self.args.N_iters + 1, desc="training-v7")
        wall_start = time.perf_counter()
        for step in progress:
            stats = self.training_step(step)
            elapsed = self.training_wall_seconds + time.perf_counter() - wall_start
            self._current_elapsed = elapsed
            if step % self.args.i_print == 0 or step == self.start_step + 1:
                record = {"step": step, **stats, "elapsed_seconds": elapsed}
                self._append_metrics(record); progress.set_postfix(
                    loss=f"{stats['loss']:.4g}", psnr=f"{stats['psnr']:.2f}")
                print(json.dumps(record))
            if self.args.i_eval > 0 and step % self.args.i_eval == 0:
                started = time.perf_counter(); val_psnr = self._validation_psnr()
                self.evaluation_wall_seconds += time.perf_counter() - started
                if val_psnr is not None:
                    optimization_elapsed = elapsed - self.evaluation_wall_seconds
                    self._append_metrics({"step": step, "val_psnr": val_psnr,
                        "elapsed_seconds": elapsed, "optimization_seconds": optimization_elapsed})
                    if self.time_to_target_psnr is None and val_psnr >= self.args.target_psnr:
                        self.time_to_target_psnr = optimization_elapsed
            if step % self.args.i_weights == 0 or step == self.args.N_iters:
                self.save_checkpoint(step)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.training_wall_seconds += time.perf_counter() - wall_start
        peak = (torch.cuda.max_memory_allocated(self.device) / 2**20
                if self.device.type == "cuda" else 0.0)
        summary = {"version": CHECKPOINT_VERSION, "seed": self.args.seed,
                   "total_training_seconds": self.training_wall_seconds,
                   "optimization_seconds_excluding_validation":
                       self.training_wall_seconds - self.evaluation_wall_seconds,
                   "validation_seconds": self.evaluation_wall_seconds,
                   "time_to_target_psnr_seconds": self.time_to_target_psnr,
                   "target_psnr": self.args.target_psnr, "peak_gpu_memory_mib": peak}
        with open(os.path.join(self.experiment_dir, "training_summary.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def render(self):
        if self.start_step == 0:
            raise ValueError("render_only requires --ft_path or checkpoints/latest.pt")
        self.field.eval()
        test_ids = None
        if self.args.render_poses:
            poses = torch.from_numpy(np.load(self.args.render_poses).astype(np.float32)).to(self.device)[:, :3, :4]
            split = "custom"
        elif self.args.render_split == "test":
            test_ids = torch.as_tensor(self.scene.test_indices, device=self.device)
            poses, split = self.scene.poses[test_ids], "test"
            render_intrinsics, render_radial = self.scene.intrinsics[test_ids], self.scene.radial_distortion[test_ids]
        else:
            poses, split = self.scene.render_poses, "path"
        train_ids = torch.as_tensor(self.scene.train_indices, device=self.device)
        if split != "test":
            intrinsic = self.scene.intrinsics[train_ids].median(dim=0).values
            render_intrinsics = intrinsic[None].expand(len(poses), -1, -1)
            render_radial = self.scene.radial_distortion[train_ids].median().expand(len(poses))
        output_dir = os.path.join(self.experiment_dir, f"render_{split}_{self.start_step:06d}")
        os.makedirs(output_dir, exist_ok=True)
        frames, compute = [], 0.0
        wall = time.perf_counter()
        with torch.no_grad():
            for index, pose in enumerate(poses):
                started = time.perf_counter()
                frame = self._render_pose(pose, render_intrinsics[index], render_radial[index])
                if self.device.type == "cuda": torch.cuda.synchronize(self.device)
                compute += time.perf_counter() - started; frames.append(frame)
                imageio.imwrite(os.path.join(output_dir, f"rgb_{index:03d}.png"),
                                (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8))
                for key in ("depth", "uncertainty", "acc"):
                    np.save(os.path.join(output_dir, f"{key}_{index:03d}.npy"), frame[key].astype(np.float32))
                if test_ids is not None:
                    gt = self.scene.images[test_ids[index], ..., :3].detach().cpu().numpy()
                    imageio.imwrite(os.path.join(output_dir, f"gt_rgb_{index:03d}.png"),
                                    (np.clip(gt, 0, 1) * 255).astype(np.uint8))
        all_depth = np.concatenate([f["depth"].reshape(-1) for f in frames]) if frames else np.array([])
        valid = all_depth[np.isfinite(all_depth) & (all_depth > 0)]
        low, high = np.percentile(valid, [2, 98]) if len(valid) else (0.0, 1.0)
        videos = {key: [] for key in ("rgb", "depth", "uncertainty")}
        for index, frame in enumerate(frames):
            rgb = (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8)
            depth = (np.clip((frame["depth"] - low) / max(high - low, 1e-8), 0, 1) * 255).astype(np.uint8)
            unc = (np.clip(frame["uncertainty"], 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_dir, f"depth_{index:03d}.png"), depth)
            imageio.imwrite(os.path.join(output_dir, f"uncertainty_{index:03d}.png"), unc)
            videos["rgb"].append(rgb); videos["depth"].append(depth); videos["uncertainty"].append(unc)
        for key, values in videos.items():
            if values: imageio.mimwrite(os.path.join(output_dir, f"{key}.mp4"), values, fps=30, quality=8)
        elapsed = time.perf_counter() - wall
        height, width, _ = self.scene.hwf
        efficiency = {"render_wall_seconds_with_io": elapsed, "render_compute_seconds": compute,
                      "frames": len(frames), "fps": len(frames) / max(compute, 1e-8),
                      "rays_per_second": len(frames) * height * width / max(compute, 1e-8)}
        with open(os.path.join(output_dir, "render_efficiency.json"), "w", encoding="utf-8") as handle:
            json.dump(efficiency, handle, indent=2)
        print(f"Rendered {len(frames)} V7 views to {output_dir}; FPS={efficiency['fps']:.3f}")
        return output_dir


__all__ = ["CHECKPOINT_VERSION", "TrainerV7", "set_seed"]
