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

from v5.robust import anneal, warmup_cosine_lr

from .data import RaySampler, load_scene, pixels_to_rays
from .losses import DepthScaleAligner, geometry_loss, heteroscedastic_color_nll
from .model import RadianceFieldV6
from .rendering import distortion_loss, render_rays
from .teacher import DecoupledTeacher, neighborhood_uncertainty_regularization


CHECKPOINT_VERSION = "uncertain-nerf-v6-1"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slice_result(result, indices):
    sliced = {}
    batch = result["rgb"].shape[0]
    for key, value in result.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == batch:
            sliced[key] = value[indices]
        else:
            sliced[key] = value
    return sliced


def _safe_logit(value):
    value = value.clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.log(value) - torch.log1p(-value)


class TrainerV6:
    def __init__(self, args):
        self.args = args
        if args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if args.hash_backend == "tcnn" and self.device.type != "cuda":
            raise ValueError("hash_backend=tcnn requires a CUDA device")
        set_seed(args.seed)
        self.scene = load_scene(
            args.datadir,
            factor=args.factor,
            holdout=args.llffhold,
            n_source_views=args.teacher_views,
            device=self.device,
            aabb_scale=args.aabb_scale,
            val_every=args.val_every,
        )
        self.field = RadianceFieldV6(
            aabb=self.scene.aabb,
            hidden_dim=args.netwidth,
            hidden_layers=args.netdepth,
            hash_levels=args.hash_levels,
            hash_min_resolution=args.hash_min_resolution,
            hash_max_resolution=args.hash_max_resolution,
            hash_features=args.hash_features,
            hash_log2_size=args.hash_log2_size,
            direction_frequencies=args.multires_views,
            appearance_dim=args.appearance_dim,
            density_bias=args.density_bias,
            hash_backend=args.hash_backend,
        ).to(self.device)
        self.appearance = nn.Embedding(len(self.scene.images), args.appearance_dim).to(self.device)
        nn.init.zeros_(self.appearance.weight)
        self.background_logit = nn.Parameter(
            _safe_logit(self.scene.training_color_mean.detach().clone())
        )
        self.teacher = DecoupledTeacher(args.teacher_photometric_weight)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.field.parameters(), "lr": args.lrate},
                {"params": self.appearance.parameters(), "lr": args.lrate_appearance},
                {"params": [self.background_logit], "lr": args.lrate_appearance},
            ],
            betas=(0.9, 0.99),
            eps=1.0e-8,
        )
        amp_enabled = bool(args.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.depth_aligner = DepthScaleAligner(
            warmup=args.depth_scale_warmup,
            interval=args.depth_scale_interval,
            ema=args.depth_scale_ema,
        )
        self.sampler = RaySampler(
            self.scene, batch_size=args.N_rand, prior_ratio=args.prior_ray_ratio
        )
        self.experiment_dir = os.path.join(args.basedir, args.expname)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.metrics_path = os.path.join(self.experiment_dir, "train_metrics.jsonl")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_step = 0
        self.training_wall_seconds = 0.0
        self.evaluation_wall_seconds = 0.0
        self._current_elapsed = 0.0
        self.time_to_target_psnr = None
        self._write_run_config()
        checkpoint = self._resolve_checkpoint()
        if checkpoint:
            self.load_checkpoint(checkpoint, load_optimizer=not args.render_only)

    def _write_run_config(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        with open(os.path.join(self.experiment_dir, "args.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(self.args), handle, ensure_ascii=False, indent=2)
        if self.args.config:
            shutil.copyfile(self.args.config, os.path.join(self.experiment_dir, "config.txt"))

    def _resolve_checkpoint(self):
        if self.args.no_reload:
            return None
        if self.args.ft_path and self.args.ft_path != "None":
            return self.args.ft_path
        latest = os.path.join(self.checkpoint_dir, "latest.pt")
        return latest if os.path.exists(latest) else None

    def _background(self, count=None):
        color = torch.sigmoid(self.background_logit)[None]
        return color if count is None else color.expand(count, -1)

    def _mean_appearance(self):
        train_ids = torch.as_tensor(self.scene.train_indices, device=self.device)
        return self.appearance(train_ids).mean(dim=0, keepdim=True)

    def _render_batch(self, batch, perturb):
        appearance = self.appearance(batch["image_ids"])
        return render_rays(
            self.field,
            batch["rays_o"],
            batch["rays_d"],
            self.scene.near,
            self.scene.far,
            self._background(len(batch["rays_o"])),
            appearance=appearance,
            n_samples=self.args.N_samples,
            n_importance=self.args.N_importance,
            perturb=perturb,
            chunk=self.args.netchunk,
            sampling_space=self.args.sampling_space,
        )

    def _teacher_indices(self, rendered):
        if self.args.teacher_ray_ratio <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        valid = torch.where(rendered["valid"] & (rendered["fine"]["acc"] > 1.0e-4))[0]
        if valid.numel() == 0:
            return valid
        count = max(1, int(round(self.args.teacher_ray_ratio * self.args.N_rand)))
        return valid[torch.randperm(valid.numel(), device=self.device)[:count]]

    def training_step(self, step):
        args = self.args
        batch = self.sampler.sample()
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(args.amp and self.device.type == "cuda")
        with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
            rendered = self._render_batch(batch, perturb=True)
            coarse, fine = rendered["coarse"], rendered["fine"]
            fine_mse = F.mse_loss(fine["rgb"], batch["target"])
            coarse_mse = F.mse_loss(coarse["rgb"], batch["target"])
            color_loss = fine_mse + args.coarse_loss_weight * coarse_mse
            color_nll = heteroscedastic_color_nll(
                fine["rgb"],
                batch["target"],
                fine["uncertainty"],
                args.variance_min,
                args.variance_max,
            )
            distortion = distortion_loss(fine)
            appearance_reg = self.appearance(batch["image_ids"]).square().mean()

            teacher_loss = fine_mse.new_zeros(())
            spatial_reg = fine_mse.new_zeros(())
            teacher_count = 0
            if args.teacher_interval > 0 and step % args.teacher_interval == 0:
                teacher_indices = self._teacher_indices(rendered)
                teacher_count = int(teacher_indices.numel())
                if teacher_count:
                    fine_teacher = _slice_result(fine, teacher_indices)
                    teacher_target, _ = self.teacher(
                        fine_teacher,
                        batch["target"][teacher_indices],
                        batch["image_ids"][teacher_indices],
                        self.scene,
                        anneal(
                            args.tukey_start,
                            args.tukey_end,
                            step,
                            args.robust_anneal_steps,
                        ),
                    )
                    teacher_loss = F.smooth_l1_loss(
                        fine["uncertainty"][teacher_indices], teacher_target
                    )
                    spatial_reg = neighborhood_uncertainty_regularization(
                        self.field,
                        fine_teacher["surface_points"],
                        epsilon=args.spatial_epsilon,
                    )

        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            geometry, geo_stats = geometry_loss(
                fine["depth"].float(),
                fine["uncertainty"].float(),
                batch["prior_depth"].float(),
                batch["prior_mask"] & rendered["valid"],
                self.depth_aligner.scale,
                args.geo_gamma,
                anneal(args.huber_start, args.huber_end, step, args.robust_anneal_steps),
            )
        prior_mask = batch["prior_mask"] & rendered["valid"]
        if step >= args.depth_scale_warmup:
            self.depth_aligner.observe(
                fine["depth"][prior_mask], batch["prior_depth"][prior_mask]
            )
        self.depth_aligner.maybe_update(step)
        total = color_loss
        total = total + args.lambda_nll * color_nll
        total = total + args.lambda_geo * geometry
        total = total + args.lambda_teacher * teacher_loss
        total = total + args.lambda_spatial * spatial_reg
        total = total + args.lambda_distortion * distortion
        total = total + args.lambda_appearance * appearance_reg
        self.scaler.scale(total).backward()
        if args.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.field.parameters()) + list(self.appearance.parameters()),
                args.grad_clip,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        base_lr = warmup_cosine_lr(
            step + 1, args.N_iters, args.lr_warmup, args.lrate, args.lrate_min
        )
        ratio = args.lrate_appearance / max(args.lrate, 1.0e-12)
        self.optimizer.param_groups[0]["lr"] = base_lr
        for group in self.optimizer.param_groups[1:]:
            group["lr"] = base_lr * ratio
        psnr = -10.0 * torch.log10(fine_mse.clamp_min(1.0e-10))
        return {
            "loss": float(total.detach()),
            "color": float(color_loss.detach()),
            "nll": float(color_nll.detach()),
            "geo": float(geometry.detach()),
            "teacher": float(teacher_loss.detach()),
            "spatial": float(spatial_reg.detach()),
            "distortion": float(distortion.detach()),
            "psnr": float(psnr.detach()),
            "lr": base_lr,
            "scale": self.depth_aligner.scale,
            "teacher_count": teacher_count,
            "geo_count": geo_stats["count"],
        }

    def _append_metrics(self, record):
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _validation_psnr(self):
        if len(self.scene.val_indices) == 0:
            return None
        was_training = self.field.training
        self.field.eval()
        indices = list(map(int, self.scene.val_indices))
        if self.args.eval_max_views > 0:
            indices = indices[: self.args.eval_max_views]
        psnr_values = []
        with torch.no_grad():
            for index in indices:
                frame = self._render_pose(
                    self.scene.poses[index],
                    self.scene.intrinsics[index],
                    self.scene.radial_distortion[index],
                )
                target = self.scene.images[index, ..., :3].detach().cpu().numpy()
                mse = float(np.square(frame["rgb"] - target).mean())
                psnr_values.append(-10.0 * math.log10(max(mse, 1.0e-10)))
        if was_training:
            self.field.train()
        return float(np.mean(psnr_values))

    def train(self):
        if self.args.render_only:
            return self.render()
        self.field.train()
        self.appearance.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        progress = trange(self.start_step + 1, self.args.N_iters + 1, desc="training-v6")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        for step in progress:
            stats = self.training_step(step)
            elapsed = self.training_wall_seconds + time.perf_counter() - wall_start
            self._current_elapsed = elapsed
            if step % self.args.i_print == 0 or step == self.start_step + 1:
                record = {"step": step, **stats, "elapsed_seconds": elapsed}
                self._append_metrics(record)
                progress.set_postfix(
                    loss=f"{stats['loss']:.4g}", psnr=f"{stats['psnr']:.2f}", lr=f"{stats['lr']:.2e}"
                )
                print(json.dumps(record))
            if self.args.i_eval > 0 and step % self.args.i_eval == 0:
                evaluation_before = self.evaluation_wall_seconds
                evaluation_start = time.perf_counter()
                val_psnr = self._validation_psnr()
                self.evaluation_wall_seconds += time.perf_counter() - evaluation_start
                if val_psnr is not None:
                    optimization_elapsed = elapsed - evaluation_before
                    record = {
                        "step": step,
                        "val_psnr": val_psnr,
                        "elapsed_seconds": elapsed,
                        "optimization_seconds": optimization_elapsed,
                    }
                    self._append_metrics(record)
                    if self.time_to_target_psnr is None and val_psnr >= self.args.target_psnr:
                        self.time_to_target_psnr = optimization_elapsed
            if step % self.args.i_weights == 0 or step == self.args.N_iters:
                self.save_checkpoint(step)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.training_wall_seconds += time.perf_counter() - wall_start
        peak = (
            torch.cuda.max_memory_allocated(self.device) / 2**20
            if self.device.type == "cuda"
            else 0.0
        )
        summary = {
            "version": CHECKPOINT_VERSION,
            "seed": self.args.seed,
            "total_training_seconds": self.training_wall_seconds,
            "optimization_seconds_excluding_validation": self.training_wall_seconds - self.evaluation_wall_seconds,
            "validation_seconds": self.evaluation_wall_seconds,
            "time_to_target_psnr_seconds": self.time_to_target_psnr,
            "target_psnr": self.args.target_psnr,
            "peak_gpu_memory_mib": peak,
        }
        with open(os.path.join(self.experiment_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def checkpoint_state(self, step):
        return {
            "version": CHECKPOINT_VERSION,
            "global_step": int(step),
            "dataset_fingerprint": self.scene.fingerprint,
            "field": self.field.state_dict(),
            "appearance": self.appearance.state_dict(),
            "background_logit": self.background_logit.detach(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "depth_aligner": self.depth_aligner.state_dict(),
            "training_wall_seconds": max(self.training_wall_seconds, self._current_elapsed),
            "evaluation_wall_seconds": self.evaluation_wall_seconds,
            "time_to_target_psnr": self.time_to_target_psnr,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "aabb": self.scene.aabb.detach().cpu(),
            "split": {
                "train": self.scene.train_indices.tolist(),
                "val": self.scene.val_indices.tolist(),
                "test": self.scene.test_indices.tolist(),
            },
        }

    def save_checkpoint(self, step):
        path = os.path.join(self.checkpoint_dir, f"step_{step:06d}.pt")
        torch.save(self.checkpoint_state(step), path)
        shutil.copyfile(path, os.path.join(self.checkpoint_dir, "latest.pt"))
        print(f"Saved V6 checkpoint: {path}")

    def load_checkpoint(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"Expected {CHECKPOINT_VERSION}; V5 and V6 checkpoints are incompatible")
        if state.get("dataset_fingerprint") != self.scene.fingerprint:
            raise ValueError("Checkpoint dataset fingerprint does not match the current scene")
        expected_split = {
            "train": self.scene.train_indices.tolist(),
            "val": self.scene.val_indices.tolist(),
            "test": self.scene.test_indices.tolist(),
        }
        if state.get("split") != expected_split:
            raise ValueError("Checkpoint train/validation/test split does not match the current V6 config")
        self.field.load_state_dict(state["field"])
        self.appearance.load_state_dict(state["appearance"])
        self.background_logit.data.copy_(state["background_logit"].to(self.device))
        self.depth_aligner.load_state_dict(state.get("depth_aligner", {}))
        self.training_wall_seconds = float(state.get("training_wall_seconds", 0.0))
        self.evaluation_wall_seconds = float(state.get("evaluation_wall_seconds", 0.0))
        self._current_elapsed = self.training_wall_seconds
        self.time_to_target_psnr = state.get("time_to_target_psnr")
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state.get("scaler", {}))
            rng = state.get("rng", {})
            if rng:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.random.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
        self.start_step = int(state["global_step"])
        print(f"Loaded V6 checkpoint {path} at step {self.start_step}")

    def _render_pose(self, pose, intrinsic, radial_distortion=0.0):
        height, width, _ = self.scene.hwf
        ys, xs = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing="ij",
        )
        count = height * width
        image_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        rays_o, rays_d = pixels_to_rays(
            pose[None],
            intrinsic[None],
            image_ids,
            ys.reshape(-1),
            xs.reshape(-1),
            radial_distortion=radial_distortion,
        )
        background = self._background(count)
        appearance = self._mean_appearance().expand(count, -1)
        outputs = {"rgb": [], "depth": [], "uncertainty": [], "acc": []}
        for start in range(0, count, self.args.chunk):
            end = min(start + self.args.chunk, count)
            result = render_rays(
                self.field,
                rays_o[start:end],
                rays_d[start:end],
                self.scene.near,
                self.scene.far,
                background[start:end],
                appearance=appearance[start:end],
                n_samples=self.args.N_samples,
                n_importance=self.args.N_importance,
                perturb=False,
                chunk=self.args.netchunk,
                sampling_space=self.args.sampling_space,
            )["fine"]
            for key in outputs:
                outputs[key].append(result[key].detach().cpu())
        return {
            key: torch.cat(value).reshape(height, width, -1 if key == "rgb" else 1).squeeze(-1).numpy()
            for key, value in outputs.items()
        }

    def render(self):
        if self.start_step == 0:
            raise ValueError("render_only requires --ft_path or checkpoints/latest.pt")
        self.field.eval()
        self.appearance.eval()
        test_ids = None
        if self.args.render_poses:
            poses = torch.from_numpy(np.load(self.args.render_poses).astype(np.float32)).to(self.device)
            poses = poses[:, :3, :4]
            split = "custom"
        elif self.args.render_split == "test":
            test_ids = torch.as_tensor(self.scene.test_indices, device=self.device)
            poses = self.scene.poses[test_ids]
            render_intrinsics = self.scene.intrinsics[test_ids]
            render_radial = self.scene.radial_distortion[test_ids]
            split = "test"
        else:
            poses = self.scene.render_poses
            split = "path"
        train_ids = torch.as_tensor(self.scene.train_indices, device=self.device)
        if split != "test":
            intrinsic = self.scene.intrinsics[train_ids].median(dim=0).values
            radial = self.scene.radial_distortion[train_ids].median()
            render_intrinsics = intrinsic[None].expand(len(poses), -1, -1)
            render_radial = radial.expand(len(poses))
        output_dir = os.path.join(self.experiment_dir, f"render_{split}_{self.start_step:06d}")
        os.makedirs(output_dir, exist_ok=True)
        if len(poses) == 0:
            raise ValueError(f"The requested {split} split contains no camera poses")
        frames = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        render_start = time.perf_counter()
        render_compute_seconds = 0.0
        with torch.no_grad():
            for index, pose in enumerate(poses):
                frame_start = time.perf_counter()
                frame = self._render_pose(pose, render_intrinsics[index], render_radial[index])
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                render_compute_seconds += time.perf_counter() - frame_start
                frames.append(frame)
                imageio.imwrite(
                    os.path.join(output_dir, f"rgb_{index:03d}.png"),
                    (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8),
                )
                for key in ("depth", "uncertainty", "acc"):
                    np.save(os.path.join(output_dir, f"{key}_{index:03d}.npy"), frame[key].astype(np.float32))
                imageio.imwrite(
                    os.path.join(output_dir, f"uncertainty_{index:03d}.png"),
                    (np.clip(frame["uncertainty"], 0, 1) * 255).astype(np.uint8),
                )
                imageio.imwrite(
                    os.path.join(output_dir, f"acc_{index:03d}.png"),
                    (np.clip(frame["acc"], 0, 1) * 255).astype(np.uint8),
                )
                if test_ids is not None:
                    gt = self.scene.images[test_ids[index], ..., :3].detach().cpu().numpy()
                    imageio.imwrite(
                        os.path.join(output_dir, f"gt_rgb_{index:03d}.png"),
                        (np.clip(gt, 0, 1) * 255).astype(np.uint8),
                    )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        all_depth = np.concatenate([frame["depth"].reshape(-1) for frame in frames])
        valid_depth = all_depth[np.isfinite(all_depth) & (all_depth > 0)]
        low, high = np.percentile(valid_depth, [2, 98]) if len(valid_depth) else (0.0, 1.0)
        rgb_video, depth_video, uncertainty_video = [], [], []
        for index, frame in enumerate(frames):
            rgb8 = (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8)
            depth8 = (
                np.clip((frame["depth"] - low) / max(high - low, 1.0e-8), 0, 1) * 255
            ).astype(np.uint8)
            uncertainty8 = (np.clip(frame["uncertainty"], 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_dir, f"depth_{index:03d}.png"), depth8)
            rgb_video.append(rgb8)
            depth_video.append(depth8)
            uncertainty_video.append(uncertainty8)
        if frames:
            imageio.mimwrite(os.path.join(output_dir, "rgb.mp4"), rgb_video, fps=30, quality=8)
            imageio.mimwrite(os.path.join(output_dir, "depth.mp4"), depth_video, fps=30, quality=8)
            imageio.mimwrite(
                os.path.join(output_dir, "uncertainty.mp4"), uncertainty_video, fps=30, quality=8
            )
        elapsed = time.perf_counter() - render_start
        height, width, _ = self.scene.hwf
        efficiency = {
            "render_wall_seconds_with_io": elapsed,
            "render_compute_seconds": render_compute_seconds,
            "frames": len(frames),
            "fps": len(frames) / max(render_compute_seconds, 1.0e-8),
            "rays_per_second": len(frames) * height * width / max(render_compute_seconds, 1.0e-8),
        }
        with open(os.path.join(output_dir, "render_efficiency.json"), "w", encoding="utf-8") as handle:
            json.dump(efficiency, handle, indent=2)
        print(f"Rendered {len(frames)} V6 views to {output_dir}; FPS={efficiency['fps']:.3f}")
        return output_dir


__all__ = ["CHECKPOINT_VERSION", "TrainerV6", "set_seed"]
