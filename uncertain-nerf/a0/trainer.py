"""Training, checkpointing and rendering for the original NeRF A0 baseline."""

import json
import math
import os
import random
import shutil
import time

import imageio.v2 as imageio
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange

from .data import UniformRaySampler, load_a0_scene
from .model import OriginalNeRF, global_gradient_norm
from .rendering import get_rays, render_rays


CHECKPOINT_VERSION = "puri-nerf-a0-original-1"


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def original_learning_rate(step, initial, decay_thousands):
    return float(initial) * 0.1 ** (
        float(step) / (float(decay_thousands) * 1000.0))


class A0Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto" else args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        set_seed(args.seed)
        self.scene = load_a0_scene(
            args.datadir, factor=args.factor, holdout=args.llffhold,
            val_every=args.val_every, device=self.device)
        self.model = OriginalNeRF(
            depth=args.netdepth, width=args.netwidth,
            fine_depth=args.netdepth_fine, fine_width=args.netwidth_fine,
            point_frequencies=args.multires,
            view_frequencies=args.multires_views,
            use_viewdirs=args.use_viewdirs,
            output_channels=5 if args.N_importance > 0 else 4).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lrate, betas=(0.9, 0.999))
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(args.amp and self.device.type == "cuda"))
        self.sampler = UniformRaySampler(self.scene, args.N_rand)
        self.experiment_dir = os.path.join(args.basedir, args.expname)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.metrics_path = os.path.join(
            self.experiment_dir, "train_metrics.jsonl")
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
        values = vars(self.args).copy()
        filename = "render_args.json" if self.args.render_only else "args.json"
        with open(os.path.join(self.experiment_dir, filename), "w",
                  encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, ensure_ascii=False)
        if not self.args.render_only and self.args.config:
            shutil.copyfile(
                self.args.config, os.path.join(self.experiment_dir, "config.txt"))

    def _resolve_checkpoint(self):
        if self.args.ft_path:
            return self.args.ft_path
        latest = os.path.join(self.checkpoint_dir, "latest.pt")
        if not self.args.no_reload and os.path.isfile(latest):
            return latest
        return None

    def _render_batch(self, batch, perturb):
        return render_rays(
            self.model, batch["rays_o"], batch["rays_d"],
            self.scene.height, self.scene.width, self.scene.focal,
            n_samples=self.args.N_samples,
            n_importance=self.args.N_importance,
            perturb=perturb,
            raw_noise_std=(self.args.raw_noise_std if perturb else 0.0),
            white_background=self.args.white_bkgd,
            netchunk=self.args.netchunk,
            use_ndc=not self.args.no_ndc)

    def training_step(self, step):
        batch = self.sampler.sample()
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(self.args.amp and self.device.type == "cuda")
        with torch.amp.autocast(
                device_type=self.device.type, enabled=amp_enabled):
            rendered = self._render_batch(batch, perturb=True)
            fine_mse = F.mse_loss(rendered["fine"]["rgb"], batch["target"])
            coarse_mse = F.mse_loss(
                rendered["coarse"]["rgb"], batch["target"])
            loss = fine_mse + self.args.coarse_loss_weight * coarse_mse
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        density_grad_norm = global_gradient_norm(
            self.model.density_parameters())
        color_grad_norm = global_gradient_norm(self.model.color_parameters())
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        learning_rate = original_learning_rate(
            step, self.args.lrate, self.args.lrate_decay)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        fine = rendered["fine"]
        with torch.no_grad():
            psnr = -10.0 * torch.log10(fine_mse.clamp_min(1.0e-10))
            valid_depth = fine["depth"][fine["acc"] > 1.0e-6]
            depth_mean = (valid_depth.mean() if valid_depth.numel()
                          else fine["depth"].new_zeros(()))
        return {
            "mode": "a0_original", "loss": float(loss.detach()),
            "loss_rgb_fine": float(fine_mse.detach()),
            "loss_rgb_coarse": float(coarse_mse.detach()),
            "psnr": float(psnr), "lr": learning_rate,
            "opacity_mean": float(fine["acc"].detach().mean()),
            "depth_mean": float(depth_mean),
            "mean_weight_sum": float(
                fine["weights"].detach().sum(dim=-1).mean()),
            "mean_terminal_transmittance": float(
                fine["terminal_transmittance"].detach().mean()),
            "gradient_norm_density": density_grad_norm,
            "gradient_norm_color": color_grad_norm,
            "gradient_norm_geometry_loss": 0.0,
            "gradient_norm_uncertainty": 0.0,
        }

    def _append_metrics(self, record):
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _render_pose(self, pose):
        rays_o, rays_d = get_rays(
            self.scene.height, self.scene.width, self.scene.focal, pose)
        flat_o, flat_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        outputs = {key: [] for key in ("rgb", "depth", "acc")}
        for start in range(0, len(flat_o), self.args.chunk):
            end = min(start + self.args.chunk, len(flat_o))
            result = render_rays(
                self.model, flat_o[start:end], flat_d[start:end],
                self.scene.height, self.scene.width, self.scene.focal,
                n_samples=self.args.N_samples,
                n_importance=self.args.N_importance,
                perturb=False, raw_noise_std=0.0,
                white_background=self.args.white_bkgd,
                netchunk=self.args.netchunk,
                use_ndc=not self.args.no_ndc)["fine"]
            for key in outputs:
                outputs[key].append(result[key].detach().cpu())
        height, width = self.scene.height, self.scene.width
        return {
            "rgb": torch.cat(outputs["rgb"]).reshape(
                height, width, 3).numpy(),
            "depth": torch.cat(outputs["depth"]).reshape(
                height, width).numpy(),
            "acc": torch.cat(outputs["acc"]).reshape(
                height, width).numpy(),
            "uncertainty": np.zeros((height, width), dtype=np.float32),
        }

    def _validation_psnr(self):
        if not len(self.scene.val_indices):
            return None
        was_training = self.model.training
        self.model.eval()
        indices = list(map(int, self.scene.val_indices))
        if self.args.eval_max_views > 0:
            indices = indices[:self.args.eval_max_views]
        values = []
        with torch.no_grad():
            for index in indices:
                frame = self._render_pose(self.scene.poses[index])
                target = self.scene.images[index].detach().cpu().numpy()
                mse = float(np.square(frame["rgb"] - target).mean())
                values.append(-10.0 * math.log10(max(mse, 1.0e-10)))
        if was_training:
            self.model.train()
        return float(np.mean(values))

    def checkpoint_state(self, step):
        return {
            "version": CHECKPOINT_VERSION,
            "global_step": int(step),
            "dataset_fingerprint": self.scene.fingerprint,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "training_wall_seconds": max(
                self.training_wall_seconds, self._current_elapsed),
            "evaluation_wall_seconds": self.evaluation_wall_seconds,
            "time_to_target_psnr": self.time_to_target_psnr,
            "split": {
                "train": self.scene.train_indices.tolist(),
                "val": self.scene.val_indices.tolist(),
                "test": self.scene.test_indices.tolist(),
            },
            "rng": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
        }

    def save_checkpoint(self, step):
        path = os.path.join(
            self.checkpoint_dir, f"step_{int(step):06d}.pt")
        torch.save(self.checkpoint_state(step), path)
        shutil.copyfile(path, os.path.join(self.checkpoint_dir, "latest.pt"))
        print(f"Saved A0 checkpoint: {path}")
        return path

    def load_checkpoint(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"Expected {CHECKPOINT_VERSION}; V5--V8 checkpoints are incompatible")
        if state.get("dataset_fingerprint") != self.scene.fingerprint:
            raise ValueError("A0 checkpoint dataset fingerprint mismatch")
        expected_split = {
            "train": self.scene.train_indices.tolist(),
            "val": self.scene.val_indices.tolist(),
            "test": self.scene.test_indices.tolist(),
        }
        if state.get("split") != expected_split:
            raise ValueError("A0 checkpoint split mismatch")
        self.model.load_state_dict(state["model"])
        self.training_wall_seconds = float(
            state.get("training_wall_seconds", 0.0))
        self.evaluation_wall_seconds = float(
            state.get("evaluation_wall_seconds", 0.0))
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
        print(f"Loaded A0 checkpoint {path} at step {self.start_step}")

    def train(self):
        if self.args.render_only:
            return self.render()
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        progress = trange(
            self.start_step + 1, self.args.N_iters + 1,
            desc="training-a0-original")
        wall_start = time.perf_counter()
        for step in progress:
            stats = self.training_step(step)
            elapsed = (
                self.training_wall_seconds + time.perf_counter() - wall_start)
            self._current_elapsed = elapsed
            if step % self.args.i_print == 0 or step == self.start_step + 1:
                record = {"step": step, **stats, "elapsed_seconds": elapsed}
                self._append_metrics(record)
                progress.set_postfix(
                    loss=f"{stats['loss']:.4g}", psnr=f"{stats['psnr']:.2f}")
                print(json.dumps(record))
            if self.args.i_eval > 0 and step % self.args.i_eval == 0:
                started = time.perf_counter()
                val_psnr = self._validation_psnr()
                duration = time.perf_counter() - started
                self.evaluation_wall_seconds += duration
                elapsed += duration
                self._current_elapsed = elapsed
                if val_psnr is not None:
                    optimization_elapsed = (
                        elapsed - self.evaluation_wall_seconds)
                    self._append_metrics({
                        "step": step, "val_psnr": val_psnr,
                        "elapsed_seconds": elapsed,
                        "optimization_seconds": optimization_elapsed})
                    if (self.time_to_target_psnr is None
                            and val_psnr >= self.args.target_psnr):
                        self.time_to_target_psnr = optimization_elapsed
            if step % self.args.i_weights == 0 or step == self.args.N_iters:
                self.save_checkpoint(step)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.training_wall_seconds += time.perf_counter() - wall_start
        peak = (torch.cuda.max_memory_allocated(self.device) / 2**20
                if self.device.type == "cuda" else 0.0)
        summary = {
            "version": CHECKPOINT_VERSION, "mode": "a0_original",
            "seed": self.args.seed,
            "total_training_seconds": self.training_wall_seconds,
            "optimization_seconds_excluding_validation": (
                self.training_wall_seconds - self.evaluation_wall_seconds),
            "validation_seconds": self.evaluation_wall_seconds,
            "time_to_target_psnr_seconds": self.time_to_target_psnr,
            "target_psnr": self.args.target_psnr,
            "peak_gpu_memory_mib": peak,
            "parameter_count": self.model.parameter_count,
        }
        with open(os.path.join(
                self.experiment_dir, "training_summary.json"), "w",
                encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def render(self):
        if self.start_step == 0:
            raise ValueError("render_only requires an A0 checkpoint")
        self.model.eval()
        ground_truth_ids = None
        if self.args.render_split in ("train", "val", "test"):
            split = self.args.render_split
            indices = {
                "train": self.scene.train_indices,
                "val": self.scene.val_indices,
                "test": self.scene.test_indices,
            }[split]
            if not len(indices):
                raise ValueError(f"The A0 {split} split is empty")
            ground_truth_ids = torch.as_tensor(indices, device=self.device)
            poses = self.scene.poses[ground_truth_ids]
        else:
            split = "path"
            poses = self.scene.render_poses
        output_dir = os.path.join(
            self.experiment_dir, f"render_{split}_{self.start_step:06d}")
        os.makedirs(output_dir, exist_ok=True)
        frames, compute = [], 0.0
        wall = time.perf_counter()
        with torch.no_grad():
            for output_index, pose in enumerate(poses):
                started = time.perf_counter()
                frame = self._render_pose(pose)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                compute += time.perf_counter() - started
                frames.append(frame)
                imageio.imwrite(
                    os.path.join(output_dir, f"rgb_{output_index:03d}.png"),
                    (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8))
                for key in ("depth", "uncertainty", "acc"):
                    np.save(os.path.join(
                        output_dir, f"{key}_{output_index:03d}.npy"),
                        frame[key].astype(np.float32))
                if ground_truth_ids is not None:
                    target = self.scene.images[
                        ground_truth_ids[output_index]].detach().cpu().numpy()
                    imageio.imwrite(
                        os.path.join(
                            output_dir, f"gt_rgb_{output_index:03d}.png"),
                        (np.clip(target, 0, 1) * 255).astype(np.uint8))
        all_depth = np.concatenate([
            frame["depth"].reshape(-1) for frame in frames])
        valid_depth = all_depth[
            np.isfinite(all_depth) & (all_depth > 0)]
        low, high = (np.percentile(valid_depth, [2, 98])
                     if len(valid_depth) else (0.0, 1.0))
        videos = {key: [] for key in ("rgb", "depth", "uncertainty")}
        for output_index, frame in enumerate(frames):
            rgb = (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8)
            depth = (np.clip(
                (frame["depth"] - low) / max(high - low, 1.0e-8),
                0, 1) * 255).astype(np.uint8)
            uncertainty = np.zeros_like(depth, dtype=np.uint8)
            imageio.imwrite(os.path.join(
                output_dir, f"depth_{output_index:03d}.png"), depth)
            imageio.imwrite(os.path.join(
                output_dir, f"uncertainty_{output_index:03d}.png"), uncertainty)
            videos["rgb"].append(rgb)
            videos["depth"].append(depth)
            videos["uncertainty"].append(uncertainty)
        for key, values in videos.items():
            imageio.mimwrite(
                os.path.join(output_dir, f"{key}.mp4"), values,
                fps=30, quality=8)
        elapsed = time.perf_counter() - wall
        efficiency = {
            "render_wall_seconds_with_io": elapsed,
            "render_compute_seconds": compute, "frames": len(frames),
            "fps": len(frames) / max(compute, 1.0e-8),
            "rays_per_second": (
                len(frames) * self.scene.height * self.scene.width
                / max(compute, 1.0e-8)),
        }
        with open(os.path.join(
                output_dir, "render_efficiency.json"), "w",
                encoding="utf-8") as handle:
            json.dump(efficiency, handle, indent=2)
        print(f"Rendered {len(frames)} A0 {split} views to {output_dir}")
        return output_dir


__all__ = [
    "A0Trainer", "CHECKPOINT_VERSION", "original_learning_rate", "set_seed",
]
