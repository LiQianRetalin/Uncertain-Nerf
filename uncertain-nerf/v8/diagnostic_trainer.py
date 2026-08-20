"""Trainer for the mandatory V8 baseline and V7.5 acceptance gate."""

import json
import math
import os
import random
import shutil
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from v5.losses import DepthScaleAligner
from v5.robust import anneal, warmup_cosine_lr
from v6.trainer import _safe_logit, _slice_result, set_seed
from v7.losses import termination_distribution_loss, uncertainty_color_nll
from v7.rendering import distortion_loss
from v7.teacher import VisibilityAwareTeacher
from v7.trainer import TrainerV7

from .diagnostic_data import FixedDiagnosticRaySampler, load_scene
from .diagnostic_model import DiagnosticRadianceField


CHECKPOINT_VERSION = "puri-nerf-v8-diagnostic-1"


class _DisabledAppearance(nn.Module):
    def forward(self, image_ids):
        return torch.empty(len(image_ids), 0, device=image_ids.device)


def _global_grad_norm(parameters):
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().square().sum())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def assert_uq_gradient_isolation(loss_uq, reconstruction_parameters):
    """Assert d L_UQ / d theta_recon = 0 without changing ``.grad`` buffers."""
    parameters = tuple(reconstruction_parameters)
    gradients = torch.autograd.grad(
        loss_uq, parameters, retain_graph=True, allow_unused=True)
    leaking = []
    for index, gradient in enumerate(gradients):
        if gradient is not None and torch.count_nonzero(gradient.detach()).item():
            leaking.append(index)
    if leaking:
        raise RuntimeError(
            "STOP_UQ_GRADIENT_LEAK: UQ loss reaches reconstruction parameters "
            f"at indices {leaking[:8]}")


class DiagnosticTrainer(TrainerV7):
    """Matched trainer whose only baseline/V7.5 difference is the side UQ head."""

    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto" else args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if args.hash_backend == "tcnn" and self.device.type != "cuda":
            raise ValueError("hash_backend=tcnn requires a CUDA device")
        set_seed(args.seed)
        self.scene = load_scene(
            args.datadir, factor=args.factor, holdout=args.llffhold,
            n_source_views=args.teacher_views, device=self.device,
            aabb_scale=args.aabb_scale, val_every=args.val_every)
        self.field = DiagnosticRadianceField(
            aabb=self.scene.aabb, mode=args.mode, hidden_dim=args.netwidth,
            hidden_layers=args.netdepth, hash_levels=args.hash_levels,
            hash_min_resolution=args.hash_min_resolution,
            hash_max_resolution=args.hash_max_resolution,
            hash_features=args.hash_features,
            hash_log2_size=args.hash_log2_size,
            direction_frequencies=args.multires_views,
            density_bias=args.density_bias, hash_backend=args.hash_backend,
            mip_enabled=not args.disable_mip).to(self.device)
        self.appearance = _DisabledAppearance().to(self.device)
        self.background_logit = nn.Parameter(
            _safe_logit(self.scene.training_color_mean.detach().clone()))
        self.teacher = VisibilityAwareTeacher(args.teacher_photometric_weight)
        self.occupancy_grid = None

        reconstruction_parameters = list(self.field.reconstruction_parameters())
        self.reconstruction_optimizer = torch.optim.Adam([
            {"params": reconstruction_parameters, "lr": args.lrate},
            {"params": [self.background_logit], "lr": args.lrate_background},
        ], betas=(0.9, 0.99), eps=1.0e-8)
        uq_parameters = list(self.field.uncertainty_parameters())
        self.uq_optimizer = (torch.optim.Adam(
            uq_parameters, lr=args.lrate_uq, betas=(0.9, 0.99), eps=1.0e-8)
            if uq_parameters else None)
        # Keep this alias only for inherited checkpoint/render helpers.
        self.optimizer = self.reconstruction_optimizer
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(args.amp and self.device.type == "cuda"))
        self.depth_aligner = DepthScaleAligner(
            warmup=args.depth_scale_warmup,
            interval=args.depth_scale_interval, ema=args.depth_scale_ema)
        self.sampler = FixedDiagnosticRaySampler(
            self.scene, args.N_rand, args.prior_ray_ratio,
            args.edge_ray_ratio, args.sampling_probability_floor)
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

    def _uq_loss(self, step, rendered, fine, batch):
        if self.mode != "v7_5":
            return fine["rgb"].new_zeros(()), 0
        nll = uncertainty_color_nll(
            fine["rgb"], batch["target"], fine["uncertainty"],
            self.args.variance_min, self.args.variance_max)
        teacher_loss = nll.new_zeros(())
        teacher_count = 0
        teacher_due = (
            step >= self.args.teacher_start_step
            and self.args.teacher_interval > 0
            and step % self.args.teacher_interval == 0)
        if teacher_due:
            indices = self._teacher_indices(rendered)
            teacher_count = int(indices.numel())
            if teacher_count:
                fine_teacher = _slice_result(fine, indices)
                target, _ = self.teacher(
                    fine_teacher, batch["target"][indices],
                    batch["image_ids"][indices], self.scene,
                    anneal(self.args.tukey_start, self.args.tukey_end, step,
                           self.args.robust_anneal_steps),
                    self.field, self._background(1), self.scene.near,
                    self.scene.far,
                    visibility_samples=self.args.teacher_visibility_samples,
                    visibility_relative_tolerance=
                        self.args.teacher_visibility_relative_tolerance,
                    visibility_absolute_tolerance=
                        self.args.teacher_visibility_absolute_tolerance,
                    visibility_min_acc=self.args.teacher_visibility_min_acc,
                    chunk=self.args.netchunk, occupancy_grid=None)
                teacher_loss = F.smooth_l1_loss(
                    fine["uncertainty"][indices], target)
        return (self.args.lambda_uq_nll * nll
                + self.args.lambda_uq_teacher * teacher_loss), teacher_count

    def _write_stop_report(self, filename, step, message):
        report_dir = os.path.join(self.experiment_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        path = os.path.join(report_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "# V8 automatic stop report\n\n"
                f"- mode: `{self.mode}`\n"
                f"- step: `{int(step)}`\n"
                f"- checkpoint version: `{CHECKPOINT_VERSION}`\n"
                f"- reason: `{message}`\n\n"
                "Training was stopped before another optimizer step. "
                "Do not continue to the next V8 stage.\n")
        print(f"Wrote stop report: {path}")
        return path

    def training_step(self, step):
        args = self.args
        batch = self.sampler.sample()
        self.reconstruction_optimizer.zero_grad(set_to_none=True)
        if self.uq_optimizer is not None:
            self.uq_optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(args.amp and self.device.type == "cuda")
        with torch.amp.autocast(device_type=self.device.type,
                                enabled=amp_enabled):
            rendered = self._render_batch(batch, perturb=True)
            coarse, fine = rendered["coarse"], rendered["fine"]
            fine_mse = F.mse_loss(fine["rgb"], batch["target"])
            coarse_mse = F.mse_loss(coarse["rgb"], batch["target"])
            color_loss = fine_mse + args.coarse_loss_weight * coarse_mse
            distortion = distortion_loss(fine)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            prior_mask = batch["prior_mask"] & rendered["valid"]
            geometry, geo_stats = termination_distribution_loss(
                fine["weights"].float(),
                fine["terminal_transmittance"].float(),
                fine["z_vals"].float(), batch["prior_depth"].float(),
                prior_mask, self.depth_aligner.scale,
                args.depth_distribution_relative_sigma,
                args.depth_distribution_interval_sigma)
        reconstruction_loss = (color_loss + args.lambda_geo * geometry
                               + args.lambda_distortion * distortion)
        # UQ-only work (notably teacher ray selection) must not advance the
        # stochastic stream used by future reconstruction batches.  Otherwise
        # the two modes would see different rays after teacher_start_step even
        # though UQ never enters the reconstruction graph.
        uq_rng_devices = (
            [self.device.index
             if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=uq_rng_devices):
            uq_loss, teacher_count = self._uq_loss(
                step, rendered, fine, batch)
        if self.uq_optimizer is not None and (
                step == 1 or step % args.uq_isolation_check_interval == 0):
            try:
                assert_uq_gradient_isolation(
                    uq_loss,
                    list(self.field.reconstruction_parameters())
                    + [self.background_logit])
            except RuntimeError as error:
                if "STOP_UQ_GRADIENT_LEAK" in str(error):
                    self._write_stop_report(
                        "STOP_UQ_GRADIENT_LEAK.md", step, str(error))
                raise

        geometry_grad_norm = 0.0
        if (args.lambda_geo != 0 and geometry.requires_grad
                and (step == 1 or step % args.i_print == 0)):
            geometry_gradients = torch.autograd.grad(
                args.lambda_geo * geometry,
                tuple(self.field.reconstruction_parameters()),
                retain_graph=True, allow_unused=True)
            geometry_grad_norm = math.sqrt(sum(
                float(gradient.detach().float().pow(2).sum())
                for gradient in geometry_gradients
                if gradient is not None))

        # The renderer concatenates reconstruction and uncertainty channels, so
        # their otherwise isolated branches still share a small autograd node.
        # Backpropagating the two losses separately would either traverse a
        # freed graph or require retain_graph=True on every V7.5 step.  A single
        # backward pass avoids both problems; the isolation assertion above
        # guarantees that uq_loss contributes no reconstruction gradients.
        self.scaler.scale(reconstruction_loss + uq_loss).backward()
        self.scaler.unscale_(self.reconstruction_optimizer)
        if self.uq_optimizer is not None:
            self.scaler.unscale_(self.uq_optimizer)

        density_parameters = []
        color_parameters = []
        for name, parameter in self.field.reconstruction_named_parameters():
            if name.startswith(("position_encoder.", "trunk.", "density_head.")):
                density_parameters.append(parameter)
            if name.startswith(("feature_head.", "color_head.")):
                color_parameters.append(parameter)
        density_grad_norm = _global_grad_norm(density_parameters)
        color_grad_norm = _global_grad_norm(color_parameters)
        uq_grad_norm = _global_grad_norm(self.field.uncertainty_parameters())
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.field.reconstruction_parameters())
                + [self.background_logit], args.grad_clip)
            if self.uq_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.field.uncertainty_parameters()), args.grad_clip)
        self.scaler.step(self.reconstruction_optimizer)
        if self.uq_optimizer is not None:
            self.scaler.step(self.uq_optimizer)
        self.scaler.update()

        if step >= args.depth_scale_warmup:
            self.depth_aligner.observe(
                fine["depth"][prior_mask], batch["prior_depth"][prior_mask])
        self.depth_aligner.maybe_update(step)
        base_lr = warmup_cosine_lr(
            step + 1, args.N_iters, args.lr_warmup,
            args.lrate, args.lrate_min)
        self.reconstruction_optimizer.param_groups[0]["lr"] = base_lr
        self.reconstruction_optimizer.param_groups[1]["lr"] = base_lr * (
            args.lrate_background / max(args.lrate, 1.0e-12))
        if self.uq_optimizer is not None:
            uq_lr = warmup_cosine_lr(
                step + 1, args.N_iters, args.lr_warmup,
                args.lrate_uq, args.lrate_uq_min)
            self.uq_optimizer.param_groups[0]["lr"] = uq_lr

        with torch.no_grad():
            psnr = -10.0 * torch.log10(fine_mse.clamp_min(1.0e-10))
            valid_depth = fine["depth"][fine["acc"] > 1.0e-6]
            depth_mean = (valid_depth.mean() if valid_depth.numel()
                          else fine["depth"].new_zeros(()))
            if self.mode == "v7_5":
                u_mean = fine["uncertainty"].mean()
                p_u_gt_half = (fine["uncertainty"] > 0.5).float().mean()
            else:
                u_mean = fine["uncertainty"].new_zeros(())
                p_u_gt_half = u_mean
        return {
            "mode": self.mode,
            "loss": float((reconstruction_loss + uq_loss).detach()),
            "loss_reconstruction": float(reconstruction_loss.detach()),
            "loss_rgb": float(color_loss.detach()),
            "loss_geometry": float(geometry.detach()),
            "loss_distortion": float(distortion.detach()),
            "loss_uq": float(uq_loss.detach()),
            "psnr": float(psnr), "lr": base_lr,
            "scale": self.depth_aligner.scale,
            "opacity_mean": float(fine["acc"].detach().mean()),
            "depth_mean": float(depth_mean),
            "mean_weight_sum": float(fine["weights"].detach().sum(-1).mean()),
            "mean_terminal_transmittance": float(
                fine["terminal_transmittance"].detach().mean()),
            "u_mean": float(u_mean), "m_mean": 1.0,
            "p_u_gt_0_5": float(p_u_gt_half), "p_m_lt_0_5": 0.0,
            "gradient_norm_density": density_grad_norm,
            "gradient_norm_color": color_grad_norm,
            "gradient_norm_geometry_loss": geometry_grad_norm,
            "gradient_norm_uncertainty": uq_grad_norm,
            "teacher_count": teacher_count,
            "geo_count": geo_stats["count"]}

    def checkpoint_state(self, step):
        return {
            "version": CHECKPOINT_VERSION, "mode": self.mode,
            "global_step": int(step),
            "dataset_fingerprint": self.scene.fingerprint,
            "field": self.field.state_dict(),
            "background_logit": self.background_logit.detach(),
            "reconstruction_optimizer": self.reconstruction_optimizer.state_dict(),
            "uq_optimizer": (self.uq_optimizer.state_dict()
                             if self.uq_optimizer is not None else None),
            "scaler": self.scaler.state_dict(),
            "depth_aligner": self.depth_aligner.state_dict(),
            "training_wall_seconds": max(
                self.training_wall_seconds, self._current_elapsed),
            "evaluation_wall_seconds": self.evaluation_wall_seconds,
            "time_to_target_psnr": self.time_to_target_psnr,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.random.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None)},
            "aabb": self.scene.aabb.detach().cpu(),
            "split": {"train": self.scene.train_indices.tolist(),
                      "val": self.scene.val_indices.tolist(),
                      "test": self.scene.test_indices.tolist()}}

    def save_checkpoint(self, step):
        path = os.path.join(self.checkpoint_dir, f"step_{step:06d}.pt")
        torch.save(self.checkpoint_state(step), path)
        shutil.copyfile(path, os.path.join(self.checkpoint_dir, "latest.pt"))
        print(f"Saved V8 {self.mode} checkpoint: {path}")

    def load_checkpoint(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"Expected {CHECKPOINT_VERSION}; historical checkpoints are incompatible")
        if state.get("mode") != self.mode:
            raise ValueError(
                f"Checkpoint mode {state.get('mode')} does not match {self.mode}")
        if state.get("dataset_fingerprint") != self.scene.fingerprint:
            raise ValueError("Checkpoint dataset fingerprint does not match the scene")
        expected = {"train": self.scene.train_indices.tolist(),
                    "val": self.scene.val_indices.tolist(),
                    "test": self.scene.test_indices.tolist()}
        if state.get("split") != expected:
            raise ValueError("Checkpoint split does not match the diagnostic config")
        self.field.load_state_dict(state["field"])
        self.background_logit.data.copy_(
            state["background_logit"].to(self.device))
        self.depth_aligner.load_state_dict(state.get("depth_aligner", {}))
        self.training_wall_seconds = float(
            state.get("training_wall_seconds", 0.0))
        self.evaluation_wall_seconds = float(
            state.get("evaluation_wall_seconds", 0.0))
        self._current_elapsed = self.training_wall_seconds
        self.time_to_target_psnr = state.get("time_to_target_psnr")
        if load_optimizer:
            self.reconstruction_optimizer.load_state_dict(
                state["reconstruction_optimizer"])
            if self.uq_optimizer is not None and state.get("uq_optimizer"):
                self.uq_optimizer.load_state_dict(state["uq_optimizer"])
            self.scaler.load_state_dict(state.get("scaler", {}))
            rng = state.get("rng", {})
            if rng:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.random.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
        self.start_step = int(state["global_step"])
        print(f"Loaded V8 {self.mode} checkpoint {path} at {self.start_step}")

    @staticmethod
    def _histogram_image(values, title, bins=40, value_range=(0.0, 1.0)):
        counts, edges = np.histogram(values, bins=bins, range=value_range)
        canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 20, 20), 2, cv2.LINE_AA)
        maximum = max(int(counts.max()), 1)
        for index, count in enumerate(counts):
            left = 36 + int(index * 568 / bins)
            right = 36 + int((index + 1) * 568 / bins) - 1
            top = 430 - int(350 * int(count) / maximum)
            cv2.rectangle(canvas, (left, top), (right, 430),
                          (70, 120, 220), -1)
        cv2.rectangle(canvas, (36, 80), (604, 430), (30, 30, 30), 1)
        return canvas, {"edges": edges.tolist(), "counts": counts.tolist()}

    @staticmethod
    def _scatter_image(x, y, title, x_label, y_label, limit=5000):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) > limit:
            indices = np.linspace(0, len(x) - 1, limit).astype(np.int64)
            x, y = x[indices], y[indices]
        canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, x_label, (260, 468), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, y_label, (8, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (20, 20, 20), 1, cv2.LINE_AA)
        if len(x):
            x_low, x_high = float(x.min()), float(x.max())
            y_low, y_high = float(y.min()), float(y.max())
            x_den = max(x_high - x_low, 1.0e-8)
            y_den = max(y_high - y_low, 1.0e-8)
            px = 48 + ((x - x_low) / x_den * 550).astype(np.int32)
            py = 425 - ((y - y_low) / y_den * 340).astype(np.int32)
            for xx, yy in zip(px, py):
                cv2.circle(canvas, (int(xx), int(yy)), 1, (190, 80, 50), -1)
        cv2.rectangle(canvas, (48, 85), (598, 425), (30, 30, 30), 1)
        return canvas

    def save_diagnostics(self, step):
        """Save the mandatory V7.5 distributions and residual relationships."""
        was_training = self.field.training
        self.field.eval()
        with torch.no_grad():
            batch = self.sampler.sample()
            rendered = self._render_batch(batch, perturb=False)
            fine = rendered["fine"]
            rgb_residual = (fine["rgb"] - batch["target"]).abs().mean(-1)
            u = (fine["uncertainty"] if self.mode == "v7_5"
                 else torch.zeros_like(fine["acc"]))
            m = torch.ones_like(u)
            prior_mask = batch["prior_mask"] & rendered["valid"]
            geometry_residual = (
                fine["depth"][prior_mask]
                - float(self.depth_aligner.scale) * batch["prior_depth"][prior_mask]
            ).abs()
            geometry_u = u[prior_mask]
        if was_training:
            self.field.train()
        output_dir = os.path.join(
            self.experiment_dir, f"diagnostics_{step:06d}")
        os.makedirs(output_dir, exist_ok=True)
        arrays = {"u": u.cpu().numpy(), "m": m.cpu().numpy(),
                  "opacity": fine["acc"].cpu().numpy(),
                  "rgb_residual": rgb_residual.cpu().numpy(),
                  "geometry_residual": geometry_residual.cpu().numpy(),
                  "geometry_u": geometry_u.cpu().numpy()}
        plots = {}
        for key, title in (("u", "u histogram"), ("m", "m histogram"),
                           ("opacity", "opacity histogram")):
            image, histogram = self._histogram_image(arrays[key], title)
            cv2.imwrite(os.path.join(output_dir, f"{key}_histogram.png"), image)
            plots[f"{key}_histogram"] = histogram
        cv2.imwrite(os.path.join(output_dir, "rgb_residual_vs_uncertainty.png"),
                    self._scatter_image(arrays["u"], arrays["rgb_residual"],
                                        "RGB residual vs uncertainty",
                                        "uncertainty", "RGB residual"))
        cv2.imwrite(os.path.join(
            output_dir, "geometry_residual_vs_uncertainty.png"),
            self._scatter_image(arrays["geometry_u"],
                                arrays["geometry_residual"],
                                "geometry residual vs uncertainty",
                                "uncertainty", "geometry residual"))
        summary = {
            "mode": self.mode, "step": int(step),
            "u_mean": float(arrays["u"].mean()), "m_mean": 1.0,
            "p_u_gt_0_5": float((arrays["u"] > 0.5).mean()),
            "p_m_lt_0_5": 0.0,
            "opacity_mean": float(arrays["opacity"].mean()),
            "mean_weight_sum": float(arrays["opacity"].mean()),
            "mean_terminal_transmittance": float(
                fine["terminal_transmittance"].cpu().numpy().mean()),
            "rgb_residual_mean": float(arrays["rgb_residual"].mean()),
            "geometry_residual_mean": (float(arrays["geometry_residual"].mean())
                                       if len(arrays["geometry_residual"]) else None),
            **plots}
        with open(os.path.join(output_dir, "diagnostic_summary.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return output_dir

    def train(self):
        if self.args.render_only:
            return self.render()
        self.field.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        progress = trange(
            self.start_step + 1, self.args.N_iters + 1,
            desc=f"training-v8-{self.mode}")
        wall_start = time.perf_counter()
        for step in progress:
            stats = self.training_step(step)
            elapsed = self.training_wall_seconds + time.perf_counter() - wall_start
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
                self.evaluation_wall_seconds += time.perf_counter() - started
                if val_psnr is not None:
                    optimization_elapsed = elapsed - self.evaluation_wall_seconds
                    self._append_metrics({
                        "step": step, "val_psnr": val_psnr,
                        "elapsed_seconds": elapsed,
                        "optimization_seconds": optimization_elapsed})
                    if (self.time_to_target_psnr is None
                            and val_psnr >= self.args.target_psnr):
                        self.time_to_target_psnr = optimization_elapsed
            if step % self.args.i_weights == 0 or step == self.args.N_iters:
                self.save_checkpoint(step)
            if (self.args.i_diagnostics > 0
                    and (step % self.args.i_diagnostics == 0
                         or step == self.args.N_iters)):
                self.save_diagnostics(step)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.training_wall_seconds += time.perf_counter() - wall_start
        peak = (torch.cuda.max_memory_allocated(self.device) / 2**20
                if self.device.type == "cuda" else 0.0)
        summary = {
            "version": CHECKPOINT_VERSION, "mode": self.mode,
            "seed": self.args.seed,
            "total_training_seconds": self.training_wall_seconds,
            "optimization_seconds_excluding_validation":
                self.training_wall_seconds - self.evaluation_wall_seconds,
            "validation_seconds": self.evaluation_wall_seconds,
            "time_to_target_psnr_seconds": self.time_to_target_psnr,
            "target_psnr": self.args.target_psnr,
            "peak_gpu_memory_mib": peak}
        with open(os.path.join(self.experiment_dir, "training_summary.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def render(self):
        """Render path or a complete train/val/test split for matched metrics."""
        if self.start_step == 0:
            raise ValueError(
                "render_only requires --ft_path or checkpoints/latest.pt")
        self.field.eval()
        ground_truth_ids = None
        if self.args.render_poses:
            poses = torch.from_numpy(
                np.load(self.args.render_poses).astype(np.float32)
            ).to(self.device)[:, :3, :4]
            split = "custom"
        elif self.args.render_split in ("train", "val", "test"):
            split = self.args.render_split
            scene_indices = {
                "train": self.scene.train_indices,
                "val": self.scene.val_indices,
                "test": self.scene.test_indices,
            }[split]
            if len(scene_indices) == 0:
                raise ValueError(f"The {split} split is empty")
            ground_truth_ids = torch.as_tensor(
                scene_indices, device=self.device)
            poses = self.scene.poses[ground_truth_ids]
            render_intrinsics = self.scene.intrinsics[ground_truth_ids]
            render_radial = self.scene.radial_distortion[ground_truth_ids]
        else:
            poses, split = self.scene.render_poses, "path"

        if ground_truth_ids is None:
            train_ids = torch.as_tensor(
                self.scene.train_indices, device=self.device)
            intrinsic = self.scene.intrinsics[train_ids].median(dim=0).values
            render_intrinsics = intrinsic[None].expand(len(poses), -1, -1)
            render_radial = self.scene.radial_distortion[train_ids].median().expand(
                len(poses))

        output_dir = os.path.join(
            self.experiment_dir,
            f"render_{split}_{self.start_step:06d}")
        os.makedirs(output_dir, exist_ok=True)
        frames, compute = [], 0.0
        wall = time.perf_counter()
        with torch.no_grad():
            for index, pose in enumerate(poses):
                started = time.perf_counter()
                frame = self._render_pose(
                    pose, render_intrinsics[index], render_radial[index])
                if self.mode == "baseline":
                    # Baseline has no uncertainty prediction; keep the reused
                    # renderer's five-channel compatibility sentinel out of
                    # exported metrics and visualizations.
                    frame["uncertainty"] = np.zeros_like(
                        frame["uncertainty"])
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                compute += time.perf_counter() - started
                frames.append(frame)
                imageio.imwrite(
                    os.path.join(output_dir, f"rgb_{index:03d}.png"),
                    (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8))
                for key in ("depth", "uncertainty", "acc"):
                    np.save(os.path.join(
                        output_dir, f"{key}_{index:03d}.npy"),
                        frame[key].astype(np.float32))
                if ground_truth_ids is not None:
                    target = self.scene.images[
                        ground_truth_ids[index], ..., :3].detach().cpu().numpy()
                    imageio.imwrite(
                        os.path.join(output_dir, f"gt_rgb_{index:03d}.png"),
                        (np.clip(target, 0, 1) * 255).astype(np.uint8))

        all_depth = (np.concatenate([
            frame["depth"].reshape(-1) for frame in frames])
            if frames else np.array([]))
        valid_depth = all_depth[
            np.isfinite(all_depth) & (all_depth > 0)]
        low, high = (np.percentile(valid_depth, [2, 98])
                     if len(valid_depth) else (0.0, 1.0))
        videos = {key: [] for key in ("rgb", "depth", "uncertainty")}
        for index, frame in enumerate(frames):
            rgb = (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8)
            depth = (np.clip(
                (frame["depth"] - low) / max(high - low, 1.0e-8), 0, 1
            ) * 255).astype(np.uint8)
            uncertainty = (
                np.clip(frame["uncertainty"], 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(output_dir, f"depth_{index:03d}.png"), depth)
            imageio.imwrite(
                os.path.join(output_dir, f"uncertainty_{index:03d}.png"),
                uncertainty)
            videos["rgb"].append(rgb)
            videos["depth"].append(depth)
            videos["uncertainty"].append(uncertainty)
        for key, values in videos.items():
            if values:
                imageio.mimwrite(
                    os.path.join(output_dir, f"{key}.mp4"), values,
                    fps=30, quality=8)

        elapsed = time.perf_counter() - wall
        height, width, _ = self.scene.hwf
        efficiency = {
            "render_wall_seconds_with_io": elapsed,
            "render_compute_seconds": compute,
            "frames": len(frames),
            "fps": len(frames) / max(compute, 1.0e-8),
            "rays_per_second": (
                len(frames) * height * width / max(compute, 1.0e-8)),
        }
        with open(os.path.join(output_dir, "render_efficiency.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(efficiency, handle, indent=2)
        print(
            f"Rendered {len(frames)} V8 {self.mode} {split} views to "
            f"{output_dir}; FPS={efficiency['fps']:.3f}")
        return output_dir


__all__ = ["CHECKPOINT_VERSION", "DiagnosticTrainer",
           "assert_uq_gradient_isolation"]
