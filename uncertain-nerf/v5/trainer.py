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

from .data import RaySampler, load_scene, pixels_to_rays
from .losses import DepthScaleAligner, geometry_loss
from .model import RadianceFieldV5
from .rendering import render_rays
from .robust import LossCalibrator, anneal, warmup_cosine_lr
from .teacher import TeacherEstimator, uncertainty_regularization


CHECKPOINT_VERSION = "uncertain-nerf-v5-1"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _field(args, aabb):
    return RadianceFieldV5(
        aabb=aabb,
        hidden_dim=args.netwidth,
        hidden_layers=args.netdepth,
        dropout=args.dropout,
        hash_levels=args.hash_levels,
        hash_min_resolution=args.hash_min_resolution,
        hash_max_resolution=args.hash_max_resolution,
        hash_features=args.hash_features,
        hash_log2_size=args.hash_log2_size,
        direction_frequencies=args.multires_views,
    )


def _slice_result(result, indices):
    sliced = {}
    batch = result["rgb"].shape[0]
    for key, value in result.items():
        sliced[key] = value[indices] if torch.is_tensor(value) and value.ndim and value.shape[0] == batch else value
    return sliced


class Trainer:
    def __init__(self, args):
        self.args = args
        if args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        set_seed(args.seed)
        self.scene = load_scene(
            args.datadir,
            factor=args.factor,
            holdout=args.llffhold,
            n_source_views=args.teacher_views,
            device=self.device,
        )
        self.coarse = _field(args, self.scene.aabb).to(self.device)
        self.fine = _field(args, self.scene.aabb).to(self.device)
        self.background_logits = nn.Parameter(
            torch.zeros(len(self.scene.images), 3, device=self.device)
        )
        self.teacher = TeacherEstimator(
            mc_samples=args.mc_samples,
            kappa=args.teacher_kappa,
            variance_beta=args.variance_beta,
        ).to(self.device)
        parameters = list(self.coarse.parameters()) + list(self.fine.parameters())
        parameters += [self.background_logits] + list(self.teacher.parameters())
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=args.lrate,
            betas=(0.9, 0.99),
            eps=1.0e-8,
        )
        amp_enabled = bool(args.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.calibrator = LossCalibrator(
            rho=args.loss_ema_rho, calibration_steps=args.loss_calibration_steps
        )
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
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_step = 0
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

    def _background(self, image_ids=None):
        colors = torch.sigmoid(self.background_logits)
        if image_ids is None:
            train_ids = torch.as_tensor(self.scene.train_indices, device=self.device)
            return colors[train_ids].mean(dim=0, keepdim=True)
        return colors[image_ids]

    def _render_batch(self, batch, perturb):
        background = self._background(batch["image_ids"])
        return render_rays(
            self.coarse,
            self.fine,
            batch["rays_o"],
            batch["rays_d"],
            self.scene.near,
            self.scene.far,
            background,
            n_samples=self.args.N_samples,
            n_importance=self.args.N_importance,
            perturb=perturb,
            chunk=self.args.netchunk,
            reliability_strength=self.args.reliability_strength,
        )

    def training_step(self, step):
        args = self.args
        batch = self.sampler.sample()
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(args.amp and self.device.type == "cuda")
        with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
            rendered = self._render_batch(batch, perturb=True)
            coarse, fine = rendered["coarse"], rendered["fine"]
            color_loss = F.mse_loss(fine["rgb"], batch["target"])
            if fine is not coarse:
                color_loss = color_loss + F.mse_loss(coarse["rgb"], batch["target"])
            teacher_count = max(1, int(round(args.teacher_ray_ratio * args.N_rand)))
            teacher_indices = torch.randperm(args.N_rand, device=self.device)[:teacher_count]
            fine_teacher = _slice_result(fine, teacher_indices)
            teacher_target, teacher_stats = self.teacher(
                self.fine,
                fine_teacher,
                batch["rays_d"][teacher_indices],
                self._background(batch["image_ids"][teacher_indices]),
                batch["target"][teacher_indices],
                batch["image_ids"][teacher_indices],
                self.scene,
                tukey_multiplier=anneal(
                    args.tukey_start, args.tukey_end, step, args.robust_anneal_steps
                ),
                query_chunk=args.netchunk,
            )
            uncertainty_loss = F.l1_loss(
                fine["uncertainty"][teacher_indices], teacher_target
            )
        # Keep the spatial derivative calculation in full precision.
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            regularization, reg_stats = uncertainty_regularization(
                self.fine,
                fine,
                teacher_indices,
                eta_amplitude=args.reg_amplitude,
                eta_spatial=args.reg_spatial,
            )
            geometry, geo_stats = geometry_loss(
                fine["depth"].float(),
                fine["uncertainty"].float(),
                batch["prior_depth"].float(),
                batch["prior_mask"],
                self.depth_aligner.scale,
                args.geo_gamma,
                anneal(args.huber_start, args.huber_end, step, args.robust_anneal_steps),
            )
        self.depth_aligner.observe(
            fine["depth"][batch["prior_mask"]],
            batch["prior_depth"][batch["prior_mask"]],
        )
        self.depth_aligner.maybe_update(step)
        loss_parts = {
            "color": color_loss,
            "geo": geometry,
            "u": uncertainty_loss,
            "reg": regularization,
        }
        weights = self.calibrator.update(
            step,
            loss_parts,
            args.teacher_ray_ratio,
            {"geo": args.alpha_geo, "u": args.alpha_u, "reg": args.alpha_reg},
        )
        total = color_loss + weights["geo"] * geometry
        total = total + weights["u"] * uncertainty_loss + weights["reg"] * regularization
        self.scaler.scale(total).backward()
        if args.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.coarse.parameters()) + list(self.fine.parameters()), args.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        lr = warmup_cosine_lr(
            step + 1, args.N_iters, args.lr_warmup, args.lrate, args.lrate_min
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        psnr = -10.0 * torch.log10(F.mse_loss(fine["rgb"], batch["target"]).clamp_min(1.0e-10))
        return {
            "loss": float(total.detach()),
            "color": float(color_loss.detach()),
            "geo": float(geometry.detach()),
            "u": float(uncertainty_loss.detach()),
            "reg": float(regularization.detach()),
            "psnr": float(psnr.detach()),
            "lr": lr,
            "scale": self.depth_aligner.scale,
            "teacher_mix": float(teacher_stats["mix"].detach()),
            "reg_spatial": float(reg_stats["spatial"].detach()),
            "geo_count": geo_stats["count"],
        }

    def train(self):
        if self.args.render_only:
            return self.render()
        self.coarse.train()
        self.fine.train()
        self.teacher.train()
        progress = trange(self.start_step + 1, self.args.N_iters + 1, desc="training")
        start_time = time.time()
        for step in progress:
            stats = self.training_step(step)
            if step % self.args.i_print == 0 or step == self.start_step + 1:
                progress.set_postfix(
                    loss=f"{stats['loss']:.4g}", psnr=f"{stats['psnr']:.2f}", lr=f"{stats['lr']:.2e}"
                )
                print(json.dumps({"step": step, **stats, "elapsed": time.time() - start_time}))
            if step % self.args.i_weights == 0 or step == self.args.N_iters:
                self.save_checkpoint(step)

    def checkpoint_state(self, step):
        state = {
            "version": CHECKPOINT_VERSION,
            "global_step": int(step),
            "dataset_fingerprint": self.scene.fingerprint,
            "coarse": self.coarse.state_dict(),
            "fine": self.fine.state_dict(),
            "background_logits": self.background_logits.detach(),
            "teacher": self.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "calibrator": self.calibrator.state_dict(),
            "depth_aligner": self.depth_aligner.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "aabb": self.scene.aabb.detach().cpu(),
        }
        return state

    def save_checkpoint(self, step):
        path = os.path.join(self.checkpoint_dir, f"step_{step:06d}.pt")
        torch.save(self.checkpoint_state(step), path)
        shutil.copyfile(path, os.path.join(self.checkpoint_dir, "latest.pt"))
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                "Incompatible checkpoint: expected an Uncertain-NeRF V5 checkpoint; "
                "legacy NeRF .tar files cannot be loaded"
            )
        if state.get("dataset_fingerprint") != self.scene.fingerprint:
            raise ValueError("Checkpoint dataset fingerprint does not match the current scene")
        self.coarse.load_state_dict(state["coarse"])
        self.fine.load_state_dict(state["fine"])
        self.background_logits.data.copy_(state["background_logits"].to(self.device))
        self.teacher.load_state_dict(state["teacher"])
        self.calibrator.load_state_dict(state.get("calibrator", {}))
        self.depth_aligner.load_state_dict(state.get("depth_aligner", {}))
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
        print(f"Loaded checkpoint {path} at step {self.start_step}")

    def _render_pose(self, pose, intrinsic, radial_distortion=0.0):
        height, width, _ = self.scene.hwf
        ys, xs = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing="ij",
        )
        count = height * width
        image_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        poses = pose[None].expand(1, -1, -1)
        intrinsics = intrinsic[None].expand(1, -1, -1)
        rays_o, rays_d = pixels_to_rays(
            poses,
            intrinsics,
            image_ids,
            ys.reshape(-1),
            xs.reshape(-1),
            radial_distortion=radial_distortion,
        )
        background = self._background().expand(count, -1)
        outputs = {"rgb": [], "depth": [], "uncertainty": []}
        for start in range(0, count, self.args.chunk):
            end = min(start + self.args.chunk, count)
            result = render_rays(
                self.coarse,
                self.fine,
                rays_o[start:end],
                rays_d[start:end],
                self.scene.near,
                self.scene.far,
                background[start:end],
                n_samples=self.args.N_samples,
                n_importance=self.args.N_importance,
                perturb=False,
                chunk=self.args.netchunk,
                reliability_strength=self.args.reliability_strength,
            )["fine"]
            for key in outputs:
                outputs[key].append(result[key].detach().cpu())
        return {
            key: torch.cat(value).reshape(height, width, -1 if key == "rgb" else 1).squeeze(-1).numpy()
            for key, value in outputs.items()
        }

    def render(self):
        if self.start_step == 0:
            raise ValueError("render_only requires --ft_path or an existing checkpoints/latest.pt")
        self.coarse.eval()
        self.fine.eval()
        self.teacher.eval()
        if self.args.render_poses:
            poses = torch.from_numpy(np.load(self.args.render_poses).astype(np.float32)).to(self.device)
            poses = poses[:, :3, :4]
            split = "custom"
        elif self.args.render_split == "test":
            test_ids = torch.as_tensor(self.scene.test_indices, device=self.device)
            poses = self.scene.poses[test_ids]
            render_intrinsics = self.scene.intrinsics[test_ids]
            render_radial_distortion = self.scene.radial_distortion[test_ids]
            split = "test"
        else:
            poses = self.scene.render_poses
            split = "path"
        output_dir = os.path.join(self.experiment_dir, f"render_{split}_{self.start_step:06d}")
        os.makedirs(output_dir, exist_ok=True)
        train_ids = torch.as_tensor(self.scene.train_indices, device=self.device)
        if split != "test":
            intrinsic = self.scene.intrinsics[train_ids].median(dim=0).values
            radial_distortion = self.scene.radial_distortion[train_ids].median()
            render_intrinsics = intrinsic[None].expand(len(poses), -1, -1)
            render_radial_distortion = radial_distortion.expand(len(poses))
        frames = []
        with torch.no_grad():
            for index, pose in enumerate(poses):
                frame = self._render_pose(
                    pose,
                    render_intrinsics[index],
                    render_radial_distortion[index],
                )
                frames.append(frame)
                imageio.imwrite(
                    os.path.join(output_dir, f"rgb_{index:03d}.png"),
                    (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8),
                )
                np.save(os.path.join(output_dir, f"depth_{index:03d}.npy"), frame["depth"].astype(np.float32))
                np.save(os.path.join(output_dir, f"uncertainty_{index:03d}.npy"), frame["uncertainty"].astype(np.float32))
        all_depth = np.concatenate([frame["depth"].reshape(-1) for frame in frames])
        valid_depth = all_depth[np.isfinite(all_depth) & (all_depth > 0)]
        low, high = np.percentile(valid_depth, [2, 98]) if len(valid_depth) else (0.0, 1.0)
        rgb_video, depth_video, uncertainty_video = [], [], []
        for index, frame in enumerate(frames):
            rgb8 = (np.clip(frame["rgb"], 0, 1) * 255).astype(np.uint8)
            depth8 = (np.clip((frame["depth"] - low) / max(high - low, 1.0e-8), 0, 1) * 255).astype(np.uint8)
            uncertainty8 = (np.clip(frame["uncertainty"], 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_dir, f"depth_{index:03d}.png"), depth8)
            imageio.imwrite(os.path.join(output_dir, f"uncertainty_{index:03d}.png"), uncertainty8)
            rgb_video.append(rgb8)
            depth_video.append(depth8)
            uncertainty_video.append(uncertainty8)
        imageio.mimwrite(os.path.join(output_dir, "rgb.mp4"), rgb_video, fps=30, quality=8)
        imageio.mimwrite(os.path.join(output_dir, "depth.mp4"), depth_video, fps=30, quality=8)
        imageio.mimwrite(os.path.join(output_dir, "uncertainty.mp4"), uncertainty_video, fps=30, quality=8)
        print(f"Rendered {len(frames)} views to {output_dir}")
        return output_dir
