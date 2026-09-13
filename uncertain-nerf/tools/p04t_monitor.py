"""P04-T opt-in, read-only shadow evidence attached to one original RU run.

No OAC score is ever passed to the trainer or its strategy.  The only changes
to normal execution are auditable camera-order bookkeeping and the 20k stop.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from tools.p04_oac_diagnostic import run_probe


RUN_ID = "P04T-android-ru-seed42-evidence-20k"
WINDOWS = {"W1": (10901, 11000), "W2": (18901, 19000)}
SELECT_N = 512
MAX_EXTRA = 300
MAX_UPDATES = 40000
MAX_SECONDS = 7200


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


class ReplayableRandomSampler(Sampler[int]):
    """RandomSampler's replacement-free randperm, retaining its current order.

DataLoader worker prefetch may advance cursor ahead of delivered batches.  A
snapshot therefore stores the externally known delivered offset, not cursor.
"""

    def __init__(self, dataset):
        self.size = len(dataset)
        self.order: list[int] | None = None
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        if self.order is None or self.cursor >= self.size:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            self.order = torch.randperm(self.size, generator=generator).tolist()
            self.cursor = 0
        while self.cursor < self.size:
            item = self.order[self.cursor]
            self.cursor += 1
            yield item

    def state_dict(self, completed_step: int) -> dict:
        delivered = (completed_step + 1) % self.size
        if delivered and self.order is None:
            raise RuntimeError("current camera permutation was lost")
        return {"size": self.size, "order": self.order if delivered else None,
                "delivered_offset": delivered, "next_step": completed_step + 1}

    def load_state_dict(self, state: dict) -> None:
        if state["size"] != self.size:
            raise RuntimeError("camera sampler dataset size changed")
        self.order = state["order"]
        self.cursor = int(state["delivered_offset"])
        if self.order is not None and sorted(self.order) != list(range(self.size)):
            raise RuntimeError("camera permutation invalid")


class P04TMonitor:
    def __init__(self, runner, schedulers, sampler: ReplayableRandomSampler):
        if os.environ["P04T_RUN_ID"] != RUN_ID:
            raise RuntimeError("P04T identity changed")
        if runner.cfg.max_steps != 30000 or runner.cfg.batch_size != 1:
            raise RuntimeError("P04T requires the original 30k/batch1 schedule")
        if not runner.puri_gs_delayed_topology_enabled or runner.ru_training is None:
            raise RuntimeError("P04T requires original delayed RU")
        if runner.world_size != 1 or runner.cfg.packed:
            raise RuntimeError("P04T probe requires one camera and unpacked projection")
        self.runner = runner
        self.schedulers = schedulers
        self.sampler = sampler
        self.root = Path(os.environ["P04T_WORK"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.root / "states"
        self.state_dir.mkdir(exist_ok=True)
        self.start_time = time.monotonic()
        self.rows: dict[str, list[dict]] = {name: [] for name in WINDOWS}
        self.ids: dict[str, np.ndarray] = {}
        self.extra = self._charged()
        self.update_count = self._updates()
        self.attempt = int(os.environ.get("P04T_ATTEMPT", "1"))
        if self.attempt not in (1, 2):
            raise RuntimeError("P04T permits at most two technical attempts")
        self.prior_gpu_seconds = float(os.environ.get("P04T_PRIOR_GPU_SECONDS", "0"))
        if self.prior_gpu_seconds < 0 or (self.attempt == 2 and self.prior_gpu_seconds <= 0):
            raise RuntimeError("P04T recovery requires accounted prior GPU time")
        self._strategy_grow = runner.cfg.strategy._grow_gs
        runner.cfg.strategy._grow_gs = self._before_grow
        self._last_step = -1

    def _charged(self) -> int:
        path = self.root / "call_budget_ledger.csv"
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(("seq", "attempt", "step", "kind", "charged", "status", "seconds"))
            return 0
        with path.open(newline="", encoding="utf-8") as stream:
            return sum(int(row["charged"]) for row in csv.DictReader(stream))

    def _updates(self) -> int:
        path = self.root / "training_updates_ledger.csv"
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(("attempt", "step", "cumulative_updates", "gaussians", "lr_means", "wall_seconds"))
            return 0
        with path.open(newline="", encoding="utf-8") as stream:
            return sum(1 for _ in csv.DictReader(stream))

    def _call(self, step: int, kind: str, fn):
        if self.extra >= MAX_EXTRA or self.prior_gpu_seconds + time.monotonic() - self.start_time >= MAX_SECONDS:
            raise RuntimeError("P04T_EXTRA_OR_GPU_BUDGET_EXHAUSTED")
        self.extra += 1
        path = self.root / "call_budget_ledger.csv"
        with path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow((self.extra, self.attempt, step, kind, 1, "RESERVED", ""))
            stream.flush()
            os.fsync(stream.fileno())
        started = time.perf_counter()
        try:
            value = fn()
            torch.cuda.synchronize()
            status = "SUCCESS"
            return value
        except Exception:
            status = "FAILED_CHARGED"
            raise
        finally:
            with path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow((self.extra, self.attempt, step, kind, 0, status,
                                             round(time.perf_counter() - started, 6)))

    def start(self, loader, iterator):
        resume = os.environ.get("P04T_RESUME_PATH")
        if not resume:
            if self.update_count or self.extra != 17:
                raise RuntimeError("nonempty P04T ledger without explicit technical resume")
            self._status("TRAINING", -1)
            return 0, iterator
        if self.attempt != 2:
            raise RuntimeError("resume is permitted only as diagnosed attempt 2")
        path = Path(resume)
        data = torch.load(path, map_location=self.runner.device, weights_only=False)
        required = {"completed_step", "splats", "gaussian_optimizers", "schedulers",
                    "strategy_state", "ru_state", "shadow_rows", "selected_ids",
                    "sampler", "rng", "cfg_sha256"}
        if not required <= data.keys() or file_sha(path) != json.loads(path.with_suffix(".json").read_text())["sha256"]:
            raise RuntimeError("P04T recovery snapshot incomplete or changed")
        if data["cfg_sha256"] != file_sha(Path(self.runner.cfg.result_dir) / "cfg.yml"):
            raise RuntimeError("P04T recovery config mismatch")
        if set(data["splats"]) != set(self.runner.splats):
            raise RuntimeError("P04T recovery Gaussian parameter names differ")
        for key, tensor in data["splats"].items():
            restored = torch.nn.Parameter(tensor.detach().clone().to(self.runner.device))
            self.runner.splats[key] = restored
            groups = self.runner.optimizers[key].param_groups
            if len(groups) != 1 or len(groups[0]["params"]) != 1:
                raise RuntimeError("P04T recovery optimizer layout changed")
            groups[0]["params"] = [restored]
        for key, optimizer in self.runner.optimizers.items():
            optimizer.load_state_dict(data["gaussian_optimizers"][key])
        for scheduler, state in zip(self.schedulers, data["schedulers"], strict=True):
            scheduler.load_state_dict(state)
        self.runner.strategy_state = data["strategy_state"]
        self.runner.ru_training.load_replay_state_dict(data["ru_state"])
        self.rows = data["shadow_rows"]
        self.ids = data["selected_ids"]
        self._last_step = data["completed_step"]
        iterator._shutdown_workers()
        self.sampler.load_state_dict(data["sampler"])
        iterator = iter(loader)
        self._set_rng(data["rng"])
        first = data["completed_step"] + 1
        if not 0 < first <= 19999:
            raise RuntimeError("P04T recovery boundary invalid")
        self._status("RECOVERED", data["completed_step"])
        return first, iterator

    def _rng(self):
        return {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(), "python": random.getstate()}

    @staticmethod
    def _set_rng(state):
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state_all(state["torch_cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])

    def _status(self, stage: str, step: int, **more):
        atomic_json(self.root / "status.json", {"run_id": RUN_ID, "stage": stage,
                    "attempt": self.attempt, "completed_step": step,
                    "cumulative_updates": self.update_count, "charged_extra_calls": self.extra,
                    "wall_seconds_this_attempt": round(time.monotonic() - self.start_time, 3),
                    "gpu_wall_upper_bound_seconds": round(self.prior_gpu_seconds + time.monotonic() - self.start_time, 3), **more})

    @staticmethod
    def _choose_ids(count: int, window: str) -> np.ndarray:
        salt = int.from_bytes(hashlib.sha256(f"42:{RUN_ID}:{window}".encode()).digest()[:8], "little")
        x = np.arange(count, dtype=np.uint64) ^ np.uint64(salt)
        with np.errstate(over="ignore"):
            x ^= x >> np.uint64(30)
            x *= np.uint64(0xbf58476d1ce4e5b9)
            x ^= x >> np.uint64(27)
            x *= np.uint64(0x94d049bb133111eb)
            x ^= x >> np.uint64(31)
        chosen = np.argpartition(x, SELECT_N - 1)[:SELECT_N]
        return np.sort(chosen.astype(np.int64))

    @staticmethod
    def _region(height: int, width: int, window: str, step: int, device):
        tags = [(hashlib.sha256(f"42:{RUN_ID}:{window}:{step}:{cell}".encode()).digest(), cell)
                for cell in range(400)]
        selected = sorted(cell for _, cell in sorted(tags)[:20])
        region = torch.zeros((height, width), device=device, dtype=torch.bool)
        for cell in selected:
            y, x = divmod(cell, 20)
            region[y * height // 20:(y + 1) * height // 20,
                   x * width // 20:(x + 1) * width // 20] = True
        return region, selected

    def _live_signature(self):
        h = hashlib.sha256()
        for name, value in sorted(self.runner.splats.items()):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
            if value.grad is not None:
                h.update(value.grad.detach().cpu().contiguous().numpy().tobytes())
        for name, value in sorted(self.runner.ru_training.head.state_dict().items()):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        for name, value in sorted(self.runner.strategy_state.items()):
            if torch.is_tensor(value):
                h.update(name.encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def after_photo_backward(self, step, info, ru_state, pixels, camera, Ks, image_ids, sh_degree):
        window = next((name for name, (start, end) in WINDOWS.items() if start <= step <= end), None)
        if window is None:
            return
        start, end = WINDOWS[window]
        if step == start:
            count = len(self.runner.splats["means"])
            if count < SELECT_N:
                raise RuntimeError("P04T fewer than 512 live Gaussians")
            self.ids[window] = self._choose_ids(count, window)
            atomic_json(self.root / f"{window}_selection.json", {
                "step": step, "live_gaussians": count, "selection_rule": "splitmix64(sha256(42:run_id:window) xor index); smallest 512",
                "probe_ids": self.ids[window].tolist(),
                "probe_sha256": hashlib.sha256(self.ids[window].tobytes()).hexdigest()})
        ids_cpu = self.ids[window]
        if ids_cpu[-1] >= len(self.runner.splats["means"]):
            raise RuntimeError("P04T selected ID invalid before event")
        ids = torch.as_tensor(ids_cpu, device=pixels.device)
        mask = ru_state.safe_mask[0, 0].detach().bool()
        zero = torch.zeros_like(mask)
        probe, _ = self._call(step, "A_exact_probe_forward", lambda: run_probe(
            info, {"A": mask, "C": zero, "D": mask}, ids, pixels.device))
        g = info["means2d"].absgrad[0, ids].detach().clone()
        g[:, 0] *= info["width"] / 2.0 * info["n_cameras"]
        g[:, 1] *= info["height"] / 2.0 * info["n_cameras"]
        z = (info["radii"][0, ids] > 0).all(-1)
        row = {"step": step, "camera": int(image_ids.item()), "z": z.cpu().numpy().astype(np.uint8),
               "g": torch.linalg.vector_norm(g, dim=-1).cpu().numpy(),
               "b": probe[0], "bm": probe[1], "m": probe[4],
               "mask_fraction": float(mask.float().mean()), "d": None}
        if step - start in (0, 33, 66, 99):
            row["d"] = self._pollution(step, window, ids, pixels, camera, Ks,
                                        sh_degree, mask, row["z"], row["b"])
        self.rows[window].append(row)
        if len(self.rows[window]) != step - start + 1:
            raise RuntimeError("P04T window step gap")

    def _pollution(self, step, window, ids, pixels, camera, Ks, sh_degree, mask, a_z, a_b):
        from fused_ssim import fused_ssim
        from gsplat import rasterization
        from puri_gs.semantic_mask import masked_photo_loss

        before = self._live_signature()
        rng_before = self._rng()
        height, width = pixels.shape[1:3]
        region, cells = self._region(height, width, window, step, pixels.device)
        md = mask | region
        target = pixels.detach().clone()
        target[0, region] = target.new_tensor([1.0, 0.0, 1.0])
        cloned = {key: value.detach().clone().requires_grad_(True)
                  for key, value in self.runner.splats.items()}
        cfg = self.runner.cfg
        def render():
            return rasterization(
                means=cloned["means"], quats=cloned["quats"],
                scales=torch.exp(cloned["scales"]), opacities=torch.sigmoid(cloned["opacities"]),
                colors=torch.cat([cloned["sh0"], cloned["shN"]], dim=1),
                viewmats=torch.linalg.inv(camera), Ks=Ks, width=width, height=height,
                packed=False, absgrad=True, sh_degree=sh_degree,
                near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                rasterize_mode="antialiased" if cfg.antialiased else "classic",
                camera_model=cfg.camera_model,
            )
        rgb, _, d_info = self._call(step, "D_clone_render_forward", render)
        loss, _, _ = masked_photo_loss(rgb, target, md[None, None].float(),
                                       fused_ssim_fn=fused_ssim, ssim_lambda=cfg.ssim_lambda)
        self._call(step, "D_clone_photo_backward", lambda: loss.backward())
        d_probe, _ = self._call(step, "D_and_R_exact_probe_forward", lambda: run_probe(
            d_info, {"A": md, "C": region, "D": md}, ids, pixels.device))
        d_grad = d_info["means2d"].absgrad[0, ids].detach().clone()
        d_grad[:, 0] *= width / 2.0
        d_grad[:, 1] *= height / 2.0
        d_z = (d_info["radii"][0, ids] > 0).all(-1).cpu().numpy().astype(np.uint8)
        if self._live_signature() != before or not self._rng_equal(rng_before, self._rng()):
            raise RuntimeError("P04T D polluted live gradients/parameters/head/strategy/RNG")
        if not np.array_equal(d_z, a_z):
            raise RuntimeError("P04T cloned D geometry/projection differs from live A")
        if not np.allclose(d_probe[0], a_b, rtol=2e-4, atol=2e-4):
            raise RuntimeError("P04T cloned D base contribution differs from live A")
        return {"g": torch.linalg.vector_norm(d_grad, dim=-1).cpu().numpy(),
                "z": d_z, "b": d_probe[0], "bm": d_probe[1], "br": d_probe[2],
                "m": d_probe[4], "cells": cells, "region_pixels": int(region.sum()),
                "live_signature": before}

    @staticmethod
    def _rng_equal(a, b):
        return (torch.equal(a["torch_cpu"], b["torch_cpu"])
                and all(torch.equal(x, y) for x, y in zip(a["torch_cuda"], b["torch_cuda"]))
                and a["python"] == b["python"]
                and a["numpy"][0] == b["numpy"][0]
                and np.array_equal(a["numpy"][1], b["numpy"][1])
                and a["numpy"][2:] == b["numpy"][2:])

    def _before_grow(self, params, optimizers, state, step):
        if step == 10000:
            count = state["count"].detach().cpu().numpy()
            grad = state["grad2d"].detach().cpu().numpy()
            atomic_json(self.root / "first_long_event.json", {
                "step": step, "long_window": "0..10000", "live_gaussians": len(params["means"]),
                "count_positive": int(np.count_nonzero(count)),
                "count_quantiles": np.quantile(count, [0,.25,.5,.75,.9,.99,1]).tolist(),
                "grad_quantiles": np.quantile(grad, [0,.25,.5,.75,.9,.99,1]).tolist(),
                "opportunity_available": False})
        for name, (start, end) in WINDOWS.items():
            if start < step < end:
                raise RuntimeError("unexpected topology inside P04T fixed window")
            if step == end:
                self._finish_window(name, state)
        return self._strategy_grow(params, optimizers, state, step)

    def _finish_window(self, window, state):
        start, end = WINDOWS[window]
        rows = self.rows[window]
        if len(rows) != 100 or [r["step"] for r in rows] != list(range(start, end + 1)):
            raise RuntimeError("P04T real event window is incomplete")
        ids = self.ids[window]
        original_count = state["count"][ids].detach().cpu().numpy()
        original_grad = state["grad2d"][ids].detach().cpu().numpy()
        expected_count = np.stack([r["z"] for r in rows]).sum(axis=0)
        expected_grad = np.stack([r["g"] for r in rows]).sum(axis=0)
        max_count = float(np.max(np.abs(original_count - expected_count)))
        max_grad = float(np.max(np.abs(original_grad - expected_grad)))
        rel_grad = float(np.max(np.abs(original_grad - expected_grad) /
                                np.maximum(np.abs(original_grad), 1e-8)))
        atomic_json(self.root / f"{window}_alignment.json", {
            "event_step": end, "read_point": "after original _update_state, before grow/prune/clear",
            "count_max_abs": max_count, "grad_max_abs": max_grad,
            "grad_max_relative": rel_grad, "count_exact": max_count == 0,
            "grad_tolerance_pre_registered": "abs<=2e-5 or relative<=2e-4",
            "grad_pass": max_grad <= 2e-5 or rel_grad <= 2e-4})
        if max_count != 0 or (max_grad > 2e-5 and rel_grad > 2e-4):
            raise RuntimeError("P04T shadow/original strategy alignment failed")
        d_steps = [r["step"] for r in rows if r["d"] is not None]
        path = self.root / f"{window}_per_step_probe.npz"
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, ids=ids, steps=np.array([r["step"] for r in rows]),
            camera=np.array([r["camera"] for r in rows]),
            z=np.stack([r["z"] for r in rows]), g=np.stack([r["g"] for r in rows]),
            b=np.stack([r["b"] for r in rows]), bm=np.stack([r["bm"] for r in rows]),
            m=np.stack([r["m"] for r in rows]),
            mask_fraction=np.array([r["mask_fraction"] for r in rows]),
            d_steps=np.array(d_steps),
            d_z=np.stack([r["d"]["z"] for r in rows if r["d"] is not None]),
            d_g=np.stack([r["d"]["g"] for r in rows if r["d"] is not None]),
            d_b=np.stack([r["d"]["b"] for r in rows if r["d"] is not None]),
            d_bm=np.stack([r["d"]["bm"] for r in rows if r["d"] is not None]),
            d_br=np.stack([r["d"]["br"] for r in rows if r["d"] is not None]),
            d_m=np.stack([r["d"]["m"] for r in rows if r["d"] is not None]),
            d_cells=np.array([r["d"]["cells"] for r in rows if r["d"] is not None]),
            d_region_pixels=np.array([r["d"]["region_pixels"] for r in rows if r["d"] is not None]),
            original_count=original_count, original_grad=original_grad)
        os.replace(temp, path)
        atomic_json(self.root / f"{window}_data_manifest.json", {"sha256": file_sha(path),
                    "steps": [start, end], "rows": len(rows), "d_steps": d_steps,
                    "selected_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()})
        self._status(f"{window}_COMPLETE", end - 1, next_event=end)
        print(f"P04T_{window}_EVENT_EVIDENCE_COMPLETE", flush=True)
        self.rows[window] = []

    def _snapshot(self, step):
        result = Path(self.runner.cfg.result_dir)
        data = {"completed_step": step, "next_step": step + 1,
                "splats": self.runner.splats.state_dict(),
                "gaussian_optimizers": {k: v.state_dict() for k, v in self.runner.optimizers.items()},
                "schedulers": [v.state_dict() for v in self.schedulers],
                "strategy_state": self.runner.strategy_state,
                "ru_state": self.runner.ru_training.replay_state_dict(),
                "shadow_rows": self.rows, "selected_ids": self.ids,
                "sampler": self.sampler.state_dict(step), "rng": self._rng(),
                "cfg_sha256": file_sha(result / "cfg.yml"),
                "feature_manifest_sha256": file_sha(Path(self.runner.cfg.feature_cache_dir) / "manifest.json"),
                "sh_degree_next": min((step + 1) // self.runner.cfg.sh_degree_interval,
                                      self.runner.cfg.sh_degree)}
        target = self.state_dir / f"complete_after_step{step}.pt"
        temp = target.with_suffix(".pt.tmp")
        torch.save(data, temp)
        verify = torch.load(temp, map_location="cpu", weights_only=False)
        if not set(data) <= set(verify) or verify["completed_step"] != step:
            raise RuntimeError("P04T snapshot readback failed")
        os.replace(temp, target)
        atomic_json(target.with_suffix(".json"), {"step": step, "sha256": file_sha(target),
                    "bytes": target.stat().st_size, "keys": sorted(data),
                    "boundary": "after all original updates; before next forward/head update"})
        print(f"P04T_ATOMIC_SNAPSHOT_PASS step={step}", flush=True)

    def after_normal_update(self, step):
        if step != self._last_step + 1 and not (self._last_step == -1 and step > 0 and self.attempt == 2):
            raise RuntimeError("P04T normal update sequence gap")
        self._last_step = step
        self.update_count += 1
        if self.update_count > MAX_UPDATES or self.prior_gpu_seconds + time.monotonic() - self.start_time > MAX_SECONDS:
            raise RuntimeError("P04T_TOTAL_UPDATE_OR_GPU_TIME_BUDGET_EXHAUSTED")
        with (self.root / "training_updates_ledger.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow((self.attempt, step, self.update_count,
                len(self.runner.splats["means"]), self.schedulers[0].get_last_lr()[0],
                round(time.monotonic() - self.start_time, 3)))
        if step in (10999, 18999, 19999):
            self._snapshot(step)
        if step % 100 == 99 or step in (11000, 19000, 19999):
            self._status("COMPLETE" if step == 19999 else "TRAINING", step)
