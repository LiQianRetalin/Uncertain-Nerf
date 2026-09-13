"""Frozen independent attribution and sampler preflight; no training updates."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tools.p04_oac_diagnostic import run_probe
from tools.p04t_monitor import ReplayableRandomSampler


ATTRIBUTION_ABS_TOL = 2e-4
ATTRIBUTION_REL_TOL = 2e-4
GRAD_ABS_TOL = 2e-6


def independent_reference(means, conics, opacities, masks):
    """Scalar CPU raster walk: pixel outer loop, Gaussian inner loop."""
    result = np.zeros((5, len(opacities)), dtype=np.float64)
    alpha_image = np.zeros((16, 16), dtype=np.float64)
    covered = {"overlap": 0, "alpha_skip": 0, "negative_sigma": 0,
               "early_stop": 0, "full_accept": 0, "full_reject": 0}
    for y in range(16):
        for x in range(16):
            trans = 1.0
            hits = 0
            for i in range(len(opacities)):
                dx = float(means[i, 0]) - (x + .5)
                dy = float(means[i, 1]) - (y + .5)
                cx, cy, cz = map(float, conics[i])
                sigma = .5 * (cx * dx * dx + cz * dy * dy) + cy * dx * dy
                if sigma < 0:
                    covered["negative_sigma"] += 1
                    continue
                alpha = min(.999, float(opacities[i]) * np.exp(-sigma))
                if alpha < 1.0 / 255.0:
                    covered["alpha_skip"] += 1
                    continue
                next_trans = trans * (1.0 - alpha)
                if next_trans <= 1e-4:
                    covered["early_stop"] += 1
                    break
                w = trans * alpha
                result[0, i] += w
                result[1, i] += w * masks["A"][y, x]
                result[2, i] += w * masks["C"][y, x]
                result[3, i] += w * masks["D"][y, x]
                result[4, i] += 1
                hits += 1
                trans = next_trans
            if hits > 1:
                covered["overlap"] += 1
            if masks["A"][y, x]:
                covered["full_accept"] += 1
            if not masks["C"][y, x]:
                covered["full_reject"] += 1
            alpha_image[y, x] = 1.0 - trans
    return result, alpha_image, covered


class Integers(Dataset):
    def __len__(self):
        return 17

    def __getitem__(self, index):
        return index


def camera_sampler_test():
    def sequence(replay):
        torch.manual_seed(42)
        dataset = Integers()
        kw = {"batch_size": 1, "num_workers": 4, "persistent_workers": True}
        if replay:
            loader = DataLoader(dataset, sampler=ReplayableRandomSampler(dataset), **kw)
        else:
            loader = DataLoader(dataset, shuffle=True, **kw)
        output = []
        for _ in range(3):
            output.extend(int(value.item()) for value in loader)
        return output
    original, replayable = sequence(False), sequence(True)
    if original != replayable:
        raise RuntimeError("P04T sampler differs from original RandomSampler across epochs")
    return original


def attribution_test(device):
    means = np.array([[8.5, 8.5], [8.5, 8.5], [8.5, 8.5], [8.5, 8.5],
                      [8.5, 8.5], [8.5, 8.5], [8.5, 8.5]], np.float32)
    conics = np.array([[.08, 0, .08], [.08, 0, .08], [.08, 0, .08],
                       [-.01, 0, -.01], [.02, 0, .02], [.02, 0, .02],
                       [.02, 0, .02]], np.float32)
    opacity = np.array([.4, .6, .0001, .5, .999, .999, .8], np.float32)
    accept = np.ones((16, 16), dtype=np.bool_)
    reject = np.zeros((16, 16), dtype=np.bool_)
    region = np.zeros((16, 16), dtype=np.bool_)
    region[4:12, 4:12] = True
    masks = {"A": accept, "C": reject, "D": region}
    reference, alpha_ref, cases = independent_reference(means, conics, opacity, masks)
    if not all(value > 0 for value in cases.values()):
        raise RuntimeError("P04T synthetic attribution cases incomplete")
    info = {"means2d": torch.tensor(means[None], device=device),
            "conics": torch.tensor(conics[None], device=device),
            "opacities": torch.tensor(opacity[None], device=device),
            "flatten_ids": torch.arange(len(opacity), device=device, dtype=torch.int32),
            "isect_offsets": torch.tensor([0], device=device, dtype=torch.int32),
            "tile_width": 1}
    values, alpha = run_probe(info, {k: torch.tensor(v, device=device) for k, v in masks.items()},
                              torch.arange(len(opacity), device=device), device)
    tested = np.stack(values).astype(np.float64)
    deviation = np.abs(tested - reference)
    allowable = ATTRIBUTION_ABS_TOL + ATTRIBUTION_REL_TOL * np.abs(reference)
    if np.any(deviation > allowable):
        raise RuntimeError("P04T per-Gaussian b/bM/bR/m attribution mismatch")
    alpha_error = float(np.max(np.abs(alpha.cpu().numpy() - alpha_ref)))
    if alpha_error > ATTRIBUTION_ABS_TOL:
        raise RuntimeError("P04T synthetic alpha mismatch")
    return {"max_abs_by_channel": deviation.max(axis=1).tolist(),
            "alpha_max_abs": alpha_error, "cases": cases}


def gradient_test(device):
    from gsplat.strategy import DefaultStrategy
    strategy = DefaultStrategy(absgrad=True, refine_scale2d_stop_iter=0)
    means2d = torch.zeros((1, 2, 2), device=device)
    means2d.absgrad = torch.tensor([[[.25, .5], [.5, .25]]], device=device)
    info = {"width": 100, "height": 60, "n_cameras": 1,
            "radii": torch.tensor([[[1., 1.], [0., 0.]]], device=device),
            "gaussian_ids": None, "means2d": means2d}
    state = {"grad2d": None, "count": None, "radii": None, "scene_scale": 1.0}
    strategy._update_state({"means": torch.zeros((2, 3), device=device)}, state, info)
    expected = np.hypot(.25 * 50, .5 * 30)
    if abs(float(state["grad2d"][0]) - expected) > GRAD_ABS_TOL:
        raise RuntimeError("P04T original AbsGrad screen-scale mismatch")
    if state["count"].tolist() != [1.0, 0.0]:
        raise RuntimeError("P04T original projection count mismatch")
    z = np.array([[1, 0], [1, 1]], np.float64)
    g = np.array([[2, 9], [4, 6]], np.float64)
    baseline = (z * g).sum(0) / np.maximum(z.sum(0), 1)
    if not np.array_equal(baseline, np.array([3.0, 6.0])):
        raise RuntimeError("P04T G0 arithmetic mismatch")
    b = np.array([[1, 0], [1, 1]], np.float64)
    bm = np.array([[.5, 0], [1, .25]], np.float64)
    u = np.divide(bm, b, out=np.zeros_like(b), where=b > 1e-12)
    A = (z * u).sum(0)
    if not np.allclose(A, [1.5, .25]):
        raise RuntimeError("P04T opportunity arithmetic mismatch")
    return {"original_screen_gradient": float(state["grad2d"][0]),
            "original_count": state["count"].tolist(), "G0": baseline.tolist(),
            "opportunity": A.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not torch.cuda.is_available():
        raise RuntimeError("P04T frozen L20 GPU is unavailable")
    ledger = args.out.parent / "call_budget_ledger.csv"
    if ledger.exists():
        raise FileExistsError(ledger)
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seq", "attempt", "step", "kind", "charged", "status", "seconds"))
        writer.writerow((1, 0, -1, "independent_reference_gpu_probe", 1, "RESERVED", ""))
        stream.flush()
        os.fsync(stream.fileno())
    device = torch.device("cuda:0")
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    try:
        camera = camera_sampler_test()
        attribution = attribution_test(device)
        gradient = gradient_test(device)
        ended.record()
        torch.cuda.synchronize()
        data = {"status": "P04T_PREFLIGHT_PASS", "tolerance_frozen_before_gpu_result": {
            "attribution_abs": ATTRIBUTION_ABS_TOL, "attribution_rel": ATTRIBUTION_REL_TOL,
            "gradient_abs": GRAD_ABS_TOL}, "independent_attribution": attribution,
            "original_gradient_and_G0": gradient, "sampler_first_3_epochs": camera,
            "gpu_elapsed_ms": started.elapsed_time(ended),
            "charged_extra_calls": 1}
        args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        status = "SUCCESS"
        print(data["status"])
    except Exception:
        status = "FAILED_CHARGED"
        raise
    finally:
        with ledger.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow((1, 0, -1, "independent_reference_gpu_probe", 0,
                                         status, ""))


if __name__ == "__main__":
    main()
