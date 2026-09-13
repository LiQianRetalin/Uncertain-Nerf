"""Bounded synchronized terminal-state timing, not a 20k quality evaluation."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gsplat import rasterization
from puri_gs.dino_features import FeatureCache
from puri_gs.semantic_mask import StaticResponsibilityHead, hard_static_mask
from tools.p04_oac_diagnostic import run_probe


ROOT = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf")
WORK = Path("/home/chenglong/P04T-work")
WEIGHT_SHA = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if json.loads((WORK / "postcheck.json").read_text())["status"] != "P04T_POSTCHECK_PASS":
        raise RuntimeError("P04T cost check requires complete validated training")
    sys.path.insert(0, str(ROOT / "external/gsplat-v1.5.3-ru/examples"))
    from datasets.colmap import Parser, Dataset
    device = torch.device("cuda:0")
    ledger = WORK / "call_budget_ledger.csv"
    with ledger.open(newline="", encoding="utf-8") as stream:
        charged = sum(int(row["charged"]) for row in csv.DictReader(stream))
    if charged != 241:
        raise RuntimeError("P04T timing starts only after exact 241-call training ledger")
    seq = charged
    def debit(kind, fn):
        nonlocal seq
        if seq >= 257:
            raise RuntimeError("P04T timing allocation 16 exceeded")
        seq += 1
        with ledger.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow((seq, 1, 19999, kind, 1, "RESERVED", ""))
            stream.flush(); os.fsync(stream.fileno())
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
            with ledger.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow((seq, 1, 19999, kind, 0, status,
                    round(time.perf_counter() - started, 6)))

    started_all = time.monotonic()
    snap = torch.load(WORK / "states/complete_after_step19999.pt", map_location="cpu", weights_only=False)
    splats = {key: value.to(device) for key, value in snap["splats"].items()}
    source = Parser(str(ROOT / "data/nerf_robustnerf/robustnerf/android"),
                    factor=4, normalize=True, test_every=8)
    split = json.loads((WORK / "outputs/P04T-android-ru-seed42-evidence-20k/dataset_split.json").read_text())
    trainset = Dataset(source, split="train", val_every=0,
                       train_keyword=split["train_keyword"], test_keyword=split["test_keyword"])
    datum = trainset[0]
    name = source.image_names[int(trainset.indices[0])]
    camera = datum["camtoworld"][None].to(device)
    K = datum["K"][None].to(device)
    height, width = datum["image"].shape[:2]
    cache = FeatureCache(ROOT / "data/PURI-GS-derived/semantic_features/android",
                         expected_weight_sha256=WEIGHT_SHA)
    head = StaticResponsibilityHead().to(device)
    head.load_state_dict(snap["ru_state"]["mask_head"])
    head.eval().requires_grad_(False)
    with torch.no_grad():
        grid = debit("timing_mask_head_forward", lambda: head(cache.load(name, 16)[None].to(device)))
        mask = hard_static_mask(F.interpolate(grid, size=(height, width), mode="bilinear",
                                                   align_corners=False), threshold=.25, kernel_size=7)[0, 0].bool()
    selected = torch.arange(512, device=device)
    zero = torch.zeros_like(mask)
    def render():
        return rasterization(means=splats["means"], quats=splats["quats"],
            scales=torch.exp(splats["scales"]), opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
            viewmats=torch.linalg.inv(camera), Ks=K, width=width, height=height,
            packed=False, absgrad=True, sh_degree=3, near_plane=.01, far_plane=1e10)
    samples = []
    for repeat in range(4):
        torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
        start_alloc = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        baseline = debit("timing_base_render", render)
        base_seconds = time.perf_counter() - t0
        base_peak = torch.cuda.max_memory_allocated() - start_alloc
        del baseline
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        start_alloc_pair = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        pair = debit("timing_pair_render", render)
        probe, _ = debit("timing_pair_probe", lambda: run_probe(
            pair[2], {"A": mask, "C": zero, "D": mask}, selected, device))
        pair_seconds = time.perf_counter() - t1
        pair_peak = torch.cuda.max_memory_allocated() - start_alloc_pair
        samples.append({"repeat": repeat, "warmup": repeat == 0,
                        "base_seconds": base_seconds, "base_plus_probe_seconds": pair_seconds,
                        "base_incremental_peak_bytes": base_peak,
                        "pair_incremental_peak_bytes": pair_peak,
                        "probe_nonzero": int((probe[0] > 1e-12).sum())})
        del pair, probe
    measured = samples[1:]
    result = {"status": "P04T_TIMING_PASS", "terminal_state_only": True,
              "image_name": name, "sample_count": 3, "warmup_count": 1,
              "charged_extra_calls_this_stage": seq - charged,
              "charged_extra_calls_total": seq,
              "base_median_ms": 1000 * statistics.median(item["base_seconds"] for item in measured),
              "base_plus_probe_median_ms": 1000 * statistics.median(item["base_plus_probe_seconds"] for item in measured),
              "paired_increment_median_ms": 1000 * statistics.median(
                  item["base_plus_probe_seconds"] - item["base_seconds"] for item in measured),
              "base_incremental_peak_bytes_median": statistics.median(item["base_incremental_peak_bytes"] for item in measured),
              "pair_incremental_peak_bytes_median": statistics.median(item["pair_incremental_peak_bytes"] for item in measured),
              "memory_interpretation": "compare incremental peaks only; allocator-cache artifacts remain possible",
              "timing_includes": "render, probe selection-map creation, GPU-to-CPU probe transfer and synchronization",
              "timing_excludes": "DINO/head, photometric backward, optimizer, topology and data preparation",
              "stage_wall_seconds": time.monotonic() - started_all,
              "samples": samples}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["status"])


if __name__ == "__main__":
    main()
