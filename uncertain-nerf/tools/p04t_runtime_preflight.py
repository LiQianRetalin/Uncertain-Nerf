"""One bounded real-state D isolation check before any P04-T training update."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from gsplat import rasterization
from puri_gs.dino_features import FeatureCache
from puri_gs.semantic_mask import StaticResponsibilityHead, hard_static_mask, masked_photo_loss
from tools.p04_oac_diagnostic import run_probe
from tools.p04t_monitor import P04TMonitor


ROOT = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf")
OLD = ROOT / "logs-puri/ru-generalization-rerun-9e292309/android_ru_30k"
DATA = ROOT / "data/nerf_robustnerf/robustnerf/android"
FEATURES = ROOT / "data/PURI-GS-derived/semantic_features/android"
WEIGHT_SHA = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    import sys
    sys.path.insert(0, str(ROOT / "external/gsplat-v1.5.3-ru/examples"))
    from datasets.colmap import Parser, Dataset
    from fused_ssim import fused_ssim

    device = torch.device("cuda:0")
    ledger = args.out.parent / "call_budget_ledger.csv"
    with ledger.open(newline="", encoding="utf-8") as stream:
        charged = sum(int(row["charged"]) for row in csv.DictReader(stream))
    if charged not in (1, 5, 10):
        raise RuntimeError("P04T real-state preflight has an unapproved retry boundary")
    count = charged
    def charge(kind, fn):
        nonlocal count
        if count >= 17:
            raise RuntimeError("P04T preflight allocation exceeded")
        count += 1
        with ledger.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow((count, 0, -1, kind, 1, "RESERVED", ""))
            stream.flush(); os.fsync(stream.fileno())
        try:
            result = fn()
            torch.cuda.synchronize()
            status = "SUCCESS"
            return result
        except Exception:
            status = "FAILED_CHARGED"
            raise
        finally:
            with ledger.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow((count, 0, -1, kind, 0, status, ""))

    split = json.loads((OLD / "dataset_split.json").read_text())
    source = Parser(str(DATA), factor=4, normalize=True, test_every=8)
    trainset = Dataset(source, split="train", val_every=0,
                       train_keyword=split["train_keyword"], test_keyword=split["test_keyword"])
    if [source.image_names[int(i)] for i in trainset.indices] != split["train"]:
        raise RuntimeError("P04T original Android TRAIN split changed")
    datum = trainset[0]
    camera = datum["camtoworld"][None].to(device)
    K = datum["K"][None].to(device)
    pixels = datum["image"][None].to(device).float() / 255.0
    height, width = pixels.shape[1:3]
    feature = FeatureCache(FEATURES, expected_weight_sha256=WEIGHT_SHA)
    head = StaticResponsibilityHead().to(device)
    head.load_state_dict(torch.load(OLD / "aux/mask_head_step29999.pt", map_location="cpu", weights_only=True))
    head.eval().requires_grad_(False)
    name = source.image_names[int(trainset.indices[0])]
    with torch.no_grad():
        grid = charge("real_state_mask_head_forward", lambda: head(feature.load(name, 36)[None].to(device)))
        probability = F.interpolate(grid, size=(height, width), mode="bilinear", align_corners=False)
        mask = hard_static_mask(probability, threshold=.25, kernel_size=7)[0, 0].bool()
    ckpt = torch.load(OLD / "ckpts/ckpt_29999_rank0.pt", map_location="cpu", weights_only=True)
    splats = torch.nn.ParameterDict({key: torch.nn.Parameter(value.to(device))
                                     for key, value in ckpt["splats"].items()})
    cfg = SimpleNamespace(near_plane=.01, far_plane=1e10, antialiased=False,
                          camera_model="pinhole", ssim_lambda=.2)
    fake = SimpleNamespace(runner=SimpleNamespace(splats=splats,
        ru_training=SimpleNamespace(head=head), strategy_state={"count": None}, cfg=cfg))
    fake._call = lambda step, kind, fn: charge("real_state_" + kind, fn)
    fake._rng = P04TMonitor._rng.__get__(fake)
    fake._rng_equal = P04TMonitor._rng_equal
    fake._live_signature = P04TMonitor._live_signature.__get__(fake)
    fake._region = P04TMonitor._region

    def render():
        return rasterization(
            means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
            viewmats=torch.linalg.inv(camera), Ks=K, width=width, height=height,
            packed=False, absgrad=True, sh_degree=3, near_plane=.01, far_plane=1e10)
    rgb, _, info = charge("real_state_A_render_forward", render)
    loss, _, _ = masked_photo_loss(rgb, pixels, mask[None, None].float(),
                                   fused_ssim_fn=fused_ssim, ssim_lambda=.2)
    charge("real_state_A_photo_backward", lambda: loss.backward())
    ids_cpu = P04TMonitor._choose_ids(len(splats["means"]), "W1")
    ids = torch.as_tensor(ids_cpu, device=device)
    zero = torch.zeros_like(mask)
    values, _ = charge("real_state_A_exact_probe_forward", lambda: run_probe(
        info, {"A": mask, "C": zero, "D": mask}, ids, device))
    z = (info["radii"][0, ids] > 0).all(-1).cpu().numpy().astype("uint8")
    d = P04TMonitor._pollution(fake, 10901, "W1", ids, pixels, camera, K, 3, mask, z, values[0])
    output = {"status": "P04T_REAL_STATE_D_ISOLATION_PASS", "source": "historical Android RU step29999; pretraining verification only",
              "image_name": name, "sampled_gaussians": len(ids_cpu),
              "A_projected_count": int(z.sum()), "D_region_pixels": d["region_pixels"],
              "D_region_contribution_positive": int((d["br"] > 1e-12).sum()),
              "live_state_sha256_before_after": d["live_signature"],
              "charged_extra_calls_total": count}
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output["status"])


if __name__ == "__main__":
    main()
