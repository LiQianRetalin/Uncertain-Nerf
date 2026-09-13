"""P04 terminal-state, no-update, bounded exact-contribution probe.

The diagnostic is independent of the production training path.  It uses the
frozen gsplat projection/tile ordering and scans every front occluder while
accumulating only a deterministic 512-Gaussian probe.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from gsplat import rasterization
from puri_gs.dino_features import FeatureCache
from puri_gs.semantic_mask import StaticResponsibilityHead, hard_static_mask, masked_photo_loss


ROOT = Path("/home/chenglong/Uncertain-Nerf/uncertain-nerf")
GSPLAT = ROOT / "external/gsplat-v1.5.3-ru"
SCENES = {
    "android": (ROOT / "logs-puri/ru-generalization-rerun-9e292309/android_ru_30k",
                ROOT / "data/nerf_robustnerf/robustnerf/android",
                ROOT / "data/PURI-GS-derived/semantic_features/android"),
    "room": (ROOT / "logs-puri/phase_r/room_ru_30k",
             ROOT / "data/mipnerf360/360_v2/room",
             ROOT / "data/PURI-GS-derived/semantic_features/room"),
}
WEIGHT_SHA = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"
MAX_CALLS = 300
PROBE_N = 512


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Ledger:
    fields = ["seq", "scene", "view", "condition", "kind", "planned", "actual_success", "charged", "status", "seconds"]

    def __init__(self, directory: Path):
        self.path = directory / "call_budget_ledger.csv"
        self.rows = list(csv.DictReader(self.path.open(newline="", encoding="utf-8"))) if self.path.exists() else []
        self.scene = self.view = self.condition = ""

    def flush(self):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def call(self, kind, fn):
        if sum(int(row["charged"]) for row in self.rows) >= MAX_CALLS:
            raise RuntimeError("P04_CALL_BUDGET_EXHAUSTED")
        row = dict(seq=len(self.rows) + 1, scene=self.scene, view=self.view,
                   condition=self.condition, kind=kind, planned=1, actual_success=0,
                   charged=1, status="RESERVED_CONSERVATIVELY", seconds="")
        self.rows.append(row)
        self.flush()  # pre-debit before any forward or backward
        start = time.perf_counter()
        try:
            value = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            row["actual_success"] = 1
            row["status"] = "SUCCESS"
            return value
        except Exception as error:
            row["status"] = "FAILED_CHARGED:" + type(error).__name__
            raise
        finally:
            row["seconds"] = round(time.perf_counter() - start, 6)
            self.flush()


@triton.jit
def exact_probe(Means, Conics, Opacities, Flat, Offsets, Select, MA, MC, MD,
                B, BA, BC, BD, Count, Alpha, W: tl.constexpr, H: tl.constexpr,
                TW: tl.constexpr, NI: tl.constexpr):
    tile = tl.program_id(0)
    lane = tl.arange(0, 256)
    x = (tile % TW) * 16 + lane % 16
    y = (tile // TW) * 16 + lane // 16
    inside = (x < W) & (y < H)
    pixel = y * W + x
    start = tl.load(Offsets + tile)
    end = tl.load(Offsets + tile + 1, mask=tile + 1 < tl.num_programs(0), other=NI)
    trans = tl.full((256,), 1.0, tl.float32)
    active = inside
    ma = tl.load(MA + pixel, mask=inside, other=0).to(tl.float32)
    mc = tl.load(MC + pixel, mask=inside, other=0).to(tl.float32)
    md = tl.load(MD + pixel, mask=inside, other=0).to(tl.float32)
    k = start
    while (k < end) & (tl.sum(active.to(tl.int32), 0) > 0):
        gid = tl.load(Flat + k)
        mx = tl.load(Means + gid * 2)
        my = tl.load(Means + gid * 2 + 1)
        cx = tl.load(Conics + gid * 3)
        cy = tl.load(Conics + gid * 3 + 1)
        cz = tl.load(Conics + gid * 3 + 2)
        opacity = tl.load(Opacities + gid)
        dx = mx - (x.to(tl.float32) + 0.5)
        dy = my - (y.to(tl.float32) + 0.5)
        sigma = 0.5 * (cx * dx * dx + cz * dy * dy) + cy * dx * dy
        alpha = tl.minimum(0.999, opacity * tl.exp(-sigma))
        eligible = active & (sigma >= 0.0) & (alpha >= (1.0 / 255.0))
        next_trans = trans * (1.0 - alpha)
        contributes = eligible & (next_trans > 1.0e-4)
        weight = tl.where(contributes, trans * alpha, 0.0)
        trans = tl.where(contributes, next_trans, trans)
        active = active & ~(eligible & (next_trans <= 1.0e-4))
        selected = tl.load(Select + gid)
        if selected >= 0:
            tl.atomic_add(B + selected, tl.sum(weight, 0))
            tl.atomic_add(BA + selected, tl.sum(weight * ma, 0))
            tl.atomic_add(BC + selected, tl.sum(weight * mc, 0))
            tl.atomic_add(BD + selected, tl.sum(weight * md, 0))
            tl.atomic_add(Count + selected, tl.sum(contributes.to(tl.float32), 0))
        k += 1
    tl.store(Alpha + pixel, 1.0 - trans, mask=inside)


def run_probe(info, masks, selected_ids, device):
    height, width = masks["A"].shape
    n = info["means2d"].shape[1]
    select = torch.full((n,), -1, device=device, dtype=torch.int32)
    select[selected_ids] = torch.arange(len(selected_ids), device=device, dtype=torch.int32)
    outputs = [torch.zeros(len(selected_ids), device=device, dtype=torch.float32) for _ in range(5)]
    alpha = torch.empty(height * width, device=device, dtype=torch.float32)
    offsets = info["isect_offsets"].reshape(-1)
    flat = info["flatten_ids"]
    exact_probe[(offsets.numel(),)](
        info["means2d"].contiguous(), info["conics"].contiguous(),
        info["opacities"].contiguous(), flat.contiguous(), offsets.contiguous(), select,
        *[masks[c].to(torch.int8).contiguous().flatten() for c in ("A", "C", "D")],
        *outputs, alpha, width, height, info["tile_width"], flat.numel(),
    )
    return [x.cpu().numpy() for x in outputs], alpha.reshape(height, width)


def scene_preflight(scene, count):
    run, data, features = SCENES[scene]
    split = json.loads((run / "dataset_split.json").read_text())
    names = sorted(split["train"])
    indices = [math.floor(j * (len(names) - 1) / (count - 1)) for j in range(count)]
    assert len(set(indices)) == count
    ckpt = run / "ckpts/ckpt_29999_rank0.pt"
    head = run / "aux/mask_head_step29999.pt"
    hist = run / "aux/residual_hist_step29999.pt"
    assert all(path.is_file() for path in (ckpt, head, hist, run / "cfg.yml", features / "manifest.json"))
    sys.path.insert(0, str(GSPLAT / "examples"))
    from datasets.colmap import Parser, Dataset
    source = Parser(str(data), factor=4, normalize=True, test_every=8)
    trainset = Dataset(source, split="train", val_every=0,
                       train_keyword=split["train_keyword"], test_keyword=split["test_keyword"])
    by_name = {source.image_names[int(i)]: int(i) for i in trainset.indices}
    assert set(by_name) == set(names)
    chosen = [dict(sorted_train_index=i, image_name=names[i],
                   image_sha256=sha(Path(source.image_paths[by_name[names[i]]]))) for i in indices]
    return dict(scene=scene, source_run=str(run), data_dir=str(data), feature_cache=str(features),
                source_step=29999, state_grade="MATCHED_TERMINAL_STATE",
                checkpoint_sha256=sha(ckpt), mask_head_sha256=sha(head), residual_hist_sha256=sha(hist),
                cfg_sha256=sha(run / "cfg.yml"), split_sha256=sha(run / "dataset_split.json"),
                selected=chosen,
                train_count=len(names))


def prepare(directory, count):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "diagnostic_manifest.json"
    if target.exists():
        raise RuntimeError("manifest already exists; do not reselect after seeing results")
    scenes = [scene_preflight(scene, count) for scene in SCENES]
    manifest = dict(schema="p04-oac-terminal-probe-v1", status="PREPARED", seed=42,
                    diagnostic_code_sha256=sha(Path(__file__)),
                    frozen_count_per_scene=count, probe_size=PROBE_N,
                    selection_rule="floor(j*(N-1)/(count-1)); sorted TRAIN filenames",
                    probe_rule="first selected view: projected radii>0, 512 evenly spaced IDs",
                    conditions={"A":"actual terminal RU hard Mask", "B":"all accepted",
                                "C":"A times 8x8 checkerboard, parity SHA256(seed42+filename)",
                                "D":"C plus central 20% x 20% A-accepted C-deleted pixels on first view; RGB magenta"},
                    score=dict(kappa=4, cap=2, min_distinct_cameras=3, b_threshold=1e-6,
                               u_threshold=0.05, zero_weight_fallback="G0", top_fraction=0.10),
                    max_forward_backward_calls=MAX_CALLS, new_training=0, parameter_updates=0,
                    scenes=scenes)
    save_json(target, manifest)
    Ledger(directory).flush()
    save_json(directory / "status.json", dict(stage="PREPARED", charged_calls=0,
                                              next_checkpoint="exact probe reference and first scene"))
    return manifest


def masks_for(head, cache, name, height, width, first, device, ledger):
    feature = cache.load(name, 36).unsqueeze(0).to(device)
    with torch.no_grad():
        grid = ledger.call("mask_head_forward", lambda: head(feature))
        probability = F.interpolate(grid, size=(height, width), mode="bilinear", align_corners=False)
        accepted = hard_static_mask(probability, threshold=0.25, kernel_size=7)[0, 0].bool()
    parity = hashlib.sha256(("42:" + name).encode()).digest()[0] & 1
    ys = torch.arange(height, device=device)[:, None] * 8 // height
    xs = torch.arange(width, device=device)[None, :] * 8 // width
    c = accepted & (((ys + xs + parity) & 1) == 0)
    d = c.clone()
    roi_pixels = 0
    if first:
        roi = torch.zeros_like(d)
        y0, y1 = int(height * .4), int(height * .6)
        x0, x1 = int(width * .4), int(width * .6)
        roi[y0:y1, x0:x1] = True
        injected = roi & accepted & ~c
        d |= injected
        roi_pixels = int(injected.sum().item())
    cache._loaded.pop(name, None)
    return {"A":accepted, "B":torch.ones_like(accepted), "C":c, "D":d}, roi_pixels


def scores(rows, cond):
    ids = rows[0]["ids"]
    z = np.stack([r["z"] for r in rows]).astype(np.float64)
    g = np.stack([r["g"][cond] for r in rows]).astype(np.float64)
    b = np.stack([r["b"] for r in rows]).astype(np.float64)
    bm = np.stack([r["bm"][cond] for r in rows]).astype(np.float64)
    m = np.stack([r["m"] for r in rows]).astype(np.float64)
    u = np.divide(bm, b, out=np.zeros_like(bm), where=b > 1e-12)
    n = z.sum(0)
    A = (z * u).sum(0)
    G0 = np.divide((z*g).sum(0), n, out=np.zeros_like(n), where=n > 0)
    valid_cameras = ((b > 1e-6) & (u >= .05) & (z > 0)).sum(0)
    opportunity = np.divide((z*g).sum(0), A, out=np.zeros_like(A), where=A > 1e-12)
    oac = np.where((A > 1e-12) & (valid_cameras >= 3),
                   np.minimum(2*G0, G0 + A/(A+4)*np.maximum(opportunity-G0, 0)), G0)
    def weighted(w):
        den = (z*w).sum(0)
        return np.divide((z*w*g).sum(0), den, out=G0.copy(), where=den > 1e-12)
    react_a = np.divide(b, m, out=np.zeros_like(b), where=m > 0)
    return dict(ids=ids, n=n, A=A, G0=G0, pixel=weighted(m), react=weighted(react_a),
                simple=weighted(react_a*u), oac=oac, support=valid_cameras,
                u=u, b=b, m=m, low_evidence=(A <= 1e-12) & (G0 > 0))


def run_scene(scene_info, directory, ledger):
    scene = scene_info["scene"]
    run, data_dir, feature_dir = SCENES[scene]
    ledger.scene = scene
    sys.path.insert(0, str(GSPLAT / "examples"))
    from datasets.colmap import Parser, Dataset
    from fused_ssim import fused_ssim
    cfg = json.loads((run / "config.yaml").read_text())
    split = json.loads((run / "dataset_split.json").read_text())
    assert cfg["training"]["data_factor"] == 4 and cfg["seed"] == 42
    parser = Parser(str(data_dir), factor=4, normalize=True, test_every=8)
    trainset = Dataset(parser, split="train", val_every=0,
                       train_keyword=split["train_keyword"], test_keyword=split["test_keyword"])
    actual = {parser.image_names[int(i)] for i in trainset.indices}
    assert actual == set(split["train"])
    name_to_local = {parser.image_names[int(i)]: j for j, i in enumerate(trainset.indices)}
    device = torch.device("cuda:0")
    ckpt = torch.load(run / "ckpts/ckpt_29999_rank0.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 29999 and set(ckpt["splats"]) == {"means","scales","quats","opacities","sh0","shN"}
    splats = torch.nn.ParameterDict({k:torch.nn.Parameter(v.to(device)) for k,v in ckpt["splats"].items()})
    head = StaticResponsibilityHead().to(device)
    head.load_state_dict(torch.load(run / "aux/mask_head_step29999.pt", map_location="cpu", weights_only=True), strict=True)
    head.eval().requires_grad_(False)
    cache = FeatureCache(feature_dir, expected_weight_sha256=WEIGHT_SHA)
    values_hash = hashlib.sha256(b"".join(ckpt["splats"][k].numpy().tobytes() for k in sorted(ckpt["splats"]))).hexdigest()
    def render(data):
        camera = data["camtoworld"][None].to(device)
        K = data["K"][None].to(device)
        height, width = data["image"].shape[:2]
        return rasterization(splats["means"], splats["quats"], torch.exp(splats["scales"]),
                             torch.sigmoid(splats["opacities"]), torch.cat((splats["sh0"], splats["shN"]),1),
                             torch.linalg.inv(camera), K, width, height, packed=False, absgrad=True,
                             sh_degree=3, near_plane=.01, far_plane=1e10)
    rows = []
    selected_ids = None
    for position, item in enumerate(scene_info["selected"]):
        name = item["image_name"]
        ledger.view = name
        data = trainset[name_to_local[name]]
        image_path = Path(parser.image_paths[int(trainset.indices[name_to_local[name]])])
        assert item["image_sha256"] == sha(image_path)
        target = data["image"][None].to(device).float() / 255.0
        height, width = target.shape[1:3]
        masks, roi_pixels = masks_for(head, cache, name, height, width, position == 0, device, ledger)
        item["real_mask_accept_fraction"] = round(float(masks["A"].float().mean()), 6)
        item["d_injected_pixels"] = roi_pixels
        target_d = target.clone()
        if position == 0 and roi_pixels:
            injected = masks["D"] & ~masks["C"]
            target_d[0, injected] = target.new_tensor([1.,0.,1.])
        ledger.condition = "geometry"
        render_rgb, render_alpha, info = ledger.call("fine_forward", lambda: render(data))
        if selected_ids is None:
            visible = torch.where((info["radii"][0] > 0).all(-1))[0]
            assert len(visible) >= PROBE_N
            selection = torch.linspace(0, len(visible)-1, PROBE_N, device=device).long()
            selected_ids = visible[selection]
            scene_info["probe_ids_sha256"] = hashlib.sha256(selected_ids.cpu().numpy().tobytes()).hexdigest()
        probe, alpha_ref = ledger.call("exact_probe_forward", lambda: run_probe(info, masks, selected_ids, device))
        alpha_delta = (alpha_ref - render_alpha[0,:,:,0]).abs()
        alpha_error = float(alpha_delta.max().item())
        if position == 0:
            save_json(directory / f"alpha_validation_{scene}.json",
                      dict(max_abs=alpha_error, mean_abs=float(alpha_delta.mean()),
                           q99_abs=float(torch.quantile(alpha_delta.flatten(),.99)),
                           pixels_over_2e4=int((alpha_delta>2e-4).sum()),
                           total_pixels=height*width))
        # CUDA __expf and Triton exp differ at rare alpha/early-stop boundaries.
        # Require distributional agreement and explicitly retain the exceptions.
        outliers = int((alpha_delta > 2e-4).sum().item())
        if (alpha_error > 2e-3 or float(alpha_delta.mean()) > 1e-6
                or outliers > max(1, math.ceil(height*width*1e-5))):
            raise RuntimeError(f"exact probe alpha mismatch: max={alpha_error} outliers={outliers}")
        item["exact_probe_alpha_max_error"] = alpha_error
        item["exact_probe_alpha_outliers_over_2e4"] = outliers
        b, ba, bc, bd, m = probe
        z = (info["radii"][0, selected_ids] > 0).all(-1).cpu().numpy().astype(np.uint8)
        gradient = {}
        for cond in ("A","B","C","D"):
            ledger.condition = cond
            if cond != "A":
                render_rgb, _, info = ledger.call("fine_forward", lambda: render(data))
            for param in splats.values():
                param.grad = None
            loss, _, _ = masked_photo_loss(render_rgb, target_d if cond == "D" else target,
                                           masks[cond][None,None].float(), fused_ssim_fn=fused_ssim)
            ledger.call("photo_backward", lambda: loss.backward())
            absgrad = info["means2d"].absgrad[0,selected_ids].detach().clone()
            absgrad[:,0] *= width/2.0
            absgrad[:,1] *= height/2.0
            gradient[cond] = torch.linalg.vector_norm(absgrad, dim=-1).cpu().numpy()
        rows.append(dict(scene=scene, name=name, ids=selected_ids.cpu().numpy(), z=z, g=gradient,
                         b=b, bm={"A":ba,"B":b,"C":bc,"D":bd}, m=m,
                         mask_accept={c:float(v.float().mean()) for c,v in masks.items()},
                         roi_pixels=roi_pixels))
        charged = sum(int(row["charged"]) for row in ledger.rows)
        save_json(directory / "status.json", dict(stage="RUNNING", scene=scene, last_view=name,
                    charged_calls=charged, remaining_calls=MAX_CALLS-charged,
                    next_checkpoint="next frozen TRAIN view or score aggregation"))
    current_hash = hashlib.sha256(b"".join(splats[k].detach().cpu().numpy().tobytes() for k in sorted(splats))).hexdigest()
    if current_hash != values_hash:
        raise RuntimeError("GAUSSIAN_PARAMETERS_CHANGED")
    scene_info["parameter_tensor_sha256_before"] = values_hash
    scene_info["parameter_tensor_sha256_after"] = current_hash
    scene_info["exact_probe_scope"] = "512 IDs projected in first selected TRAIN view; all front occluders scanned"
    return rows


def summarize(all_rows, directory, manifest):
    detail = []
    summary = []
    for scene_info in manifest["scenes"]:
        scene = scene_info["scene"]
        rows = [r for r in all_rows if r["scene"] == scene]
        calculated = {c:scores(rows,c) for c in ("A","B","C","D")}
        eligible = calculated["A"]["n"] > 0
        k = max(1,math.ceil(int(eligible.sum())*.1))
        for cond, result in calculated.items():
            order = np.lexsort((result["ids"], -result["oac"]))
            top = [int(result["ids"][j]) for j in order if eligible[j]][:k]
            summary.append(dict(scene=scene, condition=cond, state_grade="MATCHED_TERMINAL_STATE",
                                scope="512-probe", views=len(rows), candidates=int(eligible.sum()), top_k=k,
                                mean_n=float(result["n"][eligible].mean()),
                                mean_A=float(result["A"][eligible].mean()),
                                mean_u=float(result["u"][:,eligible].mean()),
                                positive_g_zero_A=int(result["low_evidence"].sum()),
                                support_gate_pass=int(((result["support"]>=3)&eligible).sum()),
                                cap_active=int(((result["oac"]>=2*result["G0"]-1e-12)&(result["oac"]>result["G0"]+1e-12)&eligible).sum()),
                                oac_top_ids=";".join(map(str,top))))
            for j,gid in enumerate(result["ids"]):
                for t,row in enumerate(rows):
                    detail.append(dict(scene=scene, condition=cond, view=row["name"], gaussian_id=int(gid),
                                       z=int(row["z"][j]), g=float(row["g"][cond][j]),
                                       b=float(row["b"][j]), bm=float(row["bm"][cond][j]),
                                       m=int(row["m"][j]), u=float(result["u"][t,j]),
                                       n=int(result["n"][j]), A=float(result["A"][j]),
                                       G0=float(result["G0"][j]), pixel=float(result["pixel"][j]),
                                       react=float(result["react"][j]), simple=float(result["simple"][j]),
                                       oac=float(result["oac"][j]), support=int(result["support"][j])))
        scene_info["top_k"] = k
    with (directory / "diagnostic_summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with gzip.open(directory / "per_view_gaussian.csv.gz","wt",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(detail[0]))
        writer.writeheader(); writer.writerows(detail)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run"))
    parser.add_argument("--out",type=Path,default=Path("/home/chenglong/P04-work"))
    parser.add_argument("--views",type=int,choices=(4,6,8),default=8)
    parser.add_argument("--resume-charged",action="store_true",
                        help="manual retry after a charged technical failure; never resets ledger")
    args=parser.parse_args()
    if args.mode=="prepare":
        prepare(args.out,args.views)
        return
    manifest_path=args.out/"diagnostic_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"]!="PREPARED":
        raise RuntimeError("P04 run already consumed; no automatic retry")
    ledger=Ledger(args.out)
    if ledger.rows:
        previous=json.loads((args.out/"status.json").read_text(encoding="utf-8"))
        if not args.resume_charged or previous["stage"]!="FAILED_PARTIAL" or any(
                row["kind"]=="photo_backward" and row["actual_success"]=="1" for row in ledger.rows):
            raise RuntimeError("charged calls exist; this retry is only for pre-score technical failure")
        old=manifest["diagnostic_code_sha256"]
        manifest["diagnostic_code_revisions"]=manifest.get("diagnostic_code_revisions",[old])+[sha(Path(__file__))]
        manifest["diagnostic_code_sha256"]=sha(Path(__file__))
        manifest["pre_render_technical_failure_charged_calls"]=sum(int(row["charged"]) for row in ledger.rows)
        save_json(manifest_path,manifest)
    all_rows=[]
    try:
        for scene_info in manifest["scenes"]:
            all_rows.extend(run_scene(scene_info,args.out,ledger))
        summarize(all_rows,args.out,manifest)
        manifest["status"]="DIAGNOSTIC_COMPLETE"
        save_json(manifest_path,manifest)
        charged=sum(int(row["charged"]) for row in ledger.rows)
        save_json(args.out/"status.json",dict(stage="DIAGNOSTIC_COMPLETE",charged_calls=charged,
                                              remaining_calls=MAX_CALLS-charged,next_checkpoint="report and independent audit"))
    except Exception as error:
        charged=sum(int(row["charged"]) for row in ledger.rows)
        save_json(args.out/"status.json",dict(stage="FAILED_PARTIAL",reason=str(error)[:500],charged_calls=charged,
                                              remaining_calls=MAX_CALLS-charged,next_checkpoint="manual evidence audit; no automatic retry"))
        raise


if __name__=="__main__":
    main()
