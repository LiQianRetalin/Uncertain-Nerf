"""Cache-only P04-T two-window score and pollution-risk analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EPS = 1e-12
SUPPORT_B = 1e-6
SUPPORT_U = .05
KAPPA = 4.0
CAP = 2.0
NATIVE_THRESHOLD = .0006
SCORES = ("G0", "pixel", "react", "simple", "oac", "G0x2")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def div(a, b, fallback=0.0):
    out = np.full(np.shape(a), fallback, dtype=np.float64)
    return np.divide(a, b, out=out, where=b > EPS)


def score(z, g, b, bm, m, cameras):
    z = z.astype(bool)
    g = g.astype(np.float64)
    b = b.astype(np.float64)
    bm = bm.astype(np.float64)
    m = m.astype(np.float64)
    u = div(bm, b)
    n = z.sum(0).astype(np.float64)
    c = (z & (b > EPS)).sum(0).astype(np.float64)
    A = (z * u).sum(0)
    numerator = (z * g).sum(0)
    G0 = div(numerator, n)
    a = div(b, m)
    def weighted(w):
        denominator = (z * w).sum(0)
        return np.divide((z * w * g).sum(0), denominator,
                         out=G0.copy(), where=denominator > EPS)
    support = np.zeros(z.shape[1], np.int32)
    for j in range(z.shape[1]):
        used = z[:, j] & (b[:, j] > SUPPORT_B) & (u[:, j] >= SUPPORT_U)
        support[j] = len(np.unique(cameras[used]))
    opportunity = div(numerator, A)
    raw = G0 + A / (A + KAPPA) * np.maximum(opportunity - G0, 0)
    gate = (A > EPS) & (support >= 3)
    oac = np.where(gate, np.minimum(CAP * G0, raw), G0)
    return {"n": n, "c": c, "A": A, "u": u, "support": support,
            "cap": gate & (raw > CAP * G0 + EPS), "gate": gate,
            "G0": G0, "pixel": weighted(m), "react": weighted(a),
            "simple": weighted(a * u), "oac": oac, "G0x2": 2 * G0}


def top_ids(ids, scores, eligible, k):
    order = np.lexsort((ids, -scores))
    return [int(ids[j]) for j in order if eligible[j]][:k]


def q(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return [None] * 7
    return [float(v) for v in np.quantile(values, [0, .1, .25, .5, .75, .9, 1])]


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("no rows for " + str(path))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    status = json.loads((work / "status.json").read_text())
    if not (status["stage"] == "COMPLETE" and status["completed_step"] == 19999):
        raise RuntimeError("P04T complete 20k identity not available")
    all_summary = []
    all_risk = []
    overview = {}
    for window, start, end in (("W1", 10901, 11000), ("W2", 18901, 19000)):
        path = work / f"{window}_per_step_probe.npz"
        manifest = json.loads((work / f"{window}_data_manifest.json").read_text())
        if sha(path) != manifest["sha256"]:
            raise RuntimeError("P04T window cache hash mismatch")
        arrays = np.load(path, allow_pickle=False)
        ids = arrays["ids"]
        steps = arrays["steps"]
        cameras = arrays["camera"]
        if len(ids) != 512 or not np.array_equal(steps, np.arange(start, end + 1)):
            raise RuntimeError("P04T window boundaries/probe count changed")
        d_steps = arrays["d_steps"]
        if not np.array_equal(d_steps, np.array([start, start + 33, start + 66, start + 99])):
            raise RuntimeError("P04T pollution moments changed")
        z = arrays["z"]
        g = arrays["g"]
        b = arrays["b"]
        bm = arrays["bm"]
        m = arrays["m"]
        native = score(z, g, b, bm, m, cameras)
        sd, sg, sb, sbm, sm = [value.copy() for value in (z, g, b, bm, m)]
        for j, step in enumerate(d_steps):
            i = int(step - start)
            sd[i], sg[i], sb[i], sbm[i], sm[i] = (
                arrays["d_z"][j], arrays["d_g"][j], arrays["d_b"][j],
                arrays["d_bm"][j], arrays["d_m"][j])
        shadow = score(sd, sg, sb, sbm, sm, cameras)
        eligible = native["n"] > 0
        if not np.array_equal(eligible, shadow["n"] > 0):
            raise RuntimeError("P04T D changed geometric eligibility")
        k = max(1, math.ceil(int(eligible.sum()) * .1))
        affected = (arrays["d_br"] > EPS).any(axis=0)
        covered = np.any(arrays["d_br"] > EPS, axis=0)
        if not np.array_equal(affected, covered):
            raise AssertionError("risk coverage internal error")
        near = eligible & (np.abs(native["G0"] - NATIVE_THRESHOLD) <= NATIVE_THRESHOLD * .25)
        threshold = {}
        top = {}
        for method in SCORES:
            before = native[method] >= NATIVE_THRESHOLD
            after = shadow[method] >= NATIVE_THRESHOLD
            before_top = set(top_ids(ids, native[method], eligible, k))
            after_top = set(top_ids(ids, shadow[method], eligible, k))
            threshold[method] = {"A_above": int((before & eligible).sum()),
                                 "shadow_above": int((after & eligible).sum()),
                                 "enter": int((~before & after & eligible).sum()),
                                 "exit": int((before & ~after & eligible).sum()),
                                 "near_A_above": int((before & near).sum())}
            top[method] = {"overlap": len(before_top & after_top),
                           "enter": len(after_top - before_top),
                           "affected_enter": len({int(ids[j]) for j in np.where(affected)[0]} & (after_top - before_top))}
        native_oac_top = set(top_ids(ids, native["oac"], eligible, k))
        native_simple_top = set(top_ids(ids, native["simple"], eligible, k))
        shadow_oac_top = set(top_ids(ids, shadow["oac"], eligible, k))
        shadow_simple_top = set(top_ids(ids, shadow["simple"], eligible, k))
        ratio_oac = div(native["oac"], native["G0"])
        raw_u_over = np.maximum(native["u"] - 1, 0)
        overview[window] = {"steps": [start, end], "probe_count": len(ids),
            "eligible_count": int(eligible.sum()), "same_K": k,
            "camera_count_distinct": len(np.unique(cameras)),
            "A_n_quantiles": q(native["n"][eligible]), "A_opportunity_quantiles": q(native["A"][eligible]),
            "A_n_over_A_quantiles": q(div(native["n"], native["A"])[eligible & (native["A"] > EPS)]),
            "projection_no_contribution_total": int((native["n"] - native["c"]).sum()),
            "contribution_rejected_total": float((native["c"] - native["A"]).sum()),
            "support_gate_pass_A": int((native["gate"] & eligible).sum()),
            "cap_active_A": int((native["cap"] & eligible).sum()),
            "oac_over_G0_quantiles": q(ratio_oac[eligible & (native["G0"] > EPS)]),
            "oac_equals_G0x2_count": int((np.isclose(native["oac"], native["G0x2"], rtol=1e-5, atol=0)
                                            & native["cap"] & eligible).sum()),
            "raw_u_over_one_max": float(raw_u_over.max()), "u_clipped": False,
            "d_region_pixels_each": arrays["d_region_pixels"].tolist(),
            "d_region_gaussians_positive_each": (arrays["d_br"] > EPS).sum(axis=1).tolist(),
            "d_region_total_contribution_each": arrays["d_br"].sum(axis=1).tolist(),
            "truly_affected_probe_count": int(affected.sum()),
            "affected_projected_probe_count": int((affected & eligible).sum()),
            "threshold": threshold, "topK_A_to_shadow": top,
            "OAC_vs_simple_A_topK_overlap": len(native_oac_top & native_simple_top),
            "OAC_vs_simple_shadow_topK_overlap": len(shadow_oac_top & shadow_simple_top),
            "OAC_only_A_top_ids": sorted(native_oac_top - native_simple_top),
            "OAC_only_shadow_top_ids": sorted(shadow_oac_top - shadow_simple_top)}
        for condition, values in (("A", native), ("D_shadow", shadow)):
            row = {"window": window, "condition": condition,
                   "eligible": int(eligible.sum()), "same_top_k": k,
                   "mean_n": float(values["n"][eligible].mean()),
                   "mean_A": float(values["A"][eligible].mean()),
                   "n_minus_c": float((values["n"] - values["c"]).sum()),
                   "c_minus_A": float((values["c"] - values["A"]).sum()),
                   "support_gate_pass": int((values["gate"] & eligible).sum()),
                   "cap_active": int((values["cap"] & eligible).sum()),
                   "affected_probes": int(affected.sum())}
            for method in SCORES:
                row[f"{method}_above_0p0006"] = int(((values[method] >= NATIVE_THRESHOLD) & eligible).sum())
            all_summary.append(row)
        for j, gid in enumerate(ids):
            if not eligible[j]:
                continue
            d0 = shadow["G0"][j] - native["G0"][j]
            do = shadow["oac"][j] - native["oac"][j]
            ds = shadow["simple"][j] - native["simple"][j]
            all_risk.append({"window": window, "gaussian_id": int(gid),
                "affected_bR": int(affected[j]), "bR_total": float(arrays["d_br"][:, j].sum()),
                "A_n": float(native["n"][j]), "A_c": float(native["c"][j]),
                "A_opportunity": float(native["A"][j]), "D_opportunity": float(shadow["A"][j]),
                "A_support": int(native["support"][j]), "D_support": int(shadow["support"][j]),
                "A_cap": int(native["cap"][j]), "D_cap": int(shadow["cap"][j]),
                "A_G0": float(native["G0"][j]), "D_G0": float(shadow["G0"][j]),
                "A_simple": float(native["simple"][j]), "D_simple": float(shadow["simple"][j]),
                "A_oac": float(native["oac"][j]), "D_oac": float(shadow["oac"][j]),
                "delta_G0": float(d0), "delta_simple": float(ds), "delta_OAC": float(do),
                "delta_OAC_minus_delta_G0": float(do - d0),
                "A_low_evidence": int(native["A"][j] < 1),
                "D_low_evidence": int(shadow["A"][j] < 1),
                "A_OAC_above_threshold": int(native["oac"][j] >= NATIVE_THRESHOLD),
                "D_OAC_above_threshold": int(shadow["oac"][j] >= NATIVE_THRESHOLD),
                "A_OAC_topK": int(int(gid) in native_oac_top),
                "D_OAC_topK": int(int(gid) in shadow_oac_top)})
    write_csv(work / "window_score_summary.csv", all_summary)
    write_csv(work / "risk_attribution.csv", all_risk)
    (work / "analysis.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n", encoding="utf-8")
    print("P04T_CACHE_ANALYSIS_PASS")


if __name__ == "__main__":
    main()
