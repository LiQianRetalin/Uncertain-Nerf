"""Descriptive cache-only attribution; no new selection or pass threshold."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.p04t_analyze import EPS, score, top_ids


def quantiles(values):
    values = np.asarray(values, np.float64)
    return {"median": float(np.median(values)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for window in ("W1", "W2"):
        x = np.load(args.work / f"{window}_per_step_probe.npz", allow_pickle=False)
        ids = x["ids"]
        native = score(x["z"], x["g"], x["b"], x["bm"], x["m"], x["camera"])
        eligible = native["n"] > 0
        k = max(1, math.ceil(int(eligible.sum()) * .1))
        top = {key: set(top_ids(ids, native[key], eligible, k))
               for key in ("G0", "pixel", "react", "simple", "oac", "G0x2")}
        idx = {int(gid): j for j, gid in enumerate(ids)}
        only_oac = np.array([idx[gid] for gid in top["oac"] - top["simple"]], dtype=np.int64)
        only_simple = np.array([idx[gid] for gid in top["simple"] - top["oac"]], dtype=np.int64)
        n, c, A = native["n"], native["c"], native["A"]
        visibility_factor = (n + 4) / (c + 4)
        rejection_factor = (c + 4) / (A + 4)
        def group(indices):
            if not len(indices):
                return {"count": 0}
            return {"count": len(indices),
                    "n_minus_c": quantiles(n[indices] - c[indices]),
                    "c_minus_A": quantiles(c[indices] - A[indices]),
                    "visibility_factor": quantiles(visibility_factor[indices]),
                    "rejection_factor": quantiles(rejection_factor[indices]),
                    "G0": quantiles(native["G0"][indices]),
                    "oac": quantiles(native["oac"][indices]),
                    "simple": quantiles(native["simple"][indices])}
        affected = (x["d_br"] > EPS).any(axis=0) & eligible
        low = A < 1
        delta = {}
        sd, sg, sb, sbm, sm = [a.copy() for a in (x["z"], x["g"], x["b"], x["bm"], x["m"])]
        for j, step in enumerate(x["d_steps"]):
            i = int(step - x["steps"][0])
            sd[i], sg[i], sb[i], sbm[i], sm[i] = (
                x["d_z"][j], x["d_g"][j], x["d_b"][j], x["d_bm"][j], x["d_m"][j])
        shadow = score(sd, sg, sb, sbm, sm, x["camera"])
        for key in ("G0", "simple", "oac"):
            difference = shadow[key] - native[key]
            delta[key] = {"affected_abs": quantiles(np.abs(difference[affected])),
                          "low_evidence_affected_abs": quantiles(np.abs(difference[affected & low])),
                          "affected_positive_count": int((difference[affected] > 0).sum()),
                          "affected_negative_count": int((difference[affected] < 0).sum())}
        extra = (shadow["oac"] - native["oac"]) - (shadow["G0"] - native["G0"])
        delta["OAC_minus_G0_extra"] = {
            "affected_abs": quantiles(np.abs(extra[affected])),
            "low_evidence_affected_abs": quantiles(np.abs(extra[affected & low]))}
        sorted_extra = sorted(((abs(float(extra[j])), int(ids[j]), float(extra[j]),
                                float(shadow["oac"][j] - native["oac"][j]),
                                float(shadow["G0"][j] - native["G0"][j]))
                               for j in np.where(affected)[0]), reverse=True)[:5]
        result[window] = {"same_K": k, "topK_overlap": {
            "OAC_G0": len(top["oac"] & top["G0"]),
            "OAC_G0x2": len(top["oac"] & top["G0x2"]),
            "simple_G0": len(top["simple"] & top["G0"]),
            "OAC_simple": len(top["oac"] & top["simple"]),
            "OAC_pixel": len(top["oac"] & top["pixel"]),
            "OAC_react": len(top["oac"] & top["react"])},
            "oac_only_vs_simple": group(only_oac),
            "simple_only_vs_oac": group(only_simple),
            "affected_count": int(affected.sum()),
            "affected_low_evidence_A_lt_1": int((affected & low).sum()),
            "delta": delta,
            "top5_abs_extra_ids": [{"id": item[1], "delta_OAC_minus_delta_G0": item[2],
                                    "delta_OAC": item[3], "delta_G0": item[4]}
                                   for item in sorted_extra]}
    (args.work / "interpretation.json").write_text(json.dumps(result, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("P04T_INTERPRETATION_PASS")


if __name__ == "__main__":
    main()
