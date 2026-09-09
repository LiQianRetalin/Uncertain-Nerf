"""The fixed A/B/H roles and decision rules of the single-surface protocol."""
from __future__ import annotations

import math
import torch

from puri_gs.coverage_recoverability import require, CRITERIA

PROTOCOL = "V3_SINGLE_SURFACE_DIAGNOSTIC_V1"
A, B = "DSC07987.JPG", "DSC07989.JPG"
LRS = dict(means=1.962743359643857e-06, scales=.005, quats=.001, opacities=.05, sh0=.0025, shN=.000125)
HISTORY = {"HISTORICAL_V3_STATUS": "QUALITY_RECOVERY_FAIL / NO_GO",
           "HISTORICAL_FOUR_VIEW_DIAGNOSTIC_STATUS": "ROI_NOT_READY",
           "HISTORICAL_REPLAY_STATUS": "REPLAY_NOT_EQUIVALENT", "HISTORICAL_RECORDS_RECLASSIFIED": False}


def validate_config(cfg):
    expected = dict(protocol=PROTOCOL, target_view=A, context_view=B, source_step=29999,
        gaussian_count=2266597, candidate_limit=8, groups=["Va", "Vb", "O"], updates_per_group=400,
        evaluation_updates=[0, 400], progress_every=50, seed=42, sh_degree=3, learning_rates=LRS, criteria=CRITERIA)
    for key, value in expected.items():
        require(cfg.get(key) == value, f"single-surface fixed setting differs: {key}")
    require(cfg["source_checkpoint_sha256"] == "14563fbbcfbb722879929be31fd19ad5e6a02939e8d2409e6a6855ce909d417c", "wrong source")
    return cfg


def candidate_views(identity):
    require(len(identity["train_basenames"]) == 161 and len(identity["test_basenames"]) == 24, "split differs")
    require(not set(identity["train_basenames"]) & set(identity["test_basenames"]), "train/test overlap")
    cameras = identity["cameras"]
    require([r["basename"] for r in cameras] == identity["train_basenames"], "camera order differs")
    anchor = next(row for row in cameras if row["basename"] == A)
    center = [anchor["camtoworld"][j][3] for j in range(3)]
    rows = [{"image_name": c["basename"], "train_view_id": c["view_id"],
             "distance_to_A": math.sqrt(sum((c["camtoworld"][j][3] - center[j]) ** 2 for j in range(3)))}
            for c in cameras if c["basename"] not in (A, B)]
    require(all(math.isfinite(r["distance_to_A"]) for r in rows), "non-finite camera distance")
    return sorted(rows, key=lambda row: (row["distance_to_A"], row["train_view_id"]))[:8]


def validate_proposal(proposal, candidates):
    from puri_gs.coverage_recoverability import canonical_sha
    require(proposal["protocol"] == PROTOCOL, "proposal protocol differs")
    require(proposal["candidate_sha256"] == canonical_sha(candidates), "candidate order changed")
    names = [row["image_name"] for row in candidates]
    h = proposal["check_view"]
    require(h in names and h not in (A, B), "H must be in the fixed training candidate list")
    rank = names.index(h)
    require(proposal["candidates_reviewed"] == rank + 1, "H must be the first qualified candidate")
    exclusions = proposal.get("earlier_candidate_exclusions", {})
    require(set(exclusions) == set(names[:rank]) and all(isinstance(v, str) and v.strip() for v in exclusions.values()),
            "record each earlier candidate's original-image rejection reason")
    require(set(proposal["polygons"]) == {A, h}, "only A training ROI and H evaluation ROI have polygons")
    for polygons in proposal["polygons"].values():
        require(isinstance(polygons, list) and polygons and all(
            isinstance(polygon, list) and len(polygon) >= 3 and all(
                isinstance(point, list) and len(point) == 2 and all(type(v) is int for v in point)
                for point in polygon) for polygon in polygons), "ROI must be a list of polygons with integer pixel vertices")
    require(bool(proposal.get("surface_description")) and bool(proposal.get("correspondence_basis")), "missing static-surface evidence")
    return h


def validate_frozen(m, c, s, *, context=False):
    require(m.ndim == 2 and m.shape == c.shape == s.shape, "M/C/S dimensions differ")
    require(torch.isfinite(c).all() and (c >= 0).all() and (c <= 1).all(), "invalid float C")
    require(((m == 0) | (m == 1)).all() and s.dtype == torch.bool, "M/S must be binary")
    require(not bool(s.any()) if context else 0 < int(s.sum()) < s.numel(), "invalid A/B intervention ROI")
    return (1 - m.float()) * (1 - c) * s


def intervention_qualification(m, c, s, render, target):
    dq = validate_frozen(m, c, s)
    require(render.shape == target.shape == (*m.shape, 3) and torch.isfinite(render).all() and torch.isfinite(target).all(),
            "invalid initial float RGB")
    nonzero = int((dq > 0).sum())
    weight = float(dq.double().sum())
    residual = float((dq[..., None].double() * (render.double() - target.double()).abs()).sum())
    return {"proposal_status": "UNCONFIRMED_PROPOSAL", "delta_Q_nonzero_pixels": nonzero,
            "delta_Q_fraction_in_S": nonzero / int(s.sum()), "delta_Q_weight_sum": weight,
            "delta_Q_full_image_mean": weight / s.numel(), "delta_Q_mean_in_S": weight / int(s.sum()),
            "delta_Q_weighted_residual_sum": residual, "roi_pixels": int(s.sum()),
            "qualified": math.isfinite(weight) and math.isfinite(residual) and weight > 0 and residual > 0}


def view_metrics(render, target, alpha, *, role, roi=None, mask=None, support=None):
    require(role in ("A", "B", "H"), "invalid view role")
    require(render.ndim == 3 and render.shape[-1] == 3 and render.shape == target.shape and alpha.shape == render.shape[:2], "RGB/alpha shape mismatch")
    require(torch.isfinite(render).all() and torch.isfinite(target).all() and torch.isfinite(alpha).all(), "non-finite evaluation")
    error = (render - target).abs().mean(-1)
    result = dict(role=role, full_mae=float(error.double().mean()), full_pixels=error.numel(), roi_mae=None,
        roi_pixels=0, outside_mae=None, outside_pixels=0, accepted_mae=None, accepted_pixels=0,
        roi_alpha_mean=None, roi_low_alpha_fraction=None, delta_Q_mae=None, delta_Q_pixels=0)
    if role == "B":
        require(roi is None or not bool(roi.any()), "B must not have a target ROI")
    else:
        require(roi is not None and roi.dtype == torch.bool and roi.shape == error.shape and
                0 < int(roi.sum()) < roi.numel(), "invalid target/evaluation ROI")
        result.update(roi_mae=float(error[roi].double().mean()), roi_pixels=int(roi.sum()),
            outside_mae=float(error[~roi].double().mean()), outside_pixels=int((~roi).sum()),
            roi_alpha_mean=float(alpha[roi].double().mean()), roi_low_alpha_fraction=float((alpha[roi] <= .3).double().mean()))
    if role in ("A", "B"):
        require(mask is not None and mask.shape == error.shape and ((mask == 0) | (mask == 1)).all(), "missing original binary Mask")
        accepted = mask.bool()
        result.update(accepted_pixels=int(accepted.sum()), accepted_mae=float(error[accepted].double().mean()) if bool(accepted.any()) else None)
    if role == "A":
        active = validate_frozen(mask, support, roi) > 0
        result.update(delta_Q_pixels=int(active.sum()), delta_Q_mae=float(error[active].double().mean()) if bool(active.any()) else None)
    return result


def decide(metrics, h):
    try:
        require(h not in (A, B), "invalid H")
        for group in ("Va", "Vb", "O"):
            for update in ("0", "400"):
                require(set(metrics[group][update]) == {A, B, h}, "missing view")
                for name, role in ((A, "A"), (B, "B"), (h, "H")):
                    row, reference = metrics[group][update][name], metrics["Va"]["0"][name]
                    require(row["role"] == role, "role mismatch")
                    fields = ["full_mae"] + (["accepted_mae"] if role != "H" else [])
                    fields += ["roi_mae", "outside_mae", "roi_alpha_mean", "roi_low_alpha_fraction"] if role != "B" else []
                    fields += ["delta_Q_mae"] if role == "A" else []
                    for key in fields:
                        require(row[key] is not None and math.isfinite(row[key]) and row[key] >= 0, f"invalid {name}:{key}")
                    for key in ("full_pixels", "roi_pixels", "outside_pixels", "accepted_pixels", "delta_Q_pixels"):
                        require(row[key] == reference[key], "evaluation domains changed")
                    require(row["full_pixels"] > 0, "empty full image")
                    require(role == "H" or row["accepted_pixels"] > 0, "empty required accepted region")
                    if role == "B":
                        require(row["roi_pixels"] == 0 and row["roi_mae"] is None and row["roi_alpha_mean"] is None, "B empty ROI is not applicable")
                    else:
                        require(row["roi_pixels"] > 0 and row["outside_pixels"] > 0 and
                                row["roi_low_alpha_fraction"] <= 1 and row["roi_alpha_mean"] <= 1 + 1e-6, "invalid ROI metrics")
                    require(role != "A" or row["delta_Q_pixels"] > 0, "empty intervention domain")
        def comparison(name, field):
            starts = [metrics[g]["0"][name][field] for g in ("Va", "Vb", "O")]
            va, vb, o = [metrics[g]["400"][name][field] for g in ("Va", "Vb", "O")]
            e0, spread = sum(starts) / 3, max(starts) - min(starts)
            return dict(initial_values=starts, E0=e0, initial_range=spread, Va=va, Vb=vb, O=o,
                        n=max(abs(va-vb), spread), improvement_vs_initial=e0-o,
                        improvement_vs_better_control=min(va, vb)-o)
        local, check = comparison(A, "roi_mae"), comparison(h, "roi_mae")
        for row, fraction in ((local, .05), (check, .02)):
            row["threshold"] = max(fraction*row["E0"], 3*row["n"], 1e-4)
            row["gain"] = row["improvement_vs_initial"] >= row["threshold"] and row["improvement_vs_better_control"] >= row["threshold"]
        protected = {}
        for name, field in ((A, "outside_mae"), (A, "accepted_mae"), (B, "full_mae"),
                            (B, "accepted_mae"), (h, "roi_mae"), (h, "outside_mae")):
            row = comparison(name, field)
            row["d_initial"] = max(.01*row["E0"], 1e-4)
            row["b_control"] = max(row["d_initial"], 3*row["n"])
            row["initial_budget_pass"] = -row["improvement_vs_initial"] <= row["d_initial"]
            row["control_budget_pass"] = -row["improvement_vs_better_control"] <= row["b_control"]
            row["pass"] = row["initial_budget_pass"] and row["control_budget_pass"]
            protected[f"{name}:{field}"] = row
        protection = all(row["pass"] for row in protected.values())
        status = ("PROTECTION_NOT_ESTABLISHED" if not protection else
                  "LOCAL_RESPONSE_WITH_CHECK_VIEW_GAIN" if local["gain"] and check["gain"] else
                  "LOCAL_RESPONSE_WITH_PROTECTION" if local["gain"] else "NO_CLEAR_LOCAL_RESPONSE")
        return dict(status=status, LOCAL_GAIN=local["gain"], CHECK_VIEW_GAIN=check["gain"],
                    PROTECTION_PASS=protection, local=local, check=check, protected_regions=protected)
    except (KeyError, ValueError, TypeError) as error:
        return {"status": "DIAGNOSTIC_INVALID", "reason": str(error)}
