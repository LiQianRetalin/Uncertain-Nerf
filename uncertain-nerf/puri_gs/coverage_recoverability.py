"""Fixed local diagnostic mathematics, coverage accounting and decision rules."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from puri_gs.ru_part_v3 import static_rescue_l1
from puri_gs.semantic_mask import masked_photo_loss

CRITERIA = dict(local_fraction=.05, check_fraction=.02, protection_fraction=.01,
                repeat_margin_multiplier=3., absolute_floor=1e-4, low_alpha_threshold=.3)
OPTIMIZATION_VIEWS = ["DSC07987.JPG", "DSC07989.JPG"]
GROUPS = ["Va", "Vb", "O"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_config(config):
    expected = {"schema_version": 1, "protocol": "V3_COVERAGE_AND_LOCAL_RECOVERABILITY_DIAGNOSTIC",
                "source_step": 29999, "gaussian_count": 2266597, "optimization_views": OPTIMIZATION_VIEWS,
                "groups": GROUPS, "updates_per_group": 400, "evaluation_updates": [0, 100, 200, 400],
                "progress_every": 50, "seed": 42, "sh_degree": 3, "criteria": CRITERIA}
    for key, value in expected.items():
        require(config.get(key) == value, f"fixed diagnostic setting differs: {key}")
    require(config.get("source_checkpoint_sha256") ==
            "14563fbbcfbb722879929be31fd19ad5e6a02939e8d2409e6a6855ce909d417c", "source checkpoint differs")
    require(config.get("neighbor_preview_count") == 8, "preview budget differs")
    return config


def diagnostic_loss(render, target, mask, support, roi, *, group, fused_ssim_fn):
    require(group in GROUPS, "unknown group")
    require(roi.shape == mask.shape == support.shape, "M/C/S shape mismatch")
    effective = support.detach()
    if group == "O":
        effective = effective + (1 - effective) * roi.detach()
    base, _, _ = masked_photo_loss(render, target, mask.detach(), fused_ssim_fn=fused_ssim_fn)
    # step=500 enables the existing recovery function for every diagnostic update.
    extra = .8 * static_rescue_l1(render, target, mask, effective, step=500)
    return base + extra, base, extra


def coverage_row(name, support, mask=None, *, mask_reason=None):
    c = support.detach().double()
    require(c.ndim == 2 and torch.isfinite(c).all() and (c >= 0).all() and (c <= 1).all(), "invalid C")
    pixels = c.numel()
    row = dict(image_name=name, source_step=29999, pixels=pixels, C_nonzero_pixels=int((c > 0).sum()),
               C_weight_sum=float(c.sum()), C_nonzero_fraction=float((c > 0).double().mean()), mean_C=float(c.mean()),
               mask_available=mask is not None, mask_unavailable_reason=mask_reason,
               mask_rejected_pixels=None, mask_rejected_fraction=None, Q_nonzero_pixels=None,
               Q_weight_sum=None, Q_nonzero_fraction=None, mean_Q=None, rejected_mean_support=None,
               C_weight_in_accepted=None, C_weight_in_rejected=None, mean_Q_when_positive=None)
    if mask is not None:
        m = mask.detach().double()
        require(m.shape == c.shape and ((m == 0) | (m == 1)).all(), "invalid final binary M")
        q = (1 - m) * c
        rejected, positive = int((m == 0).sum()), int((q > 0).sum())
        qsum = float(q.sum())
        row.update(mask_rejected_pixels=rejected, mask_rejected_fraction=rejected / pixels,
                   Q_nonzero_pixels=positive, Q_weight_sum=qsum, Q_nonzero_fraction=positive / pixels,
                   mean_Q=qsum / pixels, rejected_mean_support=qsum / rejected if rejected else None,
                   C_weight_in_accepted=float((m * c).sum()), C_weight_in_rejected=qsum,
                   mean_Q_when_positive=qsum / positive if positive else None)
    return row


def aggregate_coverage(rows):
    result = {"views_with_C": len(rows), "views_with_M": sum(row["mask_available"] for row in rows),
              "equal_view_mean": {}, "pixel_weighted_mean": {}}
    for field in ("C_nonzero_fraction", "mean_C", "mask_rejected_fraction", "Q_nonzero_fraction", "mean_Q"):
        available = [row for row in rows if row[field] is not None]
        result["equal_view_mean"][field] = sum(row[field] for row in available) / len(available) if available else None
        result["pixel_weighted_mean"][field] = (sum(row[field] * row["pixels"] for row in available) /
                                                     sum(row["pixels"] for row in available)) if available else None
    result["missing_values_excluded"] = True
    available = [row for row in rows if row["mask_available"]]
    result["rejected_pixel_weighted_mean_Q"] = (sum(row["Q_weight_sum"] for row in available) /
        sum(row["mask_rejected_pixels"] for row in available)) if any(row["mask_rejected_pixels"] for row in available) else None
    result["positive_Q_pixel_weighted_mean_Q"] = (sum(row["Q_weight_sum"] for row in available) /
        sum(row["Q_nonzero_pixels"] for row in available)) if any(row["Q_nonzero_pixels"] for row in available) else None
    return result


def temporal_summary(records, final_checks=None):
    indexed = {int(row["step"]): row for row in records}
    require(len(indexed) == len(records), "duplicate progress steps")
    counts = [row["active_samples"] for _, row in sorted(indexed.items()) if "active_samples" in row]
    require(all(isinstance(v, int) and v >= 0 for v in counts) and counts == sorted(counts), "invalid cumulative history")
    require(all(row.get("mean_Q") is None or math.isfinite(row["mean_Q"]) and row["mean_Q"] >= 0
                for row in records), "invalid progress Q")
    def cumulative_at(step):
        row = indexed.get(step)
        if row is not None and "active_samples" in row:
            return int(row["active_samples"]), f"logged cumulative count at step {step}"
        row = indexed.get(step + 1)
        if row is not None and "active_samples" in row and row.get("mean_Q") is not None:
            # Q>=0: the next row's own active indicator is recoverable from its mean.
            return int(row["active_samples"]) - int(row["mean_Q"] > 0), f"step {step+1} cumulative minus that step's Q indicator"
        return None, "boundary count unavailable"
    windows = []
    for start, end in ((500, 9999), (10000, 19999), (20000, 29999)):
        selected = [row for step, row in sorted(indexed.items()) if start <= step <= end]
        left, left_source = cumulative_at(start - 1)
        right, right_source = cumulative_at(end)
        count = right - left if left is not None and right is not None else None
        if count is not None:
            require(0 <= count <= end - start + 1, "invalid cumulative activation counts")
        metrics = {}
        for key in ("mean_Q", "added_l1", "parent_base_loss"):
            values = [row[key] for row in selected if row.get(key) is not None]
            require(all(math.isfinite(v) for v in values), "non-finite progress metrics")
            metrics[key] = {"logged_samples": len(values), "logged_sample_mean": sum(values) / len(values) if values else None,
                            "whole_window_sum": sum(values) if len(values) == end - start + 1 else None}
        windows.append(dict(start=start, end=end, logged_rows=len(selected), active_updates=count,
                            active_count_basis=[left_source, right_source], metrics=metrics))
    return {"windows": windows, "sparse_points_are_not_full_window_means": True,
            "whole_run_active_updates": (final_checks or {}).get("active_samples"),
            "whole_run_added_loss_sum": (final_checks or {}).get("added_loss_sum")}


def rank_check_views(cameras, optimization_views=OPTIMIZATION_VIEWS):
    by_name = {row["basename"]: row for row in cameras}
    anchors = np.stack([np.asarray(by_name[name]["camtoworld"])[:3, 3] for name in optimization_views])
    rows = []
    for row in cameras:
        if row["basename"] in optimization_views:
            continue
        distances = np.linalg.norm(anchors - np.asarray(row["camtoworld"])[:3, 3], axis=1)
        rows.append({"image_name": row["basename"], "distance_to_nearest_optimization_camera": float(distances.min()),
                     "distances_to_optimization_cameras": distances.tolist(), "train_view_id": row["view_id"]})
    return sorted(rows, key=lambda row: (row["distance_to_nearest_optimization_camera"], row["train_view_id"]))


def polygon_mask(width, height, polygons):
    from PIL import Image, ImageDraw
    require(isinstance(polygons, list) and polygons, "ROI needs at least one polygon")
    image = Image.new("1", (width, height))
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        require(len(polygon) >= 3, "polygon needs three vertices")
        require(all(len(p) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in p)
                    and 0 <= p[0] < width and 0 <= p[1] < height for p in polygon), "invalid pixel coordinates")
        draw.polygon([tuple(p) for p in polygon], fill=1)
    result = np.asarray(image, dtype=np.bool_).copy()
    require(0 < int(result.sum()) < width * height, "ROI or outside protection region is empty")
    return result


def region_metrics(render, target, alpha, roi, mask=None):
    require(render.shape == target.shape and render.ndim == 3 and render.shape[-1] == 3, "invalid RGB")
    require(roi.dtype == torch.bool and roi.shape == render.shape[:2], "invalid ROI layout")
    require(alpha.shape == roi.shape and (mask is None or mask.shape == roi.shape), "invalid alpha/Mask layout")
    require(torch.isfinite(render).all() and torch.isfinite(target).all() and torch.isfinite(alpha).all(), "non-finite evaluation")
    error = (render - target).abs().mean(-1)  # no clamp before metrics
    outside = ~roi
    protected = outside & mask.bool() if mask is not None else outside
    fallback = mask is not None and not bool(protected.any())
    if fallback:
        protected = outside
    def mean(values, selected):
        return float(values[selected].double().mean()) if bool(selected.any()) else None
    return {"roi_mae": mean(error, roi), "protection_mae": mean(error, protected), "full_mae": float(error.double().mean()),
            "roi_alpha_mean": mean(alpha, roi), "roi_low_alpha_fraction": mean((alpha <= .3).float(), roi),
            "roi_pixels": int(roi.sum()), "protection_pixels": int(protected.sum()),
            "protection_fallback_to_outside_S": fallback}


def diagnostic_decision(metrics, optimization_views, check_views):
    """Mechanical rules: all group starts, both controls, every protected region."""
    try:
        require(len(optimization_views) == len(check_views) == 2 and len(set(optimization_views + check_views)) == 4, "invalid view sets")
        for group in GROUPS:
            for update in ("0", "100", "200", "400"):
                require(set(metrics[group][update]) == set(optimization_views + check_views), "missing evaluation view")
                for name, row in metrics[group][update].items():
                    for key in ("roi_mae", "protection_mae", "full_mae", "roi_alpha_mean", "roi_low_alpha_fraction"):
                        value = row[key]
                        require(value is not None and math.isfinite(value) and value >= 0, "missing/invalid metric")
                    require(row["roi_pixels"] > 0 and row["protection_pixels"] > 0, "empty evaluation region")
                    require(row["roi_low_alpha_fraction"] <= 1 and row["roi_alpha_mean"] <= 1 + 1e-6, "invalid alpha metric")
                    reference = metrics["Va"]["0"][name]
                    require(all(row[key] == reference[key] for key in ("roi_pixels", "protection_pixels", "protection_fallback_to_outside_S")),
                            "evaluation regions changed")
        def comparison(names, field, fraction):
            values = {g: {u: sum(metrics[g][u][name][field] for name in names) / len(names)
                          for u in ("0", "400")} for g in GROUPS}
            starts = [values[g]["0"] for g in GROUPS]
            e0 = sum(starts) / 3
            n = max(abs(values["Va"]["400"] - values["Vb"]["400"]), max(starts) - min(starts))
            threshold = max(fraction * e0, 3 * n, 1e-4)
            best_control = min(values[g]["400"] for g in ("Va", "Vb"))
            return dict(E0=e0, initial_range=max(starts)-min(starts), n=n, threshold=threshold,
                        Va=values["Va"]["400"], Vb=values["Vb"]["400"], O=values["O"]["400"],
                        initial_values=starts, improvement_vs_initial=e0-values["O"]["400"],
                        improvement_vs_better_control=best_control-values["O"]["400"])
        local = comparison(optimization_views, "roi_mae", .05)
        check = comparison(check_views, "roi_mae", .02)
        for row in (local, check):
            row["gain"] = row["improvement_vs_initial"] >= row["threshold"] and row["improvement_vs_better_control"] >= row["threshold"]
        protected = {}
        for name in optimization_views + check_views:
            for field in (["protection_mae", "roi_mae"] if name in check_views else ["protection_mae"]):
                row = comparison([name], field, .01)
                row["pass"] = -row["improvement_vs_initial"] <= row["threshold"] and -row["improvement_vs_better_control"] <= row["threshold"]
                protected[f"{name}:{field}"] = row
        protection_pass = all(row["pass"] for row in protected.values())
        status = ("COLLATERAL_ERROR_DETECTED" if not protection_pass else
                  "PROMISING_LOCAL_RECOVERABILITY" if local["gain"] and check["gain"] else
                  "LOCAL_ONLY_RESPONSE" if local["gain"] else "NO_CLEAR_RECOVERABILITY_SIGNAL")
        return dict(status=status, LOCAL_GAIN=local["gain"], CHECK_VIEW_GAIN=check["gain"],
                    PROTECTION_PASS=protection_pass, local=local, check=check, protected_regions=protected)
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "DIAGNOSTIC_INVALID", "reason": str(error)}
