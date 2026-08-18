"""Reproducible evaluation for the metrics required by the V6 paper protocol."""

import argparse
import csv
import glob
import json
import math
import os
import re

import imageio.v2 as imageio
import numpy as np


def _number(path):
    match = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _files(directory, pattern):
    return sorted(glob.glob(os.path.join(directory, pattern)), key=_number)


def _image(path):
    value = imageio.imread(path).astype(np.float32)
    if value.max() > 1.0:
        value /= 255.0
    return value[..., :3]


def _mask(path):
    value = imageio.imread(path)
    if value.ndim == 3:
        value = value[..., 0]
    return value > 127 if value.max() > 1 else value > 0.5


def _mean(values):
    return float(np.mean(values)) if values else None


def _integral(values, coordinates):
    function = getattr(np, "trapezoid", None)
    if function is None:
        function = np.trapz
    return function(values, coordinates)


def image_metrics(pred_dir, compute_lpips=False, device="cpu"):
    predictions = _files(pred_dir, "rgb_*.png")
    targets = _files(pred_dir, "gt_rgb_*.png")
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Test render must contain matching rgb_*.png and gt_rgb_*.png files")
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError("Image evaluation requires scikit-image") from exc
    lpips_model = None
    torch = None
    if compute_lpips:
        try:
            import lpips
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("LPIPS evaluation requires the lpips package") from exc
        torch = torch_module
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    psnr_values, ssim_values, lpips_values = [], [], []
    errors, uncertainties, opacities = [], [], []
    uncertainty_files = _files(pred_dir, "uncertainty_*.npy")
    acc_files = _files(pred_dir, "acc_*.npy")
    for index, (pred_path, target_path) in enumerate(zip(predictions, targets)):
        pred, target = _image(pred_path), _image(target_path)
        squared = (pred - target) ** 2
        mse = float(squared.mean())
        psnr_values.append(-10.0 * math.log10(max(mse, 1.0e-10)))
        ssim_values.append(
            structural_similarity(pred, target, channel_axis=-1, data_range=1.0)
        )
        if lpips_model is not None:
            pred_t = torch.from_numpy(pred).permute(2, 0, 1)[None].to(device) * 2 - 1
            target_t = torch.from_numpy(target).permute(2, 0, 1)[None].to(device) * 2 - 1
            with torch.no_grad():
                lpips_values.append(float(lpips_model(pred_t, target_t)))
        if index < len(uncertainty_files):
            uncertainties.append(np.load(uncertainty_files[index]).reshape(-1))
            errors.append(squared.mean(axis=-1).reshape(-1))
        if index < len(acc_files):
            opacities.append(np.load(acc_files[index]))
    return {
        "PSNR": _mean(psnr_values),
        "SSIM": _mean(ssim_values),
        "LPIPS": _mean(lpips_values),
    }, errors, uncertainties, opacities


def _risk_curve(errors, order, coverages):
    sorted_error = errors[order]
    cumulative = np.cumsum(sorted_error)
    risks = []
    for coverage in coverages:
        count = max(1, int(round(coverage * len(errors))))
        risks.append(float(cumulative[count - 1] / count))
    return np.asarray(risks)


def _binary_auroc(labels, scores):
    labels = labels.astype(bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * ((start + 1) + end)
        start = end
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def uncertainty_metrics(
    errors,
    uncertainties,
    variance_min=1.0e-4,
    variance_max=0.05,
    bad_error_quantile=0.8,
    curve_path=None,
):
    if not errors or not uncertainties:
        return {}
    error = np.concatenate(errors).astype(np.float64)
    uncertainty = np.concatenate(uncertainties).astype(np.float64)
    finite = np.isfinite(error) & np.isfinite(uncertainty)
    error, uncertainty = error[finite], np.clip(uncertainty[finite], 0.0, 1.0)
    variance = variance_min + (variance_max - variance_min) * uncertainty
    nll = np.mean(0.5 * error / variance + 0.5 * np.log(variance))
    coverages = np.linspace(0.05, 1.0, 20)
    model_risk = _risk_curve(error, np.argsort(uncertainty), coverages)
    oracle_risk = _risk_curve(error, np.argsort(error), coverages)
    random_risk = np.full_like(model_risk, error.mean())
    ause = _integral(model_risk - oracle_risk, coverages)
    aurg = _integral(random_risk - model_risk, coverages)
    aurc = _integral(model_risk, coverages)
    bad_count = min(len(error) - 1, max(1, int(math.ceil((1.0 - bad_error_quantile) * len(error)))))
    labels = np.zeros(len(error), dtype=bool)
    labels[np.argsort(error, kind="mergesort")[-bad_count:]] = True
    auroc = _binary_auroc(labels, uncertainty)
    # UCE: equal-count uncertainty bins, predicted RMSE vs empirical RMSE.
    order = np.argsort(uncertainty)
    bins = np.array_split(order, 15)
    uce = 0.0
    calibration_rows = []
    for indices in bins:
        if len(indices) == 0:
            continue
        predicted_rmse = float(np.sqrt(variance[indices].mean()))
        empirical_rmse = float(np.sqrt(error[indices].mean()))
        fraction = len(indices) / len(error)
        uce += fraction * abs(predicted_rmse - empirical_rmse)
        calibration_rows.append((float(uncertainty[indices].mean()), predicted_rmse, empirical_rmse))
    if curve_path:
        with open(curve_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["coverage", "model_risk", "oracle_risk", "random_risk"])
            writer.writerows(zip(coverages, model_risk, oracle_risk, random_risk))
        calibration_path = os.path.splitext(curve_path)[0] + "_calibration.csv"
        with open(calibration_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["mean_uncertainty", "predicted_rmse", "empirical_rmse"])
            writer.writerows(calibration_rows)
    return {
        "NLL": float(nll),
        "AUSE": float(ause),
        "AURG": float(aurg),
        "AURC": float(aurc),
        "bad_pixel_AUROC": auroc,
        "UCE": float(uce),
    }


def artifact_metrics(pred_dir, opacities, threshold=0.1):
    masks = _files(pred_dir, "background_mask_*.png")
    if not masks:
        return {
            "background_false_opacity": None,
            "floater_pixel_rate": None,
            "artifact_note": "Provide background_mask_*.png to enable artifact metrics",
        }
    if len(masks) != len(opacities):
        raise ValueError("background masks and acc maps must have equal counts")
    false_opacity, floater_rate = [], []
    for path, acc in zip(masks, opacities):
        background = _mask(path)
        if not background.any():
            continue
        false_opacity.append(float(acc[background].mean()))
        floater_rate.append(float((acc[background] > threshold).mean()))
    return {
        "background_false_opacity": _mean(false_opacity),
        "floater_pixel_rate": _mean(floater_rate),
        "floater_opacity_threshold": float(threshold),
    }


def depth_metrics(pred_dir, gt_depth_dir):
    if not gt_depth_dir:
        return {}
    predictions = _files(pred_dir, "depth_*.npy")
    targets = _files(gt_depth_dir, "depth_*.npy")
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Predicted and GT depth directories must contain equal depth_*.npy counts")
    abs_rel, squared = [], []
    for pred_path, target_path in zip(predictions, targets):
        pred, target = np.load(pred_path), np.load(target_path)
        valid = np.isfinite(pred) & np.isfinite(target) & (pred > 0) & (target > 0)
        if not valid.any():
            continue
        residual = pred[valid] - target[valid]
        abs_rel.extend((np.abs(residual) / target[valid]).tolist())
        squared.extend((residual**2).tolist())
    return {
        "AbsRel": _mean(abs_rel),
        "depth_RMSE": float(np.sqrt(np.mean(squared))) if squared else None,
    }


def pointcloud_metrics(predicted_path, target_path, fscore_threshold=0.01):
    if not predicted_path or not target_path:
        return {}
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("Point-cloud evaluation requires scipy") from exc
    predicted = np.load(predicted_path).astype(np.float64)
    target = np.load(target_path).astype(np.float64)
    if predicted.ndim != 2 or target.ndim != 2 or predicted.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("Point clouds must be NumPy arrays with shape [N, 3]")
    pred_to_gt = cKDTree(target).query(predicted, workers=-1)[0]
    gt_to_pred = cKDTree(predicted).query(target, workers=-1)[0]
    precision = float((pred_to_gt < fscore_threshold).mean())
    recall = float((gt_to_pred < fscore_threshold).mean())
    fscore = 2 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "Chamfer_L1": float(pred_to_gt.mean() + gt_to_pred.mean()),
        "F_score": float(fscore),
        "F_score_precision": precision,
        "F_score_recall": recall,
        "F_score_threshold": float(fscore_threshold),
    }


def efficiency_metrics(pred_dir, training_summary=None):
    result = {}
    render_path = os.path.join(pred_dir, "render_efficiency.json")
    if os.path.exists(render_path):
        with open(render_path, encoding="utf-8") as handle:
            result.update(json.load(handle))
    if training_summary and os.path.exists(training_summary):
        with open(training_summary, encoding="utf-8") as handle:
            result.update(json.load(handle))
    return result


def evaluate(args):
    metrics, errors, uncertainties, opacities = image_metrics(
        args.pred_dir, compute_lpips=args.compute_lpips, device=args.device
    )
    metrics.update(
        uncertainty_metrics(
            errors,
            uncertainties,
            args.variance_min,
            args.variance_max,
            args.bad_error_quantile,
            curve_path=os.path.join(args.pred_dir, "risk_coverage.csv"),
        )
    )
    metrics.update(artifact_metrics(args.pred_dir, opacities, args.floater_threshold))
    metrics.update(depth_metrics(args.pred_dir, args.gt_depth_dir))
    metrics.update(
        pointcloud_metrics(
            args.pred_pointcloud, args.gt_pointcloud, args.fscore_threshold
        )
    )
    metrics.update(efficiency_metrics(args.pred_dir, args.training_summary))
    metrics["method"] = args.method
    metrics["seed"] = args.seed
    output = args.output or os.path.join(args.pred_dir, "metrics.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {output}")
    return metrics


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pred_dir", required=True)
    value.add_argument("--method", default="uncertain-nerf-v6")
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--compute_lpips", action="store_true")
    value.add_argument("--device", default="cpu")
    value.add_argument("--gt_depth_dir")
    value.add_argument("--pred_pointcloud")
    value.add_argument("--gt_pointcloud")
    value.add_argument("--training_summary")
    value.add_argument("--variance_min", type=float, default=1.0e-4)
    value.add_argument("--variance_max", type=float, default=0.05)
    value.add_argument("--bad_error_quantile", type=float, default=0.8)
    value.add_argument("--floater_threshold", type=float, default=0.1)
    value.add_argument("--fscore_threshold", type=float, default=0.01)
    value.add_argument("--output")
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
