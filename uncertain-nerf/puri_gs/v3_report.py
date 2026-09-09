"""Fixed V3 screening gates. Missing evidence is never promoted to a pass."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from puri_gs.ru_part_v3 import PROTOCOL, write_json
from puri_gs.static_tracks import sha256_file

QUALITY = {"psnr": 27.566728, "ssim": .868957, "lpips": .073151, "DSC07988_psnr": 19.491301}
B1 = {"psnr": 27.716728, "ssim": .873957, "lpips": .068151, "DSC07988_psnr": 20.491301}


def classify(quality, protection, resources, *, integrity=True, complete=True):
    if not integrity:
        return "INVALID_RUN"
    if any(v is False for v in quality.values()):
        return "QUALITY_RECOVERY_FAIL / NO_GO"
    if any(v is None for v in quality.values()):
        return "COMPARISON_INCOMPLETE"
    if any(v is False for v in protection.values()):
        return "PARENT_COMPARISON_FAIL"
    if any(v is False for v in resources.values()):
        return "QUALITY_PASS_RESOURCE_FAIL"
    if not complete or any(v is None for v in (*protection.values(), *resources.values())):
        return "COMPARISON_INCOMPLETE"
    return "PROMISING_SINGLE_SEED"


def read_optional(path):
    return json.loads(path.read_text()) if path.is_file() else None


def fixed_roi(root, roi_path):
    """Reuse the historical B1/RU-TAR definition; never derive a ROI from V3."""
    if roi_path is None:
        project = Path(__file__).resolve().parents[1]
        candidates = list((project / "logs-puri").glob("**/arrays/b1_selected_views_float32.npz"))
        candidates = [p for p in candidates if (p.parent / "ru_tar_selected_views_float32.npz").is_file()]
        if len(candidates) != 1:
            return None, {"status": "NOT_ASSESSABLE", "reason": "historical fixed B1/RU-TAR ROI arrays unavailable or ambiguous"}
        roi_path = candidates[0].parent
    roi_path = Path(roi_path)
    if roi_path.is_dir():
        from tools.diagnose_puri_gs_garden_alpha_coverage import top_connected_component
        b1path = roi_path / "b1_selected_views_float32.npz"
        tarpath = roi_path / "ru_tar_selected_views_float32.npz"
        with np.load(b1path) as a, np.load(tarpath) as b:
            b1 = a["DSC07988__alpha"].copy()
            tar = b["DSC07988__alpha"].copy()
        roi, info = top_connected_component((b1 >= .8) & ((b1 - tar) >= .5))
        provenance = {"status": "HISTORICAL_DEFINITION_REUSED", "b1_sha256": sha256_file(b1path),
                      "ru_tar_sha256": sha256_file(tarpath), "definition": "old top_connected_component((B1>=.8)&((B1-RU_TAR)>=.5))", **info}
    else:
        roi = np.load(roi_path, allow_pickle=False)
        if roi.dtype != np.bool_ or roi.ndim != 2:
            raise ValueError("historical ROI must be a 2D boolean numpy mask")
        provenance = {"status": "USER_SUPPLIED_FIXED_ROI", "path": str(roi_path), "sha256": sha256_file(roi_path)}
    if not roi.any():
        return None, {**provenance, "status": "NOT_ASSESSABLE", "reason": "historical ROI empty"}
    return roi, provenance


def build_report(root, *, roi_path=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    missing, invalid = [], []
    evaluations, per_view, costs, states, identities = {}, {}, {}, {}, {}
    from puri_gs.v3_parent_reference import load_reference
    reference = None
    try:
        reference = load_reference(root, verify=True)
    except (ValueError, OSError, KeyError) as error:
        invalid.append(f"historical Parent reference: {error}")
    eval_dirs = {mode: root / f"eval-{mode}" for mode in ("parent", "v3")}
    if reference:
        eval_dirs["parent"] = Path(reference["evaluation_dir"])
    for mode in ("parent", "v3"):
        reused = mode == "parent" and reference is not None
        run_dir = Path(reference["source_run"]) if reused else root / mode
        eval_dir = eval_dirs[mode]
        if reused:
            # This is an external reference, never a fabricated TRAIN_COMPLETE state.
            states[mode] = {"origin": reference["origin"], "checkpoint_sha256": reference["checkpoint_sha256"]}
            eval_fingerprint = reference["evaluation_fingerprint"]
        else:
            states[mode] = read_optional(root / f"{mode}.status.json")
            eval_state = read_optional(root / f"eval-{mode}.status.json")
            if not states[mode] or states[mode].get("status") != "TRAIN_COMPLETE":
                missing.append(f"{mode}: TRAIN_COMPLETE")
            if not eval_state or eval_state.get("status") != "EVAL_COMPLETE":
                missing.append(f"{mode}: EVAL_COMPLETE")
                continue
            eval_fingerprint = eval_state.get("evaluation_fingerprint")
        evaluations[mode] = read_optional(eval_dir / "test_metrics.json")
        with (eval_dir / "per_image_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        per_view[mode] = {r["image_name"]: {k: float(r[k]) for k in ("psnr", "ssim", "lpips")} for r in rows}
        identity_path = Path(reference["current_input_manifest"]) if reused else run_dir / "v3_input_manifest.json"
        identities[mode] = read_optional(identity_path)
        if not identities[mode] or len(rows) != 24 or set(per_view[mode]) != set(identities[mode]["test_basenames"]):
            invalid.append(f"{mode}: evaluation manifest mismatch")
        if not all(math.isfinite(v) for row in per_view[mode].values() for v in row.values()):
            invalid.append(f"{mode}: non-finite metrics")
        evaluations[mode]["DSC07988_psnr"] = per_view[mode].get("DSC07988.JPG", {}).get("psnr")
        costs[mode] = {
            "training": read_optional(run_dir / "train_metrics.json"),
            "inference": read_optional(eval_dir / "efficiency_metrics.json"),
            "environment": read_optional(run_dir / "environment.json"),
            "manifest": read_optional(run_dir / "v3_run_manifest.json"),
            "evaluation_fingerprint": eval_fingerprint,
            "checkpoint_sha256": states[mode].get("checkpoint_sha256") if states[mode] else None,
        }
    quality = {k: None for k in QUALITY}
    if "v3" in evaluations:
        for k, threshold in QUALITY.items():
            value = evaluations["v3"].get(k)
            if value is not None:
                quality[k] = value <= threshold if k == "lpips" else value >= threshold
    comparable = False
    if all(m in identities for m in ("parent", "v3")):
        comparable = identities["parent"] == identities["v3"]
        if not comparable:
            invalid.append("Parent/V3 data, cameras, initialization identity differs")
        p_seq = read_optional(root / "parent/v3_camera_sequence.json")
        v_seq = read_optional(root / "v3/v3_camera_sequence.json")
        if reference:
            smoke_seq = read_optional(root / "smoke-parent/v3_camera_sequence.json")
            if v_seq is None or len(v_seq) != 30000 or smoke_seq is None or v_seq[:len(smoke_seq)] != smoke_seq:
                invalid.append("V3 full camera sequence missing or smoke prefix changed")
        elif p_seq is None or v_seq is None or p_seq != v_seq or len(v_seq) != 30000:
            invalid.append("Parent/V3 camera sequence differs or missing")
    protection = {k: None for k in ("psnr", "ssim", "lpips")}
    delta = {}
    if comparable and all(m in evaluations for m in ("parent", "v3")):
        delta = {k: evaluations["v3"][k] - evaluations["parent"][k] for k in protection}
        protection = {k: (d <= 0 if k == "lpips" else d >= 0) for k, d in delta.items()}
    resources = {k: None for k in ("gaussian_count", "training_time_ratio", "single_rasterization", "fps_ratio")}
    ratios = {}
    if "v3" in costs:
        train = costs["v3"]["training"]
        if train:
            resources["gaussian_count"] = train["gaussian_count"] <= 2312002
        validation = read_optional(root / "eval-v3/ru_validation.json")
        if validation:
            resources["single_rasterization"] = validation["evaluation_rasterization_count_ratio"] == 1
    if comparable and all(m in costs for m in ("parent", "v3")):
        p, v = costs["parent"], costs["v3"]
        # Both runs were launched by this one-device serial wrapper. Also require
        # the recorded software/hardware environments to match exactly.
        same_env = reference is None and p["environment"] is not None and p["environment"] == v["environment"]
        if same_env and p["training"] and v["training"]:
            ratios["training_time"] = v["training"]["training_time_seconds"] / p["training"]["training_time_seconds"]
            resources["training_time_ratio"] = ratios["training_time"] < 1.10
        same_eval = p["evaluation_fingerprint"] is not None and p["evaluation_fingerprint"] == v["evaluation_fingerprint"]
        if same_eval and p["inference"] and v["inference"]:
            if p["inference"]["warmup_render_count"] == v["inference"]["warmup_render_count"] == 10:
                ratios["fps"] = v["inference"]["render_fps"] / p["inference"]["render_fps"]
                resources["fps_ratio"] = ratios["fps"] >= .95
    roi_results = {}
    if "v3" in evaluations:
        roi, roi_provenance = fixed_roi(root, roi_path)
        if roi is not None:
            for mode in evaluations:
                alpha = np.load(eval_dirs[mode] / "DSC07988_alpha.npy")
                error = np.load(eval_dirs[mode] / "DSC07988_abs_error.npy")
                if alpha.shape != roi.shape or error.shape != roi.shape:
                    raise ValueError("fixed historical ROI resolution differs")
                roi_results[mode] = {"roi_pixels": int(roi.sum()), "absolute_rgb_error_mean": float(error[roi].mean()),
                                     "alpha_mean": float(alpha[roi].mean()), "low_alpha_pixels": int((alpha[roi] <= .3).sum()),
                                     "low_alpha_area_fraction": float((alpha[roi] <= .3).mean())}
        else:
            missing.append("fixed historical hole ROI")
    else:
        roi_provenance = {"status": "NOT_ASSESSABLE"}
    evidence = read_optional(root / "v3/evidence_check/evidence_check.json")
    cache = costs.get("v3", {}).get("manifest", {}) or {}
    cache = cache.get("cache") or {}
    accounting = {"cache_prepare_seconds": cache.get("cache_prepare_seconds"),
                  "original_cache_build_seconds": cache.get("original_cache_build_seconds"),
                  "training_plus_original_cache_seconds": None}
    vtrain = costs.get("v3", {}).get("training")
    if vtrain and accounting["original_cache_build_seconds"] is not None:
        accounting["training_plus_original_cache_seconds"] = vtrain["training_time_seconds"] + accounting["original_cache_build_seconds"]
    result = {
        "status": classify(quality, protection, resources, integrity=not invalid, complete=not missing),
        "quality_gates": quality, "parent_protection_gates": protection, "resource_gates": resources,
        "thresholds": QUALITY, "B1_historical_reference": B1,
        "metrics": evaluations, "delta_vs_parent": delta, "ratios": ratios, "costs": costs,
        "parent_reference": reference,
        "cache_cost_accounting": accounting, "evidence": evidence,
        "fixed_roi": roi_results, "roi_provenance": roi_provenance,
        "missing": missing, "invalid": invalid,
        "protocol": PROTOCOL, "training_actually_completed": bool(states.get("v3") and states["v3"].get("status") == "TRAIN_COMPLETE"),
    }
    write_json(root / "screening_result.json", result)
    if per_view:
        with (root / "screening_per_view.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["mode", "image_name", "psnr", "ssim", "lpips"])
            for mode, views in per_view.items():
                for name, metrics in views.items():
                    writer.writerow([mode, name, *[metrics[k] for k in ("psnr", "ssim", "lpips")]])
    lines = ["# RU-PART-V3 Garden 开发场景筛选报告", "", f"状态：**{result['status']}**", "",
             "实现仅增加 0.8 × mean((1−M)C × RGB绝对误差)，step 500 开启。基础 RU/DSSIM/责任头和常规 ADC 保留。",
             "此处协议字段描述固定设计；是否完成训练以 training_actually_completed 和阶段状态文件为准。", "",
             "| 指标 | B1 历史参考 | 可比 Parent | V3 | V3−Parent |", "|---|---:|---:|---:|---:|"]
    for key in QUALITY:
        vals = [B1[key], evaluations.get("parent", {}).get(key), evaluations.get("v3", {}).get(key), delta.get(key)]
        lines.append(f"| {key} | " + " | ".join("未测" if v is None else f"{v:.6f}" for v in vals) + " |")
    for title, group in (("固定恢复门", quality), ("Parent 质量保护", protection), ("资源门", resources)):
        lines += ["", title + "：" + "；".join(f"{k}={'NOT_ASSESSABLE' if v is None else ('PASS' if v else 'FAIL')}" for k,v in group.items())]
    if reference:
        lines += ["", "Parent来源：历史标准RU检查点及本轮独立重评；质量比较基于原配置/数据路径/split和复现审计。",
                  "历史图像/SfM字节哈希和30k相机序列未记录，当前输入哈希不冒充历史记录。完整训练时间门为NOT_ASSESSABLE。",
                  "推理比较使用本轮GPU与软件/预热/渲染配置记录，单独判断，不受历史训练GPU差异影响。"]
    lines += ["", "证据激活：" + (evidence["status"] if evidence else "未测"),
              "", "孔洞 ROI：" + json.dumps(roi_results or roi_provenance, ensure_ascii=False),
              "", "成本：" + json.dumps(accounting, ensure_ascii=False),
              "", "缺失项：" + ("；".join(missing + invalid) or "无"),
              "", "单 seed 只能提供候选信号，不能证明统计显著性、跨场景泛化、收敛或静态标签正确性。Garden 已用于开发选择。",
              "报告完成后停止，不自动开展下一算法、多 seed、跨场景、Oracle 或重放。", "",
              "```json", json.dumps(PROTOCOL, indent=2), "```", ""]
    (root / "RU_PART_V3_SCREENING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(result["status"], root / "RU_PART_V3_SCREENING_REPORT.md")
    return result
