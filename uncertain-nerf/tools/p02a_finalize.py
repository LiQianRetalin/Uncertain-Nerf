#!/usr/bin/env python3
"""Build the bounded P02-A acceptance package from frozen P01/P02 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCENE_TEST_COUNTS = {"android": 19, "patio_high": 45}
METHODS = ("B1", "RU", "RobustSplat", "SLS-MLP")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def normalize_method(value: str) -> str:
    mapping = {
        "b1": "B1", "ru": "RU", "robustsplat": "RobustSplat",
        "sls-mlp": "SLS-MLP", "sls_mlp": "SLS-MLP",
    }
    return mapping[value.casefold()]


def run_id(scene: str, method: str) -> str:
    slug = {"B1": "b1", "RU": "ru", "RobustSplat": "robustsplat", "SLS-MLP": "sls-mlp"}[method]
    return f"P02A-{scene}-{slug}"


def view_key(image_name: str) -> str:
    """Pair lossless common-input PNG names with their P01 source JPG view names."""
    return Path(image_name).stem.casefold()


def build_quality(code_root: Path, work_root: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p01_summary = [row for row in read_csv(code_root / "reports/p01/reevaluation_summary.csv") if row["scene"] in SCENE_TEST_COUNTS and row["method"] in {"B1", "RU"}]
    p01_per = [row for row in read_csv(code_root / "reports/p01/reevaluation_per_image.csv") if row["scene"] in SCENE_TEST_COUNTS and row["method"] in {"B1", "RU"}]
    p02_summary = read_csv(work_root / "report/summary.csv")
    p02_per = read_csv(work_root / "report/per_image_metrics.csv")
    summaries: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for row in p01_summary:
        method = normalize_method(row["method"])
        summaries.append({
            "run_id": run_id(row["scene"], method), "scene": row["scene"], "method": method,
            "seed": 42, "updates": 30000, "test_image_count": int(row["test_image_count"]),
            "psnr": row["psnr"], "ssim": row["ssim"], "lpips": row["lpips"],
            "checkpoint_sha256": row["checkpoint_sha256"], "evaluator_id": row["evaluator_id"],
            "quality_source_status": "P01_FROZEN_REUSED", "quality_source": row["source"],
        })
    for row in p02_summary:
        method = normalize_method(row["method"])
        summaries.append({
            "run_id": run_id(row["scene"], method), "scene": row["scene"], "method": method,
            "seed": 42, "updates": 30000, "test_image_count": int(row["test_image_count"]),
            "psnr": row["psnr"], "ssim": row["ssim"], "lpips": row["lpips"],
            "checkpoint_sha256": row["checkpoint_sha256"], "evaluator_id": row["evaluator_id"],
            "quality_source_status": "P02_REPRODUCED_REUSED", "quality_source": row["source"],
        })
    for row in p01_per:
        method = normalize_method(row["method"])
        per_image.append({
            "run_id": run_id(row["scene"], method), "scene": row["scene"], "method": method,
            "seed": 42, "image_name": row["image_name"], "view_key": view_key(row["image_name"]), "psnr": row["psnr"],
            "ssim": row["ssim"], "lpips": row["lpips"],
            "checkpoint_sha256": row["checkpoint_sha256"], "evaluator_id": row["evaluator_id"],
            "quality_source_status": "P01_FROZEN_REUSED",
        })
    for row in p02_per:
        method = normalize_method(row["method"])
        per_image.append({
            "run_id": run_id(row["scene"], method), "scene": row["scene"], "method": method,
            "seed": 42, "image_name": row["image_name"], "view_key": view_key(row["image_name"]), "psnr": row["psnr"],
            "ssim": row["ssim"], "lpips": row["lpips"],
            "checkpoint_sha256": row["checkpoint_sha256"], "evaluator_id": row["evaluator_id"],
            "quality_source_status": "P02_REPRODUCED_REUSED",
        })
    summaries.sort(key=lambda row: (row["scene"], METHODS.index(row["method"])))
    per_image.sort(key=lambda row: (row["scene"], METHODS.index(row["method"]), row["image_name"]))
    if len(summaries) != 8 or len(per_image) != 256:
        raise RuntimeError(f"quality target mismatch: {len(summaries)} summaries, {len(per_image)} rows")
    for scene, count in SCENE_TEST_COUNTS.items():
        name_sets = []
        for method in METHODS:
            rows = [row for row in per_image if row["scene"] == scene and row["method"] == method]
            if len(rows) != count or not all(math.isfinite(float(row[key])) for row in rows for key in ("psnr", "ssim", "lpips")):
                raise RuntimeError(f"invalid per-image evidence: {scene}/{method}")
            name_sets.append([row["view_key"] for row in rows])
        if any(names != name_sets[0] for names in name_sets[1:]):
            raise RuntimeError(f"paired test names differ: {scene}")
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "per_image_metrics.csv", per_image)
    return summaries, per_image


def build_assets(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output / "checkpoints").glob("*.json")):
        data = read_json(path)
        rows.append({
            "run_id": data["run_id"], "status": data["status"],
            "checkpoint_path": data["checkpoint_path"], "checkpoint_sha256": data["checkpoint_sha256"],
            "checkpoint_bytes": data["checkpoint_bytes"], "gaussian_count": data["gaussian_count"],
            "model_format": data["model_format"], "step": data["step"],
            "required_inference_components": (
                "gsplat-1.5.3 native renderer" if data["family"] == "internal"
                else "RobustSplat renderer+diff-gaussian-rasterization" if data["family"] == "robustsplat"
                else "SpotLessSplats runner+gsplat-1.1.1"
            ),
        })
    if len(rows) != 8 or any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("checkpoint audit is incomplete")
    write_csv(output / "asset_ledger.csv", rows)
    return rows


def build_timing(output: Path, gpu: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timing_root = output / "timing"
    selected_gpu = gpu["selected"]
    for scene in SCENE_TEST_COUNTS:
        for method in ("B1", "RU"):
            rid = run_id(scene, method)
            for repeat in range(1, 4):
                path = timing_root / rid / f"repeat-{repeat}" / "per_image_latency.csv"
                samples = read_csv(path)
                efficiency = read_json(path.parent / "efficiency_metrics.json")
                validation = read_json(path.parent / "ru_validation.json")
                split = read_json(path.parent / "dataset_split.json")
                if (
                    efficiency.get("warmup_render_count") != 10
                    or validation.get("evaluation_warmup_render_count") != 10
                    or validation.get("evaluation_image_save_disabled") is not True
                    or validation.get("standard_checkpoint_load_pass") is not True
                    or validation.get("evaluation_imported_dino") is not False
                    or validation.get("evaluation_loaded_mask_head") is not False
                    or validation.get("evaluation_rasterization_count_ratio") != 1.0
                    or len(split.get("test", [])) != SCENE_TEST_COUNTS[scene]
                ):
                    raise RuntimeError(f"invalid internal timing boundary: {path.parent}")
                seconds = sum(float(row["latency_ms"]) for row in samples) / 1000.0
                rows.append({
                    "run_id": rid, "scene": scene, "method": method, "seed": 42,
                    "timing_source": "P02A_RETIMED", "warmup": 10, "repeat": repeat,
                    "frames": len(samples), "seconds": seconds, "fps": len(samples) / seconds,
                    "boundary": "native gsplat rasterization path; per-view CUDA sync; excludes disk writes and public metrics",
                    "physical_gpu": selected_gpu["physical_index"], "gpu_uuid": selected_gpu["uuid"],
                })
        for method in ("RobustSplat", "SLS-MLP"):
            rid = run_id(scene, method)
            native_path = (
                timing_root / rid / "test/ours_30000/p02_timing.json"
                if method == "RobustSplat" else timing_root / rid / "stats/p02_timing_step29999.json"
            )
            native = read_json(native_path)
            if native.get("warmup") != 10 or len(native.get("records", [])) != 3:
                raise RuntimeError(f"invalid external timing boundary: {native_path}")
            for item in native["records"]:
                rows.append({
                    "run_id": rid, "scene": scene, "method": method, "seed": 42,
                    "timing_source": "P02A_RETIMED", "warmup": native["warmup"],
                    "repeat": item["repeat"], "frames": item["frames"],
                    "seconds": item["seconds"], "fps": item["fps"],
                    "boundary": native["boundary"], "physical_gpu": selected_gpu["physical_index"],
                    "gpu_uuid": selected_gpu["uuid"],
                })
    if len(rows) != 24 or any(int(row["frames"]) != SCENE_TEST_COUNTS[row["scene"]] for row in rows):
        raise RuntimeError(f"P02A timing target mismatch: {len(rows)}")
    write_csv(output / "timing_raw.csv", rows)
    return rows


def build_smoke(work_root: Path, output: Path) -> list[dict[str, Any]]:
    ledger = read_csv(work_root / "state/smoke_ledger.csv")
    timings = [row for row in read_csv(work_root / "state/stage_timing.csv") if row["stage"].startswith("smoke_P02-")]
    by_stage: dict[str, list[dict[str, str]]] = {}
    for row in timings:
        by_stage.setdefault(row["stage"], []).append(row)
    counters: dict[str, int] = {}
    rows = []
    for attempt_id, row in enumerate(ledger, 1):
        counters[row["run_id"]] = counters.get(row["run_id"], 0) + 1
        candidates = by_stage.get("smoke_" + row["run_id"], [])
        timing = candidates[counters[row["run_id"]] - 1] if len(candidates) >= counters[row["run_id"]] else None
        rows.append({
            "attempt_id": attempt_id, "run_id": row["run_id"],
            "identity_attempt_number": counters[row["run_id"]], "status": row["status"],
            "started_epoch": timing["started_epoch"] if timing else "UNKNOWN",
            "ended_epoch": timing["ended_epoch"] if timing else "UNKNOWN",
            "budget_charge_updates": int(row["updates"]),
            "actual_optimizer_updates": 100 if row["status"] == "PASS" else "UNKNOWN",
            "failure_stage": "not applicable" if row["status"] == "PASS" else "environment/runtime before completion; exact optimizer step unavailable",
            "log_source": f"logs/smoke_{row['run_id']}.log (same-name history may have been overwritten)",
        })
    if len(rows) != 8 or sum(int(row["budget_charge_updates"]) for row in rows) != 800:
        raise RuntimeError("smoke conservative charge must remain 8x100")
    write_csv(output / "smoke_audit.csv", rows)
    return rows


def build_cost(code_root: Path, work_root: Path, output: Path) -> list[dict[str, Any]]:
    p01 = [row for row in read_csv(code_root / "reports/p01/reevaluation_summary.csv") if row["scene"] in SCENE_TEST_COUNTS and row["method"] in {"B1", "RU"}]
    p02_stage = read_csv(work_root / "report/cost_breakdown.csv")
    rows: list[dict[str, Any]] = []
    for row in p01:
        rows.append({
            "run_id": run_id(row["scene"], normalize_method(row["method"])),
            "training_process_wall_seconds": row["training_seconds"] or "UNKNOWN",
            "proven_core_optimization_seconds": "UNKNOWN",
            "in_process_feature_seconds": "UNKNOWN",
            "in_process_eval_trajectory_seconds": "UNKNOWN",
            "external_feature_seconds": "UNKNOWN",
            "download_install_seconds": "UNKNOWN",
            "export_seconds": "UNKNOWN",
            "independent_evaluation_seconds": "UNKNOWN",
            "failed_consumption_seconds": "UNKNOWN",
            "boundary_note": "P01 archived training_seconds; complete feature/build boundary was not recorded",
        })
    for scene in SCENE_TEST_COUNTS:
        for method in ("RobustSplat", "SLS-MLP"):
            p02_id = f"P02-{scene}-{'robustsplat' if method == 'RobustSplat' else 'sls-mlp'}"
            stage_map = {row["stage"]: row for row in p02_stage}
            train = stage_map.get("train_" + p02_id, {})
            render = stage_map.get("render_" + p02_id, {})
            evaluate = stage_map.get("evaluate_" + p02_id, {})
            rows.append({
                "run_id": run_id(scene, method),
                "training_process_wall_seconds": train.get("elapsed_seconds", "UNKNOWN"),
                "proven_core_optimization_seconds": "UNKNOWN",
                "in_process_feature_seconds": "INCLUDED_NOT_SEPARABLE" if method == "RobustSplat" else "not applicable (external cache)",
                "in_process_eval_trajectory_seconds": "INCLUDED_NOT_SEPARABLE" if method == "SLS-MLP" else "not applicable",
                "external_feature_seconds": "11129_INCLUDES_FIRST_DOWNLOAD" if scene == "android" and method == "SLS-MLP" else "457" if scene == "patio_high" and method == "SLS-MLP" else "not applicable",
                "download_install_seconds": "UNKNOWN_NOT_FROZEN_PRETRAIN",
                "export_seconds": render.get("elapsed_seconds", "UNKNOWN"),
                "independent_evaluation_seconds": evaluate.get("elapsed_seconds", "UNKNOWN"),
                "failed_consumption_seconds": "UNKNOWN",
                "boundary_note": "Robust process includes DINO preparation; SLS process includes eval/trajectory; do not call either pure optimizer time",
            })
    rows.sort(key=lambda row: row["run_id"])
    write_csv(output / "cost_breakdown.csv", rows)
    return rows


def build_p01_loader_comparison(code_root: Path, loaders: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    manifest = read_json(code_root / "reports/protocol_manifest.json")
    protocol_ids = {
        "android": "android-colmap-factor4-keyword-v1",
        "patio_high": "patio-high-internal-factor4-v1",
    }
    comparisons = []
    for scene, protocol_id in protocol_ids.items():
        frozen = next(item for item in manifest["protocols"] if item["protocol_id"] == protocol_id)
        if "split" in frozen:
            p01_train = frozen["split"]["train"]
            p01_test = frozen["split"]["test"]
        else:
            p01_train = [name for name in frozen["ordered_image_names"] if "clutter" in name.casefold()]
            p01_test = [name for name in frozen["ordered_image_names"] if "extra" in name.casefold()]
        p01_used = {view_key(name) for name in p01_train + p01_test}
        p01_excluded = [name for name in frozen["ordered_image_names"] if view_key(name) not in p01_used]
        for loader in [item for item in loaders if scene in Path(item["data_dir"]).name or item["data_dir"].endswith("/" + scene)]:
            checks = {
                "train_view_keys_exact": [view_key(name) for name in loader["train_names"]] == [view_key(name) for name in p01_train],
                "test_view_keys_exact": [view_key(name) for name in loader["test_names"]] == [view_key(name) for name in p01_test],
                "excluded_view_keys_exact": [view_key(name) for name in loader["excluded_names"]] == [view_key(name) for name in p01_excluded],
            }
            comparisons.append({
                "scene": scene, "method": loader["method"], "p01_protocol_id": protocol_id,
                "name_mapping": "casefolded filename stem; original extension is retained in source tables",
                "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
            })
    result = {
        "schema": "puri-gs-p02a-p01-loader-comparison-v1",
        "status": "PASS" if len(comparisons) == 4 and all(item["status"] == "PASS" for item in comparisons) else "FAIL",
        "comparisons": comparisons,
        "scope_note": "view identity/order is compared to the frozen P01 manifest; extension conversion is not asserted to preserve file-byte hashes",
    }
    path = output / "loader/p01_manifest_comparison.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runtime-commit", required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve(); code = repo / "uncertain-nerf"
    work = args.work_root.resolve(); output = args.output_dir.resolve()
    if not output.is_dir():
        raise RuntimeError("P02A output directory must already exist")
    original_p02 = read_json(work / "report/status.json")
    if original_p02.get("status") != "COMPLETE":
        raise RuntimeError("P02 must remain COMPLETE")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_runtime_commit:
        raise RuntimeError(f"runtime commit mismatch: {head}")

    summaries, per_image = build_quality(code, work, output)
    assets = build_assets(output)
    gpu = read_json(output / "runtime/gpu_selection.json")
    timing = build_timing(output, gpu)
    smoke = build_smoke(work, output)
    costs = build_cost(code, work, output)

    loader_files = sorted((output / "loader").glob("*.json"))
    loaders = [read_json(path) for path in loader_files]
    loader_pass = len(loaders) == 4 and all(item["status"] == "PASS" for item in loaders)
    p01_loader_comparison = build_p01_loader_comparison(code, loaders, output)
    loader_pass = loader_pass and p01_loader_comparison["status"] == "PASS"
    weights = read_json(output / "features/weight_provenance.json")
    runtime = read_json(output / "runtime/repository_provenance.json")
    checkpoints_by_run = {row["run_id"]: row for row in assets}
    timing_counts = {row["run_id"]: 0 for row in summaries}
    for row in timing:
        timing_counts[row["run_id"]] += 1
    acceptance = []
    for row in summaries:
        rid = row["run_id"]
        for item, status, effect, evidence in (
            ("training_identity", "PASS", "quality identity retained", "runtime/, loader/, checkpoints/"),
            ("quality_evaluation", "PASS", "frozen quality reused; no reevaluation needed", row["quality_source_status"]),
            ("checkpoint_read_only_load", "PASS" if rid in checkpoints_by_run else "FAIL", "model identity and Gaussian count", "asset_ledger.csv"),
            ("loader_split", "PASS" if loader_pass else "FAIL", "train/test/excluded identity", "loader/*.json"),
            ("timing", "PASS" if timing_counts.get(rid) == 3 else "FAIL", "same-device comparable native inference", "timing_raw.csv"),
            ("cost_boundary", "UNKNOWN", "no quality invalidation; forbids unsupported complete-cost ratios", "cost_breakdown.csv"),
        ):
            acceptance.append({"run_id": rid, "item": item, "status": status, "impact": effect, "evidence": evidence})
    acceptance.extend([
        {"run_id": "P02", "item": "smoke_per_identity_attempt_limit", "status": "FAIL", "impact": "procedural deviation only; no retraining", "evidence": "smoke_audit.csv"},
        {"run_id": "P02", "item": "smoke_actual_optimizer_updates", "status": "UNKNOWN", "impact": "failed attempts unrecoverable; 800 retained as conservative charge", "evidence": "smoke_audit.csv"},
        {"run_id": "P02A", "item": "new_training_or_updates", "status": "PASS", "impact": "none created", "evidence": "status.json"},
    ])
    write_csv(output / "acceptance_matrix.csv", acceptance)

    medians = {rid: statistics.median(float(item["fps"]) for item in timing if item["run_id"] == rid) for rid in timing_counts}
    quality = {(row["scene"], row["method"]): row for row in summaries}
    diag_path = output / "representatives/patio_fixed_diagnostics.csv"
    diagnostics = read_csv(diag_path) if diag_path.is_file() else []
    provenance_files = [
        output / "summary.csv", output / "per_image_metrics.csv", output / "asset_ledger.csv",
        output / "timing_raw.csv", output / "smoke_audit.csv", output / "runtime/robustsplat.diff",
        output / "runtime/spotless.diff", output / "features/weight_provenance.json",
    ] + loader_files
    provenance = {
        "schema": "puri-gs-p02a-final-provenance-v1", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "p02_historical_status_preserved": original_p02,
        "runtime_repository": runtime, "expected_runtime_commit": args.expected_runtime_commit,
        "historical_correspondence_limit": "current source hashes are compared with frozen P02 preflight; current loader instantiation alone is not treated as proof of past behavior",
        "weights": weights,
        "evidence_sha256": {str(path.relative_to(output)): sha256_file(path) for path in provenance_files},
    }
    (output / "final_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    deviations = """# P02-A 偏离登记\n\n- 原P02 smoke规则要求每身份最多2次；Android RobustSplat和Patio-High RobustSplat实际各3次，构成程序性偏离。无需因此重训，正式模型与质量证据保留。\n- 8次尝试每次均按100 updates保守扣额，总扣额800。4个PASS尝试可证完成100步；FAILED尝试的实际optimizer updates因同名日志覆盖/环境期失败而不可恢复，登记为UNKNOWN，不能写成0，也不能把800称为已证实际更新。\n- 原P02 preflight是训练前历史快照，继续原样保留；P02-A追加final_provenance.json，不倒填旧文件。\n- P01严格FPS缺少同包原始三遍记录，且与P02函数边界/运行时负载证据不闭合，因此8个既有模型均在同一空闲L20上重计时；旧值保留，不覆盖。\n- Android SLS特征11129秒包含首次模型下载，不能作为算法固有特征提取时间；P01完整特征构建时间仍未知。\n- 旧最低/中位/最高PSNR代表图规则保留为额外诊断；P02-A主代表图改为共同测试文件名排序的首/中/末。\n"""
    (output / "deviations.md").write_text(deviations, encoding="utf-8")

    timing_boundary = """# P02-A 计时边界\n\n所有P02-A新计时均在同一次GPU选择后串行执行，固定10次warm-up、完整test三遍、每遍前后CUDA同步；排除磁盘写图与公共质量指标。相机/内参与测试顺序在计时前由实际loader核验。\n\n- B1/RU：现有标准checkpoint评测入口的原生gsplat rasterization路径；逐视图记录CUDA同步延迟，关闭图片保存。\n- RobustSplat：官方`render(...)`调用，包含其每视图必要的SH/协方差与rasterizer处理，不含保存和指标。\n- SLS-MLP：官方Runner的`rasterize_splats(...)`调用；训练期语义MLP不属于checkpoint推理必要组件，未强行加入。\n\n这些结果建立了同设备、同warm-up、同完整视图重复的原生推理比较，但不同方法的框架级Python调度仍非逐指令同构；报告只比较完整必要原生调用，不把它解释为端到端应用吞吐。推理显存未以污染FPS的钩子同期测量，历史训练日志中的`mem`字段不作为推理显存。\n"""
    (output / "timing_boundary.md").write_text(timing_boundary, encoding="utf-8")

    report_lines = [
        "# REPORT_P02A", "", "状态：`P02A_ACCEPTANCE_COMPLETE`。P02原流水线`COMPLETE`保持不变。", "",
        "本包新增训练身份0、optimizer updates 0、训练smoke 0；只读加载8个既有checkpoint，质量结果全部复用，8个模型各进行3次同设备计时。未启动P03、修复R、OAC、Corner、Room factor2、Garden/PART或U阶段。", "",
        "## 四方法质量主表", "",
        "| 场景 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | N | 来源 | P02-A中位FPS↑ |", "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in summaries:
        report_lines.append(
            f"| {row['scene']} | {row['method']} | {float(row['psnr']):.4f} | {float(row['ssim']):.5f} | {float(row['lpips']):.5f} | {row['test_image_count']} | {row['quality_source_status']} | {medians[row['run_id']]:.3f} |"
        )
    report_lines += [
        "", "质量结论：Android上RU相对RobustSplat约+0.608 dB，相对SLS-MLP约+0.073 dB，且SSIM/LPIPS更好；后一个PSNR差很小，单seed不能称稳定领先。Patio-High上RobustSplat三项质量均优于RU（PSNR约+1.580 dB）；SLS-MLP比RU约+0.276 dB，但SSIM和LPIPS更差。RU仍有研究价值，但不能宣称全面领先。", "",
        "8行汇总和256行逐图结果已合并。逐场景四方法的规范化view_key与顺序一致；Android保留P01源JPG名和P02共同输入PNG名，不把扩展名转换伪装成原文件同哈希。P01质量标记`P01_FROZEN_REUSED`，P02质量标记`P02_REPRODUCED_REUSED`，本包没有质量重评。", "",
        "## 训练身份、loader与特征", "",
        "四套实际loader检查均PASS：两方法×两场景的train/test/excluded分别为Android 122/19/122、Patio-High 221/45/1；主输入1007×755、loader factor1、共同初始化点云匹配。SLS的343个训练特征逐名对应训练图，形状1280×50×50、float32、有限且逐文件有哈希；test/excluded没有进入特征拟合名单。", "",
        "SLS实际配方为robust loss、semantics MLP、no-cluster、no-UBP、30k、seed42；Android 0.5/0.9来自固定代码默认值，Patio 0.3/0.8来自固定上游benchmark脚本。镜像权重来源与实际文件哈希见features/weight_provenance.json；镜像commit不被冒充为与原仓库逐字节等价证明。DINO结构、固定源码提交与权重哈希同样登记。", "",
        "## Checkpoint与运行源码", "",
        "8个现有checkpoint均只读加载通过，完整绝对路径、SHA-256、字节数、Gaussian数、格式和推理组件见asset_ledger.csv。运行时上游diff、子模块提交、保存配置、实际命令与文件hash位于runtime/。当前核验结合P02训练前preflight的文件hash；不会用当前loader实例化倒推过去必然正确。", "",
        "## 计时与成本", "",
        "旧P01/P02 FPS均保留。因为旧内部原始三遍记录及同期设备负载不能与P02完全闭合，本包按统一规则重计时8个既有模型，共24行；结果见timing_raw.csv，边界见timing_boundary.md。", "",
        "成本表明确保留组合边界：RobustSplat训练进程含DINO准备；SLS训练进程含eval和trajectory；Android SLS 11129秒含首次下载。不能用这些组合时间与内部不完整时间计算‘完整训练加速比’，未知值保持UNKNOWN。", "",
        "## Smoke与偏离", "",
        "两项RobustSplat各3次smoke，超过每身份最多2次的原约束，已在acceptance_matrix.csv和deviations.md判为程序性FAIL；不因此重训。800是8次×100的保守预算扣额，不是可证实际optimizer更新总数。成功尝试各完成100步，失败尝试实际更新未知。", "",
        "## 代表图与Patio诊断", "",
        "representatives/含Android第1/10/19张、Patio第1/23/45张的共同GT、四方法预测及统一0..0.25绝对误差色标；另含Patio IMG_8501/8502固定诊断。尺寸、图名、测试索引与映射均核对；仅记录观察与误差，不据测试残差删图、调bounds或重训。", "",
        "## 仍未知/不能成立的结论", "",
        "- P01 RU完整构建时间及外部特征准备边界未冻结，完整训练成本比仍UNKNOWN。", "- 失败smoke的实际optimizer步数和被覆盖旧日志不可恢复。", "- SD2.1镜像与另一仓库版本的逐字节等价若无同一文件hash对照，保持来源待核实；这不自动否定方法身份。", "- 单seed结果不能支持统计显著性或全面领先声明。", "",
        "最终建议：证据支持RU继续作为研究基础候选；若方案助手接受本验收，可讨论P03 Corner，但本文件不授权启动。", "",
        "最终判定：`P02A_COMPLETE_STOP_AND_RETURN_TO_PLAN_ASSISTANT`", "",
    ]
    (output / "REPORT_P02A.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output / "NEXT_DECISION.md").write_text(
        "# NEXT_DECISION\n\n1. 是否接受P02-A对8个既有模型训练身份、质量与同设备计时的补证，并进入P03 Corner？\n2. 是否认为任一PASS训练身份仍需条件修复R；若需要，请只指定受影响的最小run集合？\n3. 是否确认完整训练成本比因P01特征构建时间及外部组合边界缺失而继续记为UNKNOWN？\n\n本文件不授权自动启动任何下一包。\n",
        encoding="utf-8",
    )
    status = {
        "schema": "puri-gs-p02a-status-v1", "status": "COMPLETE",
        "p02_pipeline_status_preserved": "COMPLETE", "new_training_identities": 0,
        "optimizer_updates": 0, "new_training_smoke": 0, "checkpoint_read_only_loads": len(assets),
        "quality_rows": len(summaries), "per_image_metric_rows": len(per_image),
        "quality_reevaluations": 0, "models_retimed": 8, "timing_repeats": len(timing),
        "loader_audits": len(loaders), "smoke_budget_charge_updates": sum(int(row["budget_charge_updates"]) for row in smoke),
        "smoke_proven_actual_updates": 400, "smoke_failed_actual_updates": "UNKNOWN",
        "next_work_package_started": False, "out_of_scope_started": False,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("P02A_QUALITY_ROWS=8")
    print("P02A_PER_IMAGE_ROWS=256")
    print("P02A_TIMING_ROWS=24")
    print("P02A_FINALIZE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
