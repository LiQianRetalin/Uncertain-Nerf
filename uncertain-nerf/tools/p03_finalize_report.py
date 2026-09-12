#!/usr/bin/env python3
"""Assemble, validate, visualize, and package the bounded P03 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


RUNS = (
    ("P03-corner-ru", "puri-gs-ru"),
    ("P03-corner-robustsplat", "robustsplat"),
    ("P03-corner-sls-mlp", "sls-mlp"),
)

P02A = (
    ("Android", "B1", 23.3873, 0.79437, 0.17030, 1218407, 255.693),
    ("Android", "RU", 24.6472, 0.82119, 0.16470, 551297, 483.118),
    ("Android", "RobustSplat", 24.0392, 0.78722, 0.17658, 1029224, 185.514),
    ("Android", "SLS-mlp", 24.5743, 0.81270, 0.18303, 606595, 425.784),
    ("Patio-High", "B1", 14.2167, 0.42500, 0.62149, 3235690, 110.436),
    ("Patio-High", "RU", 19.3408, 0.65173, 0.30124, 435779, 499.000),
    ("Patio-High", "RobustSplat", 20.9208, 0.68625, 0.26185, 587847, 195.139),
    ("Patio-High", "SLS-mlp", 19.6165, 0.56703, 0.41399, 442526, 459.422),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timing_path(root: Path, run_id: str) -> Path:
    if run_id.endswith("-ru"):
        return root / "report" / "timing" / run_id / "result.json"
    if run_id.endswith("-robustsplat"):
        return root / "outputs" / run_id / "attempt_1" / "test" / "ours_30000" / "p03_timing.json"
    return root / "outputs" / run_id / "attempt_1" / "stats" / "p03_timing_step29999.json"


def prediction_dir(root: Path, run_id: str) -> Path:
    if run_id.endswith("-ru"):
        return root / "report" / "native" / run_id / "float_predictions"
    if run_id.endswith("-robustsplat"):
        return root / "outputs" / run_id / "attempt_1" / "test" / "ours_30000" / "float_predictions"
    return root / "outputs" / run_id / "attempt_1" / "renders" / "float_step29999"


def memory_path(root: Path, run_id: str) -> Path:
    if run_id.endswith("-ru"):
        return root / "report" / "memory" / run_id / "result.json"
    if run_id.endswith("-robustsplat"):
        return root / "outputs" / run_id / "attempt_1" / "p03_inference_memory.json"
    return root / "outputs" / run_id / "attempt_1" / "stats" / "p03_inference_memory_step29999.json"


def _error_color(error: np.ndarray) -> np.ndarray:
    value = np.clip(error / 0.5, 0.0, 1.0)
    red = np.clip(2.0 * value, 0.0, 1.0)
    green = np.clip(2.0 - 2.0 * np.abs(value - 0.5), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - value), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def _label(image: Image.Image, text: str) -> Image.Image:
    labeled = Image.new("RGB", (image.width, image.height + 24), "white")
    labeled.paste(image, (0, 24))
    ImageDraw.Draw(labeled).text((6, 6), text, fill="black")
    return labeled


def make_representatives(root: Path, common: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    output = root / "report" / "representatives"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    labels = ("RU", "RobustSplat", "SLS-mlp")
    run_ids = tuple(run_id for run_id, _ in RUNS)
    for test_name in manifest["representative_test_names"]:
        gt_path = common / "images" / test_name
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        predictions = []
        for run_id in run_ids:
            path = prediction_dir(root, run_id) / Path(test_name).with_suffix(".npy").name
            value = np.load(path, allow_pickle=False)
            if value.shape != gt.shape or not np.isfinite(value).all():
                raise RuntimeError(f"representative prediction invalid: {run_id}/{test_name}")
            predictions.append(value)

        size = (504, 378)
        top = [_label(Image.fromarray(np.uint8(np.clip(gt, 0, 1) * 255)).resize(size), "GT")]
        bottom = [_label(Image.new("RGB", size, "white"), "absolute RGB error; common scale 0..0.5")]
        for label, value in zip(labels, predictions):
            rendered = Image.fromarray(np.uint8(np.clip(value, 0, 1) * 255)).resize(size)
            error = np.mean(np.abs(value - gt), axis=2)
            heat = Image.fromarray(np.uint8(_error_color(error) * 255)).resize(size)
            top.append(_label(rendered, label))
            bottom.append(_label(heat, f"{label} mean absolute RGB error"))
        panel = Image.new("RGB", (size[0] * 4, (size[1] + 24) * 2), "white")
        for index, image in enumerate(top):
            panel.paste(image, (index * size[0], 0))
        for index, image in enumerate(bottom):
            panel.paste(image, (index * size[0], size[1] + 24))
        path = output / f"representative_{Path(test_name).stem}.png"
        panel.save(path, optimize=True)
        records.append(
            {
                "image_name": test_name,
                "panel_file": path.name,
                "panel_sha256": sha256_file(path),
                "error_scale": "mean absolute RGB error, fixed [0,0.5]",
            }
        )
    write_csv(
        output / "manifest.csv",
        ["image_name", "panel_file", "panel_sha256", "error_scale"],
        records,
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--status", choices=("COMPLETE", "PARTIAL"), required=True)
    args = parser.parse_args()
    root = args.work_root.expanduser().resolve()
    common = args.common_dir.expanduser().resolve()
    report = root / "report"
    state = root / "state"
    frozen = read_json(report / "protocol_manifest.json")

    summaries = []
    per_image = []
    timing_rows = []
    latency_rows = []
    timing_medians: dict[str, float] = {}
    for run_id, method in RUNS:
        run_report = report / "runs" / run_id
        if (run_report / "summary.json").is_file():
            summaries.append(read_json(run_report / "summary.json"))
        per_image.extend(read_csv(run_report / "per_image_metrics.csv"))
        path = timing_path(root, run_id)
        if path.is_file():
            payload = read_json(path)
            records = payload.get("repeats", payload.get("records", []))
            per_records = payload.get("per_image", [])
            if run_id.endswith("-ru"):
                per_records = read_csv(path.parent / "per_image_latency.csv")
            for row in records:
                timing_rows.append(
                    {
                        "run_id": run_id,
                        "scene": "corner",
                        "method": method,
                        "seed": 42,
                        "warmup": payload["warmup"],
                        "repeat": row["repeat"],
                        "frames": row["frames"],
                        "seconds": row["seconds"],
                        "fps": row["fps"],
                        "boundary": payload["boundary"],
                    }
                )
            for row in per_records:
                latency_rows.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "repeat": row["repeat"],
                        "image_index": row["image_index"],
                        "image_name": row["image_name"],
                        "seconds": row["seconds"],
                    }
                )
            if records:
                timing_medians[run_id] = statistics.median(float(row["fps"]) for row in records)

    summary_fields = [
        "run_id", "scene", "method", "seed", "test_image_count", "psnr", "ssim", "lpips",
        "checkpoint_sha256", "protocol_sha256", "evaluator_id", "lpips_network",
        "lpips_normalize", "source",
    ]
    write_csv(report / "summary.csv", summary_fields, summaries)
    per_fields = list(per_image[0]) if per_image else [
        "run_id", "scene", "method", "seed", "image_index", "image_name", "psnr", "ssim",
        "lpips", "prediction_sha256", "checkpoint_sha256", "evaluator_id", "lpips_network",
        "lpips_normalize", "source",
    ]
    write_csv(report / "per_image_metrics.csv", per_fields, per_image)
    write_csv(
        report / "timing_raw.csv",
        ["run_id", "scene", "method", "seed", "warmup", "repeat", "frames", "seconds", "fps", "boundary"],
        timing_rows,
    )
    write_csv(
        report / "timing_per_image_latency.csv",
        ["run_id", "method", "repeat", "image_index", "image_name", "seconds"],
        latency_rows,
    )

    gpu_events = read_csv(state / "gpu_stage_events.csv")
    wall_events = read_csv(state / "stage_events.csv")
    write_csv(
        report / "cost_breakdown.csv",
        ["stage", "category", "run_id", "gpu", "started_epoch", "ended_epoch", "elapsed_seconds", "exit_code", "timeout_seconds", "scope"],
        [
            {**row, "scope": "GPU wall time; one selected physical L20 per stage"}
            for row in gpu_events
        ]
        + [{**row, "scope": "CPU/network wall time"} for row in wall_events],
    )
    method_cost_rows = []
    for run_id, method in RUNS:
        run_events = [row for row in gpu_events if row["run_id"] == run_id]
        feature_seconds = sum(float(row["elapsed_seconds"]) for row in run_events if row["category"] == "features")
        formal_seconds = sum(float(row["elapsed_seconds"]) for row in run_events if row["category"] == "formal_training")
        smoke_seconds = sum(float(row["elapsed_seconds"]) for row in run_events if row["category"].startswith("smoke"))
        post_seconds = sum(float(row["elapsed_seconds"]) for row in run_events if row["category"] in {"timing", "quality_render", "independent_evaluation", "inference_memory"})
        method_cost_rows.append(
            {
                "run_id": run_id,
                "method": method,
                "scene_feature_seconds": feature_seconds,
                "formal_training_and_native_export_seconds": formal_seconds,
                "complete_model_build_seconds": feature_seconds + formal_seconds,
                "smoke_seconds_separate": smoke_seconds,
                "post_training_timing_quality_memory_seconds": post_seconds,
                "definition": "fixed environments/weights ready; scene features through full training and necessary native model export",
            }
        )
    write_csv(
        report / "method_cost_summary.csv",
        [
            "run_id", "method", "scene_feature_seconds", "formal_training_and_native_export_seconds",
            "complete_model_build_seconds", "smoke_seconds_separate",
            "post_training_timing_quality_memory_seconds", "definition",
        ],
        method_cost_rows,
    )

    memory_samples = read_csv(state / "gpu_memory_samples.csv")
    checkpoint_rows = []
    for run_id, method in RUNS:
        audit_path = report / "checkpoints" / f"{run_id}.json"
        if not audit_path.is_file():
            continue
        audit = read_json(audit_path)
        memory_file = memory_path(root, run_id)
        memory = read_json(memory_file) if memory_file.is_file() else {}
        training_values = [
            int(row["memory_used_mib"])
            for row in memory_samples
            if row["stage"] == f"train_{run_id}"
        ]
        checkpoint_rows.append(
            {
                "run_id": run_id,
                "method": method,
                "step": audit["step"],
                "step_semantics": "iteration 30000" if method == "robustsplat" else "zero-based step 29999",
                "gaussian_count": audit["gaussian_count"],
                "checkpoint_sha256": audit["checkpoint_sha256"],
                "checkpoint_bytes": audit["checkpoint_bytes"],
                "training_peak_device_memory_mib": max(training_values) if training_values else "UNKNOWN",
                "training_peak_source": "2-second nvidia-smi device-memory sampling",
                "inference_peak_allocated_bytes": memory.get("max_memory_allocated_bytes", "UNKNOWN"),
                "inference_peak_reserved_bytes": memory.get("max_memory_reserved_bytes", "UNKNOWN"),
                "inference_asset_bytes": audit["checkpoint_bytes"],
                "inference_asset_scope": "native checkpoint file; optimizer-state format differences retained",
            }
        )
    write_csv(
        report / "model_vram_ledger.csv",
        [
            "run_id", "method", "step", "step_semantics", "gaussian_count", "checkpoint_sha256",
            "checkpoint_bytes", "training_peak_device_memory_mib", "training_peak_source",
            "inference_peak_allocated_bytes", "inference_peak_reserved_bytes", "inference_asset_bytes",
            "inference_asset_scope",
        ],
        checkpoint_rows,
    )

    smoke = read_csv(state / "smoke_ledger.csv")
    formal = read_csv(state / "formal_ledger.csv")
    write_csv(report / "smoke_ledger.csv", list(smoke[0]) if smoke else ["run_id", "attempt", "status", "budget_updates", "actual_step", "gpu", "exit_code", "reason"], smoke)
    write_csv(report / "run_ledger.csv", list(formal[0]) if formal else ["run_id", "attempt", "status", "seed", "configured_updates", "actual_step", "gpu", "started_epoch", "ended_epoch", "exit_code", "reason"], formal)
    gpu_seconds = sum(float(row["elapsed_seconds"]) for row in gpu_events)
    budget = {
        "schema": "puri-gs-p03-budget-actual-v1",
        "status": args.status,
        "limits": {
            "smoke_attempts_per_identity": 2,
            "smoke_updates_per_attempt": 100,
            "smoke_global_updates": 600,
            "formal_attempts_per_identity": 2,
            "formal_updates": 30000,
            "gpu_seconds": 43200,
        },
        "actual": {
            "smoke_attempts": len(smoke),
            "smoke_budget_updates": sum(int(row["budget_updates"]) for row in smoke),
            "formal_attempts": len(formal),
            "gpu_seconds": gpu_seconds,
            "gpu_hours": gpu_seconds / 3600.0,
        },
        "automatic_retry": False,
        "adaptive_tuning_from_test_results": False,
    }
    (report / "budget.json").write_text(json.dumps(budget, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    representatives = []
    if args.status == "COMPLETE":
        expected_ids = {run_id for run_id, _ in RUNS}
        if {str(row["run_id"]) for row in summaries} != expected_ids:
            raise RuntimeError("refusing COMPLETE: three independent summaries are required")
        if len(per_image) != 60 or len(timing_rows) != 9 or len(latency_rows) != 180:
            raise RuntimeError("refusing COMPLETE: 60 metrics, 9 timing passes, and 180 latencies are required")
        if len(checkpoint_rows) != 3 or gpu_seconds > 43200:
            raise RuntimeError("refusing COMPLETE: model ledger incomplete or GPU budget exceeded")
        if any(int(row["test_image_count"]) != 20 for row in summaries):
            raise RuntimeError("refusing COMPLETE: each identity requires 20 test views")
        if any(sum(row["run_id"] == run_id for row in timing_rows) != 3 for run_id in expected_ids):
            raise RuntimeError("refusing COMPLETE: each identity requires three timing passes")
        if any(sum(row["run_id"] == run_id for row in latency_rows) != 60 for run_id in expected_ids):
            raise RuntimeError("refusing COMPLETE: each identity requires 60 raw latencies")
        final_formal = {row["run_id"]: row for row in formal}
        if any(final_formal.get(run_id, {}).get("status") != "COMPLETE" for run_id in expected_ids):
            raise RuntimeError("refusing COMPLETE: formal ledger is incomplete")
        representatives = make_representatives(root, common, frozen)

    (report / "timing_boundary.md").write_text(
        "# P03 统一计时边界\n\n"
        "三方法均在模型和相机准备后进入 `eval/no_grad`，按同一冻结 test 顺序显式 warmup 10 次。"
        "每个视图依次执行 CUDA 同步、启动高精度墙钟、完整原生渲染及 clamp[0,1]、CUDA 同步。"
        "存图、公共指标与相机预加载均在计时外。完整 20 图运行 3 遍，正式 FPS 为三遍 FPS 中位数。\n\n"
        "P02-A 的旧 FPS 含两种同步粒度，只作原生路径参考；不与 P03 合并计算三场景加速均值。\n",
        encoding="utf-8",
    )

    by_id = {row["run_id"]: row for row in summaries}
    model_by_id = {row["run_id"]: row for row in checkpoint_rows}
    lines = [
        "# REPORT_P03",
        "",
        f"状态：`{args.status}`",
        "",
        "## Corner 实际结果",
        "",
        "| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Gaussian数 | 统一FPS中位数↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run_id, method in RUNS:
        row = by_id.get(run_id)
        model = model_by_id.get(run_id)
        if row and model and run_id in timing_medians:
            lines.append(
                f"| {method} | {float(row['psnr']):.4f} | {float(row['ssim']):.5f} | "
                f"{float(row['lpips']):.5f} | {model['gaussian_count']} | {timing_medians[run_id]:.3f} |"
            )
        else:
            lines.append(f"| {method} | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |")
    if len(by_id) == 3 and len(model_by_id) == 3:
        ru = by_id["P03-corner-ru"]
        ru_model = model_by_id["P03-corner-ru"]
        comparison_lines = []
        for other_id, label in (("P03-corner-robustsplat", "RobustSplat"), ("P03-corner-sls-mlp", "SLS-mlp")):
            other = by_id[other_id]
            other_model = model_by_id[other_id]
            quality_all = (
                float(ru["psnr"]) > float(other["psnr"])
                and float(ru["ssim"]) > float(other["ssim"])
                and float(ru["lpips"]) < float(other["lpips"])
            )
            gaussian_delta = (int(ru_model["gaussian_count"]) / int(other_model["gaussian_count"]) - 1.0) * 100.0
            comparison_lines.append(
                f"相对 {label}：RU 的 PSNR 差 {float(ru['psnr']) - float(other['psnr']):+.4f} dB，"
                f"SSIM 差 {float(ru['ssim']) - float(other['ssim']):+.5f}，LPIPS 差 {float(ru['lpips']) - float(other['lpips']):+.5f}；"
                f"Gaussian 数差 {gaussian_delta:+.2f}%。三项质量{'均优' if quality_all else '存在取舍'}。"
            )
        lines.extend(["", "## Corner质量—规模位置", "", *comparison_lines])
    lines.extend(
        [
            "",
            "三项均为 seed42 单次确认运行；不作统计显著性声明，也不因结果调参或补跑 seed。",
            "",
            "## 三场景定位",
            "",
            "Android 与 Patio-High 数字原样引用 P02-A；Corner 仅加入本包三个身份，不补造 B1。旧 P02-A FPS 同步粒度不统一，因此不与 P03 FPS 做跨场景精确加速平均。",
            "",
            "| 场景 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Gaussian数 | FPS参考↑ |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scene, method, psnr, ssim, lpips, gaussians, fps in P02A:
        lines.append(f"| {scene} | {method} | {psnr:.4f} | {ssim:.5f} | {lpips:.5f} | {gaussians} | {fps:.3f}* |")
    for run_id, method in RUNS:
        row = by_id.get(run_id)
        model = model_by_id.get(run_id)
        if row and model and run_id in timing_medians:
            lines.append(
                f"| Corner | {method} | {float(row['psnr']):.4f} | {float(row['ssim']):.5f} | "
                f"{float(row['lpips']):.5f} | {model['gaussian_count']} | {timing_medians[run_id]:.3f} |"
            )
    lines.extend(
        [
            "",
            "`*` P02-A 原生 FPS 仅作旧口径参考。P03 三方法使用本包统一逐视图同步口径。",
            "",
            "## 有效性、成本与限制",
            "",
            f"完成独立评测 {len(summaries)}/3、逐图 {len(per_image)}/60、计时重复 {len(timing_rows)}/9、原始逐图延迟 {len(latency_rows)}/180。",
            f"P03 新增 GPU 消耗 {gpu_seconds / 3600.0:.3f} GPU·小时（硬上限 12）；分项边界见 `cost_breakdown.csv`。",
            "每方法从场景特征开始到正式训练及必要原生模型导出结束的完整构建边界见 `method_cost_summary.csv`；smoke与后训练评测另列，阶段不重复求和。",
            "checkpoint 大小是原生文件大小，可能包含 optimizer 状态；必要推理资产按原生 checkpoint 单列，不把格式差异解释为压缩收益。",
            "共同代表图固定于训练前的 test 首/中/末，误差图统一使用 [0,0.5] 色标。完整 test 始终保留。",
            "P01/P02 旧完整构建成本仍为 UNKNOWN；Corner 成本齐全不构成三场景训练加速证明。",
            "",
            "本工作包到此停止，不启动工作包4、OAC、修复R或其他场景。",
        ]
    )
    (report / "REPORT_P03.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = {
        "schema": "puri-gs-p03-status-v1",
        "status": args.status,
        "planned_identities": [run_id for run_id, _ in RUNS],
        "evaluated_identities": [row["run_id"] for row in summaries],
        "scientific_conclusion_status": "READY_FOR_PLAN_ASSISTANT" if args.status == "COMPLETE" else "INSUFFICIENT_EVIDENCE",
        "engineering_status": "VALID" if args.status == "COMPLETE" else "PARTIAL",
        "unresolved_items": [] if args.status == "COMPLETE" else ["See failed attempt and stage ledgers"],
        "representative_count": len(representatives),
        "gpu_seconds": gpu_seconds,
        "next_work_package_started": False,
    }
    (report / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (report / "NEXT_DECISION.md").write_text(
        "# NEXT_DECISION\n\n"
        "1. Corner 的质量—Gaussian 数—构建成本位置是否足以继续以原 RU 为 Parent？\n"
        "2. 是否存在有明确实现证据、足以启动条件修复 R 的问题？\n"
        "3. 是否具备由方案助手授权工作包4有界机制检查的依据？\n\n"
        "P03 不自行回答下一包授权；未启动工作包4或 OAC。\n",
        encoding="utf-8",
    )

    package = root / f"P03-final-{args.status.lower()}.zip"
    if package.exists():
        raise RuntimeError(f"refusing to overwrite package: {package}")
    included = []
    for directory, prefix in ((report, "report"), (state, "state"), (root / "logs", "logs")):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                included.append((path, Path("P03-final") / prefix / path.relative_to(directory)))
    for directory, prefix in ((root / "outputs", "attempt_metadata/formal"), (root / "smoke", "attempt_metadata/smoke")):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.casefold() in {".json", ".txt", ".yaml", ".yml"}:
                included.append((path, Path("P03-final") / prefix / path.relative_to(directory)))
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, arcname in included:
            archive.write(source, arcname.as_posix())
    package_hash = sha256_file(package)
    checksum = package.with_suffix(package.suffix + ".sha256")
    checksum.write_text(f"{package_hash}  {package.name}\n", encoding="ascii")
    with zipfile.ZipFile(package) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"package CRC validation failed: {bad}")
    print(f"P03_REPORT={args.status}")
    print(f"P03_PACKAGE={package}")
    print(f"P03_PACKAGE_SHA256={package_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
