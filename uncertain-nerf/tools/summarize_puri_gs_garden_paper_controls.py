#!/usr/bin/env python3
"""Summarize the fixed Garden RU-Align/TAR controls; never launch experiments."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puri_gs.paper_controls import write_json_new
from tools.audit_puri_gs_paper_controls import audit_run, config_audit, read, _finite_tree
from tools.summarize_puri_gs_garden_causal_2x2 import (
    _effective_baseline, _load_run, _pairing_audit, _representative_images, _split_identity,
    paired_bootstrap_ci95,
)

METRICS = ("psnr", "ssim", "lpips")
COMPACT_COUNT = 2848877


def paired_comparison(before: dict, after: dict) -> dict:
    names = [row["image_name"] for row in before["per_image"]]
    if len(names) != 24 or [row["image_name"] for row in after["per_image"]] != names:
        raise ValueError("paired image names/order differ")
    result = {}
    for metric in METRICS:
        delta = [b[metric] - a[metric] for a, b in zip(before["per_image"], after["per_image"], strict=True)]
        higher = metric != "lpips"
        worst = min(range(24), key=lambda i: delta[i]) if higher else max(range(24), key=lambda i: delta[i])
        result[metric] = {
            "mean_difference": statistics.mean(delta), "median_difference": statistics.median(delta),
            "favorable_count": sum(d > 0 if higher else d < 0 for d in delta), "image_count": 24,
            "ci95": paired_bootstrap_ci95(delta, samples=10000, seed=42),
            "bootstrap_samples": 10000, "bootstrap_seed": 42,
            "worst_view_name": names[worst], "worst_view_difference": delta[worst],
            "DSC07988_difference": delta[names.index("DSC07988.JPG")],
            "direction": "higher_is_better" if higher else "lower_is_better",
            "per_image": [dict(image_name=n, difference=d) for n, d in zip(names, delta, strict=True)],
        }
    return {"scalar_deltas": {m: after["metrics"][m] - before["metrics"][m] for m in METRICS},
            "gaussian_delta": after["training"]["gaussian_count"] - before["training"]["gaussian_count"],
            "paired": result}


def quality_gates(b1: dict, run: dict) -> dict:
    metrics = run["metrics"]
    worst = min(b["psnr"] - a["psnr"] for a, b in zip(b1["per_image"], run["per_image"], strict=True))
    clean = {"psnr": metrics["psnr"] >= 27.566728,
             "ssim": metrics["ssim"] >= 0.868957, "lpips": metrics["lpips"] <= 0.078151,
             "worst_view": worst >= -1.0}
    pareto = {m: metrics[m] >= b1["metrics"][m] if m != "lpips" else metrics[m] <= b1["metrics"][m]
              for m in METRICS}
    return {"clean_checks": clean, "worst_view_delta_psnr": worst,
            "recovery_label": "GARDEN_CLEAN_TOLERANCE_RECOVERED" if all(clean.values()) else "GARDEN_CLEAN_TOLERANCE_NOT_RECOVERED",
            "b1_metric_checks": pareto,
            "pareto_label": "GARDEN_MEAN_QUALITY_PARETO_NOT_WORSE_THAN_B1" if all(pareto.values()) else "GARDEN_MEAN_QUALITY_NOT_PARETO_RECOVERED",
            "count_reference": COMPACT_COUNT, "count_reference_pass": run["training"]["gaussian_count"] <= COMPACT_COUNT}


def make_summary(runs: dict, representatives: dict) -> dict:
    pairs = {"RU_Align_minus_RU": ("RU", "RU-Align"), "RU_Align_minus_B1": ("B1", "RU-Align")}
    if "RU-TAR" in runs:
        pairs.update(RU_TAR_minus_RU_Align=("RU-Align", "RU-TAR"),
                     RU_TAR_minus_RU=("RU", "RU-TAR"), RU_TAR_minus_B1=("B1", "RU-TAR"))
    effects = {key: paired_comparison(runs[a], runs[b]) for key, (a, b) in pairs.items()}
    gates = {name: quality_gates(runs["B1"], runs[name]) for name in ("RU-Align", "RU-TAR") if name in runs}
    capacity = {"status": "not_available_until_RU_TAR"}
    align = effects["RU_Align_minus_RU"]
    inputs = {f"ALIGN_DELTA_{m.upper()}": align["scalar_deltas"][m] for m in METRICS}
    inputs["ALIGN_PAIRED_CI"] = {m: align["paired"][m]["ci95"] for m in METRICS}
    if "RU-TAR" in runs:
        late = effects["RU_TAR_minus_RU_Align"]
        added = late["gaussian_delta"]
        rates = {m: late["scalar_deltas"][m] / (added / 1e6) if added > 0 else None for m in METRICS}
        capacity = {"status": "defined" if added > 0 else "undefined_nonpositive_added_count",
                    "added_gaussians": added, "per_million_gaussians": rates,
                    "lpips_direction": "lower_is_better", "causal_optimality_claim": False}
        inputs.update({f"LATE_DELTA_{m.upper()}": late["scalar_deltas"][m] for m in METRICS})
        inputs.update(LATE_PAIRED_CI={m: late["paired"][m]["ci95"] for m in METRICS},
                      LATE_ADDED_GAUSSIANS=added, LATE_PSNR_PER_MILLION_GAUSSIANS=rates["psnr"])
        phase = runs["RU-TAR"]["topology"]["phases"]["late"]
        inputs.update({f"LATE_{action.upper()}_TOTAL": phase[f"{action}_count"] for action in ("split", "clone", "prune")})
    target = "RU-TAR" if "RU-TAR" in runs else "RU-Align"
    worst = effects["RU_TAR_minus_B1" if target == "RU-TAR" else "RU_Align_minus_B1"]["paired"]["psnr"]
    inputs.update(GARDEN_RECOVERY_GATE=gates[target]["recovery_label"],
                  GARDEN_75_PERCENT_B1_COUNT_GATE=gates[target]["count_reference_pass"],
                  WORST_VIEW_NAME=worst["worst_view_name"], WORST_VIEW_DELTA_PSNR=worst["worst_view_difference"],
                  DSC07988_DELTA_PSNR=worst["DSC07988_difference"])
    result = {"protocol": "garden-ru-align-tar-paper-controls-v1", "branch": "dev",
              "stage": "final" if target == "RU-TAR" else "RU-Align intermediate stop",
              "positioning": "RU-Align is a scale/timing control; RU-TAR is a fixed late-topology capacity control, not the final algorithm.",
              "runs": runs, "test_names": runs["B1"]["split"]["test"],
              "config_audit": config_audit(), "effects": effects, "gates": gates,
              "capacity": capacity, "representative_images": representatives,
              "next_stage_inputs": inputs,
              "limits": ["Single training seed; paired bootstrap describes test views at fixed checkpoints, not cross-seed uncertainty.",
                         "The late factor includes continued statistics accumulation and 20 refine events; their effects are not separated.",
                         "The original T package also includes reset, statistics and optimizer ordering; it is not fully decomposed.",
                         "Fixed 10k/20k/24k and 100/200 intervals are controls, not theoretical optima.",
                         "No claim of contribution-aware or dual-risk theory is tested."],
              "not_run": ["Android", "Room", "Patio-High", "3x3", "additional seeds", "schedule/threshold search"],
              "adaptive_topology_implemented": False, "phase_U_implemented": False,
              "stop_after_report": True}
    _finite_tree(result, "summary")
    return result


def representative_canvas(runs: dict, selected: dict, destination: Path) -> None:
    from PIL import Image, ImageChops, ImageDraw
    import numpy as np
    if destination.exists():
        raise FileExistsError(destination)
    strips = []
    labels = ["GT", "B1", "RU", "RU-Align", "RU-TAR", "abs(RU-TAR - GT)"]
    for row in selected["images"]:
        index = row["test_index_zero_based"]
        images, ground_truth = [], None
        for name in ("B1", "RU", "RU-Align", "RU-TAR"):
            path = Path(runs[name]["path"]) / "independent_eval/renders" / f"test_step29999_{index:04d}.png"
            with Image.open(path) as image:
                canvas = image.convert("RGB")
            width, height = canvas.size
            if width % 2:
                raise ValueError(f"invalid GT/render canvas: {path}")
            gt = canvas.crop((0, 0, width // 2, height))
            if ground_truth is not None and (gt.size != ground_truth.size or not np.array_equal(np.asarray(gt), np.asarray(ground_truth))):
                raise ValueError("representative ground truth differs across runs")
            ground_truth = gt
            images.append(canvas.crop((width // 2, 0, width, height)))
        assert ground_truth is not None
        panels = [ground_truth, *images, ImageChops.difference(images[-1], ground_truth)]
        small_height = round(256 * ground_truth.height / ground_truth.width)
        strip = Image.new("RGB", (256 * 6, small_height + 42), "white")
        draw = ImageDraw.Draw(strip)
        draw.text((6, 3), row["image_name"], fill="black")
        for i, (panel, label) in enumerate(zip(panels, labels, strict=True)):
            draw.text((i * 256 + 6, 22), label, fill="black")
            strip.paste(panel.resize((256, small_height), Image.Resampling.LANCZOS), (i * 256, 42))
        strips.append(strip)
    output = Image.new("RGB", (1536, sum(s.height for s in strips)), "white")
    y = 0
    for strip in strips:
        output.paste(strip, (0, y))
        y += strip.height
    with destination.open("xb") as stream:
        output.save(stream, format="PNG")


def render_markdown(summary: dict) -> str:
    lines = ["# Garden RU-Align / RU-TAR 论文对照", "", summary["positioning"], "",
             f"阶段：{summary['stage']}；branch=dev；161 train / 24 test，名称和顺序均通过。", "",
             "RU-Align 隔离相同100次 refine 下的尺度切换；RU-TAR 测量继续累计统计并增加20次 refine 的固定晚期窗口。", "",
             "| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Gaussian | 训练秒 | 峰值GiB | FPS | mean/p50/p95 ms | 推理GiB | checkpoint bytes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for name, run in summary["runs"].items():
        m, t, e, c = run["metrics"], run["training"], run["efficiency"], run["checkpoint"]
        mean = e.get("latency_mean_ms", 1000 / e["render_fps"])
        lines.append(f"|{name}|{m['psnr']:.6f}|{m['ssim']:.6f}|{m['lpips']:.6f}|{t['gaussian_count']}|{t['training_time_seconds']:.3f}|{t['peak_vram_gib']:.3f}|{e['render_fps']:.3f}|{mean:.3f}/{e['latency_p50_ms']:.3f}/{e['latency_p95_ms']:.3f}|{e['inference_vram_gib']:.3f}|{c.get('size_bytes', c.get('bytes'))}|")
    lines += ["", "## 配置、结果及追溯", ""]
    for name, run in summary["runs"].items():
        lines += [f"- {name}: `{run['path']}`；配置 `{run['path']}/config.yaml`；训练 commit `{run['git_commit']}`；评测 commit `{run['evaluation_git_commit']}`。"]
    lines += ["", "测试图：" + ", ".join(summary["test_names"]), "", "## 单变量差异和精确调度", "",
              "paper_control 仅是显式身份/日志标记；算法字段仅允许以下差异。", "", "```json",
              json.dumps({"diff": summary["config_audit"]["resolved_config_diff"], "schedules": summary["config_audit"]["schedules"]}, indent=2, ensure_ascii=False), "```",
              "", "## 固定效应与逐图配对", "", "LPIPS 差值越低越好；bootstrap 是固定 checkpoint 的视角不确定性。", "",
              "| 比较 | 指标 | 标量差 | 逐图均值差 | 中位数 | 有利/24 | 95% CI | 最差图:差值 | DSC07988差值 |",
              "|---|---|---:|---:|---:|---:|---|---|---:|"]
    for name, effect in summary["effects"].items():
        for metric, p in effect["paired"].items():
            lines.append(f"|{name}|{metric}|{effect['scalar_deltas'][metric]:.6f}|{p['mean_difference']:.6f}|{p['median_difference']:.6f}|{p['favorable_count']}/24|{p['ci95']}|{p['worst_view_name']}: {p['worst_view_difference']:.6f}|{p['DSC07988_difference']:.6f}|")
    phases = {name: run["topology"] for name, run in summary["runs"].items() if "topology" in run}
    lines += ["", "## 拓扑、质量门与单位容量收益", "", "```json",
              json.dumps({"phases": phases, "gates": summary["gates"], "capacity": summary["capacity"]}, ensure_ascii=False, indent=2), "```",
              "", "75%B1规模参考上限=2,848,877；超过上限不使实验无效，但表示未满足紧凑性参考门。", "",
              "## 固定代表图", "", summary["representative_images"]["selection_rule"], "",
              ", ".join(row["image_name"] for row in summary["representative_images"]["images"])]
    if summary.get("representative_canvas"):
        lines += ["", f"![6张固定代表图]({summary['representative_canvas']})", "", "最后一列是 RU-TAR 与 GT 的绝对 RGB 误差，未按视角单独归一化。"]
    lines += ["", "## 解释边界", "", *[f"- {item}" for item in summary["limits"]], "",
              "本阶段未运行 Android、Room、Patio-High、3×3、新 seed 或参数搜索；未实现自适应拓扑和阶段 U。RU-TAR 只是固定调度论文对照，不提升为正式算法。", "",
              "## 下一阶段设计输入", "", "```json",
              json.dumps(summary["next_stage_inputs"], ensure_ascii=False, indent=2), "```"]
    if summary["stage"] == "final":
        inputs, cap = summary["next_stage_inputs"], summary["capacity"]
        dominant = max(("split", "clone", "prune"), key=lambda action: inputs[f"LATE_{action.upper()}_TOTAL"])
        priority = "质量不足" if inputs["GARDEN_RECOVERY_GATE"] != "GARDEN_CLEAN_TOLERANCE_RECOVERED" else ("Gaussian规模" if not inputs["GARDEN_75_PERCENT_B1_COUNT_GATE"] else "同时维持质量与规模")
        lines += ["", "1. 同样100次事件下的尺度收益和24视角CI见上表；单seed无法回答跨训练种子的稳定性。",
                  f"2. 晚期增加 {cap['added_gaussians']} 个Gaussian；每百万Gaussian指标变化为 {cap['per_million_gaussians']}。该效率是描述性量，不是最优性。",
                  f"3. 晚期事件按受影响Gaussian数统计，最多的是 {dominant}；完整split/clone/prune计数见输入。",
                  f"4. 最差图 {inputs['WORST_VIEW_NAME']} 相对B1变化 {inputs['WORST_VIEW_DELTA_PSNR']:.6f} dB，应与均值差共同判断。",
                  f"5. 恢复门：{inputs['GARDEN_RECOVERY_GATE']}。",
                  "6. 2×2已见的Mask×拓扑负交互仍是解释约束；本阶段尚不能把局部负交互从其他机制单独识别。",
                  f"7. 按本阶段质量门和规模参考门，后续应优先关注：{priority}。", "",
                  "汇总到此停止，由用户将报告交给ChatGPT分析；不生成下一算法的公式、阈值或代码。"]
    else:
        lines += ["", "RU-Align 中间停点：等待用户复核并明确确认是否继续固定RU-TAR；不得自动训练或改参数。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--align-run", type=Path, required=True)
    parser.add_argument("--tar-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stem = "GARDEN_RU_ALIGN_TAR_PAPER_CONTROL" if args.tar_run else "GARDEN_RU_ALIGN_INTERIM"
    md, js = args.output_dir / f"PHASE_R_{stem}.md", args.output_dir / f"phase_r_{stem.lower()}.json"
    if md.exists() or js.exists():
        raise FileExistsError("refusing to overwrite paper-control reports")
    old = read(args.historical_report)
    quadrants = {key: _load_run(Path(old["runs"][key]["path"]), key) for key in ("Y00", "Y01", "Y10", "Y11")}
    _pairing_audit(quadrants)
    for key, run in quadrants.items():
        for field in ("git_commit", "evaluation_git_commit", "metrics", "training", "efficiency", "checkpoint"):
            if run[field] != old["runs"][key][field]:
                raise ValueError(f"historical evidence changed: {key}.{field}")
    representatives = _representative_images(quadrants)
    if representatives != old["representative_images"] or representatives["images"][0]["image_name"] != "DSC07988.JPG":
        raise ValueError("historical representative selection differs")
    runs = {run["method"]: run for run in quadrants.values()}
    runs["RU-Align"] = audit_run(args.align_run.resolve(), evaluation=True)
    if args.tar_run:
        runs["RU-TAR"] = audit_run(args.tar_run.resolve(), evaluation=True)
    for name, run in runs.items():
        if _split_identity(run["split"]) != _split_identity(runs["B1"]["split"]) or _effective_baseline(run["config"]) != _effective_baseline(runs["B1"]["config"]):
            raise ValueError(f"run is not comparable: {name}")
        if name in ("RU-Align", "RU-TAR") and run["config"]["paper_control"] != name.lower().replace("-", "_"):
            raise ValueError("mislabeled paper control")
    summary = make_summary(runs, representatives)
    if args.tar_run:
        destination = args.tar_run.resolve() / "paper_control_representatives.png"
        representative_canvas(runs, representatives, destination)
        summary["representative_canvas"] = str(destination)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_new(js, summary)
    with md.open("x", encoding="utf-8") as stream:
        stream.write(render_markdown(summary))
    print(f"PAPER-CONTROL-SUMMARY-PASS\n{md}\n{js}\nSTOP: user review required")


if __name__ == "__main__":
    main()
