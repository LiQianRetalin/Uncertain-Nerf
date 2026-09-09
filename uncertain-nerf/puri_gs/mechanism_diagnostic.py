"""Isolated RU-PART diagnostic evidence checks; importing never loads training code."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

HORIZON = 30_000
SNAPSHOT_STEP = 9_999
REPLAY_STEPS = 200
ORACLE_STEPS = 400
SOURCES = ("DSC07987.JPG", "DSC07989.JPG")
METHODS = ("B1", "DG-only", "Mask-only", "RU", "RU-Align", "RU-TAR")
DIAGNOSTIC_SNAPSHOT_SCHEMA = "puri-gs-ru-part-mechanism-diagnostic-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_reproduction(manifest_path: Path, rows_path: Path) -> dict:
    """Verify saved numerical evidence, without opening any RGB or checkpoint."""
    manifest = read_json(manifest_path)
    dataset = manifest["dataset"]
    names = dataset["test_names"]
    if (dataset["factor"], dataset["train_count"], dataset["test_count"]) != (4, 161, 24):
        raise ValueError("historical dataset protocol differs")
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError("historical test names are not 24 unique views")
    with rows_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if set(row["method"] for row in rows) != set(METHODS):
        raise ValueError("historical method inventory differs")
    summaries = {}
    maximum = 0.0
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if [row["image_name"] for row in selected] != names:
            raise ValueError(f"historical view order/count differs: {method}")
        errors = []
        for row in selected:
            actual, reference, recorded = (
                float(row[key]) for key in
                ("psnr_rerender", "psnr_reference", "psnr_abs_delta")
            )
            if not all(math.isfinite(x) for x in (actual, reference, recorded)):
                raise ValueError("nonfinite historical PSNR")
            delta = abs(actual - reference)
            if not math.isclose(delta, recorded, rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError("recorded PSNR error differs from actual subtraction")
            if row["all_finite"].lower() != "true" or delta > 0.001:
                raise ValueError("historical reproduction failed")
            errors.append(delta)
        summaries[method] = {"count": len(errors), "max_abs_delta_db": max(errors)}
        maximum = max(maximum, max(errors))
    gates = manifest["gates"]
    if gates.get("reproduction_gate") is not True:
        raise ValueError("historical manifest reproduction gate failed")
    if not math.isclose(maximum, float(gates["reproduction_max_abs_delta_db"]),
                        rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError("manifest maximum differs from per-image evidence")
    ru = manifest["checkpoints"]["RU"]
    if ru["sha256_before"] != ru["sha256_after"]:
        raise ValueError("historical RU checkpoint changed during audit")
    return {
        "status": "SAVED_REPRODUCTION_VERIFIED",
        "methods": summaries,
        "max_abs_delta_db": maximum,
        "manifest_sha256": sha256_file(manifest_path),
        "per_image_sha256": sha256_file(rows_path),
        "test_rgb_opened": False,
        "checkpoint_current_hash_verified": False,
    }


def verify_current_sources(manifest_path: Path) -> dict:
    """Server-only provenance verification. Missing old summary is explicit."""
    manifest = read_json(manifest_path)
    ru = manifest["checkpoints"]["RU"]
    result = {}
    for label, filename, expected in (
        ("ru_checkpoint", ru["path"], ru["sha256_before"]),
        ("ru_config", ru["config_path"], ru["config_sha256"]),
    ):
        path = Path(filename)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source missing or changed: {label}")
        result[label] = {"path": str(path), "sha256": expected, "verified": True}
    source = manifest["source_report"]
    path = Path(source["path"])
    if path.is_file():
        if sha256_file(path) != source["sha256"]:
            raise ValueError("existing source report hash differs")
        result["source_report"] = {"status": "verified", **source}
    else:
        result["source_report"] = {"status": "not_available", **source}
    return result


def write_new_json(path: Path, value: dict) -> None:
    # Exclusive creation: existing audit evidence must never be replaced.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def filtered_suffix(sequence, monitor_ids, source_ids):
    """Preserve original order; never resample to increase source exposure."""
    blocked = set(monitor_ids)
    result = [int(x) for x in sequence[SNAPSHOT_STEP + 1:] if int(x) not in blocked][:ORACLE_STEPS]
    if len(result) != ORACLE_STEPS:
        raise ValueError("INSUFFICIENT_SUFFIX")
    exposures = {int(source): result.count(int(source)) for source in source_ids}
    if len(exposures) != 2 or min(exposures.values()) < 2:
        raise ValueError("INSUFFICIENT_SOURCE_EXPOSURE")
    return result, exposures


class Progress:
    """One parseable line every 200 updates, plus stage completion/failure."""

    def __init__(self, path: Path, stage: str, total: int):
        if total <= 0:
            raise ValueError("progress total must be positive")
        self.stream = path.open("x", encoding="utf-8")
        self.stage, self.total = stage, total
        self.started = time.monotonic()
        self.update(0)

    def update(self, completed: int, *, status="running", error=None):
        if not 0 <= completed <= self.total:
            raise ValueError("progress count outside stage bounds")
        if completed % 200 and completed != self.total and status == "running":
            return
        record = dict(stage=self.stage, completed=completed, total=self.total,
                      status=status, error=error, pid=os.getpid(),
                      updated_at=time.strftime("%Y-%m-%d %H:%M:%S%z"),
                      elapsed_seconds=round(time.monotonic() - self.started, 1))
        self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def compare_state(left, right, *, rtol=1e-5, atol=1e-7, chunk_size=262144):
    """Compare semantic state in bounded chunks; discrete state stays exact.

    A caller must exclude wall-clock diagnostics explicitly, never optimizer,
    sampler, histogram or topology state. Summary values cannot replace tensors.
    """
    import numpy as np
    import torch

    failures = []
    tensor_count, exact_count, max_error = 0, 0, 0.0

    def walk(a, b, path):
        nonlocal tensor_count, exact_count, max_error
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray):
                failures.append(path + ": type")
                return
            walk(torch.from_numpy(a), torch.from_numpy(b), path)
        elif isinstance(a, torch.Tensor):
            tensor_count += 1
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                failures.append(path + ": shape/dtype")
                return
            if a.layout != torch.strided or b.layout != torch.strided:
                raise ValueError("diagnostic state comparator requires dense tensors")
            # Contiguous state is guaranteed by the snapshot writer.
            if not a.is_contiguous() or not b.is_contiguous():
                raise ValueError("non-contiguous snapshot tensor")
            aa, bb = a.view(-1), b.view(-1)
            exact, close = True, True
            for start in range(0, a.numel(), chunk_size):
                x = aa[start:start + chunk_size].cpu()
                y = bb[start:start + chunk_size].cpu()
                same = torch.equal(x, y)
                exact &= same
                if x.is_floating_point():
                    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                    close &= finite and bool(torch.allclose(x, y, rtol=rtol, atol=atol))
                    if finite and x.numel():
                        max_error = max(max_error, float((x.double() - y.double()).abs().max()))
                else:
                    close &= same
            exact_count += int(exact)
            if not close:
                failures.append(path + ": values")
        elif isinstance(a, dict):
            if not isinstance(b, dict) or list(a) != list(b):
                failures.append(path + ": keys/order")
                return
            for key in a:
                walk(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                failures.append(path + ": sequence")
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{index}]")
        elif isinstance(a, float):
            if type(a) is not type(b) or not math.isfinite(a) or not math.isfinite(b):
                failures.append(path + ": scalar/type")
            elif not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                failures.append(path + ": scalar")
        elif type(a) is not type(b) or a != b:
            failures.append(path + ": discrete")

    walk(left, right, "state")
    return dict(passed=not failures, failures=failures, tensors=tensor_count,
                bitwise_equal_tensors=exact_count, max_abs_tensor_error=max_error,
                rtol=rtol, atol=atol)


def snapshot_tree(value):
    """Detach state into independent contiguous CPU storage, preserving order."""
    import copy
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous().clone()
    if isinstance(value, dict):
        return {key: snapshot_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_tree(item) for item in value)
    return copy.deepcopy(value)


def optimizer_references_current(params, optimizers):
    """Fail on stale row-changing Parameter references or unexpected groups."""
    if list(params) != list(optimizers):
        raise ValueError("Gaussian parameter/optimizer order differs")
    for name, parameter in params.items():
        groups = optimizers[name].param_groups
        if len(groups) != 1 or len(groups[0]["params"]) != 1:
            raise ValueError(f"unsupported optimizer group layout: {name}")
        if groups[0]["params"][0] is not parameter:
            raise ValueError(f"stale optimizer Parameter reference: {name}")


def save_training_snapshot(
    path: Path,
    *,
    step: int,
    stage: str,
    splats: Mapping[str, Any],
    optimizers: Mapping[str, Any],
    schedulers: Sequence[Any],
    strategy_state: Mapping[str, Any],
    camera_sequence,
    intervention_mode: str,
    ru_training_state: Mapping[str, Any],
) -> None:
    """Save a post-update diagnostic state without weakening replay coverage."""
    from puri_gs.replay import camera_sequence_sha256, capture_rng_state

    if step < SNAPSHOT_STEP:
        raise ValueError("diagnostic snapshot precedes the registered boundary")
    if len(camera_sequence) != HORIZON:
        raise ValueError("diagnostic camera sequence must preserve the 30000-step horizon")
    optimizer_references_current(splats, optimizers)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"diagnostic snapshot exists: {destination}")
    import torch
    torch.save({
        "schema": DIAGNOSTIC_SNAPSHOT_SCHEMA,
        "stage": str(stage),
        "step": int(step),
        "next_step": int(step + 1),
        "intervention_mode": str(intervention_mode),
        "splats": snapshot_tree(dict(splats)),
        "optimizers": snapshot_tree({
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        }),
        "schedulers": snapshot_tree([
            scheduler.state_dict() for scheduler in schedulers
        ]),
        "strategy_state": snapshot_tree(dict(strategy_state)),
        "rng": snapshot_tree(capture_rng_state()),
        "camera_sequence": snapshot_tree(camera_sequence),
        "camera_sequence_sha256": camera_sequence_sha256(camera_sequence),
        "ru_training": snapshot_tree(dict(ru_training_state)),
    }, destination)


def load_training_snapshot(path: Path) -> dict:
    import torch
    from puri_gs.replay import camera_sequence_sha256

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {
        "schema", "stage", "step", "next_step", "intervention_mode", "splats",
        "optimizers", "schedulers", "strategy_state", "rng", "camera_sequence",
        "camera_sequence_sha256", "ru_training",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("diagnostic snapshot has an invalid top-level schema")
    if payload["schema"] != DIAGNOSTIC_SNAPSHOT_SCHEMA:
        raise ValueError("unsupported diagnostic snapshot schema")
    if payload["next_step"] != payload["step"] + 1:
        raise ValueError("diagnostic snapshot step boundary is inconsistent")
    if len(payload["camera_sequence"]) != HORIZON:
        raise ValueError("diagnostic snapshot changed the training horizon")
    if camera_sequence_sha256(payload["camera_sequence"]) != payload["camera_sequence_sha256"]:
        raise ValueError("diagnostic snapshot camera sequence hash mismatch")
    return payload


def comparable_training_state(payload: Mapping[str, Any]) -> dict:
    """Return all causal training state; omit labels but never mutable state."""
    return {
        key: payload[key]
        for key in (
            "step", "next_step", "intervention_mode", "splats", "optimizers",
            "schedulers", "strategy_state", "rng", "camera_sequence",
            "camera_sequence_sha256", "ru_training",
        )
    }


def run_restore_smoke(output_dir: Path, *, device: str = "cuda") -> dict:
    """Exercise post-update save/restore on a tiny optimizer state."""
    import random
    import numpy as np
    import torch
    from puri_gs.replay import restore_rng_state, restore_training_state

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    target = torch.device(device)
    random.seed(20260908)
    np.random.seed(20260908)
    torch.manual_seed(20260908)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(20260908)

    def make_state():
        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.randn(8, 3, device=target))
        })
        optimizers = {"means": torch.optim.Adam([params["means"]], lr=1e-2)}
        schedulers = [torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.99
        )]
        return params, optimizers, schedulers

    def advance(params, optimizers, schedulers, count):
        losses = []
        for _ in range(count):
            optimizers["means"].zero_grad(set_to_none=True)
            target_value = torch.randn_like(params["means"])
            loss = (params["means"] - target_value).square().mean()
            loss.backward()
            optimizers["means"].step()
            schedulers[0].step()
            losses.append(float(loss.detach().cpu()))
        return losses

    params, optimizers, schedulers = make_state()
    advance(params, optimizers, schedulers, 7)
    checkpoint = destination / "synthetic_step9999.pt"
    save_training_snapshot(
        checkpoint, step=SNAPSHOT_STEP, stage="U", splats=params,
        optimizers=optimizers, schedulers=schedulers,
        strategy_state={"counter": 7, "gradient": torch.ones(8, device=target)},
        camera_sequence=torch.arange(HORIZON), intervention_mode="parent",
        ru_training_state={"histogram": torch.arange(5, device=target)},
    )
    saved = load_training_snapshot(checkpoint)
    reference_losses = advance(params, optimizers, schedulers, 13)
    reference = {
        "splats": snapshot_tree(dict(params)),
        "optimizers": snapshot_tree({k: v.state_dict() for k, v in optimizers.items()}),
        "schedulers": snapshot_tree([value.state_dict() for value in schedulers]),
    }

    replay_params, replay_optimizers, replay_schedulers = make_state()
    restored_strategy = restore_training_state(
        saved, splats=replay_params, optimizers=replay_optimizers,
        schedulers=replay_schedulers, device=target,
    )
    optimizer_references_current(replay_params, replay_optimizers)
    restore_rng_state(saved["rng"])
    replay_losses = advance(replay_params, replay_optimizers, replay_schedulers, 13)
    replay = {
        "splats": snapshot_tree(dict(replay_params)),
        "optimizers": snapshot_tree({
            k: v.state_dict() for k, v in replay_optimizers.items()
        }),
        "schedulers": snapshot_tree([
            value.state_dict() for value in replay_schedulers
        ]),
    }
    comparison = compare_state(reference, replay, rtol=0.0, atol=0.0)
    loss_exact = reference_losses == replay_losses
    passed = comparison["passed"] and loss_exact and restored_strategy["counter"] == 7
    result = {
        "status": "SMOKE_ACCEPTED" if passed else "SMOKE_REJECTED",
        "device": str(target), "updates_before_snapshot": 7,
        "updates_after_snapshot": 13, "loss_bitwise_equal": loss_exact,
        "state_comparison": comparison,
        "optimizer_references_current": True,
    }
    write_new_json(destination / "smoke_result.json", result)
    return result


def export_source_probes(runner, *, step: int, source_names=SOURCES) -> dict:
    """Render only preregistered training sources without advancing mutable state."""
    import torch
    import torch.nn.functional as F
    from puri_gs.replay import capture_rng_state, restore_rng_state
    from puri_gs.semantic_mask import hard_static_mask

    if step not in (9_599, 9_799, 9_999, 10_199):
        raise ValueError("source probe step is not preregistered")
    if runner.ru_training is None:
        raise ValueError("source probes require the RU mask training state")
    training = runner.ru_training
    parser_names = list(training.parser.image_names)
    train_indices = [int(value) for value in training.trainset.indices]
    output = Path(runner.cfg.result_dir) / "diagnostic_source_probes"
    output.mkdir(parents=True, exist_ok=True)
    rng = capture_rng_state()
    head_training = training.head.training
    records = []
    try:
        training.head.eval()
        with torch.no_grad():
            for image_name in source_names:
                if image_name not in parser_names:
                    raise ValueError(f"registered source is absent: {image_name}")
                parser_index = parser_names.index(image_name)
                if parser_index not in train_indices:
                    raise ValueError(f"registered source is not a training view: {image_name}")
                train_index = train_indices.index(parser_index)
                data = training.trainset[train_index]
                pixels = data["image"].unsqueeze(0).to(runner.device) / 255.0
                camtoworld = data["camtoworld"].unsqueeze(0).to(runner.device)
                intrinsics = data["K"].unsqueeze(0).to(runner.device)
                height, width = pixels.shape[1:3]
                render, alpha, _ = runner.rasterize_splats(
                    camtoworlds=camtoworld, Ks=intrinsics, width=width, height=height,
                    sh_degree=min(step // runner.cfg.sh_degree_interval,
                                  runner.cfg.sh_degree),
                    near_plane=runner.cfg.near_plane, far_plane=runner.cfg.far_plane,
                    image_ids=torch.tensor([train_index], device=runner.device),
                    render_mode="RGB",
                )
                render = render[..., :3]
                grid = training.grid_for_step(step)
                feature = training.cache.load(image_name, grid).unsqueeze(0).to(runner.device)
                probability = F.interpolate(
                    training.head(feature), size=(height, width), mode="bilinear",
                    align_corners=False,
                )
                hard_mask = hard_static_mask(
                    probability, threshold=runner.cfg.mask_threshold,
                    kernel_size=runner.cfg.mask_erode_kernel,
                )
                residual = (render - pixels).abs().mean(dim=-1, keepdim=False)
                unmasked = (1.0 - hard_mask[:, 0].float()) * residual
                path = output / f"step{step}_{Path(image_name).stem}.pt"
                if path.exists():
                    raise FileExistsError(f"source probe exists: {path}")
                torch.save({
                    "step": step, "image_name": image_name,
                    "parser_index": parser_index, "train_index": train_index,
                    "render_rgb": render.detach().cpu().contiguous(),
                    "target_rgb": pixels.detach().cpu().contiguous(),
                    "hard_mask": hard_mask.detach().cpu().contiguous(),
                    "alpha": alpha.detach().cpu().contiguous(),
                    "unmasked_rgb_residual": unmasked.detach().cpu().contiguous(),
                }, path)
                records.append({
                    "image_name": image_name, "path": str(path),
                    "sha256": sha256_file(path), "parser_index": parser_index,
                    "train_index": train_index,
                })
    finally:
        training.head.train(head_training)
        restore_rng_state(rng)
    manifest = output / f"step{step}_manifest.json"
    write_new_json(manifest, {"step": step, "sources": records, "test_rgb_opened": False})
    return {"manifest": str(manifest), "sources": records}


def compare_replay_runs(paths: Mapping[str, Path]) -> dict:
    """Apply state, trajectory, source-render and fixed-Mask replay gates."""
    import torch

    labels = ("U", "R1", "R2")
    payloads = {label: load_training_snapshot(paths[label]) for label in labels}
    for label, payload in payloads.items():
        if payload["stage"] != label or payload["step"] != 10_199:
            raise ValueError(f"{label} terminal snapshot provenance differs")
    pairs = {}
    for left, right in (("U", "R1"), ("U", "R2"), ("R1", "R2")):
        pairs[f"{left}_vs_{right}"] = compare_state(
            comparable_training_state(payloads[left]),
            comparable_training_state(payloads[right]),
        )

    def curve(directory, name):
        with (directory / name).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return {int(row["step"]): row for row in rows if 10_000 <= int(row["step"]) <= 10_199}

    run_dirs = {label: Path(paths[label]).parent for label in labels}
    loss_curves = {label: curve(run_dirs[label], "loss_curve.csv") for label in labels}
    gaussian_curves = {
        label: curve(run_dirs[label], "Gaussian_count_curve.csv") for label in labels
    }
    expected_steps = set(range(10_000, 10_200))
    if any(set(value) != expected_steps for value in loss_curves.values()):
        raise ValueError("replay loss trajectory does not contain exactly 200 updates")
    if any(set(value) != expected_steps for value in gaussian_curves.values()):
        raise ValueError("replay Gaussian trajectory does not contain exactly 200 updates")
    loss_max = 0.0
    gaussian_exact = True
    for step in sorted(expected_steps):
        for left, right in (("U", "R1"), ("U", "R2"), ("R1", "R2")):
            lrow, rrow = loss_curves[left][step], loss_curves[right][step]
            if list(lrow) != list(rrow):
                raise ValueError("replay loss schemas differ")
            for key in lrow:
                if key != "step":
                    loss_max = max(loss_max, abs(float(lrow[key]) - float(rrow[key])))
            gaussian_exact &= gaussian_curves[left][step] == gaussian_curves[right][step]

    source_checks = {}
    for source in SOURCES:
        values = {}
        for label in labels:
            path = run_dirs[label] / "diagnostic_source_probes" / \
                f"step10199_{Path(source).stem}.pt"
            values[label] = torch.load(path, map_location="cpu", weights_only=False)
        checks = {}
        for left, right in (("U", "R1"), ("U", "R2"), ("R1", "R2")):
            a, b = values[left], values[right]
            target = a["target_rgb"]
            if not torch.equal(target, b["target_rgb"]):
                raise ValueError("source targets differ across replay runs")
            def psnr(render):
                mse = (render.double() - target.double()).square().mean()
                return float(-10.0 * torch.log10(mse))
            delta = abs(psnr(a["render_rgb"]) - psnr(b["render_rgb"]))
            checks[f"{left}_vs_{right}"] = {
                "psnr_abs_delta_db": delta,
                "hard_mask_exact": torch.equal(a["hard_mask"], b["hard_mask"]),
                "passed": delta <= 0.001 and torch.equal(a["hard_mask"], b["hard_mask"]),
            }
        source_checks[source] = checks
    passed = (
        all(value["passed"] for value in pairs.values())
        and loss_max <= 1e-6 and gaussian_exact
        and all(item["passed"] for source in source_checks.values() for item in source.values())
    )
    return {
        "status": "REPLAY_ACCEPTED" if passed else "REPLAY_REJECTED",
        "pairs": pairs, "loss_max_abs_delta": loss_max,
        "loss_gate": loss_max <= 1e-6, "gaussian_trajectory_exact": gaussian_exact,
        "source_rerenders": source_checks, "updates_compared": 200,
    }


def largest_component_8(mask):
    """Deterministic largest 8-connected component; row-major resolves ties."""
    import numpy as np

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    visited = np.zeros_like(value)
    best = []
    height, width = value.shape
    for row in range(height):
        for column in range(width):
            if not value[row, column] or visited[row, column]:
                continue
            visited[row, column] = True
            stack = [(row, column)]
            component = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for yy in range(max(0, y - 1), min(height, y + 2)):
                    for xx in range(max(0, x - 1), min(width, x + 2)):
                        if value[yy, xx] and not visited[yy, xx]:
                            visited[yy, xx] = True
                            stack.append((yy, xx))
            if len(component) > len(best):
                best = component
    result = np.zeros_like(value)
    if best:
        yy, xx = zip(*best)
        result[np.asarray(yy), np.asarray(xx)] = True
    return result


def derive_candidate_roi(probes: Sequence[Mapping[str, Any]]) -> dict:
    """Apply the single registered residual/Mask ROI rule to three probes."""
    import torch

    if [int(item["step"]) for item in probes] != [9_599, 9_799, 9_999]:
        raise ValueError("ROI requires exactly the three registered source probes")
    names = {str(item["image_name"]) for item in probes}
    if len(names) != 1:
        raise ValueError("ROI probes mix source images")
    shapes = {tuple(item["render_rgb"].shape) for item in probes}
    if len(shapes) != 1:
        raise ValueError("ROI probe image shapes differ")
    intersections = None
    thresholds = []
    residuals = []
    for item in probes:
        residual = (item["render_rgb"] - item["target_rgb"]).abs().mean(dim=-1)[0]
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("ROI residual is non-finite")
        threshold = torch.quantile(residual, 0.8)
        hard = residual > threshold
        intersections = hard if intersections is None else intersections & hard
        thresholds.append(float(threshold))
        residuals.append(residual)
    final_mask = probes[-1]["hard_mask"][0, 0].bool()
    candidate = intersections & ~final_mask
    component = largest_component_8(candidate.cpu().numpy())
    roi = torch.from_numpy(component)
    area = int(roi.sum())
    return {
        "status": "ROI_ESTABLISHED" if area >= 64 else "ROI_NOT_ESTABLISHED",
        "image_name": probes[-1]["image_name"], "roi": roi,
        "area": area, "thresholds": thresholds,
        "residual_step9999": residuals[-1],
        "candidate_before_component_area": int(candidate.sum()),
    }


def select_monitor_views(data_dir: Path, train_names: Sequence[str]) -> list[str]:
    """Select three train cameras by raw-center distance; no RGB is opened."""
    import numpy as np
    from pycolmap import SceneManager

    sparse = Path(data_dir) / "sparse" / "0"
    if not sparse.is_dir():
        sparse = Path(data_dir) / "sparse"
    manager = SceneManager(str(sparse))
    manager.load_cameras()
    manager.load_images()
    centers = {}
    for image in manager.images.values():
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = image.R()
        world_to_camera[:3, 3] = image.tvec
        centers[image.name] = np.linalg.inv(world_to_camera)[:3, 3]
    missing = sorted(set(train_names).difference(centers))
    if missing:
        raise ValueError(f"COLMAP camera mapping misses training view: {missing[0]}")
    for source in SOURCES:
        if source not in train_names:
            raise ValueError(f"registered source is not in train set: {source}")
    source_centers = [centers[name] for name in SOURCES]
    ranked = []
    for name in train_names:
        if name in SOURCES:
            continue
        distance = min(float(np.linalg.norm(centers[name] - center))
                       for center in source_centers)
        ranked.append((distance, Path(name).name, name))
    ranked.sort()
    return [item[2] for item in ranked[:3]]


def build_roi_evidence(
    u_dir: Path, track_cache: Path, data_dir: Path, output_dir: Path
) -> dict:
    """Build fixed train-only ROIs and audit the existing PART evidence cache."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from puri_gs.static_tracks import load_static_track_cache

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    payload = load_static_track_cache(track_cache)
    train_names = list(payload["train_basenames"])
    monitors = select_monitor_views(data_dir, train_names)
    monitor_ids = {train_names.index(name) for name in monitors}
    results = {}

    prepared_images = Path(data_dir) / "images_4_png"
    prepared_by_stem = {
        path.stem: path for path in prepared_images.iterdir() if path.is_file()
    }
    thumbnails = []
    for name in monitors:
        path = prepared_by_stem.get(Path(name).stem)
        if path is None:
            raise FileNotFoundError(f"monitor training image is missing: {name}")
        image = Image.open(path).convert("RGB")
        image.thumbnail((480, 360))
        thumbnails.append(image.copy())
    canvas = Image.new(
        "RGB", (sum(image.width for image in thumbnails), max(image.height for image in thumbnails))
    )
    left = 0
    for image in thumbnails:
        canvas.paste(image, (left, 0))
        left += image.width
    canvas.save(destination / "monitor_views.png")

    def ratio(numerator, denominator):
        return None if denominator == 0 else float(numerator) / float(denominator)

    for source in SOURCES:
        source_id = train_names.index(source)
        probe_paths = [
            Path(u_dir) / "diagnostic_source_probes" /
            f"step{step}_{Path(source).stem}.pt"
            for step in (9_599, 9_799, 9_999)
        ]
        if not all(path.is_file() for path in probe_paths):
            raise FileNotFoundError(f"source probes are incomplete: {source}")
        probes = [torch.load(path, map_location="cpu", weights_only=False)
                  for path in probe_paths]
        derived = derive_candidate_roi(probes)
        roi = derived.pop("roi")
        final = probes[-1]
        height, width = roi.shape
        original_grid = payload["track_evidence_binary"][source_id].bool()
        isolated_grid = torch.zeros_like(original_grid)
        original_support_counts = torch.zeros_like(original_grid, dtype=torch.int64)
        isolated_support_counts = torch.zeros_like(original_grid, dtype=torch.int64)
        source_track_count = 0
        isolated_track_count = 0
        for row in range(len(payload["track_ids"])):
            views = [int(value) for value in payload["track_view_ids"][row].tolist()
                     if int(value) >= 0]
            if source_id not in views:
                continue
            source_track_count += 1
            node = views.index(source_id)
            xy = payload["track_patch_xy"][row, node]
            x, y = int(torch.floor(xy[0])), int(torch.floor(xy[1]))
            if not (0 <= x < 36 and 0 <= y < 36):
                raise ValueError("track cache source coordinate is outside fine grid")
            original_support_counts[y, x] = max(
                original_support_counts[y, x], len(set(views))
            )
            if not monitor_ids.intersection(views):
                isolated_track_count += 1
                isolated_grid[y, x] = True
                isolated_support_counts[y, x] = max(
                    isolated_support_counts[y, x], len(set(views))
                )
        original = F.interpolate(
            original_grid[None, None].float(), size=(height, width),
            mode="bilinear", align_corners=False,
        )[0, 0]
        isolated = F.interpolate(
            isolated_grid[None, None].float(), size=(height, width),
            mode="bilinear", align_corners=False,
        )[0, 0]
        rejected = ~final["hard_mask"][0, 0].bool()
        roi_area = int(roi.sum())
        original_positive = original > 0
        isolated_positive = isolated > 0
        whole_coverage = float(original_positive.float().mean())
        roi_coverage = ratio(int((original_positive & roi).sum()), roi_area)
        isolated_roi_coverage = ratio(int((isolated_positive & roi).sum()), roi_area)
        rejected_area = int(rejected.sum())
        rejected_coverage = ratio(int((original_positive & rejected).sum()), rejected_area)
        alpha = final["alpha"][0, :, :, 0]
        residual = derived["residual_step9999"]
        strata = {}
        for label, selector in (
            ("lt_0.1", alpha < 0.1),
            ("0.1_to_0.5", (alpha >= 0.1) & (alpha < 0.5)),
            ("0.5_to_0.9", (alpha >= 0.5) & (alpha < 0.9)),
            ("ge_0.9", alpha >= 0.9),
        ):
            selected = roi & selector
            count = int(selected.sum())
            strata[label] = {
                "area": count,
                "residual_mean": None if not count else float(residual[selected].mean()),
                "evidence_mean": None if not count else float(original[selected].mean()),
                "evidence_nonzero_ratio": None if not count else float(
                    original_positive[selected].float().mean()
                ),
            }
        roi_path = destination / f"candidate_roi_{Path(source).stem}.pt"
        torch.save(roi, roi_path)
        bitmap = (roi.numpy().astype(np.uint8) * 255)
        Image.fromarray(bitmap).save(destination / f"candidate_roi_{Path(source).stem}.png")

        target = final["target_rgb"][0].clamp(0, 1).mul(255).byte().numpy()
        overlay = target.copy()
        overlay[roi.numpy()] = (
            0.35 * overlay[roi.numpy()] + 0.65 * np.array([255, 0, 0])
        ).astype(np.uint8)
        panels = [target, overlay]
        for field in (residual, final["hard_mask"][0, 0].float(), alpha, original, isolated):
            value = field.detach().float().numpy()
            maximum = float(np.quantile(value, 0.99))
            scaled = np.zeros_like(value, dtype=np.uint8) if maximum <= 0 else np.clip(
                value / maximum * 255, 0, 255
            ).astype(np.uint8)
            panels.append(np.repeat(scaled[..., None], 3, axis=2))
        contact = Image.fromarray(np.concatenate(panels, axis=1))
        if contact.width > 2800:
            contact.thumbnail((2800, 1000))
        contact.save(destination / f"roi_evidence_{Path(source).stem}.png")
        results[source] = {
            **{key: value for key, value in derived.items()
               if key != "residual_step9999"},
            "roi_bitmap": str(roi_path), "roi_bitmap_sha256": sha256_file(roi_path),
            "source_track_count": source_track_count,
            "heldout_isolated_track_count": isolated_track_count,
            "whole_image_evidence_coverage": whole_coverage,
            "roi_evidence_coverage": roi_coverage,
            "heldout_isolated_roi_evidence_coverage": isolated_roi_coverage,
            "rejected_area_evidence_coverage": rejected_coverage,
            "roi_enrichment_vs_full": None if not whole_coverage or roi_coverage is None
            else roi_coverage / whole_coverage,
            "roi_evidence_mean": None if not roi_area else float(original[roi].mean()),
            "support_camera_count_distribution_on_fine_cells": {
                scope: {
                    str(int(value)): int(count) for value, count in zip(
                        *torch.unique(grid[grid > 0], return_counts=True)
                    )
                }
                for scope, grid in (
                    ("original", original_support_counts),
                    ("heldout_isolated", isolated_support_counts),
                )
            },
            "alpha_strata": strata,
            "upstream_intermediates": {
                "raw_mutual_matches": "not_available",
                "components": "not_available",
                "valid_tracks": "available",
                "final_C": "available",
            },
        }
    result = {
        "status": "ROI_ESTABLISHED" if all(
            value["status"] == "ROI_ESTABLISHED" for value in results.values()
        ) else "ROI_NOT_ESTABLISHED",
        "source_names": list(SOURCES), "monitor_names": monitors,
        "monitor_selection": "minimum raw COLMAP camera-center distance; basename tie-break",
        "u_dir": str(Path(u_dir).resolve()),
        "test_rgb_opened": False, "sources": results,
        "static_confirmation": "pending",
    }
    write_new_json(destination / "roi_evidence.json", result)
    return result


def run_vjp_probes(runner, roi_dir: Path, *, confirmed: bool) -> dict:
    """Measure finite ROI/control VJPs without stepping any mutable training state."""
    import numpy as np
    import torch

    prefix = "O_static" if confirmed else "candidate_roi"
    output = Path(runner.cfg.result_dir)
    roi_record = read_json(Path(roi_dir) / "roi_evidence.json")
    if roi_record.get("status") != "ROI_ESTABLISHED":
        raise ValueError("VJP requires established ROIs")
    if confirmed and not (Path(roi_dir) / "static_confirmation.json").is_file():
        raise ValueError("confirmed VJP requires explicit static confirmation")
    records = {}

    def gradient_stats(info):
        families = {}
        values = {
            "means": runner.splats["means"].grad,
            "means2d": info.get("means2d").grad if info.get("means2d") is not None else None,
            "scales": runner.splats["scales"].grad,
            "opacity_logits": runner.splats["opacities"].grad,
            "SH": None,
        }
        sh_values = [runner.splats[name].grad for name in ("sh0", "shN")]
        if all(value is not None for value in sh_values):
            values["SH"] = torch.cat(
                [value.reshape(value.shape[0], -1) for value in sh_values], dim=1
            )
        for name, gradient in values.items():
            if gradient is None:
                families[name] = "not_available"
                continue
            value = gradient.detach().reshape(gradient.shape[0], -1)
            row_norm = torch.linalg.vector_norm(value.double(), dim=1)
            families[name] = {
                "finite": bool(torch.isfinite(value).all()),
                "l2_norm": float(torch.linalg.vector_norm(value.double())),
                "row_rms": float(torch.sqrt(torch.mean(row_norm.square()))),
                "nonzero_rows": int((row_norm > 0).sum()),
                "total_rows": int(len(row_norm)),
            }
        return families

    def zero_grad():
        for parameter in runner.splats.values():
            parameter.grad = None

    for source_index, source in enumerate(SOURCES):
        mask_path = Path(roi_dir) / f"{prefix}_{Path(source).stem}.pt"
        if not mask_path.is_file():
            raise FileNotFoundError(f"VJP mask is missing: {mask_path}")
        roi = torch.load(mask_path, map_location="cpu", weights_only=False).bool()
        probe_path = Path(roi_record["u_dir"]) / "diagnostic_source_probes" / \
            f"step9999_{Path(source).stem}.pt"
        if not probe_path.is_file():
            raise FileNotFoundError(f"VJP source probe is missing: {probe_path}")
        probe = torch.load(probe_path, map_location="cpu", weights_only=False)
        if tuple(roi.shape) != tuple(probe["hard_mask"].shape[-2:]):
            raise ValueError(f"VJP ROI coordinates differ: {source}")
        residual = (probe["render_rgb"] - probe["target_rgb"]).abs().mean(-1)[0]
        accepted = probe["hard_mask"][0, 0].bool() & ~roi
        count = int(roi.sum())
        candidates = torch.where(accepted.flatten())[0].cpu().numpy()
        if len(candidates) < count:
            control = None
        else:
            delta = np.abs(residual.flatten()[candidates].numpy() - float(residual[roi].mean()))
            order = np.lexsort((candidates, delta))
            selected = candidates[order[:count]]
            control = torch.zeros_like(roi)
            control.flatten()[torch.from_numpy(selected.copy())] = True
            torch.save(control, output / f"vjp_control_{Path(source).stem}.pt")

        train_names = list(runner.ru_training.parser.image_names)
        parser_index = train_names.index(source)
        indices = [int(value) for value in runner.ru_training.trainset.indices]
        train_index = indices.index(parser_index)
        data = runner.ru_training.trainset[train_index]
        pixels = data["image"].unsqueeze(0).to(runner.device) / 255.0
        camtoworld = data["camtoworld"].unsqueeze(0).to(runner.device)
        intrinsics = data["K"].unsqueeze(0).to(runner.device)
        height, width = pixels.shape[1:3]
        source_results = {}
        for region_name, region in (("roi", roi), ("control", control)):
            if region is None:
                source_results[region_name] = "not_available"
                continue
            region_device = region.to(runner.device)
            runs = []
            for probe_index in range(5):
                zero_grad()
                render, _, info = runner.rasterize_splats(
                    camtoworlds=camtoworld, Ks=intrinsics, width=width, height=height,
                    sh_degree=runner.cfg.sh_degree, near_plane=runner.cfg.near_plane,
                    far_plane=runner.cfg.far_plane,
                    image_ids=torch.tensor([train_index], device=runner.device),
                    render_mode="RGB",
                )
                render = render[..., :3]
                means2d = info.get("means2d")
                if means2d is not None and means2d.requires_grad:
                    means2d.retain_grad()
                if probe_index < 4:
                    generator = torch.Generator(device=runner.device)
                    generator.manual_seed(20260908 + source_index * 100 + probe_index)
                    signs = torch.randint(
                        0, 2, render.shape, generator=generator,
                        device=runner.device, dtype=torch.int64,
                    ).mul(2).sub(1).to(render.dtype)
                    scalar = (
                        render * signs * region_device[None, :, :, None]
                    ).sum() / math.sqrt(3 * count)
                    kind = f"rademacher_{probe_index}"
                else:
                    scalar = (
                        (render - pixels).abs() * region_device[None, :, :, None]
                    ).sum() / (3 * count)
                    kind = "roi_l1"
                scalar.backward()
                runs.append({
                    "probe": kind, "scalar": float(scalar.detach().cpu()),
                    "families": gradient_stats(info),
                })
            source_results[region_name] = runs
        family_summary = {}
        for family in ("means", "means2d", "scales", "opacity_logits", "SH"):
            if source_results["control"] == "not_available":
                family_summary[family] = "control_not_available"
                continue
            roi_family = [run["families"][family] for run in source_results["roi"]]
            control_family = [
                run["families"][family] for run in source_results["control"]
            ]
            if any(value == "not_available" for value in roi_family + control_family):
                family_summary[family] = "not_available"
                continue
            roi_random = [value["row_rms"] for value in roi_family[:4]]
            control_random = [value["row_rms"] for value in control_family[:4]]
            roi_rms = math.sqrt(sum(value * value for value in roi_random) / 4)
            control_rms = math.sqrt(sum(value * value for value in control_random) / 4)
            family_summary[family] = {
                "roi_rademacher_rms": roi_rms,
                "control_rademacher_rms": control_rms,
                "roi_control_ratio": None if control_rms == 0 else roi_rms / control_rms,
                "roi_probe_min": min(roi_random), "roi_probe_max": max(roi_random),
                "roi_probe_mean": sum(roi_random) / 4,
                "roi_probe_std": float(np.std(np.asarray(roi_random), ddof=0)),
                "roi_l1": roi_family[4], "control_l1": control_family[4],
            }
        records[source] = {
            "roi_pixels": count, "control_rule": (
                "accepted Mask pixels closest to ROI mean residual; row-major tie-break"
            ), "probes": source_results, "family_summary": family_summary,
        }
    zero_grad()
    result = {
        "status": "VJP_MEASURED_REQUIRES_INTERPRETATION",
        "roi_confirmation": "confirmed" if confirmed else "candidate_only",
        "rademacher_seed_base": 20260908, "explicit_jacobian_built": False,
        "optimizer_step_called": False, "topology_step_called": False,
        "head_or_histogram_updated": False, "pixel_contributor_count": "not_available",
        "sources": records,
    }
    write_new_json(output / "vjp_report.json", result)
    return result


def confirm_static_rois(roi_dir: Path, *, note: str) -> dict:
    """Freeze the user-confirmed whole candidate ROIs without overwriting them."""
    import torch

    root = Path(roi_dir)
    evidence = read_json(root / "roi_evidence.json")
    if evidence.get("status") != "ROI_ESTABLISHED":
        raise ValueError("cannot confirm an unestablished ROI")
    records = {}
    for source in SOURCES:
        candidate = root / f"candidate_roi_{Path(source).stem}.pt"
        value = torch.load(candidate, map_location="cpu", weights_only=False).bool()
        if int(value.sum()) < 64:
            raise ValueError("confirmed ROI is below the registered minimum")
        target = root / f"O_static_{Path(source).stem}.pt"
        if target.exists():
            raise FileExistsError(f"confirmed ROI exists: {target}")
        torch.save(value, target)
        records[source] = {
            "source_candidate": str(candidate), "path": str(target),
            "pixels": int(value.sum()), "sha256": sha256_file(target),
        }
    result = {
        "status": "STATIC_ROI_CONFIRMED", "scope": "whole_candidate_rois",
        "user_note": str(note),
        "confirmed_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "sources": records,
    }
    write_new_json(root / "static_confirmation.json", result)
    return result


def record_vjp_decision(roi_dir: Path, vjp_report: Path, *, status: str) -> dict:
    allowed = {
        "LOCAL_CONTROLLABILITY_PRESENT",
        "CONTROLLABILITY_INCONCLUSIVE",
        "LOCAL_CONTROLLABILITY_NOT_DETECTED",
    }
    if status not in allowed:
        raise ValueError("unknown VJP decision")
    report = read_json(Path(vjp_report))
    if report.get("status") != "VJP_MEASURED_REQUIRES_INTERPRETATION" \
            or report.get("roi_confirmation") != "confirmed":
        raise ValueError("formal confirmed-ROI VJP report is required")
    result = {
        "status": status, "vjp_report": str(Path(vjp_report).resolve()),
        "vjp_report_sha256": sha256_file(Path(vjp_report)),
        "decided_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    write_new_json(Path(roi_dir) / "vjp_decision.json", result)
    return result


def prepare_fixed_branch(training, camera_sequence, roi_dir: Path, *, stage: str) -> dict:
    """Lock the 400-step camera suffix and optional confirmed Oracle masks."""
    import torch
    from puri_gs.replay import camera_sequence_sha256

    if stage not in {"P1", "P2", "O"}:
        raise ValueError("unknown fixed diagnostic branch")
    root = Path(roi_dir)
    evidence = read_json(root / "roi_evidence.json")
    confirmation = read_json(root / "static_confirmation.json")
    vjp_decision = read_json(root / "vjp_decision.json")
    if evidence.get("status") != "ROI_ESTABLISHED" \
            or confirmation.get("status") != "STATIC_ROI_CONFIRMED":
        raise ValueError("fixed branches require established, confirmed static ROIs")
    if vjp_decision.get("status") != "LOCAL_CONTROLLABILITY_PRESENT":
        raise ValueError("fixed branches require the positive formal VJP gate")
    runtime_names = [
        training.parser.image_names[int(index)] for index in training.trainset.indices
    ]
    source_ids = [runtime_names.index(name) for name in SOURCES]
    monitor_ids = [runtime_names.index(name) for name in evidence["monitor_names"]]
    suffix, exposures = filtered_suffix(camera_sequence, monitor_ids, source_ids)
    oracle_masks = {}
    if stage == "O":
        for source, source_id in zip(SOURCES, source_ids):
            confirmed_path = root / f"O_static_{Path(source).stem}.pt"
            candidate_path = root / f"candidate_roi_{Path(source).stem}.pt"
            confirmed = torch.load(
                confirmed_path, map_location="cpu", weights_only=False
            ).bool()
            candidate = torch.load(
                candidate_path, map_location="cpu", weights_only=False
            ).bool()
            if confirmed.shape != candidate.shape or bool((confirmed & ~candidate).any()):
                raise ValueError("O_static must be a coordinate-matched candidate subset")
            if not bool(confirmed.any()):
                raise ValueError("O_static is empty")
            oracle_masks[source_id] = confirmed
    result = {
        "scheduled_ids": torch.tensor(suffix, dtype=torch.int64),
        "source_ids": source_ids, "monitor_ids": monitor_ids,
        "source_exposures": {runtime_names[key]: value for key, value in exposures.items()},
        "oracle_masks": oracle_masks,
        "suffix_sha256": camera_sequence_sha256(torch.tensor(suffix, dtype=torch.int64)),
        "updates_completed": 0, "effective_oracle_exposures": 0,
        "extra_l1_sum": 0.0,
    }
    write_new_json(Path(training.cfg.result_dir) / "fixed_branch_contract.json", {
        "stage": stage, "updates": ORACLE_STEPS,
        "source_names": list(SOURCES), "monitor_names": evidence["monitor_names"],
        "source_exposures": result["source_exposures"],
        "suffix_sha256": result["suffix_sha256"],
        "topology_frozen": True, "mask_head_frozen": True,
        "residual_histogram_frozen": True,
        "oracle_enabled": stage == "O",
    })
    return result


def fixed_branch_mask(training, image_id, pixels):
    """Compute the step9999 fixed hard Mask with no optimizer/histogram mutation."""
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    from puri_gs.semantic_mask import hard_static_mask

    image_name = training._image_name(image_id)
    head_training = training.head.training
    try:
        training.head.eval()
        with torch.no_grad():
            feature = training.cache.load(image_name, 16).unsqueeze(0).to(training.device)
            probability = F.interpolate(
                training.head(feature), size=tuple(pixels.shape[1:3]),
                mode="bilinear", align_corners=False,
            )
            mask = hard_static_mask(
                probability, threshold=training.cfg.mask_threshold,
                kernel_size=training.cfg.mask_erode_kernel,
            )
    finally:
        training.head.train(head_training)
    return SimpleNamespace(safe_mask=mask.detach())


def oracle_for_image(branch: Mapping[str, Any], image_id, pixels):
    import torch

    item = int(image_id.reshape(-1)[0].detach().cpu())
    value = branch["oracle_masks"].get(item)
    if value is None:
        return torch.zeros(
            (1, 1, pixels.shape[1], pixels.shape[2]),
            device=pixels.device, dtype=pixels.dtype,
        )
    if tuple(value.shape) != tuple(pixels.shape[1:3]):
        raise ValueError("Oracle coordinates differ from current training image")
    return value[None, None].to(device=pixels.device, dtype=pixels.dtype)


def export_branch_readouts(runner, branch: Mapping[str, Any], *, step: int) -> dict:
    """Export fixed train-source/held-out readouts at the 400-update endpoint."""
    import torch
    import torch.nn.functional as F
    from puri_gs.replay import capture_rng_state, restore_rng_state

    if step not in (9_999, 10_199, 10_399):
        raise ValueError("fixed branch readout step is not preregistered")
    contract = read_json(Path(runner.cfg.result_dir) / "fixed_branch_contract.json")
    names = list(SOURCES) + list(contract["monitor_names"])
    runtime_names = [
        runner.ru_training.parser.image_names[int(index)]
        for index in runner.ru_training.trainset.indices
    ]
    rng = capture_rng_state()
    rows = {}
    try:
        with torch.no_grad():
            for image_name in names:
                train_index = runtime_names.index(image_name)
                data = runner.ru_training.trainset[train_index]
                pixels = data["image"].unsqueeze(0).to(runner.device) / 255.0
                camtoworld = data["camtoworld"].unsqueeze(0).to(runner.device)
                intrinsics = data["K"].unsqueeze(0).to(runner.device)
                height, width = pixels.shape[1:3]
                render, alpha, _ = runner.rasterize_splats(
                    camtoworlds=camtoworld, Ks=intrinsics, width=width, height=height,
                    sh_degree=runner.cfg.sh_degree, near_plane=runner.cfg.near_plane,
                    far_plane=runner.cfg.far_plane,
                    image_ids=torch.tensor([train_index], device=runner.device),
                    render_mode="RGB",
                )
                render = render[..., :3].clamp(0, 1)
                render_chw = render.permute(0, 3, 1, 2)
                pixels_chw = pixels.permute(0, 3, 1, 2)
                values = {
                    "role": "source" if image_name in SOURCES else "branch-held-out monitor",
                    "full_psnr": float(runner.psnr(render_chw, pixels_chw)),
                    "full_ssim": float(runner.ssim(render_chw, pixels_chw)),
                    "full_lpips": float(runner.lpips(render_chw, pixels_chw)),
                    "full_rgb_mae": float((render - pixels).abs().mean()),
                    "alpha_mean": float(alpha.mean()),
                }
                if image_name in SOURCES:
                    roi = torch.load(
                        Path(runner.cfg.ru_part_diagnostic_roi_dir) /
                        f"O_static_{Path(image_name).stem}.pt",
                        map_location="cpu", weights_only=False,
                    ).bool().to(runner.device)
                    fixed = fixed_branch_mask(
                        runner.ru_training,
                        torch.tensor([train_index], device=runner.device), pixels,
                    ).safe_mask[0, 0].bool()
                    squared = (render[0] - pixels[0]).square().mean(-1)
                    absolute = (render[0] - pixels[0]).abs().mean(-1)
                    dilated = F.max_pool2d(
                        roi[None, None].float(), kernel_size=11, stride=1, padding=5
                    )[0, 0].bool()
                    safe = fixed & ~dilated
                    values.update({
                        "roi_pixels": int(roi.sum()),
                        "roi_psnr": float(-10.0 * torch.log10(squared[roi].mean())),
                        "roi_rgb_mae": float(absolute[roi].mean()),
                        "roi_alpha_mean": float(alpha[0, :, :, 0][roi].mean()),
                        "roi_alpha_lt_0.1_ratio": float(
                            (alpha[0, :, :, 0][roi] < 0.1).float().mean()
                        ),
                        "safe_pixels": int(safe.sum()),
                        "safe_rgb_mae": None if not bool(safe.any())
                        else float(absolute[safe].mean()),
                        "safe_psnr": None if not bool(safe.any())
                        else float(-10.0 * torch.log10(squared[safe].mean())),
                        "safe_exclusion_window": "11x11 fused-SSIM footprint",
                    })
                tensor_path = Path(runner.cfg.result_dir) / \
                    f"branch_readout_step{step}_{Path(image_name).stem}.pt"
                torch.save({
                    "step": step, "image_name": image_name,
                    "render_rgb": render.cpu().contiguous(),
                    "target_rgb": pixels.cpu().contiguous(),
                    "alpha": alpha.cpu().contiguous(),
                }, tensor_path)
                values["tensor_path"] = str(tensor_path)
                values["tensor_sha256"] = sha256_file(tensor_path)
                rows[image_name] = values
    finally:
        restore_rng_state(rng)
    result = {
        "stage": runner.cfg.ru_part_diagnostic_stage, "step": step,
        "test_rgb_opened": False, "gaussian_count": len(runner.splats["means"]),
        "updates_completed": int(branch["updates_completed"]),
        "effective_oracle_exposures": int(branch["effective_oracle_exposures"]),
        "extra_l1_sum": float(branch["extra_l1_sum"]),
        "views": rows,
    }
    write_new_json(
        Path(runner.cfg.result_dir) / f"branch_readouts_step{step}.json", result
    )
    return result


def build_final_report(
    *, audit_path: Path, replay_path: Path, roi_path: Path,
    vjp_decision_path: Path | None, branch_dirs: Mapping[str, Path] | None,
    output_dir: Path,
) -> dict:
    """Assemble the preregistered gates; never start another experiment."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    audit = read_json(audit_path)
    replay = read_json(replay_path)
    roi = read_json(roi_path)
    confirmation_path = Path(roi_path).parent / "static_confirmation.json"
    confirmation = read_json(confirmation_path) if confirmation_path.is_file() else {
        "status": "pending"
    }
    vjp = read_json(vjp_decision_path) if vjp_decision_path else {"status": "not_run"}
    result = {
        "historical_protocol": audit.get("status"),
        "replay_equivalence": replay.get("status"),
        "roi": roi.get("status"),
        "static_confirmation": confirmation.get("status", "pending"),
        "vjp": vjp.get("status"),
        "source_names": list(SOURCES), "monitor_names": roi.get("monitor_names", []),
        "branch_analysis": "not_run",
        "secondary_labels": [],
    }
    if any(
        source.get("roi_evidence_coverage") == 0
        for source in roi.get("sources", {}).values()
    ):
        result["secondary_labels"].append("EVIDENCE_COVERAGE_FAILURE")
    if branch_dirs is not None:
        branches = {}
        snapshots = {}
        for label in ("P1", "P2", "O"):
            directory = Path(branch_dirs[label])
            branches[label] = read_json(directory / "branch_readouts_step10399.json")
            for step in (9999, 10199, 10399):
                if not (directory / f"branch_readouts_step{step}.json").is_file():
                    raise FileNotFoundError(f"{label} misses step{step} readout")
            snapshots[label] = load_training_snapshot(
                directory / "diagnostic_terminal_step10399.pt"
            )
        control_state = compare_state(
            comparable_training_state(snapshots["P1"]),
            comparable_training_state(snapshots["P2"]),
        )
        view_names = list(SOURCES) + list(roi["monitor_names"])
        control_psnr_max = max(abs(
            branches["P1"]["views"][name]["full_psnr"]
            - branches["P2"]["views"][name]["full_psnr"]
        ) for name in view_names)
        control_pass = control_state["passed"] and control_psnr_max <= 0.001
        gaussian_fixed = all(
            read_json(Path(branch_dirs[label]) / "branch_readouts_step9999.json")["gaussian_count"]
            == branches[label]["gaussian_count"]
            for label in ("P1", "P2", "O")
        )

        def weighted_roi(label):
            rows = branches[label]["views"]
            denominator = sum(rows[name]["roi_pixels"] for name in SOURCES)
            return sum(rows[name]["roi_rgb_mae"] * rows[name]["roi_pixels"]
                       for name in SOURCES) / denominator

        roi_mae = {label: weighted_roi(label) for label in ("P1", "P2", "O")}
        target_pass = all(
            roi_mae["O"] <= 0.9 * roi_mae[control]
            and all(
                branches["O"]["views"][source]["roi_rgb_mae"]
                <= branches[control]["views"][source]["roi_rgb_mae"]
                for source in SOURCES
            )
            for control in ("P1", "P2")
        )
        safe_pass = all(
            sum(branches["O"]["views"][name]["safe_psnr"] for name in SOURCES) / 2
            - sum(branches[control]["views"][name]["safe_psnr"] for name in SOURCES) / 2
            >= -0.03
            for control in ("P1", "P2")
        )

        monitor_deltas = {}
        monitor_pass = True
        for control in ("P1", "P2"):
            delta = {}
            for metric in ("full_psnr", "full_ssim", "full_lpips"):
                delta[metric] = sum(
                    branches["O"]["views"][name][metric]
                    - branches[control]["views"][name][metric]
                    for name in roi["monitor_names"]
                ) / 3
            monitor_deltas[f"O_minus_{control}"] = delta
            monitor_pass &= (
                delta["full_psnr"] >= -0.03
                and delta["full_ssim"] >= -0.001
                and delta["full_lpips"] <= 0.002
            )
        fixed_contract = control_pass and gaussian_fixed
        if fixed_contract and target_pass and safe_pass and monitor_pass:
            label = "SUPERVISION_PROBE_POSITIVE"
        elif target_pass and (not safe_pass or not monitor_pass):
            label = "LOCAL_GAIN_WITH_COLLATERAL"
        else:
            label = "PROBE_INCONCLUSIVE"
        result["branch_analysis"] = {
            "label": label, "control_state": control_state,
            "control_per_view_psnr_max_abs_delta": control_psnr_max,
            "control_pass": control_pass, "gaussian_count_fixed": gaussian_fixed,
            "combined_roi_mae": roi_mae, "target_gate": target_pass,
            "safe_area_gate": safe_pass, "monitor_gate": monitor_pass,
            "monitor_deltas": monitor_deltas,
            "effective_oracle_exposures": branches["O"]["effective_oracle_exposures"],
            "extra_l1_sum": branches["O"]["extra_l1_sum"],
        }
    write_new_json(destination / "diagnostic_report.json", result)
    lines = [
        "# RU-PART 机制诊断报告", "",
        f"- 历史协议：{result['historical_protocol']}",
        f"- U/R1/R2：{result['replay_equivalence']}",
        f"- ROI：{result['roi']}",
        f"- VJP：{result['vjp']}",
        f"- P1/P2/O：{result['branch_analysis'] if isinstance(result['branch_analysis'], str) else result['branch_analysis']['label']}",
        "", "报告生成后强制停止；不得自动启动 P/R/C 或新的长训练。", "",
    ]
    (destination / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def static_unmask_l1(render, target, mask, oracle, *, ssim_lambda=0.2):
    """Only the additional full-image RGB-mean L1; original RU loss is untouched."""
    import torch
    if render.shape != target.shape or render.ndim != 4 or render.shape[0] != 1 or render.shape[-1] != 3:
        raise ValueError("expected matching batch-one BHWC RGB")
    expected = (1, 1, render.shape[1], render.shape[2])
    if tuple(mask.shape) != expected or tuple(oracle.shape) != expected:
        raise ValueError("Mask/Oracle coordinates differ from render")
    if ssim_lambda != 0.2:
        raise ValueError("diagnostic DSSIM coefficient is fixed at 0.2")
    for value in (mask, oracle):
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError("Mask and Oracle must be binary")
    q = ((1 - mask.float()) * oracle.float()).detach().permute(0, 2, 3, 1)
    # Caller must skip this function when Q=0, preserving the original loss graph.
    if not bool(torch.any(q)):
        return None
    return (1 - ssim_lambda) * (q * (render - target).abs()).mean()
