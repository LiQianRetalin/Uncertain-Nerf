"""Register the audited standard RU checkpoint without inventing a training run."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shlex

import yaml

from puri_gs.static_tracks import sha256_file
from puri_gs.ru_part_v3 import write_json

AUDITED_COMMIT = "9e29230952be700dc5b527008e2f60f05b82717d"
REEVAL_COMMIT = "3058fbb113ff78c216d63155f12ecfd27534891c"
DECISION = "REUSE_REPRODUCED_STANDARD_RU_PARENT"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime_config(path):
    """Read YAML nodes, retaining strategy types without constructing Python objects."""
    def visit(node):
        if isinstance(node, yaml.MappingNode):
            value = {key.value: visit(child) for key, child in node.value}
            if node.tag != "tag:yaml.org,2002:map":
                value["__yaml_type__"] = node.tag
            return value
        if isinstance(node, yaml.SequenceNode):
            return [visit(child) for child in node.value]
        return node.value
    value = visit(yaml.compose(Path(path).read_text(encoding="utf-8")))
    strategy = value.get("strategy", {})
    require(isinstance(strategy, dict) and strategy.get("__yaml_type__") ==
            "tag:yaml.org,2002:python/object:puri_gs.delayed_absgrad.DelayedAbsGradStrategy",
            "cfg.yml does not use the standard DelayedAbsGradStrategy")
    # Only these inert additions are absent in the audited old implementation.
    defaults = {"puri_gs_mask_enabled": "false", "puri_gs_delayed_topology_enabled": "false",
                "eval_warmup_renders": "0", "eval_disable_image_save": "false"}
    for key, default in defaults.items():
        value.setdefault(key, default)
    require(value.get("ru_v3_mode", "null") in ("null", "parent"), "non-Parent V3 mode")
    require(value.get("ru_v3_track_cache", "null") == "null", "Parent loads a track cache")
    require(value.get("ru_v3_stop_step", "29999") in ("599", "29999"), "unexpected stop step")
    for key in ("result_dir", "ru_v3_mode", "ru_v3_track_cache", "ru_v3_stop_step"):
        value.pop(key, None)
    require(strategy.get("event_recorder", "null") == "null", "unexpected strategy recorder")
    strategy.pop("event_recorder", None)
    schedule = strategy.get("schedule", {})
    require(schedule.get("refine_windows", []) == [], "nonstandard refine windows")
    schedule.pop("refine_windows", None)
    return value


def command(path):
    words = shlex.split(Path(path).read_text(encoding="utf-8"))
    gpu = next((s.split("=", 1)[1] for s in words if s.startswith("CUDA_VISIBLE_DEVICES=")), None)
    options = {}
    for index, word in enumerate(words):
        if word.startswith("--"):
            require(word not in options, f"duplicate command option {word}")
            options[word] = words[index + 1] if index + 1 < len(words) and not words[index + 1].startswith("--") else True
    return gpu, options


def evaluation_fingerprint(directory, env):
    directory = Path(directory)
    gpu, options = command(directory / "run_command.txt")
    require(gpu == str(env["gpu"]), "evaluation GPU differs from the pinned device")
    runtime = read(directory / "environment.json")
    validation = read(directory / "ru_validation.json")
    return {
        "gpu_index": gpu, "gpu_name": runtime["torch"]["gpu"],
        "python": runtime["python"], "platform": runtime["platform"], "torch": runtime["torch"],
        "packages": runtime["packages"], "gsplat_commit": runtime["gsplat"]["commit"],
        "trainer_sha256": env["trainer_sha256"],
        "warmup": validation["evaluation_warmup_render_count"],
        "image_save_disabled": validation["evaluation_image_save_disabled"],
        "data_dir": options["--data_dir"], "data_factor": options["--data_factor"],
        "test_every": options["--test_every"], "sh_degree": options["--sh_degree"],
        "rasterization_ratio": validation["evaluation_rasterization_count_ratio"],
    }


def register_reference(root, project, source, evaluation, env, checkpoint_validator):
    root, project, source, evaluation = map(lambda p: Path(p).resolve(), (root, project, source, evaluation))
    destination = root / "parent_reference.json"
    if destination.exists():
        reference = load_reference(root, verify=True)
        require(reference["source_run"] == str(source) and reference["evaluation_dir"] == str(evaluation),
                "another Parent reference is already registered")
        return reference
    require(not (root / "parent.status.json").exists() and not (root / "parent").exists(),
            "a fresh Parent stage already exists; inspect it before registering a historical reference")
    for stage in ("smoke-parent", "smoke-v3"):
        state = read(root / f"{stage}.status.json")
        require(state.get("status") == "SMOKE_COMPLETE" and state.get("exit_code") == 0,
                f"{stage} has not completed")
    old_env = read(source / "environment.json")
    require(old_env["repository_commit"] == AUDITED_COMMIT, "historical source commit has not been audited")
    current = read(project / "configs/puri_gs_ru_part_v3_parent_garden30k.yaml")
    current.pop("v3_screening")
    require(read(source / "config.yaml") == current, "historical training config differs")
    old_cfg, smoke_cfg = runtime_config(source / "cfg.yml"), runtime_config(root / "smoke-parent/cfg.yml")
    differences = {key: {"historical": old_cfg.get(key), "smoke": smoke_cfg.get(key)}
                   for key in old_cfg.keys() | smoke_cfg.keys() if old_cfg.get(key) != smoke_cfg.get(key)}
    require(not differences, "resolved cfg.yml differences: " + json.dumps(differences, ensure_ascii=False))
    require(old_cfg.get("ckpt") == "null" and old_cfg.get("init_type") == "sfm"
            and old_cfg.get("max_steps") == "30000" and old_cfg.get("puri_gs_ru_enabled") == "true",
            "historical Parent must be fresh standard RU, 30k, SfM initialized")
    train_gpu, train_options = command(source / "run_command.txt")
    require(train_gpu is not None and "--ckpt" not in train_options, "historical training command is incomplete or resumed")
    require(train_options.get("--data_dir") == env["data"], "historical data directory differs")
    require(not any("ru_part" in key or "resume" in key for key in train_options), "historical command has PART/resume options")
    identity_path = root / "smoke-parent/v3_input_manifest.json"
    identity = read(identity_path)
    require(identity == read(root / "smoke-v3/v3_input_manifest.json"), "smoke data identities differ")
    require(read(root / "smoke-parent/v3_camera_sequence.json") ==
            read(root / "smoke-v3/v3_camera_sequence.json"), "smoke camera sequences differ")
    for directory in (source, evaluation):
        split = read(directory / "dataset_split.json")
        for split_name, size in (("train", 161), ("test", 24)):
            require(split[split_name] == identity[f"{split_name}_basenames"] and len(split[split_name]) == size,
                    f"{directory.name}: {split_name} split differs")
    # These are CURRENT input hashes. They are never attributed to the old run.
    require(set(identity["train_image_sha256"]) == set(identity["train_basenames"]), "incomplete train image hashes")
    require(set(identity["sfm_sha256"]) == {"cameras.bin", "images.bin", "points3D.bin"}, "incomplete SfM hashes")
    input_hashes = {str(Path(env["data"], "images_4_png", Path(name).with_suffix(".png"))): digest
                    for name, digest in identity["train_image_sha256"].items()}
    input_hashes.update({str(Path(env["data"], "sparse/0", name)): digest
                         for name, digest in identity["sfm_sha256"].items()})
    for path, digest in input_hashes.items():
        require(sha256_file(path) == digest, f"input changed since smoke: {path}")
    checkpoint = source / "ckpts/ckpt_29999_rank0.pt"
    count = checkpoint_validator(checkpoint, 29999)
    train = read(source / "train_metrics.json")
    require(train["step"] == 29999 and train["gaussian_count"] == count, "historical train step/count mismatch")
    require(int(evaluation.with_suffix(".exitcode").read_text()) == 0, "historical re-evaluation did not exit successfully")
    eval_gpu, eval_options = command(evaluation / "run_command.txt")
    require(eval_options.get("--ckpt") == str(checkpoint), "re-evaluation loaded another checkpoint")
    require(eval_options.get("--eval_warmup_renders") == "10", "re-evaluation warmup is not 10")
    require(read(evaluation / "environment.json")["repository_commit"] == REEVAL_COMMIT,
            "re-evaluation source revision has not been audited")
    metrics = read(evaluation / "test_metrics.json")
    old_metrics = read(source / "independent_eval/test_metrics.json")
    with (evaluation / "per_image_metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 24 and sorted(r["image_name"] for r in rows) == sorted(identity["test_basenames"]),
            "re-evaluation must cover the fixed 24 views exactly once")
    for key in ("psnr", "ssim", "lpips"):
        values = [float(row[key]) for row in rows]
        require(all(math.isfinite(v) for v in (*values, metrics[key], old_metrics[key])), "non-finite evaluation metrics")
        require(abs(sum(values) / 24 - metrics[key]) <= 1e-5, "per-view and aggregate metrics differ")
    difference = abs(metrics["psnr"] - old_metrics["psnr"])
    require(difference <= .001, "historical checkpoint PSNR reproduction failed")
    validation = read(evaluation / "ru_validation.json")
    require(validation.get("standard_checkpoint_load_pass") is True and
            validation.get("evaluation_imported_dino") is False and
            validation.get("evaluation_loaded_mask_head") is False and
            not validation.get("evaluation_loaded_track_cache", False) and
            validation.get("evaluation_rasterization_count_ratio") == 1 and
            validation.get("evaluation_warmup_render_count") == 10 and
            validation.get("raw_latency_sample_count") == 24, "independent evaluator validation failed")
    efficiency = read(evaluation / "efficiency_metrics.json")
    require(efficiency["gaussian_count"] == count == metrics["num_GS"] and
            efficiency["warmup_render_count"] == 10 and efficiency["raw_latency_sample_count"] == 24 and
            math.isfinite(efficiency["render_fps"]) and efficiency["render_fps"] > 0, "invalid inference cost/count")
    files = [identity_path, checkpoint, evaluation.with_suffix(".exitcode")]
    files += [source / name for name in ("config.yaml", "cfg.yml", "environment.json", "run_command.txt",
                                        "dataset_split.json", "train_metrics.json", "independent_eval/test_metrics.json")]
    files += [evaluation / name for name in ("test_metrics.json", "per_image_metrics.csv", "environment.json",
                                             "run_command.txt", "ru_validation.json", "efficiency_metrics.json",
                                             "dataset_split.json", "DSC07988_alpha.npy", "DSC07988_abs_error.npy")]
    reference = {
        "schema_version": 1, "decision": DECISION, "origin": "HISTORICAL_CHECKPOINT_REEVALUATED",
        "source_run": str(source), "evaluation_dir": str(evaluation), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "gaussian_count": count,
        "current_input_manifest": str(identity_path), "current_input_hashes": input_hashes,
        "files_sha256": {str(path): sha256_file(path) for path in files},
        "source_commit": AUDITED_COMMIT, "registered_at_commit": env["commit"],
        "resolved_training_config_matches": True, "psnr_reproduction_abs_difference": difference,
        "quality_comparability": "RECORD_BASED_STANDARD_RU_REUSE",
        "data_basis": "same data path, split, resolved parser/normalization/SfM settings; current inputs match paired smoke; fixed checkpoint PSNR reproduced",
        "historical_input_hashes": "NOT_RECORDED", "historical_camera_sequence": "NOT_RECORDED",
        "limitations": ["Historical image/SfM byte identity and 30k camera sequence were not recorded; current hashes are not retrospective proof.",
                        "Historical training timing is not accepted as a same-device, same-instrumentation baseline."],
        "historical_training_gpu": train_gpu, "reevaluation_gpu": eval_gpu,
        "training_time_comparability": "NOT_ASSESSABLE",
        "evaluation_fingerprint": evaluation_fingerprint(evaluation, env),
        "gpu_mapping_basis": "recorded CUDA_VISIBLE_DEVICES on the same server; historical re-evaluation did not record a UUID",
    }
    write_json(destination, reference)
    return reference


def load_reference(root, *, verify=False):
    path = Path(root) / "parent_reference.json"
    if not path.is_file():
        return None
    reference = read(path)
    require(reference.get("decision") == DECISION, "Parent reference is not accepted")
    if verify:
        for filename, digest in {**reference["files_sha256"], **reference["current_input_hashes"]}.items():
            require(sha256_file(filename) == digest, f"registered Parent/input artifact changed: {filename}")
    return reference
