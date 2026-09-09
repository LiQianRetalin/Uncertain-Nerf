"""Read the existing final state; reuse isolated renderer/optimizer source blocks.

Never construct Runner, a DINO model, a replay controller, or a topology loop.
"""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

from puri_gs.coverage_recoverability import canonical_sha, read_json, require
from puri_gs.ru_part_v3 import StaticSupport, runtime_identity
from puri_gs.semantic_mask import StaticResponsibilityHead, hard_static_mask
from puri_gs.static_tracks import sha256_file


def read_runtime_yaml(path):
    import yaml
    def visit(node):
        if isinstance(node, yaml.MappingNode):
            result = {key.value: visit(child) for key, child in node.value}
            if node.tag != "tag:yaml.org,2002:map":
                result["__yaml_type__"] = node.tag
            return result
        if isinstance(node, yaml.SequenceNode):
            return [visit(child) for child in node.value]
        if node.tag == "tag:yaml.org,2002:null":
            return None
        if node.tag == "tag:yaml.org,2002:bool":
            return node.value.casefold() == "true"
        if node.tag == "tag:yaml.org,2002:int":
            return int(node.value)
        if node.tag == "tag:yaml.org,2002:float":
            return float(node.value)
        require(node.tag == "tag:yaml.org,2002:str", "unsupported YAML scalar tag")
        return node.value
    return visit(yaml.compose(Path(path).read_text(encoding="utf-8")))


def source_functions(trainer_path, *, rasterization=None, default_strategy=None):
    """Extract two already-audited pure blocks without executing trainer imports."""
    tree = ast.parse(Path(trainer_path).read_text(encoding="utf-8"))
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_splats_with_optimizers")
    start = next(i for i, n in enumerate(factory.body) if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "BS" for t in n.targets))
    end = next(i for i, n in enumerate(factory.body) if isinstance(n, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == "optimizers" for t in n.targets))
    node = ast.parse("def make_optimizers(splats, params, batch_size, world_size, sparse_grad, visible_adam):\n    pass").body[0]
    node.body = copy.deepcopy(factory.body[start:end+1]) + [ast.Return(ast.Name(id="optimizers", ctx=ast.Load()))]
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
    render = copy.deepcopy(next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "rasterize_splats"))
    render.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node, render], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "math": math, "rasterization": rasterization, "DefaultStrategy": default_strategy}
    exec(compile(module, str(trainer_path), "exec"), namespace)
    return namespace["make_optimizers"], namespace["rasterize_splats"], canonical_sha(ast.unparse(module))


def terminal_learning_rates(cfg, scene_scale):
    require(cfg["max_steps"] == 30000 and cfg["batch_size"] == 1 and cfg["steps_scaler"] == 1,
            "OPTIMIZER_CONFIG_UNRESOLVED: unexpected steps/batch scaling")
    require(not cfg["sparse_grad"] and not cfg["visible_adam"], "OPTIMIZER_CONFIG_UNRESOLVED: source backend differs")
    gamma = .01 ** (1.0 / cfg["max_steps"])
    means = cfg["means_lr"] * scene_scale
    # Exact Python multiply recurrence used by ExponentialLR; scheduler.step()
    # follows optimizer.step(), so original update 29999 uses 29999 decays.
    for _ in range(29999):
        means *= gamma
    result = {"means": means, "scales": cfg["scales_lr"], "quats": cfg["quats_lr"],
              "opacities": cfg["opacities_lr"], "sh0": cfg["sh0_lr"], "shN": cfg["shN_lr"]}
    require(all(math.isfinite(v) and v > 0 for v in result.values()), "OPTIMIZER_CONFIG_UNRESOLVED: invalid LR")
    return result


def runtime_environment(device):
    import importlib.metadata
    import platform
    import os
    import subprocess
    import gsplat
    packages = {}
    for name in ("gsplat", "numpy", "Pillow", "imageio", "PyYAML", "fused-ssim"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "UNAVAILABLE"
    package_root = Path(gsplat.__file__).resolve().parent
    installed_files = sorted(path for path in package_root.rglob("*") if path.is_file()
                             and path.suffix in (".py", ".so", ".pyd", ".cu", ".cuh", ".h", ".hpp", ".cpp"))
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "packages": packages,
            "installed_gsplat_root": str(package_root),
            "installed_gsplat_sha256": {str(path): sha256_file(path) for path in installed_files},
            "driver_versions": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def validate_splats(checkpoint, *, count, step):
    require(checkpoint.get("step") == step, "source/diagnostic checkpoint step mismatch")
    values = checkpoint["splats"]
    shapes = {"means": (count, 3), "scales": (count, 3), "quats": (count, 4),
              "opacities": (count,), "sh0": (count, 1, 3), "shN": (count, 15, 3)}
    require(set(values) == set(shapes), "nonstandard Gaussian parameter set")
    for key, shape in shapes.items():
        require(tuple(values[key].shape) == shape and values[key].dtype == torch.float32 and
                torch.isfinite(values[key]).all(), f"invalid Gaussian tensor: {key}")
    return values


class FinalStateRuntime:
    def __init__(self, paths, protocol, *, device="cuda:0"):
        self.paths, self.protocol = paths, protocol
        self.source = Path(paths["source_run"])
        self.device = torch.device(device)
        self.identity = read_json(self.source / "v3_input_manifest.json")
        self.run_manifest = read_json(self.source / "v3_run_manifest.json")
        self.raw_cfg = read_runtime_yaml(self.source / "cfg.yml")
        require(self.run_manifest["implementation_revision"] == "v3-single-q-pinned-transfer-v1", "source V3 revision differs")
        require(read_json(self.source / "config.yaml")["v3_screening"] == "v3", "source is not the V3 run")
        self.cfg = SimpleNamespace(**self.raw_cfg)
        for key in ("app_opt", "pose_opt", "depth_loss", "use_bilateral_grid", "random_bkgd",
                    "sparse_grad", "visible_adam", "opacity_reg", "scale_reg", "with_ut", "with_eval3d"):
            require(not getattr(self.cfg, key), f"unsupported active source setting: {key}")
        require(self.cfg.patch_size is None and self.cfg.normalize_world_space and self.cfg.sh_degree == 3
                and self.cfg.ssim_lambda == .2 and self.cfg.data_factor == 4 and self.cfg.camera_model == "pinhole",
                "source rendering/training settings differ")
        require(self.cfg.mask_threshold == .25 and self.cfg.mask_erode_kernel == 7 and
                self.cfg.dino_feature_dim == 384 and self.cfg.mask_hidden_dim == 16 and self.cfg.dino_fine_grid == 36,
                "source Mask settings differ")
        strategy = self.raw_cfg["strategy"]
        require(strategy["__yaml_type__"] == "tag:yaml.org,2002:python/object:puri_gs.delayed_absgrad.DelayedAbsGradStrategy"
                and strategy["absgrad"] is True, "source strategy is not standard RU AbsGrad")
        sys.path.insert(0, str(Path(paths["gsplat"]) / "examples"))
        from datasets.colmap import Parser, Dataset
        import imageio.v2 as imageio
        test_stems = {Path(name).stem for name in self.identity["test_basenames"]}
        self.opened_training_images = set()
        self.test_image_read_attempts = 0
        original_imread = imageio.imread
        def training_only_imread(path, *args, **kwargs):
            if Path(path).stem in test_stems:
                self.test_image_read_attempts += 1
                raise RuntimeError("TEST_IMAGE_ACCESS_REJECTED")
            self.opened_training_images.add(Path(path).name)
            return original_imread(path, *args, **kwargs)
        require(Path(paths["data"], "images_4_png").is_dir(), "prepared factor4 PNG directory is missing")
        imageio.imread = training_only_imread
        try:
            self.parser = Parser(paths["data"], factor=4, normalize=True, test_every=8, calibration_index=1)
        finally:
            imageio.imread = original_imread
        self.trainset = Dataset(self.parser, split="train", val_every=0)
        test_metadata_only = Dataset(self.parser, split="test", val_every=0)
        actual = runtime_identity(self.parser, self.trainset, test_metadata_only)
        require(all(actual[key] == self.identity[key] for key in actual), "source camera/split identity differs")
        require(np.array_equal(np.asarray(self.identity["normalization"]), self.parser.transform), "normalization differs")
        self.source_hashes = {}
        for local, index in enumerate(self.trainset.indices):
            name = self.parser.image_names[int(index)]
            path = Path(self.parser.image_paths[int(index)])
            digest = sha256_file(path)
            require(digest == self.identity["train_image_sha256"][name], f"training image changed: {name}")
            self.source_hashes[str(path)] = digest
        for name, expected in self.identity["sfm_sha256"].items():
            path = Path(paths["data"], "sparse/0", name)
            digest = sha256_file(path)
            require(digest == expected, f"SfM changed: {name}")
            self.source_hashes[str(path)] = digest
        self.names = self.identity["train_basenames"]
        self.local_ids = {name: i for i, name in enumerate(self.names)}
        self.cfg.data_dir = paths["data"]
        self.cfg.feature_cache_dir = paths["feature_cache"]
        self.support = StaticSupport(paths["track_cache"], identity=self.identity, cfg=self.cfg, device=self.device)
        require(self.support.audit["cache_file_sha256"] == self.run_manifest["cache"]["cache_file_sha256"], "source C cache changed")
        self.source_hashes[str(Path(paths["track_cache"]))] = self.support.audit["cache_file_sha256"]
        self.head, self.feature_cache, self.mask_problem = None, None, None
        head_path = self.source / "aux/mask_head_step29999.pt"
        try:
            from puri_gs.dino_features import FeatureCache
            metadata = read_json(self.source / "aux/dino_environment.json")
            self.feature_cache = FeatureCache(paths["feature_cache"], expected_weight_sha256=metadata["weight_sha256"])
            self.head = StaticResponsibilityHead(self.cfg.dino_feature_dim, self.cfg.mask_hidden_dim).to(self.device)
            self.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True), strict=True)
            require(all(torch.isfinite(p).all() for p in self.head.parameters()), "non-finite final head")
            self.head.eval().requires_grad_(False)
            self.source_hashes[str(head_path)] = sha256_file(head_path)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            self.head = None
            self.mask_problem = f"MASK_STATE_UNAVAILABLE: {error}"
        self.scene_scale = float(self.parser.scene_scale * 1.1 * self.cfg.global_scale)
        self.optimizer_problem = None
        try:
            self.learning_rates = terminal_learning_rates(self.raw_cfg, self.scene_scale)
        except (KeyError, ValueError) as error:
            self.learning_rates = None
            self.optimizer_problem = f"OPTIMIZER_CONFIG_UNRESOLVED: {error}"
        self.cpu_data = {}
        self.splats = None
        self.counts = {"training_rasterization": 0, "evaluation_rasterization": 0, "precheck_rasterization": 0,
                       "preparation_rasterization": 0, "gaussian_backward": 0, "optimizer_updates": 0,
                       "precheck_updates": 0, "head_updates": 0, "topology_events": 0}
        from gsplat import rasterization
        from gsplat.strategy import DefaultStrategy
        self.cfg.strategy = DefaultStrategy(absgrad=True)
        self.world_size = 1
        self.make_optimizers, self.render_function, self.extracted_source_sha = source_functions(
            Path(paths["gsplat"], "examples/simple_trainer.py"), rasterization=rasterization, default_strategy=DefaultStrategy)
        self.optimizers = {}

    @torch.no_grad()
    def mask(self, name):
        require(name in self.local_ids, "not a training view")
        if self.head is None:
            return None
        camera = self.identity["cameras"][self.local_ids[name]]
        feature = self.feature_cache.load(name, 36).unsqueeze(0).to(self.device)
        feature_path = self.feature_cache.directory / self.feature_cache.records[name]["file"]
        self.source_hashes[str(feature_path)] = sha256_file(feature_path)
        probability = self.head(feature)
        probability = torch.nn.functional.interpolate(probability, size=(camera["height"], camera["width"]),
                                                        mode="bilinear", align_corners=False)
        result = hard_static_mask(probability, threshold=self.cfg.mask_threshold, kernel_size=self.cfg.mask_erode_kernel)
        self.feature_cache._loaded.pop(name, None)  # bound CPU feature memory, never change cache files
        return result[0, 0].detach()

    def data(self, name):
        require(name in self.local_ids and name not in self.identity["test_basenames"], "TEST_IMAGE_ACCESS_REJECTED")
        if name not in self.cpu_data:
            self.cpu_data[name] = self.trainset[self.local_ids[name]]
            self.opened_training_images.add(name)
        return self.cpu_data[name]

    def load_parameters(self, *, optimizers=True):
        require(self.splats is None, "previous group parameters must be discarded")
        checkpoint = torch.load(self.paths["checkpoint"], map_location="cpu", weights_only=True)
        values = validate_splats(checkpoint, count=self.protocol["gaussian_count"], step=29999)
        import hashlib
        digest = hashlib.sha256()
        for key in sorted(values):
            digest.update(key.encode())
            digest.update(values[key].contiguous().numpy().tobytes())
        self.splats = torch.nn.ParameterDict({key: torch.nn.Parameter(value.to(self.device)) for key, value in values.items()})
        if optimizers:
            require(self.learning_rates is not None, self.optimizer_problem)
            self.optimizers = self.make_optimizers(self.splats,
                [(key, None, self.learning_rates[key]) for key in ("means", "scales", "quats", "opacities", "sh0", "shN")], 1, 1, False, False)
            require(all(not opt.state for opt in self.optimizers.values()), "new optimizer state is not empty")
        return digest.hexdigest()

    def optimizer_description(self):
        description = {key: {"class": type(opt).__module__ + "." + type(opt).__name__,
                      "settings": {name: value for name, value in opt.param_groups[0].items() if name != "params"},
                      "initial_state_entries": len(opt.state)} for key, opt in self.optimizers.items()}
        return json.loads(json.dumps(description, allow_nan=False))

    def render(self, name, *, kind):
        require(kind in ("training", "evaluation", "precheck", "preparation"), "untracked renderer purpose")
        data = self.data(name)
        target = data["image"][None].to(self.device) / 255.
        self.counts[kind + "_rasterization"] += 1
        rgb, alpha, _ = self.render_function(self, camtoworlds=data["camtoworld"][None].to(self.device),
            Ks=data["K"][None].to(self.device), width=target.shape[2], height=target.shape[1],
            sh_degree=3, near_plane=self.cfg.near_plane, far_plane=self.cfg.far_plane)
        return rgb, target, alpha

    def discard(self):
        self.optimizers = {}
        self.splats = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
