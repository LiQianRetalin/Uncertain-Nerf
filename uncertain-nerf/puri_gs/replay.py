"""Deterministic camera scheduling and full-state RU-PART replay checkpoints."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


REPLAY_SCHEMA = "puri-gs-ru-part-replay-v1"
REPLAY_SAVE_STEPS = (9_999, 13_299, 16_599, 19_899)


def fixed_camera_sequence(dataset_size: int, total_steps: int, seed: int) -> Tensor:
    if dataset_size <= 0 or total_steps <= 0:
        raise ValueError("dataset_size and total_steps must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    epochs = (total_steps + dataset_size - 1) // dataset_size
    sequence = torch.cat(
        [torch.randperm(dataset_size, generator=generator) for _ in range(epochs)]
    )[:total_steps]
    return sequence.to(dtype=torch.int64)


def camera_sequence_sha256(sequence: Tensor) -> str:
    canonical = sequence.detach().cpu().contiguous().to(torch.int64).numpy()
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("replay RNG state has an invalid schema")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available():
        cuda_states = list(state["torch_cuda"])
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("replay CUDA RNG device count differs from runtime")
        torch.cuda.set_rng_state_all(cuda_states)


def save_replay_checkpoint(
    path: str | Path,
    *,
    step: int,
    splats: Mapping[str, Tensor],
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Sequence[Any],
    strategy_state: Mapping[str, Any],
    camera_sequence: Tensor,
    intervention_mode: str,
    ru_training_state: Mapping[str, Any],
) -> None:
    if step not in REPLAY_SAVE_STEPS:
        raise ValueError("replay checkpoint step is not preregistered")
    if len(camera_sequence) <= step + 1:
        raise ValueError("camera sequence does not include the next replay step")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": REPLAY_SCHEMA,
            "step": int(step),
            "next_step": int(step + 1),
            "intervention_mode": intervention_mode,
            "splats": {name: value.detach() for name, value in splats.items()},
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            },
            "schedulers": [scheduler.state_dict() for scheduler in schedulers],
            "strategy_state": dict(strategy_state),
            "rng": capture_rng_state(),
            "camera_sequence": camera_sequence.detach().cpu(),
            "camera_sequence_sha256": camera_sequence_sha256(camera_sequence),
            "ru_training": dict(ru_training_state),
        },
        destination,
    )


def load_replay_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {
        "schema", "step", "next_step", "intervention_mode", "splats",
        "optimizers", "schedulers", "strategy_state", "rng",
        "camera_sequence", "camera_sequence_sha256", "ru_training",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("replay checkpoint has an invalid top-level schema")
    if payload["schema"] != REPLAY_SCHEMA:
        raise ValueError("unsupported replay checkpoint schema")
    if payload["next_step"] != payload["step"] + 1:
        raise ValueError("replay checkpoint next_step is inconsistent")
    if camera_sequence_sha256(payload["camera_sequence"]) != payload["camera_sequence_sha256"]:
        raise ValueError("replay camera sequence hash mismatch")
    return payload


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def restore_training_state(
    payload: Mapping[str, Any],
    *,
    splats: Any,
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Sequence[Any],
    device: str | torch.device,
) -> dict[str, Any]:
    """Restore shape-changing Gaussian state and reconnect optimizer parameters."""

    target = torch.device(device)
    saved_splats = payload["splats"]
    if set(saved_splats) != set(splats) or set(payload["optimizers"]) != set(optimizers):
        raise ValueError("replay Gaussian/optimizer keys differ from runtime")
    if len(payload["schedulers"]) != len(schedulers):
        raise ValueError("replay scheduler count differs from runtime")
    for name in list(splats.keys()):
        old = splats[name]
        new = torch.nn.Parameter(
            saved_splats[name].to(device=target, dtype=old.dtype),
            requires_grad=old.requires_grad,
        )
        splats[name] = new
        optimizer = optimizers.get(name)
        if optimizer is None:
            if old.requires_grad:
                raise ValueError(f"trainable replay parameter {name} has no optimizer")
            continue
        for group in optimizer.param_groups:
            if len(group["params"]) != 1 or group["params"][0] is not old:
                raise ValueError("replay supports one parameter per optimizer group")
            group["params"] = [new]
        optimizer.state.pop(old, None)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])
    for scheduler, state in zip(schedulers, payload["schedulers"]):
        scheduler.load_state_dict(state)
    return _to_device(payload["strategy_state"], target)
