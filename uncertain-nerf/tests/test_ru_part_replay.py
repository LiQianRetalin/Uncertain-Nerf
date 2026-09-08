import random

import numpy as np
import pytest
import torch

from puri_gs.replay import (
    REPLAY_SAVE_STEPS,
    camera_sequence_sha256,
    capture_rng_state,
    fixed_camera_sequence,
    load_replay_checkpoint,
    restore_rng_state,
    restore_training_state,
    save_replay_checkpoint,
)
from puri_gs.delayed_absgrad import DelayedAbsGradSchedule
from puri_gs.ru_part import LINEAGE_KEYS, LineageDelayedAbsGradStrategy


def test_fixed_camera_sequence_is_deterministic_and_epoch_complete():
    first = fixed_camera_sequence(7, 18, 42)
    second = fixed_camera_sequence(7, 18, 42)
    assert torch.equal(first, second)
    assert sorted(first[:7].tolist()) == list(range(7))
    assert sorted(first[7:14].tolist()) == list(range(7))
    assert len(camera_sequence_sha256(first)) == 64


def test_rng_roundtrip_restores_python_numpy_and_torch():
    random.seed(4)
    np.random.seed(5)
    torch.manual_seed(6)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_replay_checkpoint_roundtrip_and_camera_hash_guard(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    sequence = fixed_camera_sequence(7, 20_000, 42)
    path = tmp_path / "replay.pt"
    save_replay_checkpoint(
        path,
        step=REPLAY_SAVE_STEPS[0],
        splats={"means": parameter},
        optimizers={"means": optimizer},
        schedulers=[scheduler],
        strategy_state={"count": torch.ones(1)},
        camera_sequence=sequence,
        intervention_mode="current",
        ru_training_state={"mask": 1},
    )
    payload = load_replay_checkpoint(path)
    assert payload["next_step"] == 10_000
    assert payload["camera_sequence"][payload["next_step"]] == sequence[10_000]
    assert payload["optimizers"]["means"]["state"]
    payload["camera_sequence"][0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_replay_checkpoint(path)


def test_unregistered_replay_step_is_rejected(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    with pytest.raises(ValueError, match="not preregistered"):
        save_replay_checkpoint(
            tmp_path / "bad.pt",
            step=10,
            splats={"means": parameter},
            optimizers={"means": optimizer},
            schedulers=[],
            strategy_state={},
            camera_sequence=fixed_camera_sequence(2, 20, 42),
            intervention_mode="noop",
            ru_training_state={},
        )


def test_restore_reconnects_optimizer_to_shape_changed_parameter():
    old = torch.nn.Parameter(torch.tensor([9.0]))
    splats = torch.nn.ParameterDict({"means": old})
    optimizer = torch.optim.Adam([old], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    source = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    source_optimizer = torch.optim.Adam([source], lr=0.1)
    source_scheduler = torch.optim.lr_scheduler.ExponentialLR(source_optimizer, gamma=0.9)
    source.grad = torch.ones_like(source)
    source_optimizer.step()
    source_scheduler.step()
    payload = {
        "splats": {"means": source.detach()},
        "optimizers": {"means": source_optimizer.state_dict()},
        "schedulers": [source_scheduler.state_dict()],
        "strategy_state": {"count": torch.ones(2)},
    }
    state = restore_training_state(
        payload,
        splats=splats,
        optimizers={"means": optimizer},
        schedulers=[scheduler],
        device="cpu",
    )
    restored = splats["means"]
    assert restored is not old and optimizer.param_groups[0]["params"] == [restored]
    assert torch.equal(state["count"], torch.ones(2))
    before = restored.detach().clone()
    restored.grad = torch.ones_like(restored)
    optimizer.step()
    assert not torch.equal(restored.detach(), before)


def test_lineage_initialization_and_birth_identity(tmp_path):
    strategy = LineageDelayedAbsGradStrategy(
        schedule=DelayedAbsGradSchedule(), result_dir=tmp_path
    )
    state = {}
    strategy.ensure_lineage(state, 3, torch.device("cpu"))
    assert state["uid"].tolist() == [0, 1, 2]
    assert state["lineage_id"].tolist() == [0, 1, 2]
    assert state["parent_uid"].tolist() == [-1, -1, -1]
    for key in LINEAGE_KEYS:
        state[key] = torch.cat((state[key], torch.zeros(2, dtype=torch.int64)))
    strategy.assign_birth_lineage(state, [17, 23], 10_000)
    assert state["uid"].tolist() == [0, 1, 2, 3, 4]
    assert state["lineage_id"][-2:].tolist() == [3, 4]
    assert state["parent_uid"][-2:].tolist() == [-1, -1]
    assert state["track_id"][-2:].tolist() == [17, 23]
    assert state["birth_step"][-2:].tolist() == [10_000, 10_000]
