import pytest

from puri_gs.v3_gpu import GPU_PRIORITY, gpu_inventory, select_gpu


def device(index, **changes):
    return {"index": index, "uuid": f"GPU-{index}", "name": "L20",
            "memory_used_mib": 0, "utilization_percent": 0,
            "compute_process_present": False, **changes}


def test_preference_follows_first_available_in_exact_order():
    rows = [device(i) for i in range(8)]
    for expected in GPU_PRIORITY:
        assert select_gpu(rows)["index"] == expected
        rows[expected]["compute_process_present"] = True
    with pytest.raises(RuntimeError, match="No idle GPU"):
        select_gpu(rows)


@pytest.mark.parametrize("busy", [
    {"compute_process_present": True}, {"memory_used_mib": 513}, {"utilization_percent": 6},
])
def test_busy_preferred_gpu_is_skipped(busy):
    assert select_gpu([device(6, **busy), device(7)])["index"] == 7


def test_pinned_experiment_does_not_move_to_idle_lower_priority_device():
    with pytest.raises(RuntimeError, match="pinned"):
        select_gpu([device(6, compute_process_present=True), device(7)], pinned=6)
    with pytest.raises(ValueError, match="device changes"):
        select_gpu([device(6), device(7)], "7", pinned=6)
    assert select_gpu([device(6), device(7)], "auto", pinned=7)["index"] == 7


def test_explicit_index_stays_explicit_and_missing_priority_ids_are_skipped():
    assert select_gpu([device(0), device(1)])["index"] == 0
    assert select_gpu([device(0), device(1)], "1")["index"] == 1
    with pytest.raises(ValueError):
        select_gpu([device(0)], "-1")


def test_nvidia_smi_csv_maps_compute_processes_by_uuid(monkeypatch):
    replies = iter(["0, GPU-a, NVIDIA L20, 15, 0\n6, GPU-b, NVIDIA L20, 0, 0\n",
                    "GPU-b, 12345\n"])
    monkeypatch.setattr("puri_gs.v3_gpu.subprocess.check_output", lambda *a, **k: next(replies))
    rows = gpu_inventory()
    assert rows[1]["compute_process_present"]
    assert select_gpu(rows)["index"] == 0
