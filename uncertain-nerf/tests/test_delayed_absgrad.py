from puri_gs.delayed_absgrad import DelayedAbsGradSchedule
from run_puri_gs import PROJECT_ROOT


def test_fixed_delayed_boundaries_and_reset_schedule():
    schedule = DelayedAbsGradSchedule()
    assert not schedule.densification_allowed(9999)
    assert schedule.densification_allowed(10000)
    assert not schedule.should_reset(14999)
    assert schedule.should_reset(15000)
    assert schedule.densification_allowed(19999)
    assert not schedule.densification_allowed(20000)
    assert schedule.should_refine(10000)
    assert not schedule.should_refine(10001)
    assert schedule.should_reset(18000)
    assert not schedule.should_reset(12000)


def test_mask_updates_pause_for_exact_fixed_window():
    schedule = DelayedAbsGradSchedule()
    assert not schedule.mask_update_paused(15000)
    assert schedule.mask_update_paused(15001)
    assert schedule.mask_update_paused(15300)
    assert not schedule.mask_update_paused(15301)
    assert not schedule.mask_update_paused(18000)
    assert schedule.mask_update_paused(18001)


def test_ru_patch_preserves_seeded_b1_data_order_and_absgrad_contract():
    patch = (PROJECT_ROOT / "patches" / "gsplat_v1.5.3_puri_gs_ru.patch").read_text()
    config = (PROJECT_ROOT / "configs" / "puri_gs_ru_full30k.yaml").read_text()
    assert "torch.random.fork_rng" in patch
    assert "delayed_strategy_from_default" in patch
    assert '"absgrad": true' in config
    assert '"grow_grad2d": 0.0006' in config
