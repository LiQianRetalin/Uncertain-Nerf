from tools.summarize_puri_gs_ru import decide


def _run(scene, method, psnr, ssim, lpips, count, delta=0.0):
    names = [f"{index:04d}.png" for index in range(count)]
    return {
        "scene": scene,
        "method": method,
        "test": {"psnr": psnr, "ssim": ssim, "lpips": lpips},
        "train": {"training_time_seconds": 100.0 if method == "b1" else 250.0},
        "efficiency": {
            "render_fps": 100.0 if method == "b1" else 98.0,
            "gaussian_count": 1_000_000 if method == "b1" else 1_100_000,
            "inference_vram_gib": 2.0 if method == "b1" else 2.05,
        },
        "validation": {
            "standard_checkpoint_load_pass": True,
            "evaluation_imported_dino": False,
            "evaluation_loaded_mask_head": False,
            "evaluation_rasterization_count_ratio": 1.0,
            "gradient_isolation_pass": True,
        },
        "per_image": {
            name: {"psnr": 25.0 + delta, "ssim": 0.9, "lpips": 0.1}
            for name in names
        },
        "checkpoint_size_bytes": 1000 if method == "b1" else 1100,
        "dino_time": {
            "render_feature_seconds": 50.0,
            "render_feature_fraction_of_training": 0.2,
        },
        "dino_environment": {"mask_head_parameter_count": 6177},
    }


def test_phase_r_full_pass_is_deterministic():
    runs = {
        "android": {
            "b1": _run("android", "b1", 25.0, 0.90, 0.10, 19),
            "ru": _run("android", "ru", 26.0, 0.91, 0.08, 19, delta=1.0),
        },
        "room": {
            "b1": _run("room", "b1", 31.0, 0.94, 0.08, 39),
            "ru": _run("room", "ru", 30.9, 0.938, 0.085, 39, delta=-0.1),
        },
    }
    first = decide(runs)
    second = decide(runs)
    assert first == second
    assert first["decision"] == "RU_RECONSTRUCTION_PASS"
