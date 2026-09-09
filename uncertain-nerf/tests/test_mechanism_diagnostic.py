import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from puri_gs.mechanism_diagnostic import (
    HORIZON, METHODS, Progress, comparable_training_state, compare_state,
    derive_candidate_roi, export_source_probes, filtered_suffix,
    largest_component_8, load_training_snapshot, optimizer_references_current,
    build_final_report, compare_replay_runs, confirm_static_rois,
    prepare_fixed_branch, run_vjp_probes,
    save_training_snapshot, snapshot_tree, static_unmask_l1,
    verify_reproduction, write_new_json,
)

try:
    import torch
except ImportError:
    torch = None


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.json"
        self.csv = self.root / "rows.csv"
        names = [f"view{i}" for i in range(24)]
        self.manifest.write_text(json.dumps({
            "dataset": {"factor": 4, "train_count": 161, "test_count": 24, "test_names": names},
            "gates": {"reproduction_gate": True, "reproduction_max_abs_delta_db": 0},
            "checkpoints": {"RU": {"sha256_before": "a", "sha256_after": "a"}},
        }))
        with self.csv.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["method", "image_name", "psnr_rerender", "psnr_reference", "psnr_abs_delta", "all_finite"])
            for method in METHODS:
                for name in names:
                    writer.writerow([method, name, 25, 25, 0, True])

    def test_complete_evidence_does_not_claim_current_hash_or_gpu_verification(self):
        result = verify_reproduction(self.manifest, self.csv)
        self.assertFalse(result["checkpoint_current_hash_verified"])
        self.assertFalse(result["test_rgb_opened"])
        self.assertEqual(len(result["methods"]), 6)

    def test_nonfinite_and_fabricated_delta_rejected(self):
        original = self.csv.read_text()
        for changed in ("nan,25,0,True", "26,25,0,True"):
            self.csv.write_text(original.replace("25,25,0,True", changed, 1))
            with self.assertRaises(ValueError):
                verify_reproduction(self.manifest, self.csv)

    def test_missing_row_rejected(self):
        self.csv.write_text("\n".join(self.csv.read_text().splitlines()[:-1]))
        with self.assertRaises(ValueError):
            verify_reproduction(self.manifest, self.csv)

    def test_existing_result_is_preserved(self):
        target = self.root / "audit.json"
        write_new_json(target, {"original": True})
        with self.assertRaises(FileExistsError):
            write_new_json(target, {"original": False})
        self.assertEqual(json.loads(target.read_text()), {"original": True})

    def test_suffix_excludes_monitors_without_changing_order(self):
        sequence = [99] * 10000 + list(range(10)) * 100
        suffix, exposures = filtered_suffix(sequence, [2, 3, 4], [0, 1])
        self.assertEqual(len(suffix), 400)
        self.assertEqual(suffix[:7], [0, 1, 5, 6, 7, 8, 9])
        self.assertTrue(min(exposures.values()) >= 2)

    def test_insufficient_exposure_is_not_resampled(self):
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_SOURCE_EXPOSURE"):
            filtered_suffix([99] * 11000, [], [0, 1])

    def test_progress_emits_bounded_updates_and_error(self):
        path = self.root / "progress.jsonl"
        progress = Progress(path, "R1", 200)
        for step in range(1, 200):
            progress.update(step)
        progress.update(199, status="failed", error="test failure")
        progress.close()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["completed"], 199)
        self.assertEqual(rows[-1]["status"], "failed")
        with self.assertRaises(FileExistsError):
            Progress(path, "R2", 200)

    def test_early_report_records_not_run_and_hard_stop_text(self):
        audit, replay, roi = (self.root / name for name in ("audit.json", "replay.json", "roi.json"))
        audit.write_text(json.dumps({"status": "SAVED_REPRODUCTION_VERIFIED"}))
        replay.write_text(json.dumps({"status": "REPLAY_ACCEPTED"}))
        roi.write_text(json.dumps({
            "status": "ROI_NOT_ESTABLISHED", "monitor_names": [], "sources": {}
        }))
        output = self.root / "report"
        result = build_final_report(
            audit_path=audit, replay_path=replay, roi_path=roi,
            vjp_decision_path=None, branch_dirs=None, output_dir=output,
        )
        self.assertEqual(result["branch_analysis"], "not_run")
        self.assertIn("强制停止", (output / "diagnostic_report.md").read_text())

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_confirmation_is_exclusive_and_whole_candidate(self):
        (self.root / "roi_evidence.json").write_text(json.dumps({
            "status": "ROI_ESTABLISHED"
        }))
        for source in ("DSC07987.JPG", "DSC07989.JPG"):
            torch.save(
                torch.ones(8, 8, dtype=torch.bool),
                self.root / f"candidate_roi_{Path(source).stem}.pt",
            )
        result = confirm_static_rois(self.root, note="用户确认整块为静态背景")
        self.assertEqual(result["status"], "STATIC_ROI_CONFIRMED")
        with self.assertRaises(FileExistsError):
            confirm_static_rois(self.root, note="cannot overwrite")


@unittest.skipIf(torch is None, "PyTorch unavailable")
class TensorContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_comparison_rejects_discrete_mismatch_and_nonfinite(self):
        self.assertFalse(compare_state({"step": 10000}, {"step": 10001})["passed"])
        for other in (torch.ones(3), torch.full((3,), float("nan"))):
            self.assertFalse(compare_state(torch.zeros(3), other)["passed"])
        self.assertTrue(compare_state(torch.ones(3), torch.ones(3) + 1e-6)["passed"])
        self.assertFalse(compare_state(torch.zeros(3), torch.zeros(4))["passed"])

    def test_comparison_does_not_average_away_one_bad_chunk(self):
        left = torch.ones(100)
        right = left.clone()
        right[-1] = 1.01
        result = compare_state(left, right, chunk_size=7)
        self.assertFalse(result["passed"])

    def test_snapshot_is_independent_and_stale_optimizer_is_detected(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.Adam([parameter])
        parameter.sum().backward()
        optimizer.step()
        saved = snapshot_tree(optimizer.state_dict())
        optimizer.state[parameter]["exp_avg"].zero_()
        self.assertTrue(bool(saved["state"][0]["exp_avg"].any()))
        optimizer_references_current({"means": parameter}, {"means": optimizer})
        with self.assertRaisesRegex(ValueError, "stale"):
            optimizer_references_current({"means": torch.nn.Parameter(parameter.clone())}, {"means": optimizer})

    def test_unmask_uses_full_image_channel_mean_and_detaches_q(self):
        render = torch.zeros(1, 2, 2, 3, requires_grad=True)
        mask = torch.zeros(1, 1, 2, 2, requires_grad=True)
        oracle = torch.tensor([[[[1., 0.], [0., 0.]]]], requires_grad=True)
        extra = static_unmask_l1(render, torch.ones_like(render), mask, oracle)
        self.assertAlmostEqual(extra.item(), 0.2)
        extra.backward()
        self.assertIsNone(mask.grad)
        self.assertIsNone(oracle.grad)
        self.assertEqual(torch.count_nonzero(render.grad).item(), 3)

    def test_zero_q_keeps_original_loss_and_gradient(self):
        render = torch.zeros(1, 2, 2, 3, requires_grad=True)
        loss = (render - 1).square().mean()
        expected = torch.autograd.grad(loss, render, retain_graph=True)[0]
        for mask, oracle in ((torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),
                             (torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))):
            extra = static_unmask_l1(render, torch.ones_like(render), mask, oracle)
            self.assertIsNone(extra)
            actual = loss if extra is None else loss + extra
            self.assertIs(actual, loss)
            self.assertTrue(torch.equal(expected, torch.autograd.grad(actual, render, retain_graph=True)[0]))

    def test_diagnostic_snapshot_is_full_exclusive_and_comparable(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        parameter.sum().backward()
        optimizer.step()
        scheduler.step()
        destination = self.root / "terminal.pt"
        kwargs = dict(
            step=10199,
            stage="U",
            splats={"means": parameter},
            optimizers={"means": optimizer},
            schedulers=[scheduler],
            strategy_state={"grad2d": torch.ones(2)},
            camera_sequence=torch.arange(HORIZON),
            intervention_mode="parent",
            ru_training_state={"histogram": torch.ones(3), "counter": 4},
        )
        save_training_snapshot(destination, **kwargs)
        with self.assertRaises(FileExistsError):
            save_training_snapshot(destination, **kwargs)
        loaded = load_training_snapshot(destination)
        self.assertEqual(loaded["next_step"], 10200)
        self.assertTrue(compare_state(
            comparable_training_state(loaded),
            comparable_training_state(loaded),
        )["passed"])

    def test_snapshot_rejects_short_horizon_and_stale_optimizer(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter])
        base = dict(
            path=self.root / "bad.pt", step=9999, stage="U",
            splats={"means": parameter}, optimizers={"means": optimizer},
            schedulers=[], strategy_state={}, intervention_mode="parent",
            ru_training_state={},
        )
        with self.assertRaisesRegex(ValueError, "30000-step horizon"):
            save_training_snapshot(camera_sequence=torch.arange(10000), **base)
        with self.assertRaisesRegex(ValueError, "stale"):
            save_training_snapshot(
                camera_sequence=torch.arange(HORIZON),
                splats={"means": torch.nn.Parameter(torch.ones(1))},
                **{key: value for key, value in base.items() if key != "splats"},
            )

    def test_source_probe_uses_only_registered_training_views_and_restores_rng(self):
        class Trainset:
            indices = [0, 1]

            def __getitem__(self, item):
                return {
                    "image": torch.full((8, 8, 3), 255.0 * item),
                    "camtoworld": torch.eye(4),
                    "K": torch.eye(3),
                }

        class Cache:
            @staticmethod
            def load(name, grid):
                return torch.ones(1, grid, grid)

        training = SimpleNamespace(
            parser=SimpleNamespace(image_names=list(("DSC07987.JPG", "DSC07989.JPG"))),
            trainset=Trainset(), head=torch.nn.Conv2d(1, 1, 1), cache=Cache(),
            grid_for_step=lambda step: 16,
        )
        cfg = SimpleNamespace(
            result_dir=str(self.root), sh_degree_interval=1000, sh_degree=3,
            near_plane=0.01, far_plane=100.0, mask_threshold=0.25,
            mask_erode_kernel=7,
        )
        class Runner:
            device = torch.device("cpu")
            ru_training = training

            @staticmethod
            def rasterize_splats(**kwargs):
                shape = (1, kwargs["height"], kwargs["width"])
                return torch.zeros(*shape, 3), torch.ones(*shape, 1), {}

        runner = Runner()
        runner.cfg = cfg
        before = torch.get_rng_state().clone()
        result = export_source_probes(runner, step=9599)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(result["sources"]), 2)
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertFalse(manifest["test_rgb_opened"])
        for record in result["sources"]:
            payload = torch.load(record["path"], weights_only=False)
            self.assertEqual(payload["render_rgb"].shape, (1, 8, 8, 3))
            self.assertEqual(payload["hard_mask"].shape, (1, 1, 8, 8))

    def test_roi_rule_uses_strict_three_time_intersection_and_largest_component(self):
        probes = []
        for step in (9599, 9799, 9999):
            render = torch.zeros(1, 20, 20, 3)
            render[:, 1:9, 1:9] = 1
            render[:, 15:17, 15:17] = 1
            probes.append({
                "step": step, "image_name": "DSC07987.JPG",
                "render_rgb": render, "target_rgb": torch.zeros_like(render),
                "hard_mask": torch.zeros(1, 1, 20, 20),
            })
        result = derive_candidate_roi(probes)
        self.assertEqual(result["status"], "ROI_ESTABLISHED")
        self.assertEqual(result["area"], 64)
        self.assertEqual(int(result["roi"][1, 1]), 1)
        self.assertEqual(int(result["roi"][16, 16]), 0)

    def test_component_tie_uses_first_row_major_component(self):
        import numpy as np
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0:2] = True
        mask[4, 3:5] = True
        selected = largest_component_8(mask)
        self.assertTrue(selected[0, 0] and selected[0, 1])
        self.assertFalse(selected[4, 3] or selected[4, 4])

    def test_vjp_probe_never_steps_parameters_or_builds_jacobian(self):
        roi_dir = self.root / "roi"
        u_dir = self.root / "U"
        probe_dir = u_dir / "diagnostic_source_probes"
        roi_dir.mkdir()
        probe_dir.mkdir(parents=True)
        (roi_dir / "roi_evidence.json").write_text(json.dumps({
            "u_dir": str(u_dir), "status": "ROI_ESTABLISHED"
        }))
        for source in ("DSC07987.JPG", "DSC07989.JPG"):
            roi = torch.tensor([[True, False], [False, False]])
            torch.save(roi, roi_dir / f"candidate_roi_{Path(source).stem}.pt")
            torch.save({
                "render_rgb": torch.zeros(1, 2, 2, 3),
                "target_rgb": torch.ones(1, 2, 2, 3),
                "hard_mask": torch.tensor([[[[0., 1.], [1., 1.]]]]),
            }, probe_dir / f"step9999_{Path(source).stem}.pt")

        class Trainset:
            indices = [0, 1]
            def __getitem__(self, item):
                return {"image": torch.ones(2, 2, 3) * 255,
                        "camtoworld": torch.eye(4), "K": torch.eye(3)}

        splats = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.ones(2, 3)),
            "scales": torch.nn.Parameter(torch.ones(2, 3)),
            "opacities": torch.nn.Parameter(torch.ones(2)),
            "sh0": torch.nn.Parameter(torch.ones(2, 1, 3)),
            "shN": torch.nn.Parameter(torch.ones(2, 2, 3)),
        })
        original = {name: value.detach().clone() for name, value in splats.items()}
        class Runner:
            device = torch.device("cpu")
            cfg = SimpleNamespace(
                result_dir=str(self.root / "vjp"), sh_degree=3,
                near_plane=0.01, far_plane=100.0,
            )
            ru_training = SimpleNamespace(
                parser=SimpleNamespace(image_names=list(("DSC07987.JPG", "DSC07989.JPG"))),
                trainset=Trainset(),
            )
            @staticmethod
            def rasterize_splats(**kwargs):
                means2d = splats["means"][:, :2][None] * 1
                scalar = means2d.mean() + sum(
                    splats[name].mean() for name in ("scales", "opacities", "sh0", "shN")
                )
                render = scalar.expand(1, kwargs["height"], kwargs["width"], 3)
                return render, torch.ones(1, kwargs["height"], kwargs["width"], 1), {"means2d": means2d}
        runner = Runner()
        runner.splats = splats
        Path(runner.cfg.result_dir).mkdir()
        result = run_vjp_probes(runner, roi_dir, confirmed=False)
        self.assertFalse(result["explicit_jacobian_built"])
        self.assertFalse(result["optimizer_step_called"])
        for name, value in splats.items():
            self.assertTrue(torch.equal(value, original[name]))
            self.assertIsNone(value.grad)

    def test_fixed_branch_filters_monitors_and_locks_400_updates(self):
        roi_dir = self.root / "roi"
        result_dir = self.root / "branch"
        roi_dir.mkdir()
        result_dir.mkdir()
        names = ["DSC07987.JPG", "DSC07989.JPG", "m0.JPG", "m1.JPG", "m2.JPG"]
        (roi_dir / "roi_evidence.json").write_text(json.dumps({
            "status": "ROI_ESTABLISHED", "monitor_names": names[2:]
        }))
        (roi_dir / "static_confirmation.json").write_text(json.dumps({
            "status": "STATIC_ROI_CONFIRMED"
        }))
        (roi_dir / "vjp_decision.json").write_text(json.dumps({
            "status": "LOCAL_CONTROLLABILITY_PRESENT"
        }))
        for source in names[:2]:
            mask = torch.ones(8, 8, dtype=torch.bool)
            torch.save(mask, roi_dir / f"candidate_roi_{Path(source).stem}.pt")
            torch.save(mask, roi_dir / f"O_static_{Path(source).stem}.pt")
        training = SimpleNamespace(
            parser=SimpleNamespace(image_names=names),
            trainset=SimpleNamespace(indices=list(range(5))),
            cfg=SimpleNamespace(result_dir=str(result_dir)),
        )
        sequence = torch.tensor(([2, 0, 3, 1, 4] * 6000)[:HORIZON])
        branch = prepare_fixed_branch(training, sequence, roi_dir, stage="O")
        self.assertEqual(len(branch["scheduled_ids"]), 400)
        self.assertTrue(set(branch["scheduled_ids"].tolist()).isdisjoint({2, 3, 4}))
        self.assertEqual(set(branch["oracle_masks"]), {0, 1})

    def test_replay_gate_checks_200_step_curves_and_source_rerenders(self):
        parameter = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = torch.optim.Adam([parameter])
        paths = {}
        for label in ("U", "R1", "R2"):
            directory = self.root / label
            directory.mkdir()
            path = directory / "diagnostic_terminal_step10199.pt"
            save_training_snapshot(
                path, step=10199, stage=label, splats={"means": parameter},
                optimizers={"means": optimizer}, schedulers=[], strategy_state={},
                camera_sequence=torch.arange(HORIZON), intervention_mode="parent",
                ru_training_state={},
            )
            paths[label] = path
            with (directory / "loss_curve.csv").open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["step", "photo_loss"])
                writer.writerows((step, step / 100000) for step in range(10000, 10200))
            with (directory / "Gaussian_count_curve.csv").open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["step", "gaussian_count"])
                writer.writerows((step, 2) for step in range(10000, 10200))
            probe_dir = directory / "diagnostic_source_probes"
            probe_dir.mkdir()
            for source in ("DSC07987.JPG", "DSC07989.JPG"):
                torch.save({
                    "target_rgb": torch.ones(1, 2, 2, 3),
                    "render_rgb": torch.zeros(1, 2, 2, 3),
                    "hard_mask": torch.ones(1, 1, 2, 2),
                }, probe_dir / f"step10199_{Path(source).stem}.pt")
        result = compare_replay_runs(paths)
        self.assertEqual(result["status"], "REPLAY_ACCEPTED")
        self.assertEqual(result["updates_compared"], 200)


if __name__ == "__main__":
    unittest.main()
