import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import auto_test
from cases import CASES, collect_cases, select_cases, split_frames, training_config
from results import compare_metrics, compare_predictions, read_epochs, read_per_frame_scalars, write_json
from runtime import file_digest


class FrameworkTests(unittest.TestCase):
    def test_site_settings_default_and_explicit_cli_overrides(self):
        parser = auto_test.make_parser({
            "partition": "gpuA,gpuB", "cpu_partition": "cpuA", "jobs": "3",
            "cuda_module": "", "conda_env": "matpl-site",
        })
        common = ["--site-config", "site.sh", "--work-dir", "external"]
        for command in (["run"], ["base", "build"]):
            with self.subTest(command=command):
                args = parser.parse_args([*command, *common, "--partition", "3090,q4"])
                self.assertEqual(args.partition, "3090,q4")
                self.assertEqual(args.cpu_partition, "cpuA")
                self.assertEqual(args.jobs, 3)
                self.assertEqual(args.cuda_module, "")
                self.assertEqual(args.conda_env, "matpl-site")

    def run_fake_slurm(self, work, outputs):
        received = iter(outputs)

        def submit(command, *, cwd, stdout, stderr):
            code, message, result = next(received)
            stdout.write(message)
            if result is not None:
                write_json(cwd / "result.json", result)
            return SimpleNamespace(returncode=code)

        args = auto_test.make_parser().parse_args(["run", "--work-dir", str(work.parent)])
        case = next(case for case in auto_test.MATRIX if case.key == "test/BaTiO/cpu4-pppm")
        with patch.object(auto_test.subprocess, "run", side_effect=submit) as runner, \
                patch("time.sleep") as delay:
            result = auto_test.run_job(args, case, work, {})
        return result, runner.call_count, delay.call_count

    def test_slurm_balance_lookup_error_resubmits_without_losing_log(self):
        failure = ("sbatch: error: Failed to query the balance. Please contact the administrator\n"
                   "sbatch: error: Batch job submission failed: Unspecified error\n")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "case"
            result, attempts, delays = self.run_fake_slurm(work, [
                (1, failure, None), (0, "4833534\n", {"status": "PASS"}),
            ])
            self.assertEqual((result["status"], attempts, delays), ("PASS", 2, 1))
            log = (work / "submit.log").read_text(encoding="utf-8")
            self.assertIn("Failed to query the balance", log)
            self.assertIn("4833534", log)

    def test_slurm_reports_permanent_errors_without_resubmitting(self):
        failure = ("sbatch: error: Invalid account or account/partition combination specified\n"
                   "sbatch: error: Batch job submission failed: Invalid account\n")
        with tempfile.TemporaryDirectory() as tmp:
            result, attempts, delays = self.run_fake_slurm(Path(tmp) / "case", [(1, failure, None)])
            self.assertEqual((result["status"], attempts, delays), ("FAIL", 1, 0))
            self.assertIn("Slurm submission failed", result["message"])
            self.assertIn("Invalid account", result["message"])

    def test_slurm_balance_retry_stops_after_four_attempts(self):
        failure = ("sbatch: error: Failed to query the balance. Please contact the administrator\n"
                   "sbatch: error: Batch job submission failed: Unspecified error\n")
        with tempfile.TemporaryDirectory() as tmp:
            result, attempts, delays = self.run_fake_slurm(
                Path(tmp) / "case", [(1, failure, None)] * 4,
            )
            self.assertEqual((result["status"], attempts, delays), ("FAIL", 4, 3))
            self.assertIn("Failed to query the balance", result["message"])

    def test_slurm_never_resubmits_after_receiving_job_id(self):
        output = ("4833534;cluster\n"
                  "sbatch: error: Failed to query the balance. Please contact the administrator\n"
                  "sbatch: error: Batch job submission failed: Unspecified error\n")
        with tempfile.TemporaryDirectory() as tmp:
            result, attempts, delays = self.run_fake_slurm(Path(tmp) / "case", [(1, output, None)])
            self.assertEqual((result["status"], attempts, delays), ("FAIL", 1, 0))
            self.assertIn("slurm-4833534.log", result["message"])

    def test_matrix_covers_six_systems_and_eight_qnep_inference_modes(self):
        matrix = collect_cases()
        regression = select_cases(matrix, "full", ["regression"], [], [])
        self.assertEqual(len(regression), 12)
        self.assertEqual({case.system for case in regression}, set(CASES))
        inference = select_cases(matrix, "full", ["test"], [], [])
        self.assertEqual(
            {case.variant for case in inference if case.options["kspace"] in ("ewald", "pppm")},
            {f"{device}-{method}" for device in ("cpu1", "cpu4", "gpu1", "gpu2")
             for method in ("ewald", "pppm")},
        )

    def test_batio_training_preserves_qnep_and_normalizes_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "input.xyz"
            data.touch()
            (root / "nep.json").write_text(json.dumps({
                "model_type": "NEP_chargemode2", "train_data": "input.xyz",
                "model": {"descriptor": {"charge_mode": True}},
                "optimizer": {"_t_0": 6, "_t_mult": 1},
            }))
            config = training_config(root, "BaTiO", {"batch_size": 32})
            self.assertEqual(config["model_type"], "NEP_CHARGE2")
            self.assertEqual([Path(path) for path in config["train_data"]], [data.resolve()])
            self.assertNotIn("charge_mode", config["model"]["descriptor"])
            self.assertNotIn("_t_0", config["optimizer"])
            self.assertFalse(config["recover_train"])

    def test_splits_are_disjoint_and_keep_stress_and_bec(self):
        frames = [(i, f"frame {i}", "bec" if i % 2 else "stress") for i in range(100)]
        train, valid = split_frames(frames, 8, 4)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(valid), 4)
        self.assertFalse({row[0] for row in train} & {row[0] for row in valid})
        self.assertEqual({row[2] for row in valid}, {"bec", "stress"})

    def test_epoch_parser_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "epoch_train.dat"
            file.write_text("# epoch loss RMSE_F(eV/Å) last_lr_used time(s)\n"
                            "1 1 0.5 0.001 2\n", encoding="utf-8")
            metrics = read_epochs(file)
            self.assertEqual(metrics, {1: {"loss": 1.0, "RMSE_F(eV/Å)": 0.5,
                                            "last_lr_used": 0.001}})
            self.assertTrue(compare_metrics(metrics, {1: {**metrics[1], "loss": 2.0}},
                                            1e-9, 1e-9))
            file.write_text("# epoch loss time(s)\n1 NaN 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_epochs(file)

    def test_prediction_diff_identifies_frame_atom_and_component(self):
        reference = {"frame_ids": [3, 9], "natoms": [1, 2],
                     "force": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        actual = {**reference, "force": [[1, 2, 3], [4, 5, 6], [7, 8, 10]]}
        messages = compare_predictions(actual, reference, 1e-9, 1e-9)
        self.assertRegex(" ".join(messages), r"frame=9.*atom=1.*component=2")

    def test_ragged_scalar_predictions_preserve_frame_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "charge.txt"
            file.write_text("1 2 3\n4\n", encoding="utf-8")
            self.assertEqual(read_per_frame_scalars(file, [3, 1]), [[1], [2], [3], [4]])
            with self.assertRaisesRegex(ValueError, "per-frame atom counts"):
                read_per_frame_scalars(file, [2, 2])

    def test_file_digest_on_login_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "input.dat"
            file.write_bytes(b"abc")
            self.assertEqual(file_digest(file),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_training_comparison_rejects_validation_and_step_changes(self):
        base = {"epochs": {"1": {"loss": 1}}, "validation_epochs": {"1": {"loss": 2}},
                "optimizer_updates": 7}
        current = {**base, "validation_epochs": {"1": {"loss": 3}}, "optimizer_updates": 8}
        differences = auto_test.compare_values(current, base, 1e-9, 0)
        self.assertTrue(any(text.startswith("validation epoch=1") for text in differences))
        self.assertTrue(any(text.startswith("optimizer updates differ") for text in differences))

    def test_empty_work_dir_uses_bundled_baseline_and_keeps_local_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            bundled = root / "bundled.json"
            frozen = {"regression/AuAg/batch1": {"values": {"loss": 1}}}
            write_json(bundled, {"id": "bundled", "entries": frozen})
            work = root / "new-work"
            with patch.object(auto_test, "BUNDLED_MANIFEST", bundled):
                self.assertEqual(auto_test.current_base_id(work), "bundled")
                self.assertEqual(auto_test.base_index(work), frozen)
                self.assertFalse(work.exists())
                local = work / "baselines" / "local"
                local.mkdir(parents=True)
                write_json(local / "manifest.json", {"id": "local", "entries": {}})
                write_json(local.parent / "current.json", {"id": "local"})
                self.assertEqual(auto_test.current_base_id(work), "local")
                self.assertEqual(auto_test.base_index(work), {})
                self.assertEqual(auto_test.base_index(work, "bundled"), frozen)

    def test_different_device_still_compares_to_reference_at_same_tolerance(self):
        reference_runtime = {"backend": "cuda", "torch": "2.2.0", "device_names": ["3090"]}
        inputs = {"nep.json": "abc"}
        base = {"inputs": inputs, "runtime": reference_runtime,
                "values": {"epochs": {"1": {"loss": 1.0}}, "optimizer_updates": 1}}
        result = {**base, "runtime": {**reference_runtime, "device_names": ["4090"]}}
        matched = {"status": "PASS", "inputs": inputs}
        auto_test.assess_difference(matched, result, base, 1e-9, 1e-10)
        self.assertEqual(matched["status"], "PASS")
        self.assertEqual(matched["environment_differences"]["device_names"],
                         {"baseline": ["3090"], "actual": ["4090"]})
        mismatch = {"status": "PASS", "inputs": inputs}
        result["values"] = {**result["values"], "epochs": {"1": {"loss": 2.0}}}
        auto_test.assess_difference(mismatch, result, base, 1e-9, 1e-10)
        self.assertEqual(mismatch["status"], "FAIL")
        self.assertTrue(mismatch["differences"])

    def test_commands_that_write_artifacts_require_an_external_work_dir(self):
        parser = auto_test.make_parser()
        for command in (["run"], ["compare", "RUN"], ["base", "build"], ["base", "promote", "BASE"]):
            error = io.StringIO()
            with self.subTest(command=command), contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as missing:
                    parser.parse_args(command)
                self.assertEqual(missing.exception.code, 2)
                self.assertRegex(error.getvalue(), r"--work-dir\s+PATH")
                self.assertIn("required: --work-dir/--work_dir", error.getvalue())
                for flag in ("--work-dir", "--work_dir"):
                    parsed = parser.parse_args([*command, flag, "external"])
                    self.assertEqual(parsed.work_dir, Path("external"))
        self.assertEqual(parser.parse_args(["list"]).command, "list")

    def test_work_dir_guard_rejects_repository_paths_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "checkout"
            repo.mkdir()
            (repo / ".git").touch()  # Git worktrees keep a .git file.
            with self.assertRaisesRegex(ValueError, "outside.*Git"):
                auto_test.resolve_work_dir(repo / "nested" / "not-created")
            work = auto_test.resolve_work_dir(root / "external")
            self.assertFalse(work.exists())
            self.assertEqual(auto_test.workspace_path(work, "runs", "ID"), work / "runs" / "ID")
            with self.assertRaisesRegex(ValueError, "outside the work directory"):
                auto_test.workspace_path(work, "..", "checkout")

    def test_run_and_base_build_default_to_data_beside_the_framework(self):
        parser = auto_test.make_parser()
        for command in (["run"], ["base", "build"]):
            with self.subTest(command=command):
                args = parser.parse_args([*command, "--work-dir", "external"])
                self.assertEqual(args.data_root, Path(auto_test.__file__).resolve().parent / "quick_test")
                override = parser.parse_args([*command, "--work-dir", "external", "--data-root", "other"])
                self.assertEqual(override.data_root, Path("other"))

    def test_base_promotion_merges_disjoint_entries_and_rejects_stale_overlap(self):
        def candidate(root, name, key, loss):
            work = root / "candidates" / name / key
            work.mkdir(parents=True)
            write_json(work.parents[2] / "summary.json", {
                "version": auto_test.VERSIONS, "code": {"commit": "reference"},
                "base_parent": None, "passed": True, "cases": {key: {"inputs": {}}},
            })
            write_json(work / "result.json", {
                "status": "PASS", "runtime": {},
                "values": {"epochs": {"1": {"loss": loss}}, "optimizer_updates": 1},
            })

        with tempfile.TemporaryDirectory() as tmp, patch.object(auto_test, "BUNDLED_MANIFEST",
                                                                  Path("absent-reference.json")):
            root = Path(tmp).resolve()
            first, second = "regression/BaTiO/batch1", "regression/BaTiO/batch32"
            candidate(root, "initial", first, 1)
            candidate(root, "disjoint", second, 2)
            candidate(root, "stale", first, 3)
            auto_test.publish_base(root, "initial")
            auto_test.publish_base(root, "disjoint")
            self.assertEqual(set(auto_test.base_index(root)), {first, second})
            with self.assertRaisesRegex(ValueError, "base changed"):
                auto_test.publish_base(root, "stale")
            self.assertEqual(auto_test.current_base_id(root), "disjoint")


if __name__ == "__main__":
    unittest.main()
