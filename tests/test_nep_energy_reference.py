import unittest

from src.utils.nep_energy_reference import (
    normalize_lmdb_type,
    reference_for_numbers,
    make_energy_reference_metadata,
    validate_energy_reference_metadata,
    REFERENCE_TABLES,
)


class EnergyReferenceTest(unittest.TestCase):
    def test_aliases_and_tables(self):
        self.assertEqual(normalize_lmdb_type("OMAT"), "omat24")
        self.assertEqual(normalize_lmdb_type("MPtrj"), "mptrj")
        self.assertEqual(normalize_lmdb_type("MAE"), "mae")
        for values in REFERENCE_TABLES.values():
            self.assertEqual(len(values), 94)

    def test_lookup_by_atomic_number(self):
        self.assertAlmostEqual(
            reference_for_numbers("omat24", [8, 1, 8]),
            2 * REFERENCE_TABLES["omat24"][7] + REFERENCE_TABLES["omat24"][0],
        )
        with self.assertRaisesRegex(ValueError, "95"):
            reference_for_numbers("omat24", [95])

    def test_metadata_rejects_changed_table(self):
        metadata = make_energy_reference_metadata("omat24")
        self.assertEqual(validate_energy_reference_metadata(metadata), "omat24")
        metadata["tables"]["omat24"][7] += 1
        with self.assertRaisesRegex(ValueError, "omat24"):
            validate_energy_reference_metadata(metadata)


class EnergyReferenceIntegrationTest(unittest.TestCase):
    def test_lmdb_configuration_requires_type_and_serializes_it(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from src.test.test_nep_lmdb_config import _minimal_nep_json
        from src.user.input_param import InputParam
        with TemporaryDirectory() as directory:
            path = Path(directory) / "one.aselmdb"
            path.touch()
            base = _minimal_nep_json(path)
            base["model"]["fitting_net"]["network_size"] = [30, 1]
            base.pop("lmdb_type")
            with self.assertRaisesRegex(ValueError, "lmdb_type"):
                InputParam(base, "TRAIN")
            base["lmdb_type"] = "OMAT"
            param = InputParam(base, "TRAIN")
            self.assertEqual(param.lmdb_type, "omat24")
            self.assertEqual(param.to_dict()["lmdb_type"], "omat24")
            self.assertEqual(param.file_paths.model_name, "nep_model_omat24.ckpt")
            self.assertFalse(param.optimizer_param.reset_epoch)
            explicit_reset = dict(base)
            explicit_reset["optimizer"] = dict(base["optimizer"], reset_epoch=True)
            self.assertTrue(InputParam(explicit_reset, "TRAIN").optimizer_param.reset_epoch)
            base["model_name"] = "custom_omat24.ckpt"
            self.assertEqual(
                InputParam(base, "TRAIN").file_paths.model_name,
                "custom_omat24.ckpt",
            )

    def test_dataset_residual_preserves_other_labels(self):
        from src.test.test_nep_lmdb_dataset import _frame
        from src.pre_data.nep_lmdb_dataset import NepLmdbDataset
        dataset = NepLmdbDataset([], atom_types=[1, 8], cutoff_radial=6.0,
                                 energy_reference_type="omat24")
        sample = dataset._convert_frame(_frame([8, 1, 8], energy=-20.0,
                                              stress=[1, 2, 3, 4, 5, 6]))
        reference = reference_for_numbers("omat24", [8, 1, 8])
        self.assertAlmostEqual(sample["energy"].item() + reference, -20.0)
        self.assertAlmostEqual(sample["energy_raw"].item(), -20.0)
        self.assertEqual(sample["force"].shape, (3, 3))
        self.assertEqual(sample["virial"].shape, (9,))

    def test_dataset_rejects_elements_missing_from_reference_table(self):
        from src.pre_data.nep_lmdb_dataset import NepLmdbDataset
        with self.assertRaisesRegex(ValueError, "95"):
            NepLmdbDataset([], atom_types=[1, 95],
                           energy_reference_type="omat24")

    def test_zero_initial_bias_is_trainable(self):
        from src.model.nep_fitting import FittingNet
        net = FittingNet([30, 1], True, False, "tanh", 7, 0.0,
                         zero_last_bias=True)
        bias = net.layers[-1].bias
        self.assertTrue(bias.requires_grad)
        self.assertEqual(float(bias.detach().abs().max()), 0.0)


class EnergyReferenceCheckpointTest(unittest.TestCase):
    def test_checkpoint_includes_reference_metadata(self):
        from src.utils.nep_energy_reference import checkpoint_reference_for_config
        data = checkpoint_reference_for_config({"lmdb_type": "omat24"})
        self.assertEqual(data["active_type"], "omat24")
        self.assertEqual(len(data["tables"]["mae"]), 94)

    def test_text_export_adjusts_copy_of_bias_once(self):
        import torch
        from src.utils.nep_to_gpumd import energy_reference_export_state
        original = {
            "fitting_net.0.layers.1.bias": torch.tensor([[0.25]], dtype=torch.float64),
            "fitting_net.1.layers.1.bias": torch.tensor([[-0.5]], dtype=torch.float64),
        }
        metadata = make_energy_reference_metadata("omat24")
        adjusted = energy_reference_export_state(original, [8, 1], metadata,
                                                 selected_type="mptrj")
        self.assertAlmostEqual(
            adjusted["fitting_net.0.layers.1.bias"].item(),
            0.25 + REFERENCE_TABLES["mptrj"][7],
        )
        self.assertAlmostEqual(
            adjusted["fitting_net.1.layers.1.bias"].item(),
            -0.5 + REFERENCE_TABLES["mptrj"][0],
        )
        self.assertEqual(original["fitting_net.0.layers.1.bias"].item(), 0.25)


class EnergyReferenceResumeTest(unittest.TestCase):
    def test_same_type_resume_and_cross_type_start(self):
        from src.utils.nep_energy_reference import prepare_energy_reference_checkpoint
        checkpoint = {"energy_reference": make_energy_reference_metadata("omat24"),
                      "state_dict": {}}
        self.assertFalse(prepare_energy_reference_checkpoint(
            checkpoint, "omat24", [8])[0])
        self.assertTrue(prepare_energy_reference_checkpoint(
            checkpoint, "mae", [8])[0])
        checkpoint["energy_reference"]["tables"]["mae"][7] += 1
        with self.assertRaisesRegex(ValueError, "mae"):
            prepare_energy_reference_checkpoint(checkpoint, "mae", [8])

    def test_legacy_requires_explicit_migration_and_preserves_energy(self):
        import torch
        from src.utils.nep_energy_reference import prepare_energy_reference_checkpoint
        key = "fitting_net.0.layers.1.bias"
        checkpoint = {"state_dict": {key: torch.tensor([[1.2]], dtype=torch.float64)}}
        with self.assertRaisesRegex(ValueError, "migrate"):
            prepare_energy_reference_checkpoint(checkpoint, "omat24", [8])
        new_stage, source = prepare_energy_reference_checkpoint(
            checkpoint, "omat24", [8], migrate_legacy=True)
        self.assertTrue(new_stage)
        self.assertEqual(source, "legacy")
        self.assertAlmostEqual(checkpoint["state_dict"][key].item() +
                               REFERENCE_TABLES["omat24"][7], 1.2)


class EnergyReferenceTotxtTest(unittest.TestCase):
    def test_type_selector_and_output_name(self):
        from src.utils.nep_to_gpumd import parse_nep_totxt_arguments
        self.assertEqual(
            parse_nep_totxt_arguments(["model.ckpt", "-OMAT"]),
            ("model.ckpt", None, "omat24"),
        )
        self.assertEqual(
            parse_nep_totxt_arguments(["model.ckpt", "custom.txt", "-MAE"]),
            ("model.ckpt", "custom.txt", "mae"),
        )
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_nep_totxt_arguments(["model.ckpt", "-other"])


class EnergyReferenceInferenceTest(unittest.TestCase):
    def test_lmdb_test_images_keep_physical_total_energy(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from src.test.test_nep_lmdb_dataset import _frame, _write_aselmdb
        from src.pre_data.nep_data_loader import NepTestData
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.aselmdb"
            _write_aselmdb(path, {1: _frame([8, 1], energy=-10.0)})
            param = SimpleNamespace(
                file_paths=SimpleNamespace(test_data_path=[str(path)], format="lmdb"),
                atom_type=[1, 8],
                nep_param=SimpleNamespace(cutoff=[6.0, 4.0]),
                nonperiodic_vacuum_padding=None,
            )
            images = NepTestData(param).image_list
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].Ep, -10.0)
            self.assertEqual(images[0].atom_types_image.tolist(), [8, 1])


class EnergyReferenceConfigurationSafetyTest(unittest.TestCase):
    def test_residual_training_rejects_unlabelled_nep_text_initialization(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from src.test.test_nep_lmdb_config import _minimal_nep_json
        from src.user.input_param import InputParam
        with TemporaryDirectory() as directory:
            path = Path(directory) / "one.aselmdb"
            path.touch()
            config = _minimal_nep_json(path, nep_txt_file="old_total_energy.txt")
            with self.assertRaisesRegex(ValueError, "nep_txt_file"):
                InputParam(config, "TRAIN")
