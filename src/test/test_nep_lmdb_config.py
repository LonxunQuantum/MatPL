import ctypes.util
import tempfile
import unittest
from pathlib import Path

import torch

from src.pre_data.nep_lmdb_dataset import discover_aselmdb_files
from src.user.input_param import InputParam
from src.user.work_file_param import WorkFileStructure
from src.utils.op_loader import get_library_path


def _minimal_nep_json(lmdb_path, **overrides):
    config = {
        "model_type": "NEP",
        "atom_type": [1],
        "model": {
            "descriptor": {
                "cutoff": [6.0, 4.0],
                "n_max": [4, 4],
                "basis_size": [8, 8],
                "l_max": [4, 2, 1],
            },
            "fitting_net": {"network_size": [8, 1]},
        },
        "optimizer": {"optimizer": "ADAM", "epochs": 1, "batch_size": 1},
        "format": "lmdb",
        "train_data": [str(lmdb_path)],
    }
    config.update(overrides)
    return config


class DiscoverAseLmdbFilesTest(unittest.TestCase):
    def test_recursively_discovers_sorts_and_deduplicates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "z" / "first.aselmdb"
            second = root / "a" / "nested" / "second.aselmdb"
            ignored = root / "a" / "not-lmdb.txt"
            for path in (first, second, ignored):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            result = discover_aselmdb_files([root, first, root])

            self.assertEqual(result, sorted({str(first.resolve()), str(second.resolve())}))

    def test_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "No .aselmdb files"):
                discover_aselmdb_files([tmpdir])

    def test_rejects_non_lmdb_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            path.touch()
            with self.assertRaisesRegex(ValueError, "Expected an .aselmdb"):
                discover_aselmdb_files([path])

    def test_rejects_missing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                discover_aselmdb_files([missing])


class LmdbWorkFileStructureTest(unittest.TestCase):
    def test_lmdb_paths_are_expanded_for_all_dataset_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train" / "a.aselmdb"
            valid = root / "valid" / "b.aselmdb"
            test = root / "test" / "c.aselmdb"
            for path in (train, valid, test):
                path.parent.mkdir(parents=True)
                path.touch()
            paths = WorkFileStructure(
                json_dir=tmpdir,
                reserve_work_dir=False,
                reserve_feature=False,
                model_type="NEP",
            )

            paths.set_train_valid_file(
                {
                    "model_type": "NEP",
                    "format": "lmdb",
                    "train_data": [train.parent],
                    "valid_data": valid,
                    "test_data": [test],
                }
            )

            self.assertEqual(paths.train_data_path, [str(train.resolve())])
            self.assertEqual(paths.valid_data_path, [str(valid.resolve())])
            self.assertEqual(paths.test_data_path, [str(test.resolve())])


class LmdbStatisticsConfigurationTest(unittest.TestCase):
    def test_default_and_explicit_statistics_frame_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lmdb_path = Path(tmpdir) / "tiny.aselmdb"
            lmdb_path.touch()

            default_param = InputParam(_minimal_nep_json(lmdb_path), "TRAIN")
            explicit_param = InputParam(
                _minimal_nep_json(lmdb_path, lmdb_stat_frames=8192), "TRAIN"
            )

            self.assertEqual(default_param.lmdb_stat_frames, 32768)
            self.assertEqual(explicit_param.lmdb_stat_frames, 8192)
            self.assertEqual(explicit_param.to_dict()["lmdb_stat_frames"], 8192)

    def test_statistics_frame_count_must_be_a_positive_integer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lmdb_path = Path(tmpdir) / "tiny.aselmdb"
            lmdb_path.touch()
            for invalid in (0, -1, True, 1.5, "4096"):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "lmdb_stat_frames"):
                        InputParam(
                            _minimal_nep_json(
                                lmdb_path, lmdb_stat_frames=invalid
                            ),
                            "TRAIN",
                        )

    def test_non_lmdb_format_ignores_lmdb_statistics_option(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "unused.data"
            data_path.touch()

            param = InputParam(
                _minimal_nep_json(
                    data_path,
                    format="extxyz",
                    lmdb_stat_frames=0,
                ),
                "TRAIN",
            )

            self.assertEqual(param.lmdb_stat_frames, 0)
            self.assertNotIn("lmdb_stat_frames", param.to_dict())


class NepElementConfigurationTest(unittest.TestCase):
    def _param(self, atom_types, **overrides):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        lmdb_path = Path(temporary_directory.name) / "tiny.aselmdb"
        lmdb_path.touch()
        return InputParam(
            _minimal_nep_json(
                lmdb_path,
                atom_type=atom_types,
                **overrides,
            ),
            "TRAIN",
        )

    def test_accepts_all_118_periodic_table_elements(self):
        param = self._param(list(range(1, 119)))

        self.assertEqual(param.atom_type, list(range(1, 119)))

    def test_rejects_atomic_numbers_outside_periodic_table(self):
        for atom_types in ([0], [119]):
            with self.subTest(atom_types=atom_types):
                with self.assertRaisesRegex(ValueError, "1.*118"):
                    self._param(atom_types)

    def test_rejects_duplicate_nep_elements(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self._param([1, 1])

    def test_batch_max_types_is_no_longer_part_of_input_state(self):
        param = self._param([1, 8], batch_max_types=1)

        self.assertFalse(hasattr(param, "max_allow_atom_type"))
        self.assertNotIn("batch_max_types", param.to_dict())


class NepElementCapacitySourceTest(unittest.TestCase):
    def test_shared_element_limit_is_118(self):
        src_root = Path(__file__).resolve().parents[1]
        header = src_root / "op" / "include" / "nep_limits.h"

        self.assertTrue(header.is_file(), f"missing shared NEP limit header: {header}")
        self.assertIn("NEP_MAX_ELEMENT_TYPES = 118", header.read_text())

    def test_legacy_100_type_buffers_are_removed(self):
        src_root = Path(__file__).resolve().parents[1]
        source_paths = (
            src_root / "op" / "kernel" / "calculateNepNeighbor.cu",
            src_root / "op" / "kernel" / "calculateNepMbFeat_secondgradout.cu",
            src_root / "op" / "kernel" / "utilities" / "nep_mbgrad.cuh",
            src_root / "op" / "src" / "cpu_calculate_nepneighbor.cpp",
        )
        legacy_patterns = ("[100]", "resize(100", "unique_types(100")
        for source_path in source_paths:
            source = source_path.read_text()
            for pattern in legacy_patterns:
                self.assertNotIn(pattern, source, f"{pattern} remains in {source_path}")


_CUDA_DRIVER_AVAILABLE = (
    torch.cuda.is_available() and ctypes.util.find_library("cuda") is not None
)


@unittest.skipUnless(_CUDA_DRIVER_AVAILABLE, "PyTorch CUDA libraries need a GPU driver")
class CalcOpsElementCapacityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.ops.load_library(str(get_library_path("cuda")))
        cls.calc_ops = torch.ops.CalcOps_cpu
        cls.cuda_calc_ops = torch.ops.CalcOps_cuda

    @staticmethod
    def _inputs(device="cpu"):
        num_atoms = torch.tensor([1], dtype=torch.int64, device=device)
        box = torch.tensor(
            [[1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0,
              1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0]],
            dtype=torch.float64,
            device=device,
        )
        box_original = torch.eye(3, dtype=torch.float64, device=device).reshape(1, 9)
        num_cell = torch.ones((1, 3), dtype=torch.int64, device=device)
        position = torch.zeros((1, 3), dtype=torch.float64, device=device)
        return num_atoms, box, box_original, num_cell, position

    def test_calculate_maxneigh_rejects_more_than_118_types(self):
        inputs = self._inputs()
        atom_type_map = torch.zeros(1, dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, "118"):
            self.calc_ops.calculate_maxneigh(
                *inputs,
                0.25,
                0.25,
                119,
                atom_type_map,
                False,
            )

    def test_calculate_maxneigh_accepts_type_index_117(self):
        inputs = self._inputs()
        atom_type_map = torch.tensor([117], dtype=torch.int64)

        radial, angular = self.calc_ops.calculate_maxneigh(
            *inputs,
            0.25,
            0.25,
            118,
            atom_type_map,
            True,
        )

        self.assertEqual(tuple(radial.shape), (1, 118))
        self.assertEqual(tuple(angular.shape), (1, 118))
        self.assertEqual(torch.count_nonzero(radial).item(), 0)
        self.assertEqual(torch.count_nonzero(angular).item(), 0)

    def test_cuda_calculate_maxneigh_rejects_more_than_118_types(self):
        inputs = self._inputs("cuda")
        atom_type_map = torch.zeros(1, dtype=torch.int64, device="cuda")

        with self.assertRaisesRegex(RuntimeError, "118"):
            self.cuda_calc_ops.calculate_maxneigh(
                *inputs,
                0.25,
                0.25,
                119,
                atom_type_map,
                False,
            )

    def test_cuda_calculate_maxneigh_accepts_type_index_117(self):
        inputs = self._inputs("cuda")
        atom_type_map = torch.tensor([117], dtype=torch.int64, device="cuda")

        radial, angular = self.cuda_calc_ops.calculate_maxneigh(
            *inputs,
            0.25,
            0.25,
            118,
            atom_type_map,
            True,
        )

        self.assertEqual(tuple(radial.shape), (1, 118))
        self.assertEqual(tuple(angular.shape), (1, 118))
        self.assertEqual(torch.count_nonzero(radial).item(), 0)
        self.assertEqual(torch.count_nonzero(angular).item(), 0)


if __name__ == "__main__":
    unittest.main()
