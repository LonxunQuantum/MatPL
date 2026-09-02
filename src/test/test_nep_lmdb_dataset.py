import json
import os
import tempfile
import unittest
import zlib
from unittest import mock
from pathlib import Path

import lmdb
import numpy as np
import torch

from src.pre_data.nep_lmdb_dataset import AseLmdbShard, NepLmdbDataset


def _compressed_json(value):
    return zlib.compress(json.dumps(value).encode("utf-8"))


def _frame(numbers, *, energy=-1.0, stress=None, atomic_energy=None):
    natoms = len(numbers)
    frame = {
        "numbers": list(numbers),
        "positions": [[float(i), 0.25, 0.5] for i in range(natoms)],
        "pbc": [True, True, True],
        "cell": [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        "energy": energy,
        "forces": [[0.1, 0.2, 0.3] for _ in range(natoms)],
    }
    if stress is not None:
        frame["stress"] = stress
    if atomic_energy is not None:
        frame["atomic_energy"] = atomic_energy
    return frame


def _write_aselmdb(path, rows, *, nextid=None, deleted_ids=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), subdir=False, map_size=8 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            for row_id, frame in rows.items():
                value = frame if isinstance(frame, bytes) else _compressed_json(frame)
                txn.put(str(row_id).encode("ascii"), value)
            txn.put(
                b"nextid",
                _compressed_json(nextid if nextid is not None else max(rows) + 1),
            )
            if deleted_ids is not None:
                txn.put(b"deleted_ids", _compressed_json(deleted_ids))
    finally:
        env.close()


class AseLmdbShardTest(unittest.TestCase):
    def test_metadata_maps_deleted_rows_without_retaining_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deleted.aselmdb"
            _write_aselmdb(
                path,
                {1: _frame([1]), 3: _frame([8, 1])},
                nextid=4,
                deleted_ids=[2],
            )

            shard = AseLmdbShard(str(path))

            self.assertEqual(len(shard), 2)
            self.assertEqual(shard.row_id(0), 1)
            self.assertEqual(shard.row_id(1), 3)
            self.assertFalse(any(isinstance(value, lmdb.Environment) for value in vars(shard).values()))
            with self.assertRaises(IndexError):
                shard.row_id(-1)
            with self.assertRaises(IndexError):
                shard.row_id(2)


class NepLmdbDatasetTest(unittest.TestCase):
    def _dataset(self, paths, **overrides):
        options = {
            "data_paths": [str(path) for path in paths],
            "atom_types": [1, 8],
            "cutoff_radial": 6.0,
            "cutoff_angular": 4.0,
            "dtype": torch.float32,
            "index_type": torch.int64,
        }
        options.update(overrides)
        return NepLmdbDataset(**options)

    def test_length_indexing_and_worker_safe_laziness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.aselmdb"
            second = Path(tmpdir) / "second.aselmdb"
            _write_aselmdb(
                first,
                {1: _frame([1]), 3: _frame([8, 1])},
                nextid=4,
                deleted_ids=[2],
            )
            _write_aselmdb(second, {1: _frame([8])})
            dataset = self._dataset([first, second])

            self.assertEqual(len(dataset), 3)
            self.assertFalse(hasattr(dataset, "image_list"))
            self.assertEqual(len(dataset._env_cache), 0)
            self.assertEqual(dataset[1]["num_atom"].item(), 2)
            self.assertEqual(len(dataset._env_cache), 1)
            self.assertEqual(dataset[-1]["atom_type_image"].tolist(), [8])
            with self.assertRaises(IndexError):
                dataset[-4]
            with self.assertRaises(IndexError):
                dataset[3]

            state = dataset.__getstate__()
            self.assertEqual(len(state["_env_cache"]), 0)
            dataset.close()
            self.assertEqual(len(dataset._env_cache), 0)

    def test_decodes_expected_training_tensors_and_stress_convention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tensors.aselmdb"
            _write_aselmdb(
                path,
                {
                    1: _frame(
                        [1, 8],
                        energy=-3.5,
                        stress=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        atomic_energy=[-1.0, -2.5],
                    )
                },
            )
            dataset = self._dataset([path], train_ei=True, batch_max_types=5)

            sample = dataset[0]

            self.assertEqual(sample["position"].shape, (2, 3))
            self.assertEqual(sample["force"].shape, (2, 3))
            self.assertEqual(sample["atom_type_map"].tolist(), [0, 1])
            self.assertEqual(sample["atom_type_image"].tolist(), [1, 8])
            self.assertEqual(sample["num_atom"].tolist(), [2])
            self.assertEqual(sample["box"].shape, (18,))
            self.assertEqual(sample["box_original"].shape, (9,))
            self.assertEqual(sample["num_cell"].shape, (3,))
            self.assertEqual(sample["volume"].tolist(), [24.0])
            self.assertEqual(sample["energy"].tolist(), [-3.5])
            self.assertEqual(sample["ei"].tolist(), [-1.0, -2.5])
            self.assertEqual(sample["bec"].shape, (2, 9))
            self.assertEqual(sample["fragment"].tolist(), [-1, -1])
            self.assertTrue(torch.isnan(sample["fragment_charge"]).all())
            self.assertEqual(sample["charge"].tolist(), [0.0])
            self.assertEqual(sample["max_allow_atom_type"].tolist(), [5])
            self.assertEqual(sample["position"].dtype, torch.float32)
            self.assertEqual(sample["atom_type_map"].dtype, torch.int64)
            expected_virial = torch.tensor(
                [-24.0, -144.0, -120.0, -144.0, -48.0, -96.0, -120.0, -96.0, -72.0]
            )
            torch.testing.assert_close(sample["virial"], expected_virial)

    def test_samples_follow_existing_variable_length_collate_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "collate.aselmdb"
            _write_aselmdb(path, {1: _frame([1, 8]), 2: _frame([1])})
            dataset = self._dataset([path])
            with mock.patch("src.utils.op_loader.load_calc_ops", return_value=None):
                from src.pre_data.nep_data_loader import variable_length_collate_fn

            batch = variable_length_collate_fn([dataset[0], dataset[1]])

            self.assertEqual(batch["position"].shape, (3, 3))
            self.assertEqual(batch["force"].shape, (3, 3))
            self.assertEqual(batch["atom_type_map"].tolist(), [0, 1, 0])
            self.assertEqual(batch["num_atom"].flatten().tolist(), [2, 1])
            self.assertEqual(batch["num_atom_sum"].flatten().tolist(), [2, 3])

    def test_missing_stress_and_atomic_energy_follow_mask_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing_optional.aselmdb"
            _write_aselmdb(path, {1: _frame([1, 8])})

            sample = self._dataset([path], train_ei=False)[0]
            self.assertTrue(torch.isfinite(sample["ei"]).all())
            self.assertEqual(sample["ei"].tolist(), [0.0, 0.0])
            self.assertTrue(torch.all(sample["virial"] == -1e6))

            with self.assertRaisesRegex(ValueError, "atomic energies"):
                self._dataset([path], train_ei=True)[0]

    def test_environment_cache_is_lru_bounded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index in range(10):
                path = Path(tmpdir) / ("shard-{}.aselmdb".format(index))
                _write_aselmdb(path, {1: _frame([1])})
                paths.append(path)
            dataset = self._dataset(paths, max_open_shards=3)

            for index in range(len(dataset)):
                dataset[index]

            self.assertLessEqual(len(dataset._env_cache), 3)
            self.assertEqual(
                list(dataset._env_cache),
                [str(path.resolve()) for path in paths[-3:]],
            )
            dataset.close()

    def test_inherited_environment_cache_is_reopened_for_new_worker_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "worker.aselmdb"
            _write_aselmdb(path, {1: _frame([1])})
            dataset = self._dataset([path])
            dataset[0]
            parent_environment = next(iter(dataset._env_cache.values()))

            dataset._env_pid = -1
            dataset[0]

            self.assertEqual(dataset._env_pid, os.getpid())
            self.assertIsNot(next(iter(dataset._env_cache.values())), parent_environment)

    def test_unknown_elements_and_nonperiodic_frames_are_rejected_with_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unknown = Path(tmpdir) / "unknown.aselmdb"
            nonperiodic = Path(tmpdir) / "nonperiodic.aselmdb"
            _write_aselmdb(unknown, {1: _frame([6])})
            frame = _frame([1])
            frame["pbc"] = [True, False, True]
            _write_aselmdb(nonperiodic, {1: frame})

            with self.assertRaisesRegex(ValueError, r"unknown\.aselmdb: frame key 1.*atom"):
                self._dataset([unknown])[0]
            with self.assertRaisesRegex(ValueError, r"nonperiodic\.aselmdb: frame key 1.*periodic"):
                self._dataset([nonperiodic])[0]

    def test_corrupt_payload_and_invalid_shapes_are_rejected_with_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt = Path(tmpdir) / "corrupt.aselmdb"
            invalid = Path(tmpdir) / "invalid.aselmdb"
            _write_aselmdb(corrupt, {1: b"not zlib"})
            frame = _frame([1, 8])
            frame["forces"] = [[0.0, 0.0, 0.0]]
            _write_aselmdb(invalid, {1: frame})

            with self.assertRaisesRegex(ValueError, r"corrupt\.aselmdb: frame key 1"):
                self._dataset([corrupt])[0]
            with self.assertRaisesRegex(ValueError, r"invalid\.aselmdb: frame key 1.*forces"):
                self._dataset([invalid])[0]


if __name__ == "__main__":
    unittest.main()
