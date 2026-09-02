import importlib
import json
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import lmdb
import torch

from src.pre_data.nep_lmdb_dataset import (
    DistributedAtomBatchSampler,
    DistributedFrameBatchSampler,
    NepLmdbDataset,
    parse_lmdb_batch_size,
    select_stat_indices,
)


def _compressed_json(value):
    return zlib.compress(json.dumps(value).encode("utf-8"))


def _frame(natoms):
    return {
        "numbers": [1] * natoms,
        "positions": [[float(index), 0.0, 0.0] for index in range(natoms)],
        "pbc": [True, True, True],
        "cell": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "energy": -2.0 * natoms,
        "forces": [[0.0, 0.0, 0.0]] * natoms,
        "stress": [0.0] * 6,
    }


def _write_aselmdb(path, atom_counts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), subdir=False, map_size=8 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            for row_id, natoms in enumerate(atom_counts, start=1):
                txn.put(str(row_id).encode("ascii"), _compressed_json(_frame(natoms)))
            txn.put(b"nextid", _compressed_json(len(atom_counts) + 1))
    finally:
        env.close()


def _load_network_class():
    fake_findneigh = types.ModuleType("src.feature.nep_find_neigh.findneigh")
    fake_findneigh.FindNeigh = object
    sys.modules.setdefault("src.feature.nep_find_neigh.findneigh", fake_findneigh)
    import src.utils.op_loader as op_loader

    with mock.patch.object(op_loader, "load_calc_ops", return_value=SimpleNamespace()):
        module = importlib.import_module("src.PWMLFF.nep_network")
    return module.nep_network


def _load_network_module():
    _load_network_class()
    return importlib.import_module("src.PWMLFF.nep_network")


def _input_stub(root, train_paths, valid_paths, batch_size):
    return SimpleNamespace(
        file_paths=SimpleNamespace(
            format="lmdb",
            train_data_path=[str(path) for path in train_paths],
            valid_data_path=[str(path) for path in valid_paths],
            test_data_path=[],
            json_dir=str(root),
        ),
        optimizer_param=SimpleNamespace(
            batch_size=batch_size,
            train_bec=False,
            train_bec_ion=False,
            train_ei=False,
        ),
        nep_param=SimpleNamespace(cutoff=[3.0, 3.0]),
        atom_type=[1],
        max_allow_atom_type=-1,
        precision="float64",
        workers=0,
        world_size=1,
        rank=0,
        seed=37,
        data_shuffle=False,
        valid_shuffle=False,
        inference=False,
        multi_gpus=False,
        lmdb_stat_frames=4,
    )


class LmdbBatchSizeParsingTest(unittest.TestCase):
    def test_integer_and_mix_modes(self):
        self.assertEqual(parse_lmdb_batch_size(8), ("frames", 8))
        self.assertEqual(parse_lmdb_batch_size("mix:4096"), ("atoms", 4096))

    def test_invalid_values_are_contextual(self):
        for invalid in (0, -1, True, "8", "mix:0", "mix:-2", "mix:1.5", "bad"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "LMDB.*batch_size"):
                    parse_lmdb_batch_size(invalid)


class NepLmdbLoadDataIntegrationTest(unittest.TestCase):
    def _network(self, input_param):
        network_class = _load_network_class()
        network = object.__new__(network_class)
        network.input_param = input_param
        network.device = torch.device("cpu")
        return network

    def test_integer_batches_return_existing_nep_tensor_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = [root / "train-a.aselmdb", root / "train-b.aselmdb"]
            valid = [root / "valid.aselmdb"]
            _write_aselmdb(train[0], [2, 1])
            _write_aselmdb(train[1], [3, 2])
            _write_aselmdb(valid[0], [1, 2])
            network = self._network(_input_stub(root, train, valid, batch_size=2))

            energy_shift, train_loader, val_loader, stat_loader = network.load_data()

            self.assertIsInstance(train_loader.dataset, NepLmdbDataset)
            self.assertIsInstance(train_loader.batch_sampler, DistributedFrameBatchSampler)
            self.assertIsInstance(val_loader.batch_sampler, DistributedFrameBatchSampler)
            self.assertEqual(energy_shift, [-2.0])
            self.assertEqual(train_loader.dataset.avg_image_atom, 2.0)
            self.assertEqual(train_loader.dataset.max_atom_nums, 3)
            batch = next(iter(train_loader))
            expected_keys = {
                "box",
                "box_original",
                "num_cell",
                "volume",
                "atom_type_map",
                "num_atom",
                "force",
                "ei",
                "energy",
                "fragment",
                "fragment_charge",
                "charge",
                "position",
                "virial",
                "bec",
                "num_atom_sum",
            }
            self.assertEqual(set(batch), expected_keys)
            self.assertEqual(batch["num_atom"].flatten().tolist(), [2, 1])
            self.assertEqual(batch["num_atom_sum"].flatten().tolist(), [2, 3])
            self.assertEqual(batch["position"].shape, (3, 3))
            self.assertEqual(len(stat_loader.batch_sampler.sampler), 4)

    def test_mix_batches_build_cache_and_respect_atom_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = [root / "train.aselmdb"]
            valid = [root / "valid.aselmdb"]
            _write_aselmdb(train[0], [2, 1, 3, 2, 1])
            _write_aselmdb(valid[0], [1, 2])
            network = self._network(
                _input_stub(root, train, valid, batch_size="mix:4")
            )

            _, train_loader, val_loader, _ = network.load_data()

            self.assertIsInstance(train_loader.batch_sampler, DistributedAtomBatchSampler)
            self.assertIsInstance(val_loader.batch_sampler, DistributedAtomBatchSampler)
            self.assertTrue((root / ".matpl_lmdb_cache").is_dir())
            for batch in train_loader:
                self.assertLessEqual(int(batch["num_atom"].sum()), 4)

    def test_float32_is_rejected_before_fixed_double_cuda_descriptor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = [root / "train.aselmdb"]
            _write_aselmdb(train[0], [1, 2])
            input_param = _input_stub(root, train, [], batch_size=1)
            input_param.precision = "float32"
            network = self._network(input_param)

            with self.assertRaisesRegex(ValueError, "float64"):
                network.load_data()

    def test_zero_complete_integer_batches_fail_during_lmdb_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = [root / "train.aselmdb"]
            _write_aselmdb(train[0], [1])
            network = self._network(_input_stub(root, train, [], batch_size=2))

            with self.assertRaisesRegex(ValueError, "LMDB training.*complete batch"):
                network.load_data()

    def test_zero_complete_mix_batches_fail_with_context(self):
        module = _load_network_module()
        sampler = DistributedAtomBatchSampler(
            [2, 2], atom_budget=4, rank=0, world_size=2, shuffle=False
        )
        self.assertEqual(len(sampler), 0)

        with self.assertRaisesRegex(ValueError, "mix:4.*2 ranks"):
            module._require_lmdb_training_batches(
                sampler,
                dataset_size=2,
                batch_mode="atoms",
                batch_value=4,
                world_size=2,
            )

    def test_empty_multirank_stat_slice_uses_reduction_identities(self):
        module = _load_network_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.aselmdb"
            _write_aselmdb(path, [1] * 8)
            dataset = NepLmdbDataset(
                [str(path)],
                atom_types=[1],
                cutoff_radial=3.0,
                cutoff_angular=3.0,
                dtype=torch.float64,
            )
            indices = select_stat_indices(
                len(dataset), requested=1, rank=3, world_size=4, seed=37
            )
            self.assertEqual(indices, [])
            stat_loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=torch.utils.data.BatchSampler(
                    indices, batch_size=1, drop_last=False
                ),
            )

            result = module._calculate_lmdb_neighbor_scaler(
                stat_loader, 4, 8, 4, 8, 4, 2, 1, torch.device("cpu")
            )

            local_max, local_min, max_radial, min_radial, max_angular, min_angular = result
            self.assertEqual(local_max.shape, (35,))
            self.assertTrue(torch.isneginf(local_max).all())
            self.assertTrue(torch.isposinf(local_min).all())
            self.assertEqual(
                (max_radial, min_radial, max_angular, min_angular),
                (0, 0, 0, 0),
            )
            dataset.close()

    def test_epoch_is_forwarded_to_custom_and_legacy_samplers(self):
        module = _load_network_module()
        custom_sampler = mock.Mock()
        module._set_data_loader_epoch(
            SimpleNamespace(batch_sampler=custom_sampler, sampler=None), 7
        )
        custom_sampler.set_epoch.assert_called_once_with(7)

        dataset = torch.utils.data.TensorDataset(torch.arange(4))
        legacy_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=1, rank=0
        )
        legacy_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, sampler=legacy_sampler
        )
        module._set_data_loader_epoch(legacy_loader, 9)
        self.assertEqual(legacy_sampler.epoch, 9)


if __name__ == "__main__":
    unittest.main()
