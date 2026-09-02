import itertools
import json
import os
import tempfile
import unittest
import zlib
from pathlib import Path

import lmdb
import numpy as np

from src.pre_data.nep_lmdb_dataset import (
    DistributedAtomBatchSampler,
    LmdbNatomsCache,
    NepLmdbDataset,
)


def _compressed_json(value):
    return zlib.compress(json.dumps(value).encode("utf-8"))


def _frame(natoms):
    return {
        "numbers": [1] * natoms,
        "positions": [[0.0, 0.0, 0.0]] * natoms,
        "pbc": [True, True, True],
        "cell": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        "energy": -float(natoms),
        "forces": [[0.0, 0.0, 0.0]] * natoms,
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


def _flatten(batches):
    return list(itertools.chain.from_iterable(batches))


class LmdbNatomsCacheTest(unittest.TestCase):
    def _dataset(self, paths):
        return NepLmdbDataset(
            [str(path) for path in paths],
            atom_types=[1],
            cutoff_radial=3.0,
            cutoff_angular=3.0,
        )

    def test_builds_atomic_int32_sidecars_and_memory_maps_them(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shards = [root / "a.aselmdb", root / "b.aselmdb"]
            _write_aselmdb(shards[0], [2, 1])
            _write_aselmdb(shards[1], [3, 4, 2])
            cache_dir = root / "cache"
            cache = LmdbNatomsCache(self._dataset(shards), cache_dir)

            cache.build_assigned(rank=0, world_size=1)
            cache.load()

            self.assertEqual(len(cache), 5)
            self.assertEqual([cache[index] for index in range(5)], [2, 1, 3, 4, 2])
            self.assertEqual(len(list(cache_dir.glob("*.i32"))), 2)
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 2)
            self.assertEqual(list(cache_dir.glob("*.tmp-*")), [])
            self.assertTrue(all(isinstance(array, np.memmap) for array in cache._arrays))

    def test_reuses_valid_sidecars_and_invalidates_source_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = root / "reuse.aselmdb"
            _write_aselmdb(shard, [2, 5])
            cache_dir = root / "cache"
            dataset = self._dataset([shard])
            first = LmdbNatomsCache(dataset, cache_dir)
            first.build_assigned(0, 1)
            data_path = next(cache_dir.glob("*.i32"))
            manifest_path = next(cache_dir.glob("*.json"))
            initial_data_mtime = data_path.stat().st_mtime_ns
            initial_manifest = json.loads(manifest_path.read_text())

            reused = LmdbNatomsCache(self._dataset([shard]), cache_dir)
            reused.build_assigned(0, 1)
            self.assertEqual(data_path.stat().st_mtime_ns, initial_data_mtime)
            self.assertEqual(json.loads(manifest_path.read_text()), initial_manifest)

            source_stat = shard.stat()
            changed_mtime = source_stat.st_mtime_ns + 2_000_000_000
            os.utime(shard, ns=(source_stat.st_atime_ns, changed_mtime))
            rebuilt = LmdbNatomsCache(self._dataset([shard]), cache_dir)
            rebuilt.build_assigned(0, 1)
            rebuilt_manifest = json.loads(manifest_path.read_text())
            self.assertEqual(rebuilt_manifest["source_mtime_ns"], changed_mtime)
            self.assertNotEqual(rebuilt_manifest, initial_manifest)

    def test_nextid_change_invalidates_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = root / "growing.aselmdb"
            _write_aselmdb(shard, [1])
            cache_dir = root / "cache"
            first = LmdbNatomsCache(self._dataset([shard]), cache_dir)
            first.build_assigned(0, 1)
            first.load()
            self.assertEqual(list(first), [1])
            first.close()

            env = lmdb.open(str(shard), subdir=False, map_size=8 * 1024 * 1024)
            try:
                with env.begin(write=True) as txn:
                    txn.put(b"2", _compressed_json(_frame(3)))
                    txn.put(b"nextid", _compressed_json(3))
            finally:
                env.close()

            second = LmdbNatomsCache(self._dataset([shard]), cache_dir)
            second.build_assigned(0, 1)
            second.load()
            self.assertEqual(list(second), [1, 3])

    def test_rank_builds_only_assigned_shards_until_barrier_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shards = [root / ("{}.aselmdb".format(index)) for index in range(4)]
            for index, shard in enumerate(shards):
                _write_aselmdb(shard, [index + 1])
            cache_dir = root / "cache"
            dataset = self._dataset(shards)
            left = LmdbNatomsCache(dataset, cache_dir)
            right = LmdbNatomsCache(dataset, cache_dir)

            left.build_assigned(rank=0, world_size=2)
            self.assertEqual(len(list(cache_dir.glob("*.i32"))), 2)
            right.build_assigned(rank=1, world_size=2)
            self.assertEqual(len(list(cache_dir.glob("*.i32"))), 4)
            left.load()
            self.assertEqual(list(left), [1, 2, 3, 4])


class DistributedAtomBatchSamplerTest(unittest.TestCase):
    def test_batches_respect_budget_except_oversized_single_frames(self):
        natoms = [2, 3, 4, 7, 1, 12, 2, 2, 5]
        sampler = DistributedAtomBatchSampler(
            natoms, atom_budget=10, rank=0, world_size=1, seed=5, shuffle=False
        )

        batches = list(sampler)

        self.assertEqual(_flatten(batches), list(range(len(natoms))))
        for batch in batches:
            atom_count = sum(natoms[index] for index in batch)
            self.assertTrue(atom_count <= 10 or (len(batch) == 1 and atom_count > 10))
        self.assertEqual(len(sampler), len(batches))

    def test_multi_rank_batches_have_equal_steps_and_disjoint_frames(self):
        natoms = [(index % 7) + 1 for index in range(103)]
        samplers = [
            DistributedAtomBatchSampler(
                natoms,
                atom_budget=13,
                rank=rank,
                world_size=4,
                seed=17,
                shuffle=True,
                block_size=11,
            )
            for rank in range(4)
        ]
        batches = [list(sampler) for sampler in samplers]
        indices = [_flatten(rank_batches) for rank_batches in batches]

        self.assertEqual(len({len(rank_batches) for rank_batches in batches}), 1)
        self.assertTrue(all(len(sampler) == len(rank_batches) for sampler, rank_batches in zip(samplers, batches)))
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertTrue(set(indices[left]).isdisjoint(indices[right]))
        self.assertLessEqual(max(sampler.peak_completed_batches for sampler in samplers), 4)

    def test_epoch_order_is_deterministic_and_changes(self):
        natoms = [1 + index % 5 for index in range(80)]
        first = DistributedAtomBatchSampler(natoms, 10, 0, 2, seed=21, block_size=9)
        repeated = DistributedAtomBatchSampler(natoms, 10, 0, 2, seed=21, block_size=9)
        self.assertEqual(list(first), list(repeated))

        first.set_epoch(2)
        repeated.set_epoch(2)
        epoch_two = list(first)
        self.assertEqual(epoch_two, list(repeated))
        first.set_epoch(3)
        self.assertNotEqual(epoch_two, list(first))

    def test_incomplete_final_rank_group_is_dropped_without_padding(self):
        natoms = [6, 6, 6, 6, 6]
        samplers = [
            DistributedAtomBatchSampler(
                natoms, atom_budget=5, rank=rank, world_size=4, shuffle=False
            )
            for rank in range(4)
        ]
        rank_batches = [list(sampler) for sampler in samplers]

        self.assertEqual([len(batches) for batches in rank_batches], [1, 1, 1, 1])
        self.assertEqual(
            set(_flatten(rank_batches[0] + rank_batches[1] + rank_batches[2] + rank_batches[3])),
            {0, 1, 2, 3},
        )


if __name__ == "__main__":
    unittest.main()
