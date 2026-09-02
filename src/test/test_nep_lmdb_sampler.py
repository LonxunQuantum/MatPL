import itertools
import unittest

from src.pre_data.nep_lmdb_dataset import (
    BlockShuffleIndices,
    DistributedFrameBatchSampler,
)


def _flatten(batches):
    return list(itertools.chain.from_iterable(batches))


class BlockShuffleIndicesTest(unittest.TestCase):
    def test_seed_and_epoch_are_deterministic_and_cover_all_indices(self):
        first = list(BlockShuffleIndices(101, 13, seed=23, epoch=4, shuffle=True))
        repeated = list(BlockShuffleIndices(101, 13, seed=23, epoch=4, shuffle=True))
        next_epoch = list(BlockShuffleIndices(101, 13, seed=23, epoch=5, shuffle=True))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(sorted(first), list(range(101)))

    def test_validation_order_is_ascending(self):
        indices = BlockShuffleIndices(37, 8, seed=7, epoch=9, shuffle=False)
        self.assertEqual(list(indices), list(range(37)))

    def test_only_one_index_block_is_buffered(self):
        indices = BlockShuffleIndices(1003, 31, seed=7, epoch=0, shuffle=True)
        self.assertEqual(len(list(indices)), 1003)
        self.assertLessEqual(indices.peak_buffered, 31)


class DistributedFrameBatchSamplerTest(unittest.TestCase):
    def test_multi_rank_batches_are_equal_disjoint_and_globally_complete(self):
        dataset_size = 53
        batch_size = 3
        world_size = 4
        samplers = [
            DistributedFrameBatchSampler(
                dataset_size,
                batch_size,
                rank,
                world_size,
                seed=19,
                shuffle=True,
                block_size=7,
            )
            for rank in range(world_size)
        ]
        rank_batches = [list(sampler) for sampler in samplers]
        rank_indices = [_flatten(batches) for batches in rank_batches]

        self.assertEqual([len(batches) for batches in rank_batches], [4, 4, 4, 4])
        self.assertTrue(
            all(len(batch) == batch_size for batches in rank_batches for batch in batches)
        )
        for left in range(world_size):
            for right in range(left + 1, world_size):
                self.assertTrue(set(rank_indices[left]).isdisjoint(rank_indices[right]))

        retained = dataset_size // (batch_size * world_size) * batch_size * world_size
        expected_global = list(
            itertools.islice(
                BlockShuffleIndices(
                    dataset_size, 7, seed=19, epoch=0, shuffle=True
                ),
                retained,
            )
        )
        self.assertEqual(
            set(itertools.chain.from_iterable(rank_indices)), set(expected_global)
        )
        self.assertEqual(len(samplers[0]), 4)

    def test_epoch_changes_global_shuffle_consistently_across_ranks(self):
        samplers = [
            DistributedFrameBatchSampler(80, 4, rank, 2, seed=31, block_size=9)
            for rank in range(2)
        ]
        epoch_zero = [_flatten(list(sampler)) for sampler in samplers]
        for sampler in samplers:
            sampler.set_epoch(1)
        epoch_one = [_flatten(list(sampler)) for sampler in samplers]

        self.assertNotEqual(epoch_zero, epoch_one)
        self.assertEqual(len(set(epoch_one[0]) & set(epoch_one[1])), 0)
        self.assertEqual(set(epoch_one[0]) | set(epoch_one[1]), set(range(80)))

    def test_nonshuffle_rank_orders_are_ascending_and_tail_is_not_padded(self):
        samplers = [
            DistributedFrameBatchSampler(
                29, 2, rank, 4, seed=3, shuffle=False, block_size=5
            )
            for rank in range(4)
        ]
        rank_indices = [_flatten(list(sampler)) for sampler in samplers]

        self.assertEqual([len(indices) for indices in rank_indices], [6, 6, 6, 6])
        self.assertTrue(all(indices == sorted(indices) for indices in rank_indices))
        union = set().union(*(set(indices) for indices in rank_indices))
        self.assertEqual(union, set(range(24)))
        self.assertNotIn(24, union)

    def test_buffers_are_bounded_by_one_block_plus_one_super_batch(self):
        sampler = DistributedFrameBatchSampler(
            1001, 5, rank=1, world_size=4, seed=9, block_size=17
        )
        list(sampler)
        self.assertLessEqual(sampler.peak_buffered, 17 + 5 * 4)

    def test_invalid_distributed_arguments_are_rejected(self):
        invalid = [
            {"dataset_size": -1, "batch_size": 1, "rank": 0, "world_size": 1},
            {"dataset_size": 4, "batch_size": 0, "rank": 0, "world_size": 1},
            {"dataset_size": 4, "batch_size": True, "rank": 0, "world_size": 1},
            {"dataset_size": 4, "batch_size": 1, "rank": 1, "world_size": 1},
            {"dataset_size": 4, "batch_size": 1, "rank": 0, "world_size": 0},
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    DistributedFrameBatchSampler(**arguments)


if __name__ == "__main__":
    unittest.main()
